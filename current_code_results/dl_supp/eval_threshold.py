#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-evaluate saved best checkpoints with dual thresholds (0.5 vs Youden).
Detects 0.5-collapse seeds and reports mean±SD per model/dataset.
Usage: python eval_threshold.py [--model subset...]"""
import os, glob, json, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as te

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(HERE, "saved")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def val_loader(ds, seed):
    items = te.list_images(ds)
    lab = np.array([y for _, y in items])
    idx = np.arange(len(items))
    from sklearn.model_selection import train_test_split
    _, te_i = train_test_split(idx, test_size=0.3, random_state=seed, stratify=lab)
    test_items = [items[i] for i in te_i]
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return DataLoader(te.ImgDS(test_items, tf), batch_size=64, shuffle=False,
                      num_workers=0)


def metrics(y, p, th):
    pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sen = tp / (tp + fn) if tp + fn else 0.0
    spe = tn / (tn + fp) if tn + fp else 0.0
    acc = accuracy_score(y, pred)
    return acc, sen, spe


def eval_ckpt(path):
    fname = os.path.basename(path)          # {ds}_{model}_seed{seed}.pt
    ds, rest = fname.split("_", 1)
    model = rest.rsplit("_seed", 1)[0]
    seed = int(rest.rsplit("_seed", 1)[1].split(".")[0])
    ck = torch.load(path, map_location="cpu")
    m = te.build_model(model).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    ys, pr = [], []
    dl = val_loader(ds, seed)
    with torch.no_grad():
        for x, y in dl:
            with torch.cuda.amp.autocast():
                pr.append(m(x.to(DEV)).softmax(-1)[:, 1].cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(pr)
    auc = roc_auc_score(y, p)
    a05, s05, p05 = metrics(y, p, 0.5)
    fpr, tpr, ths = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    topt = float(ths[i])
    ay, sy, py_ = metrics(y, p, topt)
    coll = (s05 < 0.10) or (p05 < 0.10)
    return dict(ds=ds, model=model, seed=seed, auc=auc,
                acc05=a05, sen05=s05, spe05=p05,
                accY=ay, senY=sy, speY=py_, topt=topt, collapse05=bool(coll),
                rescued=(bool(coll) and sy >= 0.60 and py_ >= 0.60))


def main():
    files = sorted(glob.glob(os.path.join(SAVE, "*.pt")))
    rows = []
    for f in files:
        try:
            rows.append(eval_ckpt(f))
        except Exception as e:
            print("skip", os.path.basename(f), e)
    print(f"evaluated {len(rows)} checkpoints")
    # group
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["ds"])].append(r)
    print("\n=== per-seed (0.5 | Youden) ===")
    for (m, ds), rr in sorted(g.items()):
        for r in rr:
            mark = "  ⚠0.5-COLLAPSE" if r["collapse05"] else ""
            rescued = " ->Youden rescued" if r.get("rescued") else ""
            print(f"{m:16s} {ds:9s} s{r['seed']:<4} AUC {r['auc']:.3f} | "
                  f"0.5: Acc{r['acc05']:.2f} Sen{r['sen05']:.2f} Spe{r['spe05']:.2f} | "
                  f"Youden(T={r['topt']:.2f}): Acc{r['accY']:.2f} Sen{r['senY']:.2f} "
                  f"Spe{r['speY']:.2f}{mark}{rescued}")
    print("\n=== mean±SD per model/dataset ===")
    for (m, ds), rr in sorted(g.items()):
        def stat(k):
            a = np.array([r[k] for r in rr])
            return f"{a.mean():.3f}±{a.std():.3f}"
        print(f"{m:16s} {ds:9s} n={len(rr)} AUC {stat('auc')} | "
              f"0.5 Acc{stat('acc05')} Sen{stat('sen05')} Spe{stat('spe05')} | "
              f"Youden Acc{stat('accY')} Sen{stat('senY')} Spe{stat('speY')} "
              f"Topt {stat('topt')}")
    # summary json
    json.dump(rows, open(os.path.join(HERE, "eval_threshold_results.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()

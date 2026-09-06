#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DL baseline train/eval on B-mode US images (grayscale, 2-class).
Datasets:
  internal : dataset/output_expansion/{benign,magnant}_1mm/original  (546)
  busi     : dataset/BUSI/Dataset_BUSI_with_GT/{benign,magnant}
Usage:
  python train_eval.py --dataset internal --model resnet18 --seed 42
Metrics (val split): acc/sen/spe/auc.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image

DS = r"C:\Users\ZhangErHao\Desktop\论文\dataset"


def list_images(ds):
    items = []
    if ds == "internal":
        for lab, cls in [("benign", "benign"), ("malignant", "malignant")]:
            d = os.path.join(DS, "output_expansion", f"{cls}_1mm", "original")
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    items.append((os.path.join(d, f), 0 if lab == "benign" else 1))
    else:
        for lab, cls in [("benign", "benign"), ("malignant", "malignant")]:
            d = os.path.join(DS, "BUSI", "Dataset_BUSI_with_GT", cls)
            for f in os.listdir(d):
                if f.lower().endswith(".png") and "_mask" not in f.lower():
                    items.append((os.path.join(d, f), 0 if lab == "benign" else 1))
    return items


class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.tf = tf
        # decode once into memory (dataset is small)
        self.arrs = [np.asarray(Image.open(p).convert("RGB")) for p, _ in items]
        self.ys = [y for _, y in items]

    def __len__(self):
        return len(self.arrs)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.arrs[i])), self.ys[i]


def build_model(name, nclass=2):
    if name in ("resnet18", "resnet50"):
        m = getattr(models, name)(weights=None)
        state = models.ResNet18_Weights.IMAGENET1K_V1 if name == "resnet18" else models.ResNet50_Weights.IMAGENET1K_V1
        m.load_state_dict(state.get_state_dict(progress=False))
        m.fc = nn.Linear(m.fc.in_features, nclass)
        return m
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        st = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        m.load_state_dict(st.get_state_dict(progress=False))
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, nclass)
        return m
    if name.startswith("medvit"):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "vendor", "MedViT"))
        from MedViT import MedViT_small, MedViT_base, MedViT_large
        fac = {"medvit_small": MedViT_small, "medvit_base": MedViT_base,
               "medvit_large": MedViT_large}[name]
        model = fac(num_classes=nclass, pretrained=False)
        pth = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pretrained",
                           {"medvit_small":"w1.pth","medvit_base":"w2.pth",
                            "medvit_large":"w3.pth"}[name])
        if os.path.exists(pth):
            ck = torch.load(pth, map_location="cpu")
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            sd = {k: v for k, v in sd.items() if not k.startswith("proj_head")}
            res = model.load_state_dict(sd, strict=False)
            print("medvit pretrained loaded; missing", len(res.missing_keys),
                  "unexpected", len(res.unexpected_keys), flush=True)
        else:
            print("medvit pretrained NOT found:", pth, flush=True)
        return model
    raise NotImplementedError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["internal", "busi"])
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resize224", action="store_true",
                    help="scheme B: direct resize to 224 (no center crop)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--savedir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)

    items = list_images(a.dataset)
    print(f"{a.dataset}: {len(items)} items", flush=True)
    lab = np.array([y for _, y in items])
    idx = np.arange(len(items))
    tr, te = train_test_split(idx, test_size=0.3, random_state=a.seed,
                              stratify=lab)
    train_items = [items[i] for i in tr]
    test_items = [items[i] for i in te]

    if a.resize224:
        tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        train_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    else:
        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    dl_tr = DataLoader(ImgDS(train_items, train_tf), batch_size=a.bs,
                       shuffle=True, num_workers=0, pin_memory=False)
    dl_te = DataLoader(ImgDS(test_items, tf), batch_size=a.bs, shuffle=False,
                       num_workers=0, pin_memory=False)

    model = build_model(a.model).to(dev)
    if a.compile:
        try:
            model = torch.compile(model)
            print("compile ok", flush=True)
        except Exception as e:
            print("compile failed", e, flush=True)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(a.amp))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def eval_():
        model.eval(); ys, pr = [], []
        with torch.no_grad():
            for x, y in dl_te:
                with torch.cuda.amp.autocast(enabled=bool(a.amp)):
                    p = model(x.to(dev)).softmax(-1).cpu()
                ys.append(y.numpy()); pr.append(p.numpy())
        y = np.concatenate(ys); p = np.concatenate(pr)
        pred = p.argmax(1)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        auc = roc_auc_score(y, p[:, 1])
        return dict(acc=float(accuracy_score(y, pred)), sen=float(tp/(tp+fn)),
                    spe=float(tn/(tn+fp)), auc=float(auc), n=int(len(y)))

    os.makedirs(a.savedir, exist_ok=True)
    ckpt_path = os.path.join(a.savedir, f"{a.dataset}_{a.model}_seed{a.seed}.pt")
    t0 = time.time()
    best = None
    hist = []
    for ep in range(a.epochs):
        model.train()
        for x, y in dl_tr:
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(a.amp)):
                loss = crit(model(x.to(dev)), y.to(dev))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        sched.step()
        m = eval_()
        print(f"ep{ep+1} {m}", flush=True)
        hist.append({"epoch": ep + 1, **m})
        if best is None or m["auc"] > best["auc"]:
            best = m; best["best_epoch"] = ep + 1
            torch.save({"state_dict": model.state_dict(), "meta": best},
                       ckpt_path)
    best["dataset"] = a.dataset; best["model"] = a.model; best["seed"] = a.seed
    best["time_s"] = round(time.time() - t0, 1)
    hpath = a.out.replace(".json", "_hist.json")
    json.dump(hist, open(hpath, "w"))
    bi = int(np.argmax([h["auc"] for h in hist]))
    patience, pi, cnt = 3, 0, 0
    for i, h in enumerate(hist):
        if h["auc"] > hist[pi]["auc"]:
            pi = i; cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break
    print("EVAL best=%.4f@ep%d final=%.4f@ep%d patience=%.4f@ep%d scheme=%s" % (
        hist[bi]["auc"], hist[bi]["epoch"], hist[-1]["auc"], hist[-1]["epoch"],
        hist[pi]["auc"], hist[pi]["epoch"], "resize224" if a.resize224 else "centercrop"), flush=True)

    with open(a.out, "w") as f:
        json.dump(best, f)
    print("BEST", best)


if __name__ == "__main__":
    main()

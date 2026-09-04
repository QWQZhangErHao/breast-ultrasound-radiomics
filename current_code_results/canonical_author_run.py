#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANONICAL author-config run (mirrors released run_1mm_pipeline_optimized.py):
  LASSO tol=1e-4 / iter=1e4 / cv5 ; LR=saga ; XGB hist depth3 lr.1 n=100 ;
  RF 500/depth5 ; SVM lin C1 ; KNN5 ; seeds 42..4096 ; 70/30 stratified.

scale  : cumulative 0..4 mm grayscale (repo feature build), 6 metrics
fusion : 1mm grayscale base + {VFA,3 opt,13 stage4,21 bf_*}, AUC
suppl  : XGB base vs +3 -> NRI, Brier, Eavg, DCA, VFA0/VFA>0 AUC  (per seed)
stats  : scale 1 vs 0/2 paired; adjacent fusion pairs (t & Wilcoxon, delta/CI)

All per-seed values saved -> canonical_author_run.pkl
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, re, time, pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, accuracy_score, brier_score_loss,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from joblib import Parallel, delayed

DATASET = r"C:\Users\ZhangErHao\Desktop\论文\dataset"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
MODELS = ["SVM", "LR", "RF", "XGBoost", "KNN"]
PID, LAB = "patient_name", "label"
D3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]
WORKERS = 16
BUS = os.path.join(DATASET, "BUS_features")
FLOW = os.path.join(DATASET, "flow_features")


def sid(r):
    m = re.search(r"(\d+)", str(r))
    return int(m.group(1)) if m else None


def is_feat(c):
    return not any(p in c.lower() for p in ["patient_name", "file_name", "diagnostics_"])


def load_bus(mm):
    b = pd.read_csv(os.path.join(BUS, f"benign_{mm}mm.csv"))
    m = pd.read_csv(os.path.join(BUS, f"malignant_{mm}mm.csv"))
    b[LAB], m[LAB] = 0, 1
    df = pd.concat([b, m], ignore_index=True)
    df[PID] = df[PID].apply(sid)
    return df


def nonfeat(df):
    return df[[c for c in df.columns if c in (PID, LAB) or is_feat(c)]]


def suffix(df, k):
    return df.rename(columns={c: c if c in (PID, LAB) else f"{c}_{k}mm"
                              for c in df.columns})


def cumulative(k):
    d0 = nonfeat(load_bus(0))
    if k == 0:
        feats = [c for c in d0.columns if c not in (PID, LAB)]
        return d0, feats
    dk = suffix(nonfeat(load_bus(k)), k)
    dk = dk.drop(columns=[c for c in dk.columns
                          if "shape" in c.lower() and c not in (PID, LAB)],
                 errors="ignore")
    out = d0.merge(dk, on=[PID, LAB], how="inner")
    return out, [c for c in out.columns if c not in (PID, LAB)]


def build_fusion_base():
    df1, gray = cumulative(1)
    bf = pd.concat([pd.read_csv(os.path.join(FLOW, "blood_flow_features_benign.csv")),
                    pd.read_csv(os.path.join(FLOW, "blood_flow_features_malignant.csv"))],
                   ignore_index=True)
    bf = bf.rename(columns={"image_id": PID})
    bf[PID] = bf[PID].apply(sid)
    bf = bf[[PID] + [c for c in bf.columns if c.startswith("bf_")]]
    vfa = pd.concat([pd.read_csv(os.path.join(FLOW, "benign_flow_density.csv")),
                     pd.read_csv(os.path.join(FLOW, "malignant_flow_density.csv"))],
                    ignore_index=True).rename(columns={"case_id": PID})
    vfa = vfa[["flow_density", PID]]
    df = df1.merge(bf, on=PID, how="inner").merge(vfa, on=PID, how="inner")
    df[LAB] = df1.set_index(PID)[LAB].reindex(df[PID]).to_numpy()
    df = df[df[LAB].notna()].reset_index(drop=True)
    df[LAB] = df[LAB].astype(int)
    return df, gray, [c for c in df.columns if c.startswith("bf_")]


def lasso_fit(Xtr, ytr):
    s = StandardScaler(); Xs = s.fit_transform(Xtr)
    l = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=1, tol=1e-4)
    l.fit(Xs, ytr)
    mask = l.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(l.coef_))[-10:]] = True
    return s, mask


def clf(name):
    return {"SVM": lambda: SVC(kernel="linear", C=1, probability=True, random_state=42),
            "LR": lambda: LogisticRegression(penalty="l2", C=1, solver="saga",
                                             max_iter=10000, random_state=42, n_jobs=1),
            "RF": lambda: RandomForestClassifier(n_estimators=500, max_depth=5,
                                                 random_state=42, n_jobs=1),
            "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                                 learning_rate=0.1, random_state=42,
                                                 verbosity=0, n_jobs=1, tree_method="hist"),
            "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean",
                                                n_jobs=1)}[name]()


def met(y, p):
    pr = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr).ravel()
    sen = tp / (tp + fn) if tp + fn else 0.0
    spe = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"acc": accuracy_score(y, pr), "sen": sen, "spe": spe, "prec": prec,
            "f1": (2 * prec * sen / (prec + sen)) if prec + sen else 0.0,
            "auc": roc_auc_score(y, p)}


def one(key, seed, X, y, keep_prob):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                              random_state=seed, stratify=y)
    sc, mask = lasso_fit(X[tr], y[tr])
    Xt = sc.transform(X[tr])[:, mask]; Xe = sc.transform(X[te])[:, mask]
    res = {}
    for n in MODELS:
        c = clf(n); c.fit(Xt, y[tr])
        p = c.predict_proba(Xe)[:, 1]
        res[n] = met(y[te], p)
        if keep_prob and n == "XGBoost":
            res["px"] = p
            res["yte"] = y[te].copy()
            res["te"] = te
    return key, seed, res


def ci_boot(d, rng=None):
    rng = rng or np.random.default_rng(0)
    b = rng.choice(d, (5000, len(d)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    t0 = time.time()
    DATA = {}
    for k in range(5):
        df, feats = cumulative(k)
        DATA[f"scale{k}"] = (df[feats].to_numpy(), df[LAB].to_numpy().ravel())
        print(f"scale{k}: n={len(df)} feats={len(feats)}", flush=True)
    dff, gray, bf_all = build_fusion_base()
    yf = dff[LAB].to_numpy().ravel()
    sel_pkl = os.path.join(os.path.dirname(OUT), "重新验证后的论文数据",
                           "doppler_selection_results.pkl")
    with open(sel_pkl, "rb") as f:
        sel13 = list(dict.fromkeys(pickle.load(f)["stage4_passed"]))
    fus = {"fus_base": gray,
           "fus_vfa": gray + ["flow_density"],
           "fus_p3": gray + [c for c in D3 if c in dff.columns],
           "fus_p13": gray + sel13,
           "fus_p21": gray + bf_all}
    for k, cols in fus.items():
        DATA[k] = (dff[cols].to_numpy(), yf)
    v0 = (dff["flow_density"].to_numpy(dtype=float) == 0)
    print("fusion feats:", {k: len(c) for k, c in fus.items()}, flush=True)

    keys = list(DATA.keys())
    jobs = [(k, s) for k in keys for s in SEEDS]
    print(f"{len(jobs)} jobs on {WORKERS} workers", flush=True)
    res = Parallel(n_jobs=WORKERS)(
        delayed(one)(k, s, DATA[k][0], DATA[k][1],
                     k in ("fus_base", "fus_p3"))
        for k, s in jobs)

    # assemble
    aucs = {k: {m: [] for m in MODELS} for k in keys}
    full = {k: {m: {q: [] for q in ["acc", "sen", "spe", "prec", "f1", "auc"]}
                for m in MODELS} for k in keys}
    prob = {s: {"base": None, "p3": None, "yte": None, "te": None} for s in SEEDS}
    for k, s, r in res:
        for m in MODELS:
            aucs[k][m].append(r[m]["auc"])
            for q in full[k][m]:
                full[k][m][q].append(r[m][q])
        if "px" in r:
            prob[s][("base" if k == "fus_base" else "p3")] = r["px"]
            prob[s]["yte"] = r["yte"]; prob[s]["te"] = r["te"]
    A = {k: {m: np.array(aucs[k][m]) for m in MODELS} for k in keys}

    def show_auc(k):
        return " ".join(f"{A[k][m].mean():.3f}±{A[k][m].std():.3f}" for m in MODELS)

    print("\n=== scale AUC ===")
    for k in [f"scale{i}" for i in range(5)]:
        print(f"{k:8s} | {show_auc(k)}")
    print("\n=== fusion AUC ===")
    for k, tag in [("fus_base", "base(1mm gray)"), ("fus_vfa", "+VFA"),
                   ("fus_p3", "+3"), ("fus_p13", "+13"), ("fus_p21", "+21")]:
        print(f"{tag:16s} | {show_auc(k)}")

    # paired stats
    print("\n=== paired stats (XGBoost) ===")
    pair_stats = {}
    def pair(a_key, b_key, lbl):
        d = A[b_key]["XGBoost"] - A[a_key]["XGBoost"]
        md = d.mean(); sd = d.std(ddof=1); lo, hi = ci_boot(d)
        _, pt = stats.ttest_rel(A[a_key]["XGBoost"], A[b_key]["XGBoost"])
        try:
            _, pw = stats.wilcoxon(A[a_key]["XGBoost"], A[b_key]["XGBoost"])
        except ValueError:
            pw = float("nan")
        pair_stats[lbl] = dict(delta=float(md), sd=float(sd), ci=[lo, hi],
                               p_t=float(pt), p_w=float(pw))
        print(f"  {lbl}: Δ={md:+.4f} [CI {lo:+.4f},{hi:+.4f}] t-p={pt:.4f} w-p={pw:.4f}")
    pair("scale0", "scale1", "1mm-0mm"); pair("scale1", "scale2", "1mm-2mm")
    pair("fus_base", "fus_vfa", "base→+VFA"); pair("fus_base", "fus_p3", "base→+3")
    pair("fus_p3", "fus_p13", "+3→+13"); pair("fus_p3", "fus_p21", "+3→+21")
    pair("fus_p13", "fus_p21", "+13→+21")

    # supplementary (XGB base vs p3)
    nb = 10; edges = np.linspace(0, 1, nb + 1); cents = (edges[:-1] + edges[1:]) / 2
    nris, brG, brM, egG, egM, auc0m, auc1m = ([] for _ in range(7))
    ev, nev, dca_mg = [], [], []
    ths = np.arange(0.01, 1.0, 0.01)
    mg = [[] for _ in ths]
    for s in SEEDS:
        yte = prob[s]["yte"]; pg = prob[s]["base"]; pm = prob[s]["p3"]
        nris.append(float(np.sum(pm[yte == 1] > pg[yte == 1]) -
                          np.sum(pm[yte == 1] < pg[yte == 1])) / max(np.sum(yte == 1), 1) +
                    float(np.sum(pg[yte == 0] > pm[yte == 0]) -
                          np.sum(pg[yte == 0] < pm[yte == 0])) / max(np.sum(yte == 0), 1))
        brG.append(brier_score_loss(yte, pg)); brM.append(brier_score_loss(yte, pm))
        def eavg(p):
            fr = []
            for i in range(nb):
                m = (p >= edges[i]) & (p < edges[i + 1])
                if i == nb - 1: m = (p >= edges[i]) & (p <= edges[i + 1])
                fr.append(yte[m].mean() if m.sum() else np.nan)
            return np.nanmean(np.abs(np.array(fr) - cents))
        egG.append(eavg(pg)); egM.append(eavg(pm))
        ev.append(float(np.sum(pm[yte == 1] > pg[yte == 1]) -
                        np.sum(pm[yte == 1] < pg[yte == 1])) / max(np.sum(yte == 1), 1))
        nev.append(float(np.sum(pg[yte == 0] > pm[yte == 0]) -
                         np.sum(pg[yte == 0] < pm[yte == 0])) / max(np.sum(yte == 0), 1))
        for i, pt in enumerate(ths):
            def nbv(p, th):
                b = (p >= th).astype(int)
                return np.sum((b == 1) & (yte == 1)) / len(yte) - \
                       np.sum((b == 1) & (yte == 0)) / len(yte) * th / (1 - th)
            mg[i].append(nbv(pm, pt) - nbv(pg, pt))
        m0, m1 = v0[prob[s]["te"]], ~v0[prob[s]["te"]]
        if yte[m0].sum() not in (0, m0.sum()): auc0m.append(roc_auc_score(yte[m0], pm[m0]))
        if yte[m1].sum() not in (0, m1.sum()): auc1m.append(roc_auc_score(yte[m1], pm[m1]))
    mg_mean = np.array(mg).mean(1)
    ben = ths[mg_mean > 0.001]
    nri_a = np.array(nris); _, p_nri = stats.ttest_1samp(nri_a, 0)
    _, p_br = stats.ttest_rel(brG, brM)
    suppl = dict(nri=(float(nri_a.mean()), float(nri_a.std()), float(p_nri)),
                 nri_event=float(np.mean(ev)), nri_nonevent=float(np.mean(nev)),
                 brier=((float(np.mean(brG)), float(np.std(brG))),
                        (float(np.mean(brM)), float(np.std(brM))), float(p_br)),
                 eavg=(float(np.mean(egG)), float(np.mean(egM))),
                 dca_range=[float(ben[0]), float(ben[-1])] if len(ben) else [],
                 sub_v0_auc=(float(np.mean(auc0m)), float(np.std(auc0m))),
                 sub_v1_auc=(float(np.mean(auc1m)), float(np.std(auc1m))))
    print("\n=== supplementary (XGB base vs +3) ===")
    print(f"  NRI {nri_a.mean():.4f}±{nri_a.std():.4f} p={p_nri:.4f} | "
          f"event={np.mean(ev):.4f} non-event={np.mean(nev):.4f}")
    print(f"  Brier G {np.mean(brG):.4f}±{np.std(brG):.4f} → M {np.mean(brM):.4f}±{np.std(brM):.4f} p={p_br:.4f}")
    print(f"  Eavg G {np.mean(egG):.4f} vs M {np.mean(egM):.4f}")
    print(f"  DCA benefit range: {suppl['dca_range']}")
    print(f"  VFA0 AUC {np.mean(auc0m):.3f}±{np.std(auc0m):.3f} | VFA>0 {np.mean(auc1m):.3f}±{np.std(auc1m):.3f}")

    with open(os.path.join(OUT, "canonical_author_run.pkl"), "wb") as f:
        pickle.dump(dict(full=full, auc=A, pair_stats=pair_stats, suppl=suppl,
                         v0=v0.tolist()), f)
    print(f"\n[OK] {time.time()-t0:.0f}s -> canonical_author_run.pkl")


if __name__ == "__main__":
    main()

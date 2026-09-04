#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G3/G4/G5 group ablation in the current-code (tol=1e-4, saga) pipeline.
Rows: G3 multi-scale grayscale (0+2+3+4 mm); G4 = G3 + all Doppler (bf_*+VFA);
      G5 = 2 mm cumulative (0+2) + all Doppler.
Uses worker-parallel single-thread execution; 10 seeds x 5 classifiers x 6 metrics.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_author_run as ca

OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = ca.SEEDS
MODELS = ca.MODELS
WORKERS = 20
PID, LAB = ca.PID, ca.LAB


def add_doppler(gray_df, gray_cols):
    """Return df/gray set with all Doppler (bf_* + VFA) inner-merged."""
    bf = pd.concat([pd.read_csv(os.path.join(ca.FLOW, "blood_flow_features_benign.csv")),
                    pd.read_csv(os.path.join(ca.FLOW, "blood_flow_features_malignant.csv"))],
                   ignore_index=True)
    bf = bf.rename(columns={"image_id": PID})
    bf[PID] = bf[PID].apply(ca.sid)
    bf = bf[[PID] + [c for c in bf.columns if c.startswith("bf_")]]
    vfa = pd.concat([pd.read_csv(os.path.join(ca.FLOW, "benign_flow_density.csv")),
                     pd.read_csv(os.path.join(ca.FLOW, "malignant_flow_density.csv"))],
                    ignore_index=True).rename(columns={"case_id": PID})
    vfa = vfa[["flow_density", PID]]
    df = gray_df.merge(bf, on=PID, how="inner").merge(vfa, on=PID, how="inner")
    df[LAB] = gray_df.set_index(PID)[LAB].reindex(df[PID]).to_numpy()
    df = df[df[LAB].notna()].reset_index(drop=True)
    df[LAB] = df[LAB].astype(int)
    return df, gray_cols, [c for c in df.columns if c.startswith("bf_")]


def build_multi():
    d0 = ca.nonfeat(ca.load_bus(0))
    for k in (2, 3, 4):
        dk = ca.suffix(ca.nonfeat(ca.load_bus(k)), k)
        dk = dk.drop(columns=[c for c in dk.columns
                              if "shape" in c.lower() and c not in (PID, LAB)],
                     errors="ignore")
        d0 = d0.merge(dk, on=[PID, LAB], how="inner")
    return d0, [c for c in d0.columns if c not in (PID, LAB)]


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


def one(key, seed, X, y):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                              random_state=seed, stratify=y)
    sc, mask = lasso_fit(X[tr], y[tr])
    Xt = sc.transform(X[tr])[:, mask]; Xe = sc.transform(X[te])[:, mask]
    out = {}
    for n in MODELS:
        c = clf(n); c.fit(Xt, y[tr])
        out[n] = met(y[te], c.predict_proba(Xe)[:, 1])
    return key, seed, out


def main():
    t0 = time.time()
    DATA = {}
    gm, gmc = build_multi()
    DATA["G3"] = (gm[gmc].to_numpy(), gm[LAB].to_numpy().ravel())
    g3d, _, _ = add_doppler(gm, gmc)
    DATA["G4"] = (g3d[[c for c in g3d.columns if c not in (PID, LAB)]].to_numpy(),
                  g3d[LAB].to_numpy().ravel())
    s2, s2c = ca.cumulative(2)
    s2d, _, _ = add_doppler(s2, s2c)
    DATA["G5"] = (s2d[[c for c in s2d.columns if c not in (PID, LAB)]].to_numpy(),
                  s2d[LAB].to_numpy().ravel())
    for k, (X, _) in DATA.items():
        print(f"{k}: n={len(X)} feats={X.shape[1]}", flush=True)

    jobs = [(k, s) for k in DATA for s in SEEDS]
    print(f"{len(jobs)} jobs on {WORKERS} workers", flush=True)
    res = Parallel(n_jobs=WORKERS)(
        delayed(one)(k, s, DATA[k][0], DATA[k][1]) for k, s in jobs)
    full = {k: {m: {q: [] for q in ["acc", "sen", "spe", "prec", "f1", "auc"]}
                for m in MODELS} for k in DATA}
    for k, s, r in res:
        for m in MODELS:
            for q in full[k][m]:
                full[k][m][q].append(r[m][q])
    print("\n=== G-group AUC (mean±SD) ===")
    for k in DATA:
        line = " ".join(f"{np.mean(full[k][m]['auc']):.3f}±{np.std(full[k][m]['auc']):.3f}"
                        for m in MODELS)
        print(f"{k:3s} | {line}")
    with open(os.path.join(OUT, "ggroups_author.pkl"), "wb") as f:
        pickle.dump(full, f)
    print(f"[OK] {time.time()-t0:.0f}s -> ggroups_author.pkl")


if __name__ == "__main__":
    main()

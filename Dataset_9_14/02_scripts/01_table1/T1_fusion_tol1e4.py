#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- standalone paths (self-contained package) ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
DATA = str(_ROOT / '01_data')
OUTDIR = str(_ROOT / '03_outputs' / '01_table1')
_os.makedirs(OUTDIR, exist_ok=True)
# -------------------------------------------------

"""
Feasibility reproduction of 1.0.docx fusion Table using the SAME pipeline
configuration as the author's optimized runs:
  - LASSO tol = 1e-4, max_iter = 10000, cv = 5
  - LR solver = saga
  - otherwise identical seeds/hyperparameters

Rows (1 mm cumulative grayscale base = 0+1 mm):
  N0  base
  N1  + VFA (flow_density)
  N3  + 3 optimized (VFA, Energy, GLNU)
  N13 + 13 selected features (documented stage-4 set, deduplicated)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_doppler_ablation as cda

OUT = OUTDIR
SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
D3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]
WORKERS = 24


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
                                             max_iter=10000, random_state=42,
                                             n_jobs=1),
            "RF": lambda: RandomForestClassifier(n_estimators=500, max_depth=5,
                                                 random_state=42, n_jobs=1),
            "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                                 learning_rate=0.1, random_state=42,
                                                 verbosity=0, n_jobs=1, tree_method="hist"),
            "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean",
                                                n_jobs=1)}[name]()


def one(key, seed, Xd, y):
    X = Xd[key]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                              random_state=seed, stratify=y)
    sc, mask = lasso_fit(X[tr], y[tr])
    Xt = sc.transform(X[tr])[:, mask]; Xe = sc.transform(X[te])[:, mask]
    o = {}
    for n in ["SVM", "LR", "RF", "XGBoost", "KNN"]:
        c = clf(n); c.fit(Xt, y[tr])
        o[n] = roc_auc_score(y[te], c.predict_proba(Xe)[:, 1])
    return key, seed, o


def main():
    t0 = time.time()
    df, gray, dop = cda.build_1mm()
    sel13 = list(dict.fromkeys(pd.read_pickle(os.path.join(OUT, os.path.join("04_doppler_selection", "doppler_selection.pkl")))["stage4_passed"]))
    rows = {"base": gray,
            "vfa": gray + ["flow_density"],
            "plus3": gray + [c for c in D3 if c in df.columns],
            "plus13": gray + sel13}
    X = {k: df[c].to_numpy() for k, c in rows.items()}
    y = df["label"].to_numpy().ravel()
    print("sizes:", {k: v.shape[1] for k, v in X.items()}, flush=True)
    jobs = [(k, s) for k in X for s in SEEDS]
    res = Parallel(n_jobs=WORKERS)(delayed(one)(k, s, X, y) for k, s in jobs)
    auc = {k: {n: [] for n in ["SVM", "LR", "RF", "XGBoost", "KNN"]} for k in X}
    for k, s, o in res:
        for n in auc[k]: auc[k][n].append(o[n])

    print("\n=== tol=1e-4 / saga reproduction ===")
    for k, disp in [("base", "Grayscale(1mm) N0"), ("vfa", "+VFA N1"),
                    ("plus3", "+3 N3"), ("plus13", "+13 N13")]:
        cells = " ".join(f"{np.mean(auc[k][n]):.3f}±{np.std(auc[k][n]):.3f}"
                         for n in ["SVM", "LR", "RF", "XGBoost", "KNN"])
        print(f"{disp:18s} | {cells}")
    with open(os.path.join(OUT, "T1_fusion_tol1e4.pkl"), "wb") as f:
        pickle.dump({"auc": auc}, f)
    print(f"[OK] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

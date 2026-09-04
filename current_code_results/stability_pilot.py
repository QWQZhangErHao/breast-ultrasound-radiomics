#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pilot B: paper-style stability LASSO selection (tol=1e-4).
For each outer 70/30 seed, selection is done inside the train split using inner
k-fold LASSO: keep features with selection-rate>=0.6 and sign-consistency>=0.7,
then fit XGBoost(+3 optimized Doppler) and evaluate the held-out test AUC.
Compare with method A (single LassoCV) on the same seeds from
canonical_author_run.pkl (XGBoost, fusion +3).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import xgboost as xgb
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_author_run as ca

D3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]
PILOT_SEEDS = [42, 123, 2024]
RATE, SIGN = 0.6, 0.7
INNER_K = 5


def lasso_mask(Xt, yt):
    sc = StandardScaler(); Xs = sc.fit_transform(Xt)
    l = LassoCV(cv=3, random_state=42, max_iter=10000, n_jobs=1, tol=1e-4)
    l.fit(Xs, yt)
    return l.coef_


def stability_select(Xtr, ytr, seed):
    counts = np.zeros(Xtr.shape[1])
    pos = np.zeros(Xtr.shape[1]); neg = np.zeros(Xtr.shape[1])
    inner = StratifiedKFold(INNER_K, shuffle=True, random_state=seed)
    for tr, va in inner.split(Xtr, ytr):
        coef = lasso_mask(Xtr[tr], ytr[tr])
        nz = coef != 0
        counts += nz
        pos += (coef > 0); neg += (coef < 0)
    sel_rate = counts / INNER_K
    sign_ok = np.maximum(pos, neg) / np.maximum(counts, 1)
    stable = (sel_rate >= RATE) & (sign_ok >= SIGN)
    if stable.sum() == 0:                      # fallback: top-by-rate
        stable = np.zeros_like(stable, dtype=bool)
        order = np.argsort(-sel_rate)[:20]
        stable[order] = True
    return stable, int(stable.sum()), float(sel_rate.mean())


def one(seed, X, y):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                              random_state=seed, stratify=y)
    stable, n_sel, avg_rate = stability_select(X[tr], y[tr], seed)
    sc = StandardScaler(); sc.fit(X[tr])
    Xt = sc.transform(X[tr])[:, stable]; Xe = sc.transform(X[te])[:, stable]
    c = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          random_state=42, verbosity=0, n_jobs=1, tree_method="hist")
    c.fit(Xt, y[tr])
    return roc_auc_score(y[te], c.predict_proba(Xe)[:, 1]), n_sel, avg_rate


def main():
    t0 = time.time()
    df, gray, _ = ca.build_fusion_base()
    cols = gray + [c for c in D3 if c in df.columns]
    X = df[cols].to_numpy(); y = df[ca.LAB].to_numpy().ravel()
    print(f"X {X.shape}", flush=True)
    res = Parallel(n_jobs=len(PILOT_SEEDS))(
        delayed(one)(s, X, y) for s in PILOT_SEEDS)
    aucs = np.array([r[0] for r in res])
    nsel = [r[1] for r in res]; rate = [r[2] for r in res]
    print("Method B (stability LASSO): seeds", PILOT_SEEDS)
    print("  per-seed AUC:", np.round(aucs, 3), "mean±sd",
          f"{aucs.mean():.3f}±{aucs.std():.3f}")
    print("  selected feats:", nsel, "mean rate:", np.round(np.mean(rate), 2))
    # Method A on the SAME seeds from canonical run (seed order = ca.SEEDS)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "canonical_author_run.pkl"), "rb") as f:
        P = pickle.load(f)
    arrA = np.asarray(P["auc"]["fus_p3"]["XGBoost"])
    idx = [ca.SEEDS.index(s) for s in PILOT_SEEDS]
    Aa = arrA[idx]
    print("Method A (single LassoCV) same seeds XGB+3:", np.round(Aa, 3),
          "mean±sd", f"{Aa.mean():.3f}±{Aa.std():.3f}")
    print(f"[OK] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

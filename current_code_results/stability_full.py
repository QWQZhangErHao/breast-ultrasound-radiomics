#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full B (paper-style stability LASSO) on all 10 seeds, XGBoost + 3 Doppler.
Selection inside each outer train split via inner k=5 LASSO folds (tol=1e-4);
stable features = selection-rate>=0.6 & sign-consistency>=0.7.
Compares to method A (single LassoCV) from canonical_author_run.pkl (same seeds).
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
RATE, SIGN, INNER_K, WORKERS = 0.6, 0.7, 5, 8
HERE = os.path.dirname(os.path.abspath(__file__))


def lasso_coef(Xt, yt):
    s = StandardScaler(); Xs = s.fit_transform(Xt)
    l = LassoCV(cv=3, random_state=42, max_iter=10000, n_jobs=1, tol=1e-4)
    l.fit(Xs, yt)
    return l.coef_


def stability_select(Xtr, ytr, seed):
    counts = np.zeros(Xtr.shape[1]); pos = np.zeros(Xtr.shape[1]); neg = np.zeros(Xtr.shape[1])
    for tr, _ in StratifiedKFold(INNER_K, shuffle=True, random_state=seed).split(Xtr, ytr):
        coef = lasso_coef(Xtr[tr], ytr[tr])
        counts += coef != 0
        pos += coef > 0; neg += coef < 0
    rate = counts / INNER_K
    sign_ok = np.maximum(pos, neg) / np.maximum(counts, 1)
    stable = (rate >= RATE) & (sign_ok >= SIGN)
    if stable.sum() == 0:
        stable = np.zeros_like(stable, dtype=bool)
        stable[np.argsort(-rate)[:20]] = True
    return stable


def one(seed, X, y):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                              random_state=seed, stratify=y)
    stable = stability_select(X[tr], y[tr], seed)
    sc = StandardScaler(); sc.fit(X[tr])
    Xt = sc.transform(X[tr])[:, stable]; Xe = sc.transform(X[te])[:, stable]
    c = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          random_state=42, verbosity=0, n_jobs=1, tree_method="hist")
    c.fit(Xt, y[tr])
    return seed, roc_auc_score(y[te], c.predict_proba(Xe)[:, 1]), int(stable.sum())


def main():
    t0 = time.time()
    df, gray, _ = ca.build_fusion_base()
    cols = gray + [c for c in D3 if c in df.columns]
    X = df[cols].to_numpy(); y = df[ca.LAB].to_numpy().ravel()
    print(f"X {X.shape}; full-B seeds {ca.SEEDS}; workers {WORKERS}", flush=True)
    res = Parallel(n_jobs=WORKERS)(delayed(one)(s, X, y) for s in ca.SEEDS)
    B = {s: {"auc": a, "nsel": n} for s, a, n in res}
    aucB = np.array([B[s]["auc"] for s in ca.SEEDS])
    nsel = [B[s]["nsel"] for s in ca.SEEDS]
    with open(os.path.join(HERE, "canonical_author_run.pkl"), "rb") as f:
        P = pickle.load(f)
    aucA = np.asarray(P["auc"]["fus_p3"]["XGBoost"])
    print("per-seed AUC  A:", np.round(aucA, 3))
    print("per-seed AUC  B:", np.round(aucB, 3))
    print(f"A: {aucA.mean():.3f}±{aucA.std():.3f}  |  B: {aucB.mean():.3f}±{aucB.std():.3f}")
    print("B selected feats per seed:", nsel, "mean", int(np.mean(nsel)))
    with open(os.path.join(HERE, "stability_full.pkl"), "wb") as f:
        pickle.dump({"B": B, "A_seeds": aucA.tolist(), "seeds": ca.SEEDS}, f)
    print(f"[OK] {time.time()-t0:.0f}s -> stability_full.pkl")


if __name__ == "__main__":
    main()

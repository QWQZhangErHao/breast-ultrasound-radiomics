#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nested vs non-nested (author tol=1e-4) for final multimodal XGB (gray1mm+3)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import xgboost as xgb
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_author_run as ca

D3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]


def lasso_fit(Xtr, ytr):
    s = StandardScaler(); Xs = s.fit_transform(Xtr)
    l = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=1, tol=1e-4)
    l.fit(Xs, ytr)
    mask = l.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(l.coef_))[-10:]] = True
    return s, mask


def fit_p(Xtr, ytr, Xte, pre=None):
    if pre is None:
        sc, mask = lasso_fit(Xtr, ytr)
    else:
        sc, mask = pre
    Xt = sc.transform(Xtr)[:, mask]; Xe = sc.transform(Xte)[:, mask]
    c = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          random_state=42, verbosity=0, n_jobs=1, tree_method="hist")
    c.fit(Xt, ytr)
    return c.predict_proba(Xe)[:, 1]


def main():
    t0 = time.time()
    df, gray, _ = ca.build_fusion_base()
    cols = gray + [c for c in D3 if c in df.columns]
    X = df[cols].to_numpy(); y = df[ca.LAB].to_numpy().ravel()
    print(f"X {X.shape} y {y.sum()}/{len(y)}", flush=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    sc_full, mask_full = lasso_fit(X, y)
    def nested(f):
        tr, te = f
        return te, fit_p(X[tr], y[tr], X[te])
    def leaky(f):
        tr, te = f
        return te, fit_p(X[tr], y[tr], X[te], (sc_full, mask_full))
    rn = Parallel(n_jobs=5)(delayed(nested)(f) for f in folds)
    rl = Parallel(n_jobs=5)(delayed(leaky)(f) for f in folds)
    def pooled(rs):
        te = np.concatenate([r[0] for r in rs]); p = np.concatenate([r[1] for r in rs])
        return roc_auc_score(y[te], p)
    an, al = pooled(rn), pooled(rl)
    print(f"nested={an:.3f}  non-nested(leaky)={al:.3f}  inflation={al-an:+.3f}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "nested_author.txt"), "w", encoding="utf-8") as f:
        f.write(f"nested={an:.4f}\nnon_nested={al:.4f}\ninflation={al-an:+.4f}\n")
    print(f"[OK] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

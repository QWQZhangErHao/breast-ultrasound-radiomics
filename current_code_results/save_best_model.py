#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit the best (current-code) final models on the FULL cohort and save them
for downstream/testing use:
  1) XGBoost + 3 optimized Doppler  (primary, parsimonious)
  2) XGBoost + all 21 bf_ Doppler    (highest-point candidate)
Pipeline saved: StandardScaler + LASSO-mask (tol1e-4) + fitted classifier,
plus the ordered feature-column list needed to build test inputs.
"""
import os, json, time, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import xgboost as xgb

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
import canonical_author_run as ca

OUT = os.path.join(sys_path, "saved_models")
os.makedirs(OUT, exist_ok=True)
D3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]
SEED = 42


def fit_pipe(cols):
    df, gray, bf_all = ca.build_fusion_base()
    X = df[cols].to_numpy(); y = df[ca.LAB].to_numpy().ravel()
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    l = LassoCV(cv=5, random_state=SEED, max_iter=10000, n_jobs=-1, tol=1e-4)
    l.fit(Xs, y)
    mask = l.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(l.coef_))[-10:]] = True
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                            random_state=SEED, verbosity=0, n_jobs=-1, tree_method="hist")
    clf.fit(Xs[:, mask], y)
    return dict(cols=cols, scaler=sc, mask=mask, clf=clf, n=len(y),
                pos=float(y.sum()), seed=SEED)


def main():
    t0 = time.time()
    df, gray, bf_all = ca.build_fusion_base()
    # XGBoost +3
    c3 = [c for c in gray + D3 if c in df.columns]
    p3 = fit_pipe(c3)
    # XGBoost +21
    p21 = fit_pipe(gray + bf_all)
    p3["clf"]._Booster  # ensure loaded
    for name, p in [("xgb_plus3", p3), ("xgb_plus21", p21)]:
        with open(os.path.join(OUT, f"{name}.pkl"), "wb") as f:
            pickle.dump(p, f)
    # small smoke test: in-sample AUC
    from sklearn.metrics import roc_auc_score
    for name, p in [("xgb_plus3", p3), ("xgb_plus21", p21)]:
        X = df[p["cols"]].to_numpy()
        proba = p["clf"].predict_proba(p["scaler"].transform(X)[:, p["mask"]])[:, 1]
        auc = roc_auc_score(df[ca.LAB].to_numpy(), proba)
        print(f"{name}: fit on full n={p['n']}, selected={int(p['mask'].sum())} feats, "
              f"in-sample AUC={auc:.3f}")
    print("[OK] saved models ->", OUT)
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

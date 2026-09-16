#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- standalone paths (self-contained package) ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
DATA = str(_ROOT / '01_data')
OUTDIR = str(_ROOT / '03_outputs' / '03_sec321_scale')
_os.makedirs(OUTDIR, exist_ok=True)
# -------------------------------------------------

"""
CANONICAL cumulative peritumoral-scale table (matches manuscript semantics):
  k = 0   : intratumoral (0 mm) radiomics only
  k = 1..4: intratumoral + peritumoral radiomics of the ROI expanded to k mm
            (0 mm + {k} mm features, peritumoral shape features excluded),
            identical to the feature sets used for the 1-mm grayscale base of the
            multimodal fusion analysis.

Pipeline: 70/30 stratified split -> LASSO (cv5, tol1e-3) on train only -> 5
classifiers -> default-0.5 metrics, 10 seeds. Parallel over (scale x seed).
Writes metrics_by_scale_cumulative_sd.csv (long, Mean + SD).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time, re
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
BUS = os.path.join(DATA, "01_bus_features")
SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
WORKERS = 24
METRIC_NAMES = {"acc": "Accuracy", "sen": "Sensitivity", "spe": "Specificity",
                "prec": "Precision", "f1": "F1", "auc": "AUC"}
PID, LAB = "patient_name", "label"


def sid(raw):
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else None


def is_feat(col):
    return not any(p in col.lower() for p in ["patient_name", "file_name",
                                              "diagnostics_"])


def load(mm):
    b = pd.read_csv(os.path.join(BUS, f"benign_{mm}mm.csv"))
    m = pd.read_csv(os.path.join(BUS, f"malignant_{mm}mm.csv"))
    b[LAB], m[LAB] = 0, 1
    df = pd.concat([b, m], ignore_index=True)
    df[PID] = df[PID].apply(sid)
    return df


def cumulative_features(k):
    d0 = load(0)
    cols0 = [c for c in d0.columns if c != LAB and is_feat(c) and c != PID]
    d0f = d0[[PID, LAB] + cols0]
    if k == 0:
        return d0f, cols0
    dk = load(k)
    feats = [c for c in dk.columns if c != LAB and is_feat(c) and c != PID]
    feats = [c for c in feats if "shape" not in c.lower()]
    dkf = dk[[PID, LAB] + feats].rename(
        columns={c: c if c in (PID, LAB) else f"{c}_{k}mm" for c in
                 [PID, LAB] + feats})
    out = d0f.merge(dkf, on=[PID, LAB], how="inner")
    return out, [c for c in out.columns if c not in (PID, LAB)]


def lasso_select(Xtr, ytr):
    s = StandardScaler(); Xs = s.fit_transform(Xtr)
    l = LassoCV(cv=5, random_state=42, max_iter=5000, n_jobs=1, tol=1e-3)
    l.fit(Xs, ytr)
    mask = l.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(l.coef_))[-10:]] = True
    return s, mask


def clf(name):
    return {"SVM": lambda: SVC(kernel="linear", C=1, probability=True, random_state=42),
            "LR": lambda: LogisticRegression(penalty="l2", C=1, solver="liblinear",
                                             max_iter=10000, random_state=42),
            "RF": lambda: RandomForestClassifier(n_estimators=500, max_depth=5,
                                                 random_state=42, n_jobs=1),
            "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                                 learning_rate=0.1, random_state=42,
                                                 verbosity=0, n_jobs=1, tree_method="hist"),
            "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean",
                                                n_jobs=1)}[name]()


def met_at(y, p):
    pr = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr).ravel()
    sen = tp / (tp + fn) if tp + fn else 0.0
    spe = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"acc": accuracy_score(y, pr), "sen": sen, "spe": spe, "prec": prec,
            "f1": (2 * prec * sen / (prec + sen)) if prec + sen else 0.0,
            "auc": roc_auc_score(y, p)}


def one(k, seed, X, y):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                              random_state=seed, stratify=y)
    sc, mask = lasso_select(X[tr], y[tr])
    Xtr = sc.transform(X[tr])[:, mask]; Xte = sc.transform(X[te])[:, mask]
    out = {}
    for n in ["SVM", "LR", "RF", "XGBoost", "KNN"]:
        c = clf(n); c.fit(Xtr, y[tr])
        out[n] = met_at(y[te], c.predict_proba(Xte)[:, 1])
    return k, seed, out


def main():
    t0 = time.time()
    groups = {}
    y = None
    for k in range(5):
        dfk, cols = cumulative_features(k)
        groups[k] = dfk[cols].to_numpy()
        if y is None:
            y = dfk[LAB].to_numpy().ravel()
        print(f"k={k}mm: patients={len(dfk)} feats={len(cols)}", flush=True)

    jobs = [(k, s) for k in range(5) for s in SEEDS]
    print(f"{len(jobs)} jobs on {WORKERS} workers", flush=True)
    res = Parallel(n_jobs=WORKERS)(delayed(one)(k, s, groups[k], y)
                                   for k, s in jobs)
    per = {(k, s): o for k, s, o in res}

    rows = []
    for k in range(5):
        for n in ["SVM", "LR", "RF", "XGBoost", "KNN"]:
            for mkey, mname in METRIC_NAMES.items():
                arr = np.array([per[(k, s)][n][mkey] for s in SEEDS])
                rows.append({"Model": n, "Feature_Range": k, "Metric": mname,
                             "Mean": arr.mean(), "SD": arr.std()})
    long = pd.DataFrame(rows).sort_values(["Model", "Feature_Range"])
    out = os.path.join(OUTDIR, "S321_cumulative_scale.csv")
    long.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] {time.time()-t0:.0f}s -> {out}")

    # quick AUC glance
    pv = long[long["Metric"] == "AUC"].pivot(index="Model", columns="Feature_Range",
                                            values="Mean")
    print("\nAUC cumulative means:\n", pv.round(3).to_string())


if __name__ == "__main__":
    main()

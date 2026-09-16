#!/usr/bin/env python3
# --- standalone paths (self-contained package) ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
DATA = str(_ROOT / '01_data')
OUTDIR = str(_ROOT / '03_outputs' / '04_sec322_stats')
_os.makedirs(OUTDIR, exist_ok=True)
# -------------------------------------------------

"""PR-AUC and Youden threshold analysis on 1mm data."""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings("ignore")

N_SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
CV_FOLDS = 5
N_JOBS = 32

BASE_DIR = DATA
BUS_DIR = os.path.join(BASE_DIR, "01_bus_features")
FLOW_DIR = os.path.join(BASE_DIR, "02_flow_features")
OUTPUT_DIR = OUTDIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATIENT_ID_COL = "patient_name"
LABEL_COL = "label"

# Import the data loading functions from the original pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
def sanitize_id(raw):
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else None

def load_bus(mm):
    b = pd.read_csv(os.path.join(BUS_DIR, f"benign_{mm}mm.csv"))
    m = pd.read_csv(os.path.join(BUS_DIR, f"malignant_{mm}mm.csv"))
    b[LABEL_COL], m[LABEL_COL] = 0, 1
    df = pd.concat([b, m], ignore_index=True)
    df[PATIENT_ID_COL] = df[PATIENT_ID_COL].apply(sanitize_id)
    return df

def is_feature_col(col):
    cl = col.lower()
    for p in ["patient_name", "file_name", "diagnostics_"]:
        if p in cl: return False
    return True

def is_shape_col(col):
    return "shape" in col.lower()

def drop_non_feature(df):
    drop = [c for c in df.columns if c not in [PATIENT_ID_COL, LABEL_COL] and not is_feature_col(c)]
    return df.drop(columns=drop, errors="ignore")

def suffix_cols(df, suffix, exempt):
    return df.rename(columns={c: c if c in exempt else f"{c}_{suffix}" for c in df.columns})

# Build 1mm groups
bus = {mm: load_bus(mm) for mm in [0, 1, 2, 3, 4]}
bf_b = pd.read_csv(os.path.join(FLOW_DIR, "blood_flow_features_benign.csv"))
bf_m = pd.read_csv(os.path.join(FLOW_DIR, "blood_flow_features_malignant.csv"))
bf_b[LABEL_COL], bf_m[LABEL_COL] = 0, 1
bf_all = pd.concat([bf_b, bf_m], ignore_index=True)
bf_all[PATIENT_ID_COL] = bf_all["image_id"].apply(sanitize_id)
bf_feat = [c for c in bf_all.columns if c.startswith("bf_")]
bf_all = bf_all[[PATIENT_ID_COL, LABEL_COL] + bf_feat]

vfa_b = pd.read_csv(os.path.join(FLOW_DIR, "benign_flow_density.csv"))
vfa_m = pd.read_csv(os.path.join(FLOW_DIR, "malignant_flow_density.csv"))
vfa_b[LABEL_COL], vfa_m[LABEL_COL] = 0, 1
vfa_all = pd.concat([vfa_b, vfa_m], ignore_index=True)
vfa_all[PATIENT_ID_COL] = vfa_all["case_id"].apply(sanitize_id)
vfa_all["flow_density"] = vfa_all["flow_density"].astype(float)
vfa_all = vfa_all[[PATIENT_ID_COL, "flow_density", LABEL_COL]]

exempt = [PATIENT_ID_COL, LABEL_COL]
d0 = drop_non_feature(bus[0])
d1 = drop_non_feature(bus[1])
d1r = suffix_cols(d1, "1mm", exempt)
d1r = d1r.drop(columns=[c for c in d1r.columns if is_shape_col(c) and c not in exempt], errors="ignore")
g1mm = pd.merge(d0, d1r, on=exempt, how="inner")

g1mm_dopp = pd.merge(g1mm, bf_all, on=[PATIENT_ID_COL, LABEL_COL], how="inner")
g1mm_dopp = pd.merge(g1mm_dopp, vfa_all[[PATIENT_ID_COL, "flow_density"]], on=PATIENT_ID_COL, how="inner")

non_feat = [c for c in exempt if c in g1mm_dopp.columns] + ["flow_density"]
feat_cols = [c for c in g1mm_dopp.columns if c not in non_feat]

X = g1mm_dopp[feat_cols].values
y = g1mm_dopp[LABEL_COL].values.ravel()

print("=== PR-AUC and Youden Analysis (1mm + Doppler) ===")
print(f"Data: {X.shape[0]} samples, {X.shape[1]} features\n")

all_results = []
for seed in N_SEEDS:
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[tr])
    lasso = LassoCV(cv=CV_FOLDS, random_state=seed, max_iter=10000, n_jobs=N_JOBS, tol=1e-4)
    lasso.fit(X_tr_s, y[tr])
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True
    X_tr = scaler.transform(X[tr])[:, mask]
    X_te = scaler.transform(X[te])[:, mask]

    clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed, verbosity=0, n_jobs=N_JOBS, tree_method="hist")
    clf.fit(X_tr, y[tr])
    y_proba = clf.predict_proba(X_te)[:, 1]

    # ROC-AUC
    roc_auc = roc_auc_score(y[te], y_proba)

    # PR-AUC
    pr_auc = average_precision_score(y[te], y_proba)

    # Youden threshold
    fpr, tpr, thresholds = roc_curve(y[te], y_proba)
    youden_idx = np.argmax(tpr - fpr)
    youden_th = thresholds[youden_idx]
    youden_val = tpr[youden_idx] - fpr[youden_idx]

    # Precision-recall curve
    prec, rec, th = precision_recall_curve(y[te], y_proba)

    all_results.append({
        "seed": seed,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "youden_threshold": youden_th,
        "youden_sensitivity": tpr[youden_idx],
        "youden_specificity": 1 - fpr[youden_idx],
        "youden_index": youden_val,
    })

# Summary
print(f"{'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 57)
for metric in ["roc_auc", "pr_auc", "youden_threshold", "youden_sensitivity", "youden_specificity"]:
    vals = [r[metric] for r in all_results]
    print(f"{metric:<25} {np.mean(vals):.4f}  {np.std(vals):.4f}  {np.min(vals):.4f}  {np.max(vals):.4f}")

# Save
with open(os.path.join(OUTPUT_DIR, "S322_pr_youden.txt"), "w") as f:
    f.write("PR-AUC and Youden Threshold Analysis\n")
    f.write(f"Model: XGBoost, 1mm peritumoral + 3 Doppler features\n")
    f.write(f"Validation: 10 repeated 70/30 splits (nested LASSO)\n\n")
    f.write(f"{'Metric':<25} {'Mean':>8} {'Std':>8}\n")
    f.write("-" * 42 + "\n")
    for m in ["roc_auc", "pr_auc", "youden_threshold", "youden_sensitivity", "youden_specificity"]:
        v = [r[m] for r in all_results]
        f.write(f"{m:<25} {np.mean(v):.4f}  {np.std(v):.4f}\n")

print(f"\nSaved to: {os.path.join(OUTPUT_DIR, 'pr_youden_results.txt')}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1mm Peritumoral Scale — OPTIMIZED for RTX 5070 Ti
===================================================
- GPU-accelerated XGBoost (gpu_hist)
- Parallel 10-seed loop
- All 32 CPU threads
"""

import os, sys, pickle, time, re, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# =============================================================================
# OPTIMIZED CONFIG
# =============================================================================
N_JOBS = 32
N_SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
CV_FOLDS = 5

os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

DESKTOP = r"C:\Users\ZhangErHao\Desktop"
_avail = [d for d in os.listdir(DESKTOP) if d.startswith("dataset")]
BASE_DIR = os.path.join(DESKTOP, _avail[0]) if _avail else os.path.join(DESKTOP, "dataset")
OUTPUT_DIR = os.path.join(DESKTOP, "重新验证后的论文数据")
BUS_DIR = os.path.join(BASE_DIR, "BUS_features")
FLOW_DIR = os.path.join(BASE_DIR, "flow_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATIENT_ID_COL = "patient_name"
LABEL_COL = "label"

CLASSIFIERS = {
    "SVM": lambda: SVC(kernel="linear", C=1.0, probability=True, random_state=42),
    "LR": lambda: LogisticRegression(penalty="l2", C=1.0, solver="saga", max_iter=10000, random_state=42, n_jobs=N_JOBS),
    "RF": lambda: RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42, n_jobs=N_JOBS),
    "XGBoost": lambda: xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, verbosity=0,
        n_jobs=N_JOBS,
        tree_method="hist",
    ),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=N_JOBS),
}

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

def build_groups():
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
    d2 = drop_non_feature(bus[2])

    g1 = d0.copy()
    d1r = suffix_cols(d1, "1mm", exempt)
    d1r = d1r.drop(columns=[c for c in d1r.columns if is_shape_col(c) and c not in exempt], errors="ignore")
    g1mm = pd.merge(d0, d1r, on=exempt, how="inner")
    d2r = suffix_cols(d2, "2mm", exempt)
    d2r = d2r.drop(columns=[c for c in d2r.columns if is_shape_col(c) and c not in exempt], errors="ignore")
    g2mm = pd.merge(d0, d2r, on=exempt, how="inner")

    g1mm_dopp = pd.merge(g1mm, bf_all, on=[PATIENT_ID_COL, LABEL_COL], how="inner")
    g1mm_dopp = pd.merge(g1mm_dopp, vfa_all[[PATIENT_ID_COL, "flow_density"]], on=PATIENT_ID_COL, how="inner")
    g2mm_dopp = pd.merge(g2mm, bf_all, on=[PATIENT_ID_COL, LABEL_COL], how="inner")
    g2mm_dopp = pd.merge(g2mm_dopp, vfa_all[[PATIENT_ID_COL, "flow_density"]], on=PATIENT_ID_COL, how="inner")

    return {
        "G1_intratumoral_0mm": g1,
        "G1_peritumoral_1mm": g1mm,
        "G2_peritumoral_2mm": g2mm,
        "G4_1mm_Doppler": g1mm_dopp,
        "G5_2mm_Doppler": g2mm_dopp,
    }

def lasso_select(X_train, y_train, feature_names):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    lasso = LassoCV(cv=CV_FOLDS, random_state=42, max_iter=10000, n_jobs=N_JOBS, tol=1e-4)
    lasso.fit(Xs, y_train)
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True
    return scaler, mask, [feature_names[i] for i in np.where(mask)[0]]

def run_seed(seed, gname, df):
    exempt = [PATIENT_ID_COL, LABEL_COL, "flow_density"]
    non_feat = [c for c in exempt if c in df.columns]
    feat_cols = [c for c in df.columns if c not in non_feat]
    X = df[feat_cols].values
    y = df[LABEL_COL].values.ravel()
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    scaler, mask, sel = lasso_select(X[tr_idx], y[tr_idx], feat_cols)
    X_tr_s = scaler.transform(X[tr_idx])[:, mask]
    X_te_s = scaler.transform(X[te_idx])[:, mask]
    results = {}
    for name, clf_fn in CLASSIFIERS.items():
        clf = clf_fn()
        clf.fit(X_tr_s, y[tr_idx])
        y_proba = clf.predict_proba(X_te_s)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te_idx], y_pred).ravel()
        results[name] = {
            "auc": round(roc_auc_score(y[te_idx], y_proba), 4),
            "acc": round(accuracy_score(y[te_idx], y_pred), 4),
            "sen": round(tp/(tp+fn) if (tp+fn)>0 else 0, 4),
            "spe": round(tn/(tn+fp) if (tn+fp)>0 else 0, 4),
        }
    return seed, gname, results

def main():
    t0 = time.time()
    print("=" * 70)
    print("1mm vs 2mm — OPTIMIZED (32 threads + RTX 5070 Ti)")
    print("=" * 70)

    groups = build_groups()
    for n, df in groups.items():
        print(f"  {n}: {df.shape}")

    print(f"\nRunning {len(N_SEEDS)} seeds × {len(groups)} groups = {len(N_SEEDS)*len(groups)} jobs...")
    tasks = [(s, g, df) for g, df in groups.items() for s in N_SEEDS]
    results = Parallel(n_jobs=min(N_JOBS, 20), verbose=5)(
        delayed(run_seed)(s, g, df) for s, g, df in tasks
    )

    all_r = {}
    for seed, gname, res in results:
        all_r.setdefault(gname, {})[seed] = res

    print(f"\n{'Group':<30} {'SVM':>10} {'LR':>10} {'RF':>10} {'XGB':>10} {'KNN':>10}")
    print("-" * 80)
    for gname in groups:
        row = f"{gname:<30}"
        for clf in CLASSIFIERS:
            aucs = [all_r[gname][s][clf]["auc"] for s in N_SEEDS]
            row += f" {np.mean(aucs):.3f}±{np.std(aucs):.3f}"
        print(row)

    print(f"\n=== DOPPLER GAIN ===")
    for clf in CLASSIFIERS:
        auc1 = [all_r["G1_peritumoral_1mm"][s][clf]["auc"] for s in N_SEEDS]
        auc1d = [all_r["G4_1mm_Doppler"][s][clf]["auc"] for s in N_SEEDS]
        auc2 = [all_r["G2_peritumoral_2mm"][s][clf]["auc"] for s in N_SEEDS]
        auc2d = [all_r["G5_2mm_Doppler"][s][clf]["auc"] for s in N_SEEDS]
        d1 = np.mean(auc1d) - np.mean(auc1)
        d2 = np.mean(auc2d) - np.mean(auc2)
        _, p1 = stats.ttest_rel(auc1, auc1d)
        _, p2 = stats.ttest_rel(auc2, auc2d)
        print(f"{clf:10s}: 1mm Δ={d1:+.3f}(p={p1:.4f}) | 2mm Δ={d2:+.3f}(p={p2:.4f})")

    # XGBoost final result
    xgb1 = [all_r["G1_peritumoral_1mm"][s]["XGBoost"]["auc"] for s in N_SEEDS]
    xgb1d = [all_r["G4_1mm_Doppler"][s]["XGBoost"]["auc"] for s in N_SEEDS]
    print(f"\n🏆 FINAL (XGBoost + 1mm + nested CV):")
    print(f"   Grayscale: {np.mean(xgb1):.3f}±{np.std(xgb1):.3f}")
    print(f"   +Doppler:  {np.mean(xgb1d):.3f}±{np.std(xgb1d):.3f}")
    d = np.mean(xgb1d) - np.mean(xgb1)
    _, p = stats.ttest_rel(xgb1, xgb1d)
    print(f"   ΔAUC={d:+.3f}, p={p:.4f}")
    print(f"⏱ Total: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

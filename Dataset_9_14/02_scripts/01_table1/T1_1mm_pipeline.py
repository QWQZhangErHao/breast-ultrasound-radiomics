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
Run 1mm peritumoral scale with nested CV — directly addresses reviewer concern.
Adds 1mm groups alongside existing 2mm groups for head-to-head comparison.
Uses the same validation framework as validate_pipeline.py.
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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from joblib import Parallel, delayed

try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx._count_physical_cores_win32 = lambda: max(1, (os.cpu_count() or 4) // 2)
except Exception:
    pass

warnings.filterwarnings("ignore")

SEED = 42
N_SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
CV_FOLDS = 5
N_JOBS = -1

BASE_DIR = DATA
OUTPUT_DIR = OUTDIR
BUS_DIR = os.path.join(BASE_DIR, "01_bus_features")
FLOW_DIR = os.path.join(BASE_DIR, "02_flow_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATIENT_ID_COL = "patient_name"
LABEL_COL = "label"

CLASSIFIERS = {
    "SVM": lambda: SVC(kernel="linear", C=1.0, probability=True, random_state=SEED),
    "LR": lambda: LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=10000, random_state=SEED),
    "RF": lambda: RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED, verbosity=0, n_jobs=-1, tree_method="hist"),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),
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
        if p in cl:
            return False
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

    # Flow features
    bf_b = pd.read_csv(os.path.join(FLOW_DIR, "blood_flow_features_benign.csv"))
    bf_m = pd.read_csv(os.path.join(FLOW_DIR, "blood_flow_features_malignant.csv"))
    bf_b[LABEL_COL], bf_m[LABEL_COL] = 0, 1
    bf_all = pd.concat([bf_b, bf_m], ignore_index=True)
    bf_all[PATIENT_ID_COL] = bf_all["image_id"].apply(sanitize_id)
    bf_feat = [c for c in bf_all.columns if c.startswith("bf_")]
    bf_all = bf_all[[PATIENT_ID_COL, LABEL_COL] + bf_feat]

    # VFA
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

    # Group 1: B (0mm only) — intratumoral baseline
    g1 = d0.copy()

    # Group 2: 0mm + 1mm peritumoral (THE NEW 1mm GROUP)
    d1r = suffix_cols(d1, "1mm", exempt)
    d1r = d1r.drop(columns=[c for c in d1r.columns if is_shape_col(c) and c not in exempt], errors="ignore")
    g1mm = pd.merge(d0, d1r, on=exempt, how="inner")

    # Group 3: 0mm + 2mm peritumoral (original reference)
    d2r = suffix_cols(d2, "2mm", exempt)
    d2r = d2r.drop(columns=[c for c in d2r.columns if is_shape_col(c) and c not in exempt], errors="ignore")
    g2mm = pd.merge(d0, d2r, on=exempt, how="inner")

    # Group 4: 0mm + 1mm + Doppler
    g1mm_dopp = pd.merge(g1mm, bf_all, on=[PATIENT_ID_COL, LABEL_COL], how="inner")
    g1mm_dopp = pd.merge(g1mm_dopp, vfa_all[[PATIENT_ID_COL, "flow_density"]], on=PATIENT_ID_COL, how="inner")

    # Group 5: 0mm + 2mm + Doppler (original multimodal)
    g2mm_dopp = pd.merge(g2mm, bf_all, on=[PATIENT_ID_COL, LABEL_COL], how="inner")
    g2mm_dopp = pd.merge(g2mm_dopp, vfa_all[[PATIENT_ID_COL, "flow_density"]], on=PATIENT_ID_COL, how="inner")

    groups = {
        "G1_intratumoral_0mm": g1,
        "G1_peritumoral_1mm": g1mm,
        "G2_peritumoral_2mm": g2mm,
        "G4_1mm_Doppler": g1mm_dopp,
        "G5_2mm_Doppler": g2mm_dopp,
    }
    return groups


# --- Feature selection (LASSO on training set only) ---
def lasso_select(X_train, y_train, feature_names):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    lasso = LassoCV(cv=CV_FOLDS, random_state=SEED, max_iter=5000, n_jobs=N_JOBS, tol=1e-3)
    lasso.fit(Xs, y_train)
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        top = min(10, len(lasso.coef_))
        mask[np.argsort(np.abs(lasso.coef_))[-top:]] = True
    sel_feat = [feature_names[i] for i in np.where(mask)[0]]
    return scaler, mask, sel_feat, lasso.alpha_


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "acc": round(accuracy_score(y_true, y_pred), 4),
        "auc": round(roc_auc_score(y_true, y_proba), 4),
        "sen": round(tp / (tp + fn) if (tp+fn) > 0 else 0, 4),
        "spe": round(tn / (tn + fp) if (tn+fp) > 0 else 0, 4),
        "prec": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def run_seed(seed, group_name, df):
    exempt = [PATIENT_ID_COL, LABEL_COL, "flow_density"]
    non_feat = [c for c in exempt if c in df.columns]
    feat_cols = [c for c in df.columns if c not in non_feat]
    X = df[feat_cols].values
    y = df[LABEL_COL].values.ravel()
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    scaler, mask, sel_feat, alpha = lasso_select(X_tr, y_tr, feat_cols)
    X_tr_s = scaler.transform(X_tr)[:, mask]
    X_te_s = scaler.transform(X_te)[:, mask]
    results = {}
    for name, clf_fn in CLASSIFIERS.items():
        clf = clf_fn()
        clf.fit(X_tr_s, y_tr)
        y_proba = clf.predict_proba(X_te_s)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        results[name] = compute_metrics(y_te, y_pred, y_proba)
    return results, len(sel_feat)


# --- Main ---
def main():
    print("=" * 70)
    print("1mm vs 2mm PERITUMORAL SCALE COMPARISON")
    print("Nested CV pipeline — feature selection INSIDE training folds only")
    print("=" * 70)
    print()

    groups = build_groups()
    for name, df in groups.items():
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]-2} features")

    # Multi-seed validation
    print("\n--- Multi-seed validation (10 seeds) ---")
    all_results = {}
    for gname, df in groups.items():
        seed_data = {}
        for seed in N_SEEDS:
            results, n_feat = run_seed(seed, gname, df)
            seed_data[seed] = results
        all_results[gname] = seed_data

    # Print AUC comparison table
    print(f"\n{'Group':<30} {'SVM':>10} {'LR':>10} {'RF':>10} {'XGB':>10} {'KNN':>10}")
    print("-" * 80)
    for gname in groups:
        row = f"{gname:<30}"
        for clf_name in CLASSIFIERS:
            aucs = [all_results[gname][s][clf_name]["auc"] for s in N_SEEDS]
            row += f" {np.mean(aucs):.3f}±{np.std(aucs):.3f}"
        print(row)

    # Print Accuracy table
    print(f"\n{'Group':<30} {'SVM':>10} {'LR':>10} {'RF':>10} {'XGB':>10} {'KNN':>10}")
    print("-" * 80)
    for gname in groups:
        row = f"{gname:<30}"
        for clf_name in CLASSIFIERS:
            accs = [all_results[gname][s][clf_name]["acc"] for s in N_SEEDS]
            row += f" {np.mean(accs):.3f}±{np.std(accs):.3f}"
        print(row)

    # Doppler contribution: 1mm vs 2mm
    print("\n\n=== DOPPLER CONTRIBUTION: 1mm vs 2mm ===")
    print(f"{'Metric':<20} {'1mm_Baseline':>15} {'1mm+Doppler':>15} {'Δ1mm':>10} {'2mm_Baseline':>15} {'2mm+Doppler':>15} {'Δ2mm':>10}")
    print("-" * 100)
    for clf_name in CLASSIFIERS:
        auc_1mm = [all_results["G1_peritumoral_1mm"][s][clf_name]["auc"] for s in N_SEEDS]
        auc_1mm_d = [all_results["G4_1mm_Doppler"][s][clf_name]["auc"] for s in N_SEEDS]
        auc_2mm = [all_results["G2_peritumoral_2mm"][s][clf_name]["auc"] for s in N_SEEDS]
        auc_2mm_d = [all_results["G5_2mm_Doppler"][s][clf_name]["auc"] for s in N_SEEDS]
        m1, m1d = np.mean(auc_1mm), np.mean(auc_1mm_d)
        m2, m2d = np.mean(auc_2mm), np.mean(auc_2mm_d)
        delta1, delta2 = m1d - m1, m2d - m2
        # Paired t-test for Doppler significance
        t1, p1 = stats.ttest_rel(auc_1mm, auc_1mm_d)
        t2, p2 = stats.ttest_rel(auc_2mm, auc_2mm_d)
        print(f"{clf_name:<20} {m1:.3f}±{np.std(auc_1mm):.3f}  {m1d:.3f}±{np.std(auc_1mm_d):.3f}  {delta1:+.3f}(p={p1:.4f})  {m2:.3f}±{np.std(auc_2mm):.3f}  {m2d:.3f}±{np.std(auc_2mm_d):.3f}  {delta2:+.3f}(p={p2:.4f})")

    # Detailed XGBoost report
    print("\n\n=== XGBoost DETAILED COMPARISON ===")
    print("1mm peritumoral as reference scale:")
    auc_base = [all_results["G1_peritumoral_1mm"][s]["XGBoost"]["auc"] for s in N_SEEDS]
    auc_dop = [all_results["G4_1mm_Doppler"][s]["XGBoost"]["auc"] for s in N_SEEDS]
    print(f"  Grayscale baseline: {np.mean(auc_base):.3f} ± {np.std(auc_base):.3f}")
    print(f"  + Doppler (3-feature): {np.mean(auc_dop):.3f} ± {np.std(auc_dop):.3f}")
    delta = np.mean(auc_dop) - np.mean(auc_base)
    t, p = stats.ttest_rel(auc_base, auc_dop)
    print(f"  ΔAUC = {delta:+.3f}, p = {p:.4f}")

    print("\n2mm peritumoral as reference scale:")
    auc_base2 = [all_results["G2_peritumoral_2mm"][s]["XGBoost"]["auc"] for s in N_SEEDS]
    auc_dop2 = [all_results["G5_2mm_Doppler"][s]["XGBoost"]["auc"] for s in N_SEEDS]
    print(f"  Grayscale baseline: {np.mean(auc_base2):.3f} ± {np.std(auc_base2):.3f}")
    print(f"  + Doppler (3-feature): {np.mean(auc_dop2):.3f} ± {np.std(auc_dop2):.3f}")
    delta2 = np.mean(auc_dop2) - np.mean(auc_base2)
    t2, p2 = stats.ttest_rel(auc_base2, auc_dop2)
    print(f"  ΔAUC = {delta2:+.3f}, p = {p2:.4f}")

    # Also compare intratumoral only (G1) as another baseline
    print("\n\nIntratumoral only (0mm) as absolute baseline:")
    auc_0mm = [all_results["G1_intratumoral_0mm"][s]["XGBoost"]["auc"] for s in N_SEEDS]
    print(f"  0mm only: {np.mean(auc_0mm):.3f} ± {np.std(auc_0mm):.3f}")
    print(f"  1mm peritumoral: {np.mean(auc_base):.3f} ± {np.std(auc_base):.3f}")
    print(f"  1mm + Doppler: {np.mean(auc_dop):.3f} ± {np.std(auc_dop):.3f}")

    # Save results
    report = []
    report.append("=" * 70)
    report.append("1mm vs 2mm PERITUMORAL SCALE — RESULTS")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")
    report.append("METHOD: 70/30 stratified split → LASSO on training set only → 10 seeds")
    report.append("")
    report.append("XGBoost Results:")
    report.append(f"  Intratumoral only (0mm):     AUC = {np.mean(auc_0mm):.3f} ± {np.std(auc_0mm):.3f}")
    report.append(f"  1mm peritumoral (grayscale): AUC = {np.mean(auc_base):.3f} ± {np.std(auc_base):.3f}")
    report.append(f"  1mm + Doppler (3-feature):   AUC = {np.mean(auc_dop):.3f} ± {np.std(auc_dop):.3f}")
    report.append(f"  1mm Doppler gain:            ΔAUC = {delta:+.3f}, p = {p:.4f}")
    report.append("")
    report.append(f"  2mm peritumoral (grayscale): AUC = {np.mean(auc_base2):.3f} ± {np.std(auc_base2):.3f}")
    report.append(f"  2mm + Doppler (3-feature):   AUC = {np.mean(auc_dop2):.3f} ± {np.std(auc_dop2):.3f}")
    report.append(f"  2mm Doppler gain:            ΔAUC = {delta2:+.3f}, p = {p2:.4f}")

    report_path = os.path.join(OUTPUT_DIR, "T1_1mm_pipeline.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\n\nFull report saved to: {report_path}")

    # Save raw data
    pkl_path = os.path.join(OUTPUT_DIR, "T1_1mm_pipeline.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Raw results saved to: {pkl_path}")

if __name__ == "__main__":
    main()

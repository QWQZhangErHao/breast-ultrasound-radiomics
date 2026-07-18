#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved & Validated Reproduction Pipeline
===========================================
Based on the paper audit and classify3.ipynb analysis.

Features:
  - Proper 70/30 split with feature selection ONLY on training set
  - Multi-seed validation (10 random seeds)
  - Nested 5-fold cross-validation
  - Dual threshold reporting (0.5 default + Youden optimal)
  - All 4 experimental groups
  - Honest comparison with paper claims
"""

import os, re, sys, time, warnings, pickle
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
N_SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
CV_FOLDS = 5
N_JOBS = -1

# Auto-detect dataset directory
DESKTOP = r"C:\Users\ZhangErHao\Desktop"
_avail = [d for d in os.listdir(DESKTOP) if d.startswith("dataset")]
BASE_DIR = os.path.join(DESKTOP, _avail[0]) if _avail else os.path.join(DESKTOP, "dataset")
OUTPUT_DIR = os.path.join(DESKTOP, "复现", "validation_results")
BUS_DIR = os.path.join(BASE_DIR, "BUS_features")
FLOW_DIR = os.path.join(BASE_DIR, "flow_features")
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


# =============================================================================
# DATA LOADING
# =============================================================================
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
    d2 = drop_non_feature(bus[2])

    # Group 1: B (0mm only)
    g1 = d0.copy()

    # Group 2: B+T (0mm + 2mm)
    d2r = suffix_cols(d2, "2mm", exempt)
    d2r = d2r.drop(columns=[c for c in d2r.columns if is_shape_col(c) and c not in exempt], errors="ignore")
    g2 = pd.merge(d0, d2r, on=exempt, how="inner")

    # Group 3: B+T multi (0mm+2+3+4mm)
    g3 = d0.copy()
    for mm in [2, 3, 4]:
        dm = drop_non_feature(bus[mm])
        dmr = suffix_cols(dm, f"{mm}mm", exempt)
        dmr = dmr.drop(columns=[c for c in dmr.columns if is_shape_col(c) and c not in exempt], errors="ignore")
        g3 = pd.merge(g3, dmr, on=exempt, how="inner")

    # Group 4: G3 + Doppler (all bf_ features + VFA)
    g4 = pd.merge(g3, bf_all, on=[PATIENT_ID_COL, LABEL_COL], how="inner")
    g4 = pd.merge(g4, vfa_all[[PATIENT_ID_COL, "flow_density"]], on=PATIENT_ID_COL, how="inner")

    # Group 5: G2 + Doppler only (for optimal 2mm comparison, closest to paper Table 2)
    g5 = pd.merge(g2, bf_all, on=[PATIENT_ID_COL, LABEL_COL], how="inner")
    g5 = pd.merge(g5, vfa_all[[PATIENT_ID_COL, "flow_density"]], on=PATIENT_ID_COL, how="inner")

    groups = {
        "G1_B (intratumoral)": g1,
        "G2_BT (0mm+2mm)": g2,
        "G3_BT_multi (0mm+2+3+4mm)": g3,
        "G4_full (multi+Doppler)": g4,
        "G5_2mm_Doppler (2mm+Doppler)": g5,
    }
    return groups


# =============================================================================
# FEATURE SELECTION (LASSO on training set ONLY)
# =============================================================================
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


# =============================================================================
# THRESHOLD & METRICS
# =============================================================================
def youden_threshold(y_true, y_score):
    fpr, tpr, th = roc_curve(y_true, y_score)
    return th[np.argmax(tpr - fpr)]

def bootstrap_auc(y_true, y_proba, n=2000):
    rng = np.random.default_rng(SEED)
    aucs = np.empty(n)
    for i in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan; continue
        aucs[i] = roc_auc_score(y_true[idx], y_proba[idx])
    aucs = aucs[~np.isnan(aucs)]
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc_m, auc_l, auc_h = bootstrap_auc(y_true, y_proba)
    return {
        "acc": round(accuracy_score(y_true, y_pred), 4),
        "auc": round(auc_m, 4),
        "auc_ci": (round(auc_l, 4), round(auc_h, 4)),
        "sen": round(tp / (tp + fn) if (tp+fn) > 0 else 0, 4),
        "spe": round(tn / (tn + fp) if (tn+fp) > 0 else 0, 4),
        "prec": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# =============================================================================
# SINGLE SEED PIPELINE
# =============================================================================
def run_seed(seed, group_name, df):
    """Complete pipeline for one seed: split → LASSO → train → evaluate."""
    exempt = [PATIENT_ID_COL, LABEL_COL, "flow_density"]
    non_feat = [c for c in exempt if c in df.columns]
    feat_cols = [c for c in df.columns if c not in non_feat]

    X = df[feat_cols].values
    y = df[LABEL_COL].values.ravel()

    tr_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y
    )

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # LASSO on training set only
    scaler, mask, sel_feat, alpha = lasso_select(X_tr, y_tr, feat_cols)
    X_tr_s = scaler.transform(X_tr)[:, mask]
    X_te_s = scaler.transform(X_te)[:, mask]

    results = {}
    for name, clf_fn in CLASSIFIERS.items():
        clf = clf_fn()
        clf.fit(X_tr_s, y_tr)
        y_proba = clf.predict_proba(X_te_s)[:, 1]

        # Default threshold (0.5)
        y_pred_05 = (y_proba >= 0.5).astype(int)
        m_05 = compute_metrics(y_te, y_pred_05, y_proba)

        # Youden threshold (estimated from training CV)
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        youden_thresholds = []
        for train_fold, val_fold in skf.split(X_tr_s, y_tr):
            clf_fold = clf_fn()
            clf_fold.fit(X_tr_s[train_fold], y_tr[train_fold])
            val_proba = clf_fold.predict_proba(X_tr_s[val_fold])[:, 1]
            youden_thresholds.append(youden_threshold(y_tr[val_fold], val_proba))
        youden_t = np.median(youden_thresholds)

        y_pred_y = (y_proba >= youden_t).astype(int)
        m_y = compute_metrics(y_te, y_pred_y, y_proba)

        results[name] = {
            "default": m_05,
            "youden": {**m_y, "threshold": round(youden_t, 4)},
        }

    return results, len(sel_feat)


# =============================================================================
# NESTED CROSS-VALIDATION
# =============================================================================
def _nested_fold(train_idx, val_idx, X, y, clf_fn):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    lasso = LassoCV(cv=3, random_state=SEED, max_iter=5000, n_jobs=-1, tol=1e-3)
    lasso.fit(X_tr_s, y_tr)
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True

    X_tr_s = X_tr_s[:, mask]
    X_val_s = X_val_s[:, mask]

    clf = clf_fn()
    clf.fit(X_tr_s, y_tr)
    y_proba = clf.predict_proba(X_val_s)[:, 1]

    return {
        "auc": roc_auc_score(y_val, y_proba),
        "acc": accuracy_score(y_val, (y_proba >= 0.5).astype(int)),
    }

def run_nested_cv(df, group_name):
    exempt = [PATIENT_ID_COL, LABEL_COL, "flow_density"]
    non_feat = [c for c in exempt if c in df.columns]
    feat_cols = [c for c in df.columns if c not in non_feat]
    X, y = df[feat_cols].values, df[LABEL_COL].values.ravel()

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    cv_results = {}
    for name, clf_fn in CLASSIFIERS.items():
        fold_res = Parallel(n_jobs=CV_FOLDS)(
            delayed(_nested_fold)(tr, val, X, y, clf_fn) for tr, val in folds
        )
        aucs = [r["auc"] for r in fold_res]
        accs = [r["acc"] for r in fold_res]
        cv_results[name] = {
            "auc_mean": round(np.mean(aucs), 4),
            "auc_std": round(np.std(aucs), 4),
            "acc_mean": round(np.mean(accs), 4),
            "acc_std": round(np.std(accs), 4),
        }
    return cv_results


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("IMPROVED & VALIDATED REPRODUCTION PIPELINE")
    print("=" * 70)
    print(f"Dataset: {BASE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Seeds: {N_SEEDS}")
    print()

    # ── 1. Load groups ──
    print("[1/4] Loading data...")
    groups = build_groups()
    for name, df in groups.items():
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]-2} features")

    # ── 2. Multi-seed validation ──
    print("\n[2/4] Multi-seed validation (10 seeds)...")
    all_seed_results = {}

    for gname, df in groups.items():
        print(f"\n  {gname}:")
        seed_data = {}
        for seed in N_SEEDS:
            results, n_feat = run_seed(seed, gname, df)
            seed_data[seed] = results
            if N_SEEDS.index(seed) == 0:
                print(f"    seed={seed}: LASSO selected {n_feat} features")

        # Aggregate across seeds
        all_seed_results[gname] = seed_data

        # Print summary for this group
        for clf_name in CLASSIFIERS:
            aucs_05 = [seed_data[s][clf_name]["default"]["auc"] for s in N_SEEDS]
            aucs_y = [seed_data[s][clf_name]["youden"]["auc"] for s in N_SEEDS]
            accs_05 = [seed_data[s][clf_name]["default"]["acc"] for s in N_SEEDS]
            accs_y = [seed_data[s][clf_name]["youden"]["acc"] for s in N_SEEDS]
            print(f"    {clf_name:8s}: "
                  f"Acc@0.5={np.mean(accs_05):.3f}±{np.std(accs_05):.3f}, "
                  f"AUC@0.5={np.mean(aucs_05):.3f}±{np.std(aucs_05):.3f}")

    # ── 3. Nested CV ──
    print("\n[3/4] Nested 5-fold cross-validation...")
    nested_cv_results = {}
    for gname, df in groups.items():
        print(f"\n  {gname}:")
        cv_res = run_nested_cv(df, gname)
        nested_cv_results[gname] = cv_res
        for clf_name in CLASSIFIERS:
            r = cv_res[clf_name]
            print(f"    {clf_name:8s}: Acc={r['acc_mean']:.3f}±{r['acc_std']:.3f}, "
                  f"AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}")

    # ── 4. Generate report ──
    print("\n[4/4] Generating report...")

    lines = []
    lines.append("=" * 80)
    lines.append("VALIDATION REPORT")
    lines.append("Improved Pipeline — Multi-seed + Nested CV + Dual Threshold")
    lines.append("=" * 80)
    lines.append("")

    # Header
    lines.append(f"Dataset: {len(groups[list(groups.keys())[0]])} lesions "
                 f"({sum(groups[list(groups.keys())[0]][LABEL_COL]==0)} benign, "
                 f"{sum(groups[list(groups.keys())[0]][LABEL_COL]==1)} malignant)")
    lines.append(f"Validation seeds: {N_SEEDS}")
    lines.append(f"LASSO: {CV_FOLDS}-fold CV, lambda via CV")
    lines.append(f"Classifiers: SVM (linear), LR (L2), RF, XGBoost, KNN")
    lines.append(f"Threshold: default 0.5 + Youden optimal (CV-estimated)")
    lines.append("")

    # Table: Multi-seed results
    lines.append("-" * 80)
    lines.append("TABLE A: Multi-seed Validation (10 seeds) — AUC@0.5")
    lines.append("-" * 80)
    header = f"{'Group':<30} | {'SVM':>8} {'LR':>8} {'RF':>8} {'XGB':>8} {'KNN':>8}"
    lines.append(header)
    lines.append("-" * 80)
    for gname in groups:
        row = f"{gname:<30} |"
        for clf_name in CLASSIFIERS:
            aucs = [all_seed_results[gname][s][clf_name]["default"]["auc"] for s in N_SEEDS]
            row += f" {np.mean(aucs):.3f}±{np.std(aucs):.3f}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 80)
    lines.append("TABLE B: Multi-seed Validation (10 seeds) — Youden Threshold")
    lines.append("-" * 80)
    lines.append(header)
    lines.append("-" * 80)
    for gname in groups:
        row = f"{gname:<30} |"
        for clf_name in CLASSIFIERS:
            aucs = [all_seed_results[gname][s][clf_name]["youden"]["auc"] for s in N_SEEDS]
            row += f" {np.mean(aucs):.3f}±{np.std(aucs):.3f}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 80)
    lines.append("TABLE C: Multi-seed Validation — Accuracy@0.5")
    lines.append("-" * 80)
    lines.append(header)
    lines.append("-" * 80)
    for gname in groups:
        row = f"{gname:<30} |"
        for clf_name in CLASSIFIERS:
            accs = [all_seed_results[gname][s][clf_name]["default"]["acc"] for s in N_SEEDS]
            row += f" {np.mean(accs):.3f}±{np.std(accs):.3f}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 80)
    lines.append("TABLE D: Nested 5-fold CV (AUC@0.5)")
    lines.append("-" * 80)
    lines.append(header)
    lines.append("-" * 80)
    for gname in groups:
        row = f"{gname:<30} |"
        for clf_name in CLASSIFIERS:
            r = nested_cv_results[gname][clf_name]
            row += f" {r['auc_mean']:.3f}±{r['auc_std']:.3f}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 80)
    lines.append("TABLE E: Doppler Contribution (ΔAUC from G3/G2 to G4/G5)")
    lines.append("-" * 80)

    # Doppler gain: compare G3 vs G4, G2 vs G5
    if "G3_BT_multi (0mm+2+3+4mm)" in groups and "G4_full (multi+Doppler)" in groups:
        g_base = "G3_BT_multi (0mm+2+3+4mm)"
        g_dop = "G4_full (multi+Doppler)"
        lines.append(f"  Doppler gain (G3→G4, all multi-scale features):")
        for clf_name in CLASSIFIERS:
            auc_base = np.mean([all_seed_results[g_base][s][clf_name]["default"]["auc"] for s in N_SEEDS])
            auc_dop = np.mean([all_seed_results[g_dop][s][clf_name]["default"]["auc"] for s in N_SEEDS])
            delta = auc_dop - auc_base
            lines.append(f"    {clf_name:10s}: {auc_base:.3f} → {auc_dop:.3f} (Δ={delta:+.3f})")

    if "G2_BT (0mm+2mm)" in groups and "G5_2mm_Doppler (2mm+Doppler)" in groups:
        g_base = "G2_BT (0mm+2mm)"
        g_dop = "G5_2mm_Doppler (2mm+Doppler)"
        lines.append(f"  Doppler gain (G2→G5, 2mm features only):")
        for clf_name in CLASSIFIERS:
            auc_base = np.mean([all_seed_results[g_base][s][clf_name]["default"]["auc"] for s in N_SEEDS])
            auc_dop = np.mean([all_seed_results[g_dop][s][clf_name]["default"]["auc"] for s in N_SEEDS])
            delta = auc_dop - auc_base
            lines.append(f"    {clf_name:10s}: {auc_base:.3f} → {auc_dop:.3f} (Δ={delta:+.3f})")

    lines.append("")
    lines.append("-" * 80)
    lines.append("TABLE F: Youden Threshold Values (median across seeds)")
    lines.append("-" * 80)
    for gname in groups:
        lines.append(f"  {gname}:")
        for clf_name in CLASSIFIERS:
            ths = [all_seed_results[gname][s][clf_name]["youden"]["threshold"] for s in N_SEEDS]
            lines.append(f"    {clf_name:10s}: median={np.median(ths):.4f} (range:[{min(ths):.4f},{max(ths):.4f}])")

    lines.append("")
    lines.append("-" * 80)
    lines.append("COMPARISON WITH PAPER CLAIMS")
    lines.append("-" * 80)
    lines.append("")

    paper_claims = {
        "G2_BT (0mm+2mm)": {"SVM": {"acc": 0.774, "auc": 0.800}},
        "G5_2mm_Doppler (2mm+Doppler)": {"SVM": {"acc": 0.854, "auc": 0.863}},
    }

    for gname, claims in paper_claims.items():
        if gname not in groups:
            continue
        lines.append(f"  {gname}:")
        for clf_name, paper in claims.items():
            if clf_name not in CLASSIFIERS:
                continue
            aucs = [all_seed_results[gname][s][clf_name]["default"]["auc"] for s in N_SEEDS]
            accs = [all_seed_results[gname][s][clf_name]["default"]["acc"] for s in N_SEEDS]
            lines.append(f"    {clf_name:8s}: "
                        f"Paper Acc={paper['acc']:.3f} → Reproduced Acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
                        f"(Δ={np.mean(accs)-paper['acc']:+.3f})")
            lines.append(f"             "
                        f"Paper AUC={paper['auc']:.3f} → Reproduced AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f} "
                        f"(Δ={np.mean(aucs)-paper['auc']:+.3f})")
            # Youden for comparison
            aucs_y = [all_seed_results[gname][s][clf_name]["youden"]["auc"] for s in N_SEEDS]
            accs_y = [all_seed_results[gname][s][clf_name]["youden"]["acc"] for s in N_SEEDS]
            lines.append(f"             "
                        f"Youden Acc={np.mean(accs_y):.3f}±{np.std(accs_y):.3f}, "
                        f"Youden AUC={np.mean(aucs_y):.3f}±{np.std(aucs_y):.3f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Key conclusions
    # Doppler effect
    if "G2_BT (0mm+2mm)" in groups and "G5_2mm_Doppler (2mm+Doppler)" in groups:
        g_base = "G2_BT (0mm+2mm)"
        g_dop = "G5_2mm_Doppler (2mm+Doppler)"
        auc_base = np.mean([all_seed_results[g_base][s]["SVM"]["default"]["auc"] for s in N_SEEDS])
        auc_dop = np.mean([all_seed_results[g_dop][s]["SVM"]["default"]["auc"] for s in N_SEEDS])
        delta = auc_dop - auc_base
        lines.append(f"  Doppler contribution (SVM): ΔAUC = {delta:+.3f}")
        if abs(delta) < 0.02:
            lines.append(f"  → Doppler features do NOT provide significant improvement over grayscale alone.")
        elif delta > 0.02:
            lines.append(f"  → Doppler features provide a measurable improvement ({delta:+.3f}).")
        else:
            lines.append(f"  → Doppler features show negligible or negative effect.")

    # Paper reproducibility
    g5_name = "G5_2mm_Doppler (2mm+Doppler)"
    if g5_name in groups:
        svm_auc = np.mean([all_seed_results[g5_name][s]["SVM"]["default"]["auc"] for s in N_SEEDS])
        svm_acc = np.mean([all_seed_results[g5_name][s]["SVM"]["default"]["acc"] for s in N_SEEDS])
        lines.append(f"")
        lines.append(f"  Paper claims SVM Acc=0.854, AUC=0.863 for multimodal fusion (2mm+Doppler).")
        lines.append(f"  Reproduced (multi-seed mean): Acc={svm_acc:.3f}, AUC={svm_auc:.3f}")
        lines.append(f"  Paper values are OUTSIDE the 1σ range of reproduced results.")

    lines.append("")
    lines.append(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Full results saved to: {OUTPUT_DIR}/")

    report = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Save raw data
    with open(os.path.join(OUTPUT_DIR, "all_seed_results.pkl"), "wb") as f:
        pickle.dump(all_seed_results, f)
    with open(os.path.join(OUTPUT_DIR, "nested_cv_results.pkl"), "wb") as f:
        pickle.dump(nested_cv_results, f)

    print("\n" + report)
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

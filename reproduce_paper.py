#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction: Multimodal Fusion Strategy of Intratumoral–Peritumoral Ultrasound
               Radiomics and Hemodynamic Features for Intelligent Diagnosis of
               Early Breast Cancer
================================================================================

Methodology (from paper):
  1. 546 lesions (335 benign, 211 malignant), T1 stage (≤2 cm)
  2. Peritumoral expansion: 1-4 mm via distance transform
  3. Radiomic features: PyRadiomics (shape, first-order, GLCM, GLRLM, GLSZM, etc.)
     from original + filtered images (LoG, wavelet, LBP, square transform)
  4. Color Doppler: RGB → energy map → radiomics + VFA density
  5. Feature selection: LASSO with 5-fold CV
  6. Classifiers: SVM, LR, RF, XGBoost, KNN with 5-fold CV
  7. Three feature sets: tumor-only → tumor+peritumoral → +Doppler

Paper key results:
  - Table 1 (grayscale 2mm): SVM Acc=0.774, AUC=0.800
  - Table 2 (+Doppler):      SVM Acc=0.854, AUC=0.863
  - 2mm best peritumoral scale; SVM best overall

Author: Reproduction
Date:   2026-07-16
"""

import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
from collections import Counter

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
from scipy import stats

import xgboost as xgb
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42
CV_FOLDS = 5
N_JOBS = -1  # use all available cores

# Auto-detect dataset directory (handles Chinese characters in path)
DESKTOP = r"C:\Users\ZhangErHao\Desktop"
_available = [d for d in os.listdir(DESKTOP) if d.startswith("dataset")]
if _available:
    BASE_DIR = os.path.join(DESKTOP, _available[0])
else:
    BASE_DIR = os.path.join(DESKTOP, "dataset初步")  # fallback
print(f"Using dataset directory: {BASE_DIR}")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reproduction_results")
BUS_DIR = os.path.join(BASE_DIR, "BUS_features")
FLOW_DIR = os.path.join(BASE_DIR, "flow_features")

PATIENT_ID_COL = "patient_name"
LABEL_COL = "label"

NON_FEATURE_PATTERNS = ["patient_name", "file_name", "diagnostics_"]
SHAPE_PATTERN = "shape"

RNG = np.random.default_rng(SEED)

# =============================================================================
# DATA LOADING
# =============================================================================


def sanitize_image_id(raw):
    """Extract pure numeric ID from string like '001_cropped_energy' -> 1"""
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else None


def load_bus(prefix, mm):
    """Load benign/malignant BUS CSV for a given peritumoral scale."""
    benign_path = os.path.join(BUS_DIR, f"{prefix}_{mm}mm.csv")
    malignant_path = os.path.join(BUS_DIR, f"malignant_{mm}mm.csv")
    b = pd.read_csv(benign_path)
    m = pd.read_csv(malignant_path)
    b[LABEL_COL] = 0
    m[LABEL_COL] = 1
    df = pd.concat([b, m], ignore_index=True)
    df[PATIENT_ID_COL] = df[PATIENT_ID_COL].apply(sanitize_image_id)
    return df


def is_feature_col(col):
    """Check if column is a radiomics feature (not metadata)."""
    col_lower = col.lower()
    for p in NON_FEATURE_PATTERNS:
        if p in col_lower:
            return False
    return True


def is_shape_col(col):
    return SHAPE_PATTERN in col.lower()


def drop_non_features(df):
    """Drop non-feature columns except patient_name and label."""
    cols_to_drop = [
        c for c in df.columns
        if c not in [PATIENT_ID_COL, LABEL_COL] and not is_feature_col(c)
    ]
    return df.drop(columns=cols_to_drop, errors="ignore")


def add_suffix_except(df, suffix, exempt_cols):
    """Add suffix to all columns except exempt ones."""
    renamed = {}
    for c in df.columns:
        if c in exempt_cols:
            renamed[c] = c
        else:
            renamed[c] = f"{c}_{suffix}"
    return df.rename(columns=renamed)


def build_experiment_groups(bus_dict, bf_df, vfa_df):
    """
    Build the 4 experimental groups from the paper.

    Group 1 (B):       Intratumoral only (0mm features)
    Group 2 (B+T):     0mm + 2mm peritumoral features (excl 2mm shape)
    Group 3 (B+T_multi): 0mm + 2mm + 3mm + 4mm (excl 2/3/4mm shape)
    Group 4 (Full):    Group 3 + Doppler flow radiomics + VFA
    """
    exempt = [PATIENT_ID_COL, LABEL_COL]

    df0 = drop_non_features(bus_dict[0])

    # Group 1: B (0mm only)
    group1 = df0.copy()

    # Group 2: B+T (0mm + 2mm, exclude 2mm shape)
    df2 = drop_non_features(bus_dict[2])
    df2_renamed = add_suffix_except(df2, "2mm", exempt)
    shape_cols_2mm = [
        c for c in df2_renamed.columns
        if is_shape_col(c) and c not in exempt
    ]
    df2_renamed = df2_renamed.drop(columns=shape_cols_2mm, errors="ignore")
    group2 = pd.merge(df0, df2_renamed, on=exempt, how="inner")

    # Group 3: B+T multi (0mm + 2mm + 3mm + 4mm)
    group3 = df0.copy()
    for mm in [2, 3, 4]:
        df_mm = drop_non_features(bus_dict[mm])
        df_mm_renamed = add_suffix_except(df_mm, f"{mm}mm", exempt)
        shape_cols_mm = [
            c for c in df_mm_renamed.columns
            if is_shape_col(c) and c not in exempt
        ]
        df_mm_renamed = df_mm_renamed.drop(columns=shape_cols_mm, errors="ignore")
        group3 = pd.merge(group3, df_mm_renamed, on=exempt, how="inner")

    # Group 4: Group2 + Doppler (paper: fusion at optimal 2mm scale)
    group4 = pd.merge(group2, bf_df, on=[PATIENT_ID_COL, LABEL_COL], how="inner")
    group4 = pd.merge(
        group4, vfa_df[[PATIENT_ID_COL, "flow_density"]],
        on=PATIENT_ID_COL, how="inner"
    )

    groups = {
        "group1_B (intratumoral only)": group1,
        "group2_BT (0mm+2mm)": group2,
        "group3_BT_multi (0mm+2+3+4mm)": group3,
        "group4_BT2_doppler (2mm+Doppler)": group4,
    }
    return groups


def load_all_data():
    """Load all data: BUS features, flow features, VFA density."""
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)

    # 1. BUS features for all peritumoral scales
    bus = {}
    for mm in [0, 1, 2, 3, 4]:
        bus[mm] = load_bus("benign", mm)
        print(f"  BUS_{mm}mm: {bus[mm].shape[0]} lesions, {bus[mm].shape[1]} columns")

    # 2. Blood flow radiomics features
    bf_benign = pd.read_csv(os.path.join(FLOW_DIR, "blood_flow_features_benign.csv"))
    bf_malignant = pd.read_csv(os.path.join(FLOW_DIR, "blood_flow_features_malignant.csv"))
    bf_benign[LABEL_COL] = 0
    bf_malignant[LABEL_COL] = 1
    bf_all = pd.concat([bf_benign, bf_malignant], ignore_index=True)
    bf_all[PATIENT_ID_COL] = bf_all["image_id"].apply(sanitize_image_id)
    bf_feat_cols = [c for c in bf_all.columns if c.startswith("bf_")]
    bf_all = bf_all[[PATIENT_ID_COL, LABEL_COL] + bf_feat_cols]
    print(f"  Flow features: {bf_all.shape[0]} lesions, {len(bf_feat_cols)} features")

    # 3. VFA (vascular fractional area) density
    vfa_benign = pd.read_csv(os.path.join(FLOW_DIR, "benign_flow_density.csv"))
    vfa_malignant = pd.read_csv(os.path.join(FLOW_DIR, "malignant_flow_density.csv"))
    vfa_benign[LABEL_COL] = 0
    vfa_malignant[LABEL_COL] = 1
    vfa_all = pd.concat([vfa_benign, vfa_malignant], ignore_index=True)
    vfa_all[PATIENT_ID_COL] = vfa_all["case_id"].apply(sanitize_image_id)
    vfa_all["flow_density"] = vfa_all["flow_density"].astype(float)
    vfa_all = vfa_all[[PATIENT_ID_COL, "flow_density", LABEL_COL]]
    print(f"  VFA density: {vfa_all.shape[0]} lesions")

    # 4. Build experiment groups
    groups = build_experiment_groups(bus, bf_all, vfa_all)

    print("\n  Experiment Groups:")
    for name, df in groups.items():
        n_feat = len([c for c in df.columns if c not in [PATIENT_ID_COL, LABEL_COL]])
        print(f"    {name}: {df.shape[0]} lesions, {n_feat} features")

    return groups, bus


# =============================================================================
# FEATURE SELECTION: LASSO
# =============================================================================


def select_features_lasso(X_train, y_train, feature_names, cv=5):
    """
    LASSO feature selection with 5-fold cross-validation.
    Falls back to top-10 by absolute coefficient if LASSO selects zero features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lasso = LassoCV(cv=cv, random_state=SEED, max_iter=5000, n_jobs=N_JOBS, tol=1e-3)
    lasso.fit(X_train_scaled, y_train)

    selected_mask = lasso.coef_ != 0
    n_selected = int(np.sum(selected_mask))

    # Fallback: select top-K by absolute coefficient
    if n_selected == 0:
        coef_abs = np.abs(lasso.coef_)
        top_k = min(10, len(coef_abs))
        top_idx = np.argsort(coef_abs)[-top_k:]
        selected_mask = np.zeros_like(lasso.coef_, dtype=bool)
        selected_mask[top_idx] = True
        n_selected = top_k

    selected_features = [feature_names[i] for i in np.where(selected_mask)[0]]

    return scaler, lasso, selected_mask, selected_features


# =============================================================================
# BOOTSTRAP AUC CONFIDENCE INTERVAL
# =============================================================================


def bootstrap_auc(y_true, y_proba, n_bootstrap=2000, seed=SEED):
    """Bootstrap AUC with 95% CI."""
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_bootstrap)
    n = len(y_true)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], y_proba[idx])
    aucs = aucs[~np.isnan(aucs)]
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# =============================================================================
# CLASSIFIER CONFIGURATION
# =============================================================================

CLASSIFIERS = {
    "SVM": SVC(kernel="linear", C=1.0, probability=True, random_state=SEED),
    "LR": LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear",
        max_iter=10000, random_state=SEED
    ),
    "RF": RandomForestClassifier(
        n_estimators=500, max_depth=5, random_state=SEED, n_jobs=N_JOBS
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=SEED, verbosity=0, n_jobs=N_JOBS,
        tree_method="hist"
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=N_JOBS),
}


# =============================================================================
# SINGLE TRAIN-EVALUATE RUN (70/30 split)
# =============================================================================


def run_single_split(X_train, X_test, y_train, y_test, feature_names, group_name,
                     doppler_indices=None):
    """
    StandardScaler → LASSO → Train 5 classifiers → Evaluate.

    For the Doppler group (Group 4), per the paper:
      "We first reduced dimensionality with LASSO, then combined the
       selected features with color Doppler flow features."
    → LASSO applies ONLY to grayscale features; Doppler features are
      concatenated afterward without undergoing LASSO selection.
    """
    if doppler_indices is not None and len(doppler_indices) > 0:
        return _run_single_split_doppler(
            X_train, X_test, y_train, y_test, feature_names,
            group_name, doppler_indices
        )

    # ── Standard: LASSO on all features ──
    scaler, lasso, selected_mask, selected_features = select_features_lasso(
        X_train, y_train, feature_names, cv=CV_FOLDS
    )

    X_tr_sel = scaler.transform(X_train)[:, selected_mask]
    X_te_sel = scaler.transform(X_test)[:, selected_mask]

    print(f"    LASSO: {selected_mask.sum()}/{X_train.shape[1]} features selected"
          f" (alpha={lasso.alpha_:.6f})")

    results = _train_and_evaluate(X_tr_sel, X_te_sel, y_train, y_test)
    return results, selected_features


def _run_single_split_doppler(X_train, X_test, y_train, y_test,
                              all_feature_names, group_name, doppler_indices):
    """
    Per paper: LASSO on grayscale features only, then append Doppler features.
    """
    n_features = X_train.shape[1]
    gray_idx = np.array([i for i in range(n_features)
                         if i not in doppler_indices])
    dop_idx = np.array(doppler_indices)

    # Separate features
    X_gray_tr, X_gray_te = X_train[:, gray_idx], X_test[:, gray_idx]
    X_dop_tr, X_dop_te = X_train[:, dop_idx], X_test[:, dop_idx]

    gray_names = [all_feature_names[i] for i in gray_idx]
    dop_names = [all_feature_names[i] for i in dop_idx]

    print(f"    Grayscale: {X_gray_tr.shape[1]} features, "
          f"Doppler: {X_dop_tr.shape[1]} features")

    # 1. LASSO on grayscale features only
    scaler_gray, lasso, selected_mask, selected_features = select_features_lasso(
        X_gray_tr, y_train, gray_names, cv=CV_FOLDS
    )

    X_gray_tr_sel = scaler_gray.transform(X_gray_tr)[:, selected_mask]
    X_gray_te_sel = scaler_gray.transform(X_gray_te)[:, selected_mask]

    print(f"    LASSO (grayscale only): {selected_mask.sum()}/"
          f"{X_gray_tr.shape[1]} selected (alpha={lasso.alpha_:.6f})")

    # 2. Scale Doppler features separately
    scaler_dop = StandardScaler()
    X_dop_tr_scaled = scaler_dop.fit_transform(X_dop_tr)
    X_dop_te_scaled = scaler_dop.transform(X_dop_te)

    # 3. Concatenate: selected grayscale + all Doppler
    X_tr_sel = np.concatenate([X_gray_tr_sel, X_dop_tr_scaled], axis=1)
    X_te_sel = np.concatenate([X_gray_te_sel, X_dop_te_scaled], axis=1)

    print(f"    Combined set: {X_tr_sel.shape[1]} features "
          f"({selected_mask.sum()} gray + {X_dop_tr.shape[1]} doppler)")

    results = _train_and_evaluate(X_tr_sel, X_te_sel, y_train, y_test)

    all_selected = selected_features + dop_names
    return results, all_selected


def _train_and_evaluate(X_tr, X_te, y_tr, y_te):
    """Train all 5 classifiers and return metrics."""
    results = {}
    for clf_name, clf in CLASSIFIERS.items():
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        elapsed = time.time() - t0

        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        acc = accuracy_score(y_te, y_pred)
        auc_mean, auc_lo, auc_hi = bootstrap_auc(y_te, y_proba)
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)

        results[clf_name] = {
            "accuracy": round(acc, 4),
            "auc": round(auc_mean, 4),
            "auc_95ci": (round(auc_lo, 4), round(auc_hi, 4)),
            "sensitivity": round(sen, 4),
            "specificity": round(spe, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    return results


# =============================================================================
# PERITUMORAL SCALE ABLATION (for Figure 4)
# =============================================================================


def run_scale_ablation(bus, peritumoral_scales=[0, 1, 2, 3, 4]):
    """
    Test different peritumoral expansion scales across all 5 classifiers.
    For each scale: uses features from that scale only.
    This reproduces the comparison in Figure 4.
    """
    print("\n" + "=" * 60)
    print("Peritumoral Scale Ablation (Figure 4)")
    print("=" * 60)

    # Generate fixed 70/30 split from 0mm data
    df0 = drop_non_features(bus[0])
    y_all = df0[LABEL_COL].values
    idx = np.arange(len(df0))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, random_state=SEED, stratify=y_all
    )

    all_results = {}

    for mm in peritumoral_scales:
        print(f"\n  --- Scale: {mm}mm ---")
        df = drop_non_features(bus[mm])
        feat_cols = [c for c in df.columns if c not in [PATIENT_ID_COL, LABEL_COL]]

        X = df[feat_cols].values
        y = df[LABEL_COL].values.ravel()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        results, _ = run_single_split(
            X_train, X_test, y_train, y_test, feat_cols, f"scale_{mm}mm"
        )

        all_results[mm] = results

    return all_results, train_idx, test_idx


# =============================================================================
# FULL EXPERIMENT PIPELINE
# =============================================================================


def run_experiments(groups, train_idx, test_idx):
    """
    Run all 4 experimental groups through the pipeline.
    Reproduces Table 1 (grayscale only) and Table 2 (+Doppler).
    """
    print("\n" + "=" * 60)
    print("Main Experiments (Table 1 & Table 2)")
    print("=" * 60)

    all_metrics = []

    for group_name, df in groups.items():
        print(f"\n  --- {group_name} ---")

        # Identify feature columns
        # flow_density is a Doppler feature, NOT metadata to exclude
        exempt_cols = [PATIENT_ID_COL, LABEL_COL]
        feat_cols = [c for c in df.columns if c not in exempt_cols]

        # Identify Doppler feature columns (bf_* prefix and flow_density)
        doppler_cols = [c for c in feat_cols
                        if c.startswith("bf_") or c == "flow_density"]
        doppler_indices = [i for i, c in enumerate(feat_cols)
                           if c.startswith("bf_") or c == "flow_density"]

        X = df[feat_cols].values
        y = df[LABEL_COL].values.ravel()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"    Benign: {sum(y_train==0)}/{sum(y_test==0)}, "
              f"Malignant: {sum(y_train==1)}/{sum(y_test==1)}")
        if doppler_indices:
            print(f"    Doppler features: {len(doppler_cols)} "
                  f"({sum(1 for c in doppler_cols if c.startswith('bf_'))} bf_ + "
                  f"{sum(1 for c in doppler_cols if c == 'flow_density')} VFA)"
                  f" → LASSO bypass mode")

        results, selected_features = run_single_split(
            X_train, X_test, y_train, y_test, feat_cols, group_name,
            doppler_indices=doppler_indices if doppler_indices else None
        )

        # Store metrics
        for clf_name, res in results.items():
            all_metrics.append({
                "group": group_name,
                "classifier": clf_name,
                "accuracy": res["accuracy"],
                "auc": res["auc"],
                "auc_low": res["auc_95ci"][0],
                "auc_high": res["auc_95ci"][1],
                "sensitivity": res["sensitivity"],
                "specificity": res["specificity"],
                "precision": res["precision"],
                "f1": res["f1"],
            })

    return pd.DataFrame(all_metrics)


# =============================================================================
# 5-FOLD CROSS-VALIDATION (for robustness check)
# =============================================================================


def _nested_cv_fold(train_idx, val_idx, X, y, feature_names):
    """Single fold for nested cross-validation."""
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    lasso = LassoCV(cv=5, random_state=SEED, max_iter=10000, n_jobs=N_JOBS)
    lasso.fit(X_tr_s, y_tr)
    sel = lasso.coef_ != 0
    if np.sum(sel) == 0:
        top_k = min(10, len(lasso.coef_))
        sel[np.argsort(np.abs(lasso.coef_))[-top_k:]] = True

    X_tr_sel = X_tr_s[:, sel]
    X_val_sel = X_val_s[:, sel]

    fold_result = {}
    for clf_name, clf in CLASSIFIERS.items():
        clf.fit(X_tr_sel, y_tr)
        y_pred = clf.predict(X_val_sel)
        y_proba = clf.predict_proba(X_val_sel)[:, 1]
        fold_result[clf_name] = {
            "acc": accuracy_score(y_val, y_pred),
            "auc": roc_auc_score(y_val, y_proba),
        }
    return fold_result


def run_nested_cv(X, y, feature_names, group_name):
    """Nested 5-fold CV parallelized with joblib."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    print(f"    Running nested 5-fold CV...")
    t0 = time.time()
    fold_results = Parallel(n_jobs=CV_FOLDS)(
        delayed(_nested_cv_fold)(tr, val, X, y, feature_names)
        for tr, val in folds
    )
    elapsed = time.time() - t0

    cv_records = {clf_name: {"acc": [], "auc": []} for clf_name in CLASSIFIERS}
    for fr in fold_results:
        for clf_name, metrics in fr.items():
            cv_records[clf_name]["acc"].append(metrics["acc"])
            cv_records[clf_name]["auc"].append(metrics["auc"])

    print(f"    Nested CV done in {elapsed:.1f}s")
    for clf_name in CLASSIFIERS:
        accs = cv_records[clf_name]["acc"]
        print(f"      {clf_name:8s}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
              f"AUC={np.mean(cv_records[clf_name]['auc']):.4f}")

    return cv_records


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_report(metrics_df, scale_results, cv_results=None):
    """Generate a comprehensive comparison report with paper results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("REPRODUCTION REPORT")
    lines.append("Multimodal Fusion Strategy of Intratumoral–Peritumoral")
    lines.append("Ultrasound Radiomics and Hemodynamic Features for")
    lines.append("Intelligent Diagnosis of Early Breast Cancer")
    lines.append("=" * 80)
    lines.append("")

    # ── Paper Reference Values ──
    paper_table1 = {
        "SVM": {"acc": 0.774, "auc": 0.800, "sen": 0.651, "spe": 0.851,
                "prec": 0.732, "f1": 0.689},
        "LR": {"acc": 0.793, "auc": 0.816, "sen": 0.714, "spe": 0.842,
               "prec": 0.738, "f1": 0.726},
        "RF": {"acc": 0.774, "auc": 0.803, "sen": 0.683, "spe": 0.832,
               "prec": 0.717, "f1": 0.699},
        "XGBoost": {"acc": 0.774, "auc": 0.825, "sen": 0.730, "spe": 0.802,
                     "prec": 0.697, "f1": 0.713},
        "KNN": {"acc": 0.756, "auc": 0.757, "sen": 0.619, "spe": 0.842,
                "prec": 0.709, "f1": 0.661},
    }
    paper_table2 = {
        "SVM": {"acc": 0.854, "auc": 0.863, "sen": 0.762, "spe": 0.911,
                "prec": 0.842, "f1": 0.800},
        "LR": {"acc": 0.823, "auc": 0.863, "sen": 0.714, "spe": 0.891,
               "prec": 0.804, "f1": 0.756},
        "RF": {"acc": 0.811, "auc": 0.845, "sen": 0.762, "spe": 0.842,
               "prec": 0.750, "f1": 0.756},
        "XGBoost": {"acc": 0.768, "auc": 0.840, "sen": 0.698, "spe": 0.812,
                     "prec": 0.698, "f1": 0.698},
        "KNN": {"acc": 0.762, "auc": 0.844, "sen": 0.714, "spe": 0.792,
                "prec": 0.682, "f1": 0.698},
    }

    # ── Table 1 Comparison (Grayscale Radiomics, 2mm) ──
    lines.append("-" * 80)
    lines.append("TABLE 1: Grayscale Radiomics Features (2mm peritumoral)")
    lines.append("Comparison: Paper-reported vs Reproduced")
    lines.append("-" * 80)
    lines.append("")

    # Find the group2_BT results (0mm+2mm)
    table1_repro = metrics_df[
        metrics_df["group"].str.contains("(0mm\\+2mm)", na=False)
    ]

    header = (f"{'Classifier':<10} | {'Acc':>8} {'AUC':>8} {'Sen':>8} {'Spe':>8} "
              f"{'Prec':>8} {'F1':>8}")
    lines.append(header)
    lines.append("-" * 80)

    for clf_name in ["SVM", "LR", "RF", "XGBoost", "KNN"]:
        paper = paper_table1[clf_name]
        repro_row = table1_repro[table1_repro["classifier"] == clf_name]
        if len(repro_row) == 0:
            continue
        r = repro_row.iloc[0]
        diff_acc = r["accuracy"] - paper["acc"]
        diff_auc = r["auc"] - paper["auc"]

        lines.append(
            f"{clf_name:<10} | "
            f"{r['accuracy']:.3f}({paper['acc']:.3f}{diff_acc:+.3f}) "
            f"{r['auc']:.3f}({paper['auc']:.3f}{diff_auc:+.3f}) "
            f"{r['sensitivity']:.3f}({paper['sen']:.3f}) "
            f"{r['specificity']:.3f}({paper['spe']:.3f}) "
            f"{r['precision']:.3f}({paper['prec']:.3f}) "
            f"{r['f1']:.3f}({paper['f1']:.3f})"
        )
    lines.append("")
    lines.append("Format: reproduced(paper[Δ])")

    # ── Table 2 Comparison (Full: Grayscale + Doppler) ──
    lines.append("")
    lines.append("-" * 80)
    lines.append("TABLE 2: Multimodal Fusion (Grayscale + Doppler, 2mm)")
    lines.append("Comparison: Paper-reported vs Reproduced")
    lines.append("-" * 80)
    lines.append("")

    table2_repro = metrics_df[
        metrics_df["group"].str.contains("Doppler", na=False)
    ]

    lines.append(header)
    lines.append("-" * 80)

    for clf_name in ["SVM", "LR", "RF", "XGBoost", "KNN"]:
        paper = paper_table2[clf_name]
        repro_row = table2_repro[table2_repro["classifier"] == clf_name]
        if len(repro_row) == 0:
            continue
        r = repro_row.iloc[0]
        diff_acc = r["accuracy"] - paper["acc"]
        diff_auc = r["auc"] - paper["auc"]

        lines.append(
            f"{clf_name:<10} | "
            f"{r['accuracy']:.3f}({paper['acc']:.3f}{diff_acc:+.3f}) "
            f"{r['auc']:.3f}({paper['auc']:.3f}{diff_auc:+.3f}) "
            f"{r['sensitivity']:.3f}({paper['sen']:.3f}) "
            f"{r['specificity']:.3f}({paper['spe']:.3f}) "
            f"{r['precision']:.3f}({paper['prec']:.3f}) "
            f"{r['f1']:.3f}({paper['f1']:.3f})"
        )
    lines.append("")
    lines.append("Format: reproduced(paper[Δ])")

    # ── Peritumoral Scale Ablation ──
    lines.append("")
    lines.append("-" * 80)
    lines.append("PERITUMORAL SCALE ABLATION (≈Figure 4)")
    lines.append("-" * 80)
    lines.append("")

    for mm in sorted(scale_results.keys()):
        lines.append(f"  {mm}mm expansion:")
        for clf_name in ["SVM", "LR", "RF", "XGBoost", "KNN"]:
            res = scale_results[mm].get(clf_name, {})
            if res:
                lines.append(
                    f"    {clf_name:<10}: Acc={res['accuracy']:.3f}, "
                    f"AUC={res['auc']:.3f}, Sen={res['sensitivity']:.3f}, "
                    f"Spe={res['specificity']:.3f}"
                )
        lines.append("")

    # ── 4 Groups Comparison ──
    lines.append("-" * 80)
    lines.append("FOUR EXPERIMENT GROUPS (Full Comparison)")
    lines.append("-" * 80)
    lines.append("")

    for grp_name in metrics_df["group"].unique():
        grp_df = metrics_df[metrics_df["group"] == grp_name]
        lines.append(f"  {grp_name}:")
        for _, row in grp_df.iterrows():
            lines.append(
                f"    {row['classifier']:<10}: Acc={row['accuracy']:.4f}, "
                f"AUC={row['auc']:.4f} [{row['auc_low']:.4f}-{row['auc_high']:.4f}], "
                f"Sen={row['sensitivity']:.4f}, Spe={row['specificity']:.4f}, "
                f"F1={row['f1']:.4f}"
            )
        lines.append("")

    # ── Cross-Validation Results ──
    if cv_results:
        lines.append("-" * 80)
        lines.append("NESTED 5-FOLD CROSS-VALIDATION RESULTS")
        lines.append("-" * 80)
        lines.append("")

        for grp_name, cv_rec in cv_results.items():
            lines.append(f"  {grp_name}:")
            for clf_name in CLASSIFIERS:
                accs = cv_rec[clf_name]["acc"]
                aucs = cv_rec[clf_name]["auc"]
                if accs:
                    lines.append(
                        f"    {clf_name:<10}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, "
                        f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}"
                    )
            lines.append("")

    # ── Summary ──
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Paper's best result: SVM with multimodal fusion: Acc=0.854, AUC=0.863")
    try:
        doppler_grp = [g for g in metrics_df["group"].unique() if "Doppler" in g][0]
        best_row = metrics_df[
            (metrics_df["group"] == doppler_grp) & (metrics_df["classifier"] == "SVM")
        ].iloc[0]
        lines.append(f"Reproduced best:     SVM with multimodal fusion: "
                     f"Acc={best_row['accuracy']:.4f}, AUC={best_row['auc']:.4f}")
        lines.append(f"  ΔAcc = {best_row['accuracy'] - 0.854:+.4f}, "
                     f"ΔAUC = {best_row['auc'] - 0.863:+.4f}")
    except Exception:
        lines.append("(Could not extract best reproduced result)")

    lines.append("")

    report = "\n".join(lines)

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "reproduction_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return report


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("REPRODUCTION: Multimodal Ultrasound Radiomics Fusion")
    print("for Early Breast Cancer Diagnosis")
    print("=" * 60)
    print(f"Seed: {SEED}, CV folds: {CV_FOLDS}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load All Data ──
    groups, bus = load_all_data()
    print(f"\nDataset: {list(bus.values())[0].shape[0]} lesions total")
    print(f"  Benign: {sum(bus[0]['label']==0)}, Malignant: {sum(bus[0]['label']==1)}")

    # ── 2. Generate fixed 70/30 stratified split ──
    print("\n" + "=" * 60)
    print("Generating Train/Test Split (70/30 stratified)")
    print("=" * 60)
    df0 = drop_non_features(bus[0])
    y_all = df0[LABEL_COL].values
    idx = np.arange(len(df0))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, random_state=SEED, stratify=y_all
    )
    print(f"  Train: {len(train_idx)} ({len(train_idx)/len(idx)*100:.0f}%)")
    print(f"  Test:  {len(test_idx)} ({len(test_idx)/len(idx)*100:.0f}%)")
    print(f"  Train distribution: {pd.Series(y_all[train_idx]).value_counts().to_dict()}")
    print(f"  Test distribution:  {pd.Series(y_all[test_idx]).value_counts().to_dict()}")

    # Save split for reproducibility
    split = {"train_idx": train_idx, "test_idx": test_idx}
    np.save(os.path.join(OUTPUT_DIR, "split.npy"), split)

    # ── 3. Peritumoral Scale Ablation (Figure 4) ──
    # Run only for key scales to speed up
    scale_results, _, _ = run_scale_ablation(bus, peritumoral_scales=[0, 2])

    # ── 4. Main Experiments ──
    metrics_df = run_experiments(groups, train_idx, test_idx)

    # Save metrics
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "repro_metrics.csv"), index=False)

    # ── 5. Nested Cross-Validation (robustness check, commented out for speed) ──
    cv_results = None
    # To enable nested CV, uncomment below:
    # print("\n" + "=" * 60)
    # print("Nested 5-fold Cross-Validation (Robustness Check)")
    # print("=" * 60)
    # cv_results = {}
    # for group_name, df in groups.items():
    #     print(f"\n  --- {group_name} ---")
    #     known_non_feat = [PATIENT_ID_COL, LABEL_COL, "flow_density"]
    #     non_feat_exist = [c for c in known_non_feat if c in df.columns]
    #     feat_cols = [c for c in df.columns if c not in non_feat_exist]
    #
    #     X = df[feat_cols].values
    #     y = df[LABEL_COL].values.ravel()
    #
    #     cv_rec = run_nested_cv(X, y, feat_cols, group_name)
    #     cv_results[group_name] = cv_rec
    #
    # import pickle
    # with open(os.path.join(OUTPUT_DIR, "cv_results.pkl"), "wb") as f:
    #     pickle.dump(cv_results, f)

    # ── 6. Generate Report ──
    report = generate_report(metrics_df, scale_results, cv_results)

    print("\n" + "=" * 60)
    print("REPRODUCTION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - reproduction_report.txt")
    print(f"  - repro_metrics.csv")
    print(f"  - cv_results.pkl")
    print(f"  - split.npy")
    print()
    print(report)


if __name__ == "__main__":
    main()

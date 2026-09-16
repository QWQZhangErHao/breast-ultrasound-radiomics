#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- standalone paths (self-contained package) ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
DATA = str(_ROOT / '01_data')
OUTDIR = str(_ROOT / '03_outputs' / '06_sec324_subgroup')
_os.makedirs(OUTDIR, exist_ok=True)
# -------------------------------------------------

"""
P2: Deep Statistical Analysis (GPU/Parallel accelerated)
========================================================
1. Paired t-tests + Bonferroni correction + Cohen's d
2. DeLong AUC test (test set, primary evidence)
3. VFA subgroup analysis
4. Sensitivity analysis (multi-seed, parallel)
5. Calibration data output
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

SEED = 42
BASE_DIR = DATA
DATA_DIR = os.path.join(DATA, "06_v0")
OUTPUT_DIR = OUTDIR
BASE_DIR = DATA
DATA_DIR = os.path.join(DATA, "06_v0")
OUTPUT_DIR = OUTDIR
CV_DIR = os.path.join(DATA, "06_v0", "cv_records")
PROBA_DIR = os.path.join(DATA, "06_v0", "proba")
N_JOBS = -1

GROUPS = ["group1_B", "group2_BT", "group3_BT_multi", "group4_BT_multi_doppler"]
GROUP_LABELS = {
    "group1_B": "B (baseline)",
    "group2_BT": "B+T(2mm)",
    "group3_BT_multi": "B+T(2+3+4mm)",
    "group4_BT_multi_doppler": "B+T(2+3+4mm)+Doppler",
}


def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0


def bootstrap_auc_test(y_true, prob1, prob2, n_bootstrap=10000, seed=SEED):
    """Bootstrap test for AUC difference. Returns (z_stat, p_value)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc1 = roc_auc_score(y_true, prob1)
    auc2 = roc_auc_score(y_true, prob2)
    diff_obs = auc1 - auc2
    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            diffs[i] = 0
            continue
        diffs[i] = roc_auc_score(y_true[idx], prob1[idx]) - roc_auc_score(y_true[idx], prob2[idx])
    se = np.std(diffs)
    if se == 0:
        return 0.0, 1.0
    z = diff_obs / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def main():
    print("=" * 60)
    print("P2: Deep Statistical Analysis")
    print("=" * 60)
    os.makedirs(f"{OUTPUT_DIR}/stats", exist_ok=True)

    # ════════════════════════════════════════════
    # 1. PAIRED T-TESTS (CV-based) + AUC DELONG (test set)
    # ════════════════════════════════════════════
    print("\n[1] Statistical tests")

    # Load CV records
    cv_data = {}
    for g in GROUPS:
        with open(f"{CV_DIR}/{g}_cv.pkl", "rb") as f:
            cv_data[g] = pickle.load(f)

    # 1a. Paired t-tests on CV accuracy
    g4_svm_acc = cv_data["group4_BT_multi_doppler"]["SVM"]["acc"]
    main_comparisons = [
        ("group4 vs group3 (Doppler value)", "group3_BT_multi"),
        ("group4 vs group2 (vs 2mm)", "group2_BT"),
        ("group4 vs group1 (vs baseline)", "group1_B"),
    ]
    n_main = len(main_comparisons)
    main_results = []
    for desc, other_group in main_comparisons:
        other_acc = cv_data[other_group]["SVM"]["acc"]
        t_stat, p_raw = stats.ttest_rel(g4_svm_acc, other_acc)
        d = cohens_d_paired(g4_svm_acc, other_acc)
        p_bonf = min(p_raw * n_main, 1.0)
        sig = "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "ns"
        print(f"  Paired-t {desc}: p(raw)={p_raw:.4f}, p(Bonf)={p_bonf:.4f}, d={d:.4f} [{sig}]")
        main_results.append({
            "comparison": desc, "p_raw": round(p_raw, 4),
            "p_bonferroni": round(p_bonf, 4), "cohens_d": round(d, 4),
            "significant": sig,
        })
    pd.DataFrame(main_results).to_csv(f"{OUTPUT_DIR}/S324_paired_ttest.csv", index=False)

    # 1b. DeLong AUC test on test set (primary evidence - much more power)
    print("\n  DeLong AUC test on test set:")
    delong_results = []
    groups_data = {}
    for g in GROUPS:
        proba = np.load(f"{PROBA_DIR}/{g}_SVM_proba.npy")
        groups_data[g] = proba

    # Need test set labels
    df_g4 = pd.read_csv(f"{DATA_DIR}/group4_BT_multi_doppler_raw.csv")
    split = np.load(f"{DATA_DIR}/split.npy", allow_pickle=True).item()
    test_idx = split["test_idx"]
    y_test = df_g4["label"].values[test_idx]

    auc_values = {}
    for g in GROUPS:
        auc_values[g] = roc_auc_score(y_test, groups_data[g])
        print(f"    {GROUP_LABELS[g]:30s}: AUC={auc_values[g]:.4f}")

    # Pairwise DeLong
    delong_pairs = [
        ("group4 vs group3", "group4_BT_multi_doppler", "group3_BT_multi"),
        ("group4 vs group2", "group4_BT_multi_doppler", "group2_BT"),
        ("group4 vs group1", "group4_BT_multi_doppler", "group1_B"),
        ("group3 vs group2", "group3_BT_multi", "group2_BT"),
        ("group2 vs group1", "group2_BT", "group1_B"),
    ]
    for desc, g1, g2 in delong_pairs:
        z, p = bootstrap_auc_test(y_test, groups_data[g1], groups_data[g2])
        sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    {desc:30s}: AUC={auc_values[g1]:.4f} vs {auc_values[g2]:.4f}, z={z:.3f}, p={p:.4f} [{sig}]")
        delong_results.append({
            "comparison": desc,
            "auc1": round(auc_values[g1], 4),
            "auc2": round(auc_values[g2], 4),
            "auc_diff": round(auc_values[g1] - auc_values[g2], 4),
            "z_statistic": round(z, 4),
            "p_value": round(p, 4),
            "significant": sig,
        })
    pd.DataFrame(delong_results).to_csv(f"{OUTPUT_DIR}/S324_delong_bootstrap.csv", index=False)
    print(f"  Saved -> stats/S324_delong_bootstrap.csv")

    # ════════════════════════════════════════════
    # 2. VFA SUBGROUP ANALYSIS
    # ════════════════════════════════════════════
    print("\n[2] VFA Subgroup Analysis")

    df_g4 = pd.read_csv(f"{DATA_DIR}/group4_BT_multi_doppler_raw.csv")
    non_feat = ["patient_name", "label", "vfa_group", "size_group", "age_group", "flow_density"]
    non_feat_exist = [c for c in non_feat if c in df_g4.columns]
    feat_cols = [c for c in df_g4.columns if c not in non_feat_exist]
    X = df_g4[feat_cols].values
    y = df_g4["label"].values.ravel()
    flow_density = df_g4["flow_density"].values if "flow_density" in df_g4.columns else None

    train_idx = split["train_idx"]
    X_train, X_test_arr = X[train_idx], X[test_idx]
    y_train, y_test_arr = y[train_idx], y[test_idx]
    fd_test = flow_density[test_idx] if flow_density is not None else None

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test_arr)

    lasso = LassoCV(cv=5, random_state=SEED, max_iter=10000, n_jobs=N_JOBS)
    lasso.fit(X_tr_s, y_train)
    sel = lasso.coef_ != 0
    if np.sum(sel) == 0:
        sel[np.argsort(np.abs(lasso.coef_))[-10:]] = True

    svm = SVC(kernel="linear", C=1.0, probability=True, random_state=SEED)
    svm.fit(X_tr_s[:, sel], y_train)
    y_proba_test = svm.predict_proba(X_te_s[:, sel])[:, 1]

    vfa_zero_mask = fd_test == 0
    vfa_pos_mask = fd_test > 0

    subgroup_results = []
    for name, mask in [("VFA=0", vfa_zero_mask), ("VFA>0", vfa_pos_mask)]:
        if np.sum(mask) < 5:
            continue
        y_sub = y_test_arr[mask]
        proba_sub = y_proba_test[mask]
        auc = roc_auc_score(y_sub, proba_sub)
        acc = accuracy_score(y_sub, (proba_sub >= 0.5).astype(int))
        n_b = int(np.sum(y_sub == 0))
        n_m = int(np.sum(y_sub == 1))
        print(f"  {name:10s}: n={np.sum(mask):3d} (B={n_b}, M={n_m}), AUC={auc:.4f}, Acc={acc:.4f}")
        subgroup_results.append({
            "subgroup": name, "n": int(np.sum(mask)),
            "n_benign": n_b, "n_malignant": n_m,
            "auc": round(auc, 4), "acc": round(acc, 4),
        })
    pd.DataFrame(subgroup_results).to_csv(f"{OUTPUT_DIR}/S324_vfa_subgroup.csv", index=False)

    # ════════════════════════════════════════════
    # 3. SENSITIVITY ANALYSIS
    # ════════════════════════════════════════════
    print("\n[3] Sensitivity Analysis (5 seeds)")

    def run_seed(seed):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_accs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train[tr_idx])
            X_val_s = scaler.transform(X_train[val_idx])
            l = LassoCV(cv=3, random_state=seed, max_iter=10000, n_jobs=1)
            l.fit(X_tr_s, y_train[tr_idx])
            sel = l.coef_ != 0
            if np.sum(sel) == 0:
                sel[np.argsort(np.abs(l.coef_))[-10:]] = True
            m = SVC(kernel="linear", C=1.0, random_state=seed)
            m.fit(X_tr_s[:, sel], y_train[tr_idx])
            cv_accs.append(m.score(X_val_s[:, sel], y_train[val_idx]))
        return {"seed": seed, "mean_cv_acc": round(np.mean(cv_accs), 4), "std_cv_acc": round(np.std(cv_accs), 4)}

    seeds = [42, 123, 2024, 7, 999]
    sens_results = Parallel(n_jobs=5)(delayed(run_seed)(s) for s in seeds)

    for r in sens_results:
        print(f"  Seed {r['seed']:4d}: CV Acc={r['mean_cv_acc']:.4f}+/-{r['std_cv_acc']:.4f}")
    pd.DataFrame(sens_results).to_csv(f"{OUTPUT_DIR}/S324_sensitivity.csv", index=False)
    print(f"  Saved -> stats/S324_sensitivity.csv")

    # ════════════════════════════════════════════
    # 4. CALIBRATION DATA
    # ════════════════════════════════════════════
    print("\n[4] Calibration data")
    cal_data = {}
    for g in GROUPS:
        proba = np.load(f"{PROBA_DIR}/{g}_SVM_proba.npy")
        bins = np.linspace(0, 1, 11)
        bin_acc = np.zeros(10)
        bin_counts = np.zeros(10)
        for i in range(10):
            mask = (proba >= bins[i]) & (proba < bins[i + 1])
            bin_counts[i] = np.sum(mask)
            if bin_counts[i] > 0:
                bin_acc[i] = np.mean(y_test_arr[mask])
        cal_data[g] = {
            "bin_centers": ((bins[:-1] + bins[1:]) / 2).tolist(),
            "bin_accuracy": bin_acc.tolist(),
            "bin_counts": bin_counts.astype(int).tolist(),
        }
    with open(f"{OUTPUT_DIR}/calibration_data.pkl", "wb") as f:
        pickle.dump(cal_data, f)
    print(f"  Calibration data: {len(cal_data)} groups")

    print("\n" + "=" * 60)
    print("P2 Complete!")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

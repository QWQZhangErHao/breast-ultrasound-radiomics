#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the 1-mm grayscale + 3-Doppler multimodal ROC for all 5 classifiers
(SVM / LR / RF / XGBoost / KNN) using the de-identified feature matrices in this
repository (`data/`). Mirrors the "mother" ablation pipeline
(compute_doppler_ablation.py): per seed, 70/30 stratified split -> LASSO on the
training fold over the FULL row (grayscale + Doppler, LassoCV tol=1e-3,
max_iter=5000) -> XGBoost etc. -> collect held-out probabilities.

Expected output (10 seeds):
    SVM      0.804 +/- 0.042
    LR       0.812 +/- 0.045
    RF       0.821 +/- 0.028
    XGBoost  0.827 +/- 0.028
    KNN      0.735 +/- 0.024

Usage:
    python reproduce/run_radiomics_roc_5models.py            # write results/roc_1mm_plus3_5models.pkl
    python reproduce/run_radiomics_roc_5models.py --seeds 2  # quick 2-seed smoke test
"""
import os, re, time, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(REPO, "data", "features")
FLOW_DIR = os.path.join(REPO, "data", "flow")
OUT = os.path.join(REPO, "results", "roc_1mm_plus3_5models.pkl")

PID, LABEL = "patient_name", "label"
SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
GRID = np.linspace(0, 1, 401)
DOPPLER_3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]
MODELS = ["SVM", "LR", "RF", "XGBoost", "KNN"]
SEED = 42


def sid(r):
    m = re.search(r"(\d+)", str(r))
    return int(m.group(1)) if m else None


def load_bus(mm):
    b = pd.read_csv(os.path.join(FEAT_DIR, f"benign_{mm}mm.csv"))
    m = pd.read_csv(os.path.join(FEAT_DIR, f"malignant_{mm}mm.csv"))
    b[LABEL], m[LABEL] = 0, 1
    df = pd.concat([b, m], ignore_index=True)
    df[PID] = df[PID].apply(sid)
    return df


def is_feature(c):
    return not any(p in c.lower() for p in ["patient_name", "file_name", "diagnostics_"])


def build_1mm():
    d0 = load_bus(0)
    d0 = d0[[c for c in d0.columns if c in (PID, LABEL) or is_feature(c)]]
    d1 = load_bus(1)
    d1 = d1[[c for c in d1.columns if c in (PID, LABEL) or is_feature(c)]]
    d1 = d1.rename(columns={c: c if c in (PID, LABEL) else f"{c}_1mm" for c in d1.columns})
    d1 = d1.drop(columns=[c for c in d1.columns if "shape" in c.lower() and c not in (PID, LABEL)],
                 errors="ignore")
    gray = pd.merge(d0, d1, on=[PID, LABEL], how="inner")
    gray_feats = [c for c in gray.columns if c not in (PID, LABEL)]
    bf = pd.concat([pd.read_csv(os.path.join(FLOW_DIR, "blood_flow_features_benign.csv")),
                    pd.read_csv(os.path.join(FLOW_DIR, "blood_flow_features_malignant.csv"))],
                   ignore_index=True).rename(columns={"image_id": PID})
    bf = bf[[c for c in bf.columns if c == PID or c.startswith("bf_")]]
    bf[PID] = bf[PID].apply(sid)
    vfa = pd.concat([pd.read_csv(os.path.join(FLOW_DIR, "benign_flow_density.csv")),
                     pd.read_csv(os.path.join(FLOW_DIR, "malignant_flow_density.csv"))],
                    ignore_index=True)
    vfa = vfa[[c for c in vfa.columns if c in ("case_id", "flow_density")]].rename(columns={"case_id": PID})
    m0 = gray.merge(bf, on=PID, how="inner")
    m1 = m0.merge(vfa, on=PID, how="inner")
    m1[LABEL] = gray.set_index(PID)[LABEL].reindex(m1[PID]).to_numpy()
    m1 = m1[m1[LABEL].notna()].reset_index(drop=True)
    m1[LABEL] = m1[LABEL].astype(int)
    gray_cols = [c for c in gray_feats if c in m1.columns]
    return m1, gray_cols


def make_clf(name):
    if name == "SVM":
        return SVC(kernel="linear", C=1.0, probability=True, random_state=SEED)
    if name == "LR":
        return LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                  max_iter=10000, random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED)
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 random_state=SEED, verbosity=0, tree_method="hist")
    return KNeighborsClassifier(n_neighbors=5, metric="euclidean")


def interp_tpr(y, proba):
    fpr, tpr, _ = roc_curve(y, proba)
    d = pd.DataFrame({"f": fpr, "t": tpr}).groupby("f", as_index=False)["t"].max().sort_values("f")
    out = np.interp(GRID, d["f"].to_numpy(), d["t"].to_numpy())
    out[0], out[-1] = 0.0, 1.0
    return out


def run_one(seed, X, y):
    t0 = time.time()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler()
    Xs = sc.fit_transform(X[tr])
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000, n_jobs=None, tol=1e-3)
    lasso.fit(Xs, y[tr])
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True
    Xtr = sc.transform(X[tr])[:, mask]
    Xte = sc.transform(X[te])[:, mask]
    out = {}
    for name in MODELS:
        clf = make_clf(name)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        out[name] = {"auc": roc_auc_score(y[te], proba), "tpr": interp_tpr(y[te], proba)}
    print(f"  seed {seed}: " + " ".join(f"{m}={out[m]['auc']:.4f}" for m in MODELS), flush=True)
    return seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="number of seeds to run")
    a = ap.parse_args()
    use_seeds = SEEDS[:a.seeds]
    t_all = time.time()
    m1, gray_cols = build_1mm()
    cols = gray_cols + [c for c in DOPPLER_3 if c in m1.columns]
    miss = [c for c in DOPPLER_3 if c not in m1.columns]
    if miss:
        raise SystemExit("doppler missing: %s" % miss)
    X = m1[cols].to_numpy()
    y = m1[LABEL].to_numpy().ravel()
    print(f"rows={len(m1)} features={len(cols)} labels={np.bincount(y).tolist()}", flush=True)
    n_jobs = min(len(use_seeds), max(2, (os.cpu_count() or 8) - 8))
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(run_one)(s, X, y) for s in use_seeds)
    by_seed = {s: o for s, o in res}
    pkg = {"fpr": GRID, "models": MODELS, "tpr_mean": {}, "tpr_std": {}, "aucs": {},
           "means": {}, "stds": {}}
    print("=" * 60, flush=True)
    for name in MODELS:
        aucs = np.array([by_seed[s][name]["auc"] for s in use_seeds])
        tpr = np.array([by_seed[s][name]["tpr"] for s in use_seeds])
        pkg["aucs"][name] = aucs
        pkg["tpr_mean"][name] = tpr.mean(0)
        pkg["tpr_std"][name] = tpr.std(0)
        pkg["means"][name] = float(aucs.mean())
        pkg["stds"][name] = float(aucs.std())
        print(f"{name:9s}: {aucs.mean():.4f} +/- {aucs.std():.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(pkg, f)
    print(f"[{time.time()-t_all:.0f}s] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

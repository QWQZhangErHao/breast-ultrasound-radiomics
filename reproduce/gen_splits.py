# -*- coding: utf-8 -*-
"""
Export per-seed test y_true + probabilities for 'base' (1mm grayscale) and
'+3 Doppler' rows across 5 classifiers, using the mother-exact nested pipeline
(LassoCV tol=1e-3, max_iter=5000 over the full row). Saves to results/
splits_base_plus3.pkl. No feature re-extraction, no DL.
"""
import os, re, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUS = os.path.join(REPO, "data", "features")
FLOW = os.path.join(REPO, "data", "flow")
OUT = os.path.join(REPO, "results", "splits_base_plus3.pkl")
PID, LABEL = "patient_name", "label"
SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
DOPPLER_3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]
MODELS = ["SVM", "LR", "RF", "XGBoost", "KNN"]

def sid(r):
    m = re.search(r"(\d+)", str(r)); return int(m.group(1)) if m else None

def load_bus(mm):
    b = pd.read_csv(os.path.join(BUS, f"benign_{mm}mm.csv"))
    m = pd.read_csv(os.path.join(BUS, f"malignant_{mm}mm.csv"))
    b[LABEL], m[LABEL] = 0, 1
    df = pd.concat([b, m], ignore_index=True)
    df[PID] = df[PID].apply(sid)
    return df

def is_feature(c):
    return not any(p in c.lower() for p in ["patient_name", "file_name", "diagnostics_"])

def build():
    d0 = load_bus(0); d0 = d0[[c for c in d0.columns if c in (PID, LABEL) or is_feature(c)]]
    d1 = load_bus(1); d1 = d1[[c for c in d1.columns if c in (PID, LABEL) or is_feature(c)]]
    d1 = d1.rename(columns={c: c if c in (PID, LABEL) else f"{c}_1mm" for c in d1.columns})
    d1 = d1.drop(columns=[c for c in d1.columns if "shape" in c.lower() and c not in (PID, LABEL)], errors="ignore")
    gray = pd.merge(d0, d1, on=[PID, LABEL], how="inner")
    gray_feats = [c for c in gray.columns if c not in (PID, LABEL)]
    bf = pd.concat([pd.read_csv(os.path.join(FLOW, "blood_flow_features_benign.csv")),
                    pd.read_csv(os.path.join(FLOW, "blood_flow_features_malignant.csv"))],
                   ignore_index=True).rename(columns={"image_id": PID})
    bf = bf[[c for c in bf.columns if c == PID or c.startswith("bf_")]]
    bf[PID] = bf[PID].apply(sid)
    vfa = pd.concat([pd.read_csv(os.path.join(FLOW, "benign_flow_density.csv")),
                     pd.read_csv(os.path.join(FLOW, "malignant_flow_density.csv"))], ignore_index=True)
    vfa = vfa[[c for c in vfa.columns if c in ("case_id", "flow_density")]].rename(columns={"case_id": PID})
    m0 = gray.merge(bf, on=PID, how="inner")
    m1 = m0.merge(vfa, on=PID, how="inner")
    m1[LABEL] = gray.set_index(PID)[LABEL].reindex(m1[PID]).to_numpy()
    m1 = m1[m1[LABEL].notna()].reset_index(drop=True)
    m1[LABEL] = m1[LABEL].astype(int)
    gray_cols = [c for c in gray_feats if c in m1.columns]
    rows = {"base": gray_cols, "plus3": gray_cols + [c for c in DOPPLER_3 if c in m1.columns]}
    return m1, rows

def make_clf(name):
    if name == "SVM":
        return SVC(kernel="linear", C=1.0, probability=True, random_state=SEED)
    if name == "LR":
        return LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=10000, random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED)
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 random_state=SEED, verbosity=0, tree_method="hist")
    return KNeighborsClassifier(n_neighbors=5, metric="euclidean")

def run_one(seed, Xrow, y):
    t0 = time.time()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    y_te = y[te]
    out = {"y_te": y_te, "rows": {}}
    for rname, cols in Xrow.items():
        Xr = Xrow[rname]
        sc = StandardScaler()
        Xs = sc.fit_transform(Xr[tr])
        lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000, n_jobs=None, tol=1e-3)
        lasso.fit(Xs, y[tr])
        mask = lasso.coef_ != 0
        if mask.sum() == 0:
            mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True
        Xtr = sc.transform(Xr[tr])[:, mask]
        Xte = sc.transform(Xr[te])[:, mask]
        probs = {}
        for name in MODELS:
            clf = make_clf(name)
            clf.fit(Xtr, y[tr])
            probs[name] = clf.predict_proba(Xte)[:, 1]
        out["rows"][rname] = probs
    aucs = {r: {m: roc_auc_score(out["y_te"], out["rows"][r][m]) for m in MODELS}
            for r in out["rows"]}
    print(f"  seed {seed}: {time.time()-t0:.1f}s " +
          " ".join(f"{r}/{m}={aucs[r][m]:.4f}" for r in ("base",) for m in MODELS), flush=True)
    return seed, out

def main():
    t_all = time.time()
    m1, rows = build()
    Xrow = {k: m1[c].to_numpy() for k, c in rows.items()}
    y = m1[LABEL].to_numpy().ravel()
    print(f"rows={len(m1)} base={len(rows['base'])} plus3={len(rows['plus3'])} "
          f"labels={np.bincount(y).tolist()}", flush=True)
    n_jobs = min(len(SEEDS), max(2, (os.cpu_count() or 8) - 8))
    print(f"running {len(SEEDS)} seeds x 2 rows x 5 clf (n_jobs={n_jobs})", flush=True)
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(run_one)(s, Xrow, y) for s in SEEDS)
    pkg = {"seeds": SEEDS, "splits": {s: o for s, o in res}}
    with open(OUT, "wb") as f:
        pickle.dump(pkg, f)
    print(f"[{time.time()-t_all:.0f}s] saved -> {OUT}", flush=True)

if __name__ == "__main__":
    main()

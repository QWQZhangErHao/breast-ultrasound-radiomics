# -*- coding: utf-8 -*-
# --- standalone paths (self-contained package) ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
DATA = str(_ROOT / '01_data')
OUTDIR = str(_ROOT / '03_outputs' / '07_all_unified')
_os.makedirs(OUTDIR, exist_ok=True)
# -------------------------------------------------

"""Unified pipeline (unified env): computes Table 1/2 and Sec 3.2.1/3.2.2/3.2.4 in one run.

Unified settings:
  env: unified env (python 3.12.13 / numpy 2.5.1 / scikit-learn 1.9.0 / xgboost 3.3.0)
  splits: 10 x 70/30 stratified, seeds 42,123,2024,7,999,314,271,1618,2048,4096
  feature selection: LASSO inside training folds only (tol=1e-3, cv=5, max_iter=10000)
  models: SVM(lin,C1) / LR(saga,l2) / RF(500,d5) / XGBoost(hist,100,d3,lr.1) / KNN(5)
  SD: ddof=1 everywhere
  thresholds: 0.5 and Youden reported separately
  subgroups: cohort counts + per-fold test-set AUC (VFA=0 / VFA>0)
  outputs: unified_results.json / .pkl
"""
import json
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

try:
    import joblib.externals.loky.backend.context as _loky
    _loky._count_physical_cores_win32 = lambda: max(1, (os.cpu_count() or 4) // 2)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = DATA
OUT = OUTDIR
SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
if os.environ.get("UNIFIED_SEEDS"):
    SEEDS = SEEDS[:int(os.environ["UNIFIED_SEEDS"])]
MODELS = ["SVM", "LR", "RF", "XGBoost", "KNN"]
PID, LAB = "patient_name", "label"
D3 = ["flow_density", "bf_firstorder_Energy", "bf_glrlm_GrayLevelNonUniformity"]
LASSO_TOL = 1e-3
WORKERS = 16
BUS = os.path.join(DATASET, "01_bus_features")
FLOW = os.path.join(DATASET, "02_flow_features")


def sid(r):
    m = re.search(r"(\d+)", str(r))
    return int(m.group(1)) if m else None


def is_feat(c):
    return not any(p in c.lower() for p in ["patient_name", "file_name", "diagnostics_"])


def load_bus(mm):
    b = pd.read_csv(os.path.join(BUS, f"benign_{mm}mm.csv"))
    m = pd.read_csv(os.path.join(BUS, f"malignant_{mm}mm.csv"))
    b[LAB], m[LAB] = 0, 1
    df = pd.concat([b, m], ignore_index=True)
    df[PID] = df[PID].apply(sid)
    return df


def nonfeat(df):
    return df[[c for c in df.columns if c in (PID, LAB) or is_feat(c)]]


def suffix(df, k):
    return df.rename(columns={c: c if c in (PID, LAB) else f"{c}_{k}mm" for c in df.columns})


def cumulative(k):
    d0 = nonfeat(load_bus(0))
    if k == 0:
        return d0, [c for c in d0.columns if c not in (PID, LAB)]
    dk = suffix(nonfeat(load_bus(k)), k)
    dk = dk.drop(columns=[c for c in dk.columns if "shape" in c.lower() and c not in (PID, LAB)],
                 errors="ignore")
    out = d0.merge(dk, on=[PID, LAB], how="inner")
    return out, [c for c in out.columns if c not in (PID, LAB)]


def build_fusion_base():
    df1, gray = cumulative(1)
    bf = pd.concat([pd.read_csv(os.path.join(FLOW, "blood_flow_features_benign.csv")),
                    pd.read_csv(os.path.join(FLOW, "blood_flow_features_malignant.csv"))],
                   ignore_index=True).rename(columns={"image_id": PID})
    bf[PID] = bf[PID].apply(sid)
    bf = bf[[PID] + [c for c in bf.columns if c.startswith("bf_")]]
    vfa = pd.concat([pd.read_csv(os.path.join(FLOW, "benign_flow_density.csv")),
                     pd.read_csv(os.path.join(FLOW, "malignant_flow_density.csv"))],
                    ignore_index=True).rename(columns={"case_id": PID})
    vfa = vfa[["flow_density", PID]]
    df = df1.merge(bf, on=PID, how="inner").merge(vfa, on=PID, how="inner")
    df[LAB] = df1.set_index(PID)[LAB].reindex(df[PID]).to_numpy()
    df = df[df[LAB].notna()].reset_index(drop=True)
    df[LAB] = df[LAB].astype(int)
    return df, gray, [c for c in df.columns if c.startswith("bf_")]


def lasso_fit(Xtr, ytr):
    s = StandardScaler()
    Xs = s.fit_transform(Xtr)
    l = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=1, tol=LASSO_TOL)
    l.fit(Xs, ytr)
    mask = l.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(l.coef_))[-10:]] = True
    return s, mask


def clf(name):
    return {"SVM": lambda: SVC(kernel="linear", C=1, probability=True, random_state=42),
            "LR": lambda: LogisticRegression(penalty="l2", C=1, solver="saga",
                                             max_iter=10000, random_state=42, n_jobs=1),
            "RF": lambda: RandomForestClassifier(n_estimators=500, max_depth=5,
                                                 random_state=42, n_jobs=1),
            "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                                 learning_rate=0.1, random_state=42,
                                                 verbosity=0, n_jobs=1, tree_method="hist"),
            "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=1)}[name]()


def met(y, p):
    pr = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr).ravel()
    fpr, tpr, th = roc_curve(y, p)
    t = float(th[int(np.argmax(tpr - fpr))])
    pr_y = (p >= t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y, pr_y).ravel()
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                acc=accuracy_score(y, pr),
                sen=tp / (tp + fn) if tp + fn else 0.0,
                spe=tn / (tn + fp) if tn + fp else 0.0,
                prec=precision_score(y, pr, zero_division=0),
                f1=f1_score(y, pr, zero_division=0),
                topt=t,
                sen_y=tp2 / (tp2 + fn2) if tp2 + fn2 else 0.0,
                spe_y=tn2 / (tn2 + fp2) if tn2 + fp2 else 0.0)


def one(key, seed, X, y, v0):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    sc, mask = lasso_fit(X[tr], y[tr])
    Xt = sc.transform(X[tr])[:, mask]
    Xe = sc.transform(X[te])[:, mask]
    out = {"seed": seed, "te": te, "yte": y[te].copy(), "v0": v0[te].copy(), "nfeat": int(mask.sum())}
    for n in MODELS:
        c = clf(n)
        c.fit(Xt, y[tr])
        p = c.predict_proba(Xe)[:, 1]
        out[n] = met(y[te], p)
        out["p_" + n] = p
    return key, seed, out


def main():
    t0 = time.time()
    SCALE, VS = {}, {}
    for k in range(5):
        df, feats = cumulative(k)
        SCALE[f"scale{k}"] = (df[feats].to_numpy(), df[LAB].to_numpy().ravel())
        print(f"scale{k}: n={len(df)} feats={len(feats)}", flush=True)
    dff, gray, bf_all = build_fusion_base()
    yf = dff[LAB].to_numpy().ravel()
    sel_pkl = os.path.join(DATA, os.path.join("04_doppler_selection", "doppler_selection.pkl"))
    sel13 = list(dict.fromkeys(pickle.load(open(sel_pkl, "rb"))["stage4_passed"]))
    fus = {"base": gray, "vfa": gray + ["flow_density"],
           "p3": gray + [c for c in D3 if c in dff.columns], "p13": gray + sel13}
    for k, cols in fus.items():
        SCALE[f"fus_{k}"] = (dff[cols].to_numpy(), yf)
    v0 = (dff["flow_density"].to_numpy(dtype=float) == 0)
    VS["fus_base"] = v0
    for k in ("fus_vfa", "fus_p3", "fus_p13"):
        VS[k] = v0
    print("feat sets:", {k: len(c) for k, c in fus.items()}, "| VFA=0 count:", int(v0.sum()), flush=True)

    keys = [f"scale{i}" for i in range(5)] + [f"fus_{k}" for k in fus]
    jobs = [(k, s) for k in keys for s in SEEDS]
    print(f"{len(jobs)} jobs on {WORKERS} workers", flush=True)
    res = Parallel(n_jobs=WORKERS)(
        delayed(one)(k, s, SCALE[k][0], SCALE[k][1], VS.get(k, np.zeros(len(SCALE[k][1]), bool)))
        for k, s in jobs)

    R = {k: {s: None for s in SEEDS} for k in keys}
    for k, s, r in res:
        R[k][s] = r

    def arr(k, s, m, q):
        return R[k][s][m][q]

    def ms(v):
        a = np.asarray(v, float)
        return float(a.mean()), float(a.std(ddof=1))

    out = {"_env": {p: __import__(p).__version__ for p in ("numpy", "sklearn", "xgboost")},
           "_lasso_tol": LASSO_TOL, "_seeds": SEEDS, "_n": int(len(yf)),
           "_vfa0": int(v0.sum()), "table1": {}, "table2": {}, "sec321": {}, "sec322": {}, "sec324": {}}

    for k, tag in (("fus_base", "G1"), ("fus_vfa", "G1+VFA"), ("fus_p3", "G1+D3"), ("fus_p13", "G1+D13")):
        out["table1"][tag] = {m: dict(zip(("mean", "sd"), ms([arr(k, s, m, "auc") for s in SEEDS])))
                              for m in MODELS}
    out["table2"]["radiomics"] = {
        m: dict(zip(("mean", "sd"), ms([arr("fus_base", s, m, "auc") for s in SEEDS]))) for m in MODELS}
    x = "XGBoost"
    for q in ("acc", "sen", "spe", "sen_y", "spe_y", "topt", "ap"):
        out["table2"][q] = dict(zip(("mean", "sd"), ms([arr("fus_base", s, x, q) for s in SEEDS])))
        out["sec322"][q + "_p3"] = dict(zip(("mean", "sd"), ms([arr("fus_p3", s, x, q) for s in SEEDS])))
    for m in MODELS:
        d = [arr("fus_p3", s, m, "auc") - arr("fus_base", s, m, "auc") for s in SEEDS]
        from scipy import stats as st
        out["sec322"][f"dAUC_{m}"] = dict(mean=float(np.mean(d)), sd=float(np.std(d, ddof=1)),
                                          p=float(st.ttest_rel([arr("fus_p3", s, m, "auc") for s in SEEDS],
                                                               [arr("fus_base", s, m, "auc") for s in SEEDS]).pvalue))
    for i in range(5):
        out["sec321"][f"{i}mm"] = {m: dict(zip(("mean", "sd"),
                                              ms([arr(f"scale{i}", s, m, "auc") for s in SEEDS])))
                                   for m in ("XGBoost", "SVM")}
    for k in ("fus_base", "fus_p3"):
        for lbl, sel in (("vfa0", True), ("vfapos", False)):
            vals = {}
            for m in MODELS:
                a = []
                for s in SEEDS:
                    mask = (R[k][s]["v0"] == sel)
                    y = R[k][s]["yte"][mask]
                    p = R[k][s]["p_" + m][mask]
                    if len(np.unique(y)) > 1:
                        a.append(roc_auc_score(y, p))
                vals[m] = dict(zip(("mean", "sd"), ms(a)))
            out["sec324"][f"{k}_{lbl}"] = vals
    out["sec324"]["counts"] = dict(vfa0=int(v0.sum()), vfapos=int((~v0).sum()),
                                   n=int(len(yf)), n_test_per_seed=int(v0.sum() * 0.3))

    json.dump(out, open(os.path.join(OUT, "ALL_unified_results.json"), "w", encoding="utf-8"),
              indent=2, default=float)
    pickle.dump({"out": out, "R": R, "featsets": {k: len(c) for k, c in fus.items()}},
                open(os.path.join(OUT, "ALL_unified_results.pkl"), "wb"))
    print("\n=== Unified Table 1 ===")
    for tag in ("G1", "G1+VFA", "G1+D3", "G1+D13"):
        print("%-9s %s" % (tag, " ".join("%s %.3f±%.3f" % (m, out["table1"][tag][m]["mean"],
                                                           out["table1"][tag][m]["sd"]) for m in MODELS)))
    print("\n=== Table 2 radiomics row ===")
    print(" AUC %.3f±%.3f | Sen@0.5 %.1f±%.1f | Spe@0.5 %.1f±%.1f | Sen@Youden %.1f±%.1f | "
          "Spe@Youden %.1f±%.1f | Topt %.3f±%.3f | PR-AUC %.3f±%.3f"
          % (out["table2"]["radiomics"]["XGBoost"]["mean"], out["table2"]["radiomics"]["XGBoost"]["sd"],
             out["table2"]["sen"]["mean"] * 100, out["table2"]["sen"]["sd"] * 100,
             out["table2"]["spe"]["mean"] * 100, out["table2"]["spe"]["sd"] * 100,
             out["table2"]["sen_y"]["mean"] * 100, out["table2"]["sen_y"]["sd"] * 100,
             out["table2"]["spe_y"]["mean"] * 100, out["table2"]["spe_y"]["sd"] * 100,
             out["table2"]["topt"]["mean"], out["table2"]["topt"]["sd"],
             out["table2"]["ap"]["mean"], out["table2"]["ap"]["sd"]))
    print("\n=== delta-AUC (+3 vs base, per-fold paired) ===")
    for m in MODELS:
        d = out["sec322"][f"dAUC_{m}"]
        print("  %-8s %+.4f ± %.4f  p=%.4f" % (m, d["mean"], d["sd"], d["p"]))
    print("\n=== Sec 3.2.4 VFA subgroups (per-fold test set) ===")
    for k in ("fus_base", "fus_p3"):
        for lbl in ("vfa0", "vfapos"):
            v = out["sec324"][f"{k}_{lbl}"]
            print("  %-9s %-7s XGBoost %.4f±%.4f" % (k, lbl, v["XGBoost"]["mean"], v["XGBoost"]["sd"]))
    print("\n[OK] %.0fs -> unified_results.json / .pkl" % (time.time() - t0))


if __name__ == "__main__":
    main()

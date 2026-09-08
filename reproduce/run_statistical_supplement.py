# -*- coding: utf-8 -*-
"""
Lightweight statistical supplement (Plan A) using real per-split predictions
exported by gen_splits.py (no DL retraining / no feature re-extraction).
Input : results/splits_base_plus3.pkl
Output: prints + Desktop/statistical_supplement.json
"""
import os, pickle, json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy import stats

PKL = os.path.join(REPO, "results", "splits_base_plus3.pkl")
JSON = os.path.join(REPO, "results", "statistical_supplement.json")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["SVM", "LR", "RF", "XGBoost", "KNN"]


def _placement_v(y, pa, pb):
    """case placement (V01) for positive rows, control placement (V10) rows."""
    idx = np.arange(len(y))
    cases = idx[y == 1]
    ctrls = idx[y == 0]
    n1 = len(cases); n0 = len(ctrls)
    va_c, vb_c = np.empty(n1), np.empty(n1)   # V01 per case
    for k, i in enumerate(cases):
        ac = (pa[i] > pa[ctrls]).sum() + 0.5 * (pa[i] == pa[ctrls]).sum()
        bc = (pb[i] > pb[ctrls]).sum() + 0.5 * (pb[i] == pb[ctrls]).sum()
        va_c[k] = ac / n0; vb_c[k] = bc / n0
    va_t, vb_t = np.empty(n0), np.empty(n0)   # V10 per control
    for k, j in enumerate(ctrls):
        at = (pa[j] < pa[cases]).sum() + 0.5 * (pa[j] == pa[cases]).sum()
        bt = (pb[j] < pb[cases]).sum() + 0.5 * (pb[j] == pb[cases]).sum()
        va_t[k] = at / n1; vb_t[k] = bt / n1
    return (va_c.mean(), vb_c.mean()), (va_c, vb_c, va_t, vb_t), (n1, n0)


def delong_paired(y, a, b):
    """Correct DeLong test for correlated ROC curves (same cases+controls)."""
    (th_a, th_b), comp, ns = _placement_v(np.asarray(y), np.asarray(a), np.asarray(b))
    va_c, vb_c, va_t, vb_t = comp
    n1, n0 = ns
    cov_c = np.cov(np.vstack([va_c, vb_c]))   # 2x2 among cases
    cov_t = np.cov(np.vstack([va_t, vb_t]))   # 2x2 among controls
    s2_a = cov_c[0, 0] / n1 + cov_t[0, 0] / n0
    s2_b = cov_c[1, 1] / n1 + cov_t[1, 1] / n0
    cov_ab = cov_c[0, 1] / n1 + cov_t[0, 1] / n0
    s = np.sqrt(max(s2_a + s2_b - 2 * cov_ab, 0.0))
    if s == 0 or not np.isfinite(s):
        return th_a, th_b, 1.0
    z = (th_a - th_b) / s
    return th_a, th_b, float(2 * (1 - stats.norm.cdf(abs(z))))


def metrics_at(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    acc = (tp + tn) / len(y)
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    return acc, sen, spe


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    t_opt = thr[i]
    if not np.isfinite(t_opt) or t_opt > 1:
        t_opt = 0.5
    _, sen, spe = metrics_at(y, p, thr=t_opt)
    return t_opt, sen, spe


def main():
    data = pickle.load(open(PKL, "rb"))
    seeds = data["seeds"]
    splits = data["splits"]
    out = {"seeds": seeds}

    # ---- AUC means (sanity vs manuscript) ----
    aucs = {m: {"base": [], "plus3": []} for m in MODELS}
    for s in seeds:
        sp = splits[s]
        for m in MODELS:
            aucs[m]["base"].append(roc_auc_score(sp["y_te"], sp["rows"]["base"][m]))
            aucs[m]["plus3"].append(roc_auc_score(sp["y_te"], sp["rows"]["plus3"][m]))
    print("AUC mean +/- SD (base -> plus3)")
    for m in MODELS:
        b = np.array(aucs[m]["base"]); p3 = np.array(aucs[m]["plus3"])
        print(f"  {m:9s}: {b.mean():.4f}+/-{b.std():.4f}  ->  {p3.mean():.4f}+/-{p3.std():.4f}")

    # ---- Task 1: paired DeLong per split (plus3 vs base) ----
    print("\n[Task 1] Paired DeLong p (per split, +3 vs base)")
    delong = {}
    paired = {}
    for m in MODELS:
        ps = []
        for s in seeds:
            sp = splits[s]
            _, _, p = delong_paired(sp["y_te"], sp["rows"]["plus3"][m], sp["rows"]["base"][m])
            ps.append(p)
        ps = np.array(ps)
        delong[m] = {"p_per_split": ps.tolist(),
                     "median_p": float(np.median(ps)),
                     "sig_folds": int(np.sum(ps < 0.05))}
        # manuscript-style paired tests over seed AUC deltas
        d = np.array(aucs[m]["plus3"]) - np.array(aucs[m]["base"])
        _, pt = stats.ttest_rel(aucs[m]["plus3"], aucs[m]["base"])
        pw = stats.wilcoxon(aucs[m]["plus3"], aucs[m]["base"]) if np.any(d != 0) else (0.0, 1.0)
        paired[m] = {"deltaAUC": float(d.mean()), "sd_delta": float(d.std(ddof=1)),
                     "p_paired_t": float(pt), "p_wilcoxon": float(getattr(pw, "pvalue", pw[1]))}
        print(f"  {m:9s}: DeLong median p={np.median(ps):.4f} sig_folds={np.sum(ps<0.05)}/10 | "
              f"paired-t ΔAUC={d.mean():+.4f} p={pt:.4f} | wilcoxon p={getattr(pw,'pvalue',pw[1]):.4f}")
    out["delong"] = delong
    out["paired"] = paired

    # ---- Task 2: XGBoost grayscale baseline Youden calibration (fill Table 2) ----
    tops, s_y, sp_y = [], [], []
    for s in seeds:
        sp = splits[s]
        t, sen, spe = youden(sp["y_te"], sp["rows"]["base"]["XGBoost"])
        tops.append(t); s_y.append(sen); sp_y.append(spe)
    xgb_y = {"Topt": [float(np.mean(tops)), float(np.std(tops, ddof=1))],
             "Sen_Youden": [float(np.mean(s_y)), float(np.std(s_y, ddof=1))],
             "Spe_Youden": [float(np.mean(sp_y)), float(np.std(sp_y, ddof=1))]}
    out["xgb_base_youden"] = xgb_y
    print("\n[Task 2] XGBoost grayscale-base Youden (fill Table 2 XGBoost row)")
    print(f"  Topt        = {np.mean(tops):.3f} +/- {np.std(tops,ddof=1):.3f}")
    print(f"  Sen(Youden) = {np.mean(s_y):.3f} +/- {np.std(s_y,ddof=1):.3f}")
    print(f"  Spe(Youden) = {np.mean(sp_y):.3f} +/- {np.std(sp_y,ddof=1):.3f}")

    # ---- Task 3: XGBoost grayscale baseline at 0.5 ----
    accs, sens, spes = [], [], []
    for s in seeds:
        sp = splits[s]
        a, se, sp_ = metrics_at(sp["y_te"], sp["rows"]["base"]["XGBoost"], 0.5)
        accs.append(a); sens.append(se); spes.append(sp_)
    xgb05 = {"Acc": [float(np.mean(accs)), float(np.std(accs, ddof=1))],
             "Sen": [float(np.mean(sens)), float(np.std(sens, ddof=1))],
             "Spe": [float(np.mean(spes)), float(np.std(spes, ddof=1))]}
    out["xgb_base_05"] = xgb05
    print("\n[Task 3] XGBoost grayscale-base at default 0.5")
    print(f"  Acc = {np.mean(accs):.3f} +/- {np.std(accs,ddof=1):.3f}")
    print(f"  Sen = {np.mean(sens):.3f} +/- {np.std(sens,ddof=1):.3f}")
    print(f"  Spe = {np.mean(spes):.3f} +/- {np.std(spes,ddof=1):.3f}")

    json.dump(out, open(JSON, "w"), indent=2, default=float)
    print("\nsaved ->", JSON)


if __name__ == "__main__":
    main()

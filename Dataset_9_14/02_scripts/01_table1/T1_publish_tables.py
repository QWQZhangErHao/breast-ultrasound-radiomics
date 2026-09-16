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

"""Post-process canonical_author_run.pkl -> paired stats (all classifiers),
Bonferroni note, and tidy CSVs (scale six-metrics & fusion AUC) for archive."""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats

OUT = OUTDIR
with open(os.path.join(OUT, "T1_canonical_main.pkl"), "rb") as f:
    P = pickle.load(f)
A = P["auc"]           # {key:{model: array(10)}}
full = P["full"]       # {key:{model:{metric:[]}}}
SUPPL = P["suppl"]

MODELS = ["SVM", "LR", "RF", "XGBoost", "KNN"]
SCALE = [f"scale{i}" for i in range(5)]
FUS = [("fus_base", "base(1mm gray)"), ("fus_vfa", "+VFA"), ("fus_p3", "+3"),
       ("fus_p13", "+13"), ("fus_p21", "+21")]
PAIRS = [("scale0", "scale1", "scale 0->1mm"), ("scale1", "scale2", "scale 1->2mm"),
         ("fus_base", "fus_vfa", "base->+VFA"), ("fus_base", "fus_p3", "base->+3"),
         ("fus_vfa", "fus_p3", "+VFA->+3"), ("fus_p3", "fus_p13", "+3->+13"),
         ("fus_p13", "fus_p21", "+13->+21"), ("fus_p3", "fus_p21", "+3->+21")]

lines = []
rows = []

def ci(d, rng=None):
    rng = rng or np.random.default_rng(0)
    b = rng.choice(d, (5000, len(d)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

lines.append("CANONICAL (current released code; tol=1e-4, saga, 10 seeds)")
lines.append("Bonferroni reference for the 10 pre-planned model-level comparisons: alpha = 0.005.")
lines.append("")
for a, b, lbl in PAIRS:
    for m in MODELS:
        Aa, Bb = np.asarray(A[a][m]), np.asarray(A[b][m])
        d = Bb - Aa
        md = float(d.mean()); sd = float(d.std(ddof=1)); lo, hi = ci(d)
        _, pt = stats.ttest_rel(Aa, Bb)
        try:
            _, pw = stats.wilcoxon(Aa, Bb)
        except ValueError:
            pw = float("nan")
        sig = "Y" if pt < 0.05 else "n"
        rows.append(dict(comparison=lbl, model=m,
                         meanA=round(Aa.mean(), 4), meanB=round(Bb.mean(), 4),
                         delta=round(md, 4), sd=round(sd, 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         p_t=round(float(pt), 4), p_w=round(float(pw), 4),
                         nominal_05=sig, survives_alpha0005=("Y" if pt < 0.005 else "n")))
    # line summary XGB for brevity in log
    m = "XGBoost"
    Aa, Bb = np.asarray(A[a][m]), np.asarray(A[b][m]); d = Bb - Aa
    _, pt = stats.ttest_rel(Aa, Bb)
    lines.append(f"{lbl:14s} XGB: {Aa.mean():.3f}->{Bb.mean():.3f} d={d.mean():+.4f} "
                 f"t-p={pt:.4f}")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "T1_publish_pair_stats.csv"), index=False,
          encoding="utf-8-sig")

# tidy scale six-metrics long
recs = []
for key in SCALE:
    mm = key.replace("scale", "")
    for m in MODELS:
        for metric in ["acc", "sen", "spe", "prec", "f1", "auc"]:
            a = np.asarray(full[key][m][metric])
            recs.append(dict(Scale=f"{mm}mm", Model=m, Metric=metric,
                             Mean=round(float(a.mean()), 4), SD=round(float(a.std()), 4)))
pd.DataFrame(recs).to_csv(os.path.join(OUT, "T1_publish_scale6.csv"), index=False,
                          encoding="utf-8-sig")
# fusion AUC
recs2 = []
for key, tag in FUS:
    for m in MODELS:
        a = np.asarray(A[key][m])
        recs2.append(dict(FeatureSet=tag, Model=m,
                          AUC=round(float(a.mean()), 4), SD=round(float(a.std()), 4)))
pd.DataFrame(recs2).to_csv(os.path.join(OUT, "T1_publish_fusion_auc.csv"), index=False,
                           encoding="utf-8-sig")

lines.append("")
lines.append("Supplementary (XGB grayscale vs +3):")
s = SUPPL
lines.append(f"  NRI {s['nri'][0]:.3f}±{s['nri'][1]:.3f} p={s['nri'][2]:.4f} "
             f"event={s['nri_event']:.3f} non-event={s['nri_nonevent']:.3f}")
lines.append(f"  Brier G {s['brier'][0][0]:.3f}±{s['brier'][0][1]:.3f} -> M "
             f"{s['brier'][1][0]:.3f}±{s['brier'][1][1]:.3f} p={s['brier'][2]:.4f}")
lines.append(f"  Eavg G {s['eavg'][0]:.3f} vs M {s['eavg'][1]:.3f}")
lines.append(f"  DCA benefit range {s['dca_range']}  | VFA0 AUC "
             f"{s['sub_v0_auc'][0]:.3f}±{s['sub_v0_auc'][1]:.3f} ; VFA>0 "
             f"{s['sub_v1_auc'][0]:.3f}±{s['sub_v1_auc'][1]:.3f}")
txt = "\n".join(lines)
with open(os.path.join(OUT, "T1_publish_report.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
print(txt)
print("\n[OK] wrote canonical_pair_stats.csv / scale6 / fusion_auc / report.txt")

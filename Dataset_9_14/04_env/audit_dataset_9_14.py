# -*- coding: utf-8 -*-
"""Deep audit of Dataset_9_14: structure, naming, isolation, inputs, outputs, runnability."""
import csv
import hashlib
import json
import os
import pickle
import py_compile
import re
import subprocess
import sys
import tempfile

import numpy as np

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable                                          # audit runs with the same interpreter
DS = ""                                                      # optional source dataset for the md5 cross-check
CJK = re.compile(r"[\u4e00-\u9fff]")
res = []


def chk(name, ok, detail=""):
    res.append((bool(ok), name, detail))
    print("%-5s %-56s %s" % ("PASS" if ok else "FAIL", name, detail))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------- A. structure ----------------
got = sorted(d for d in os.listdir(P) if os.path.isdir(os.path.join(P, d)))
chk("A1 top folders = 01_data/02_scripts/03_outputs/04_env", got == ["01_data", "02_scripts", "03_outputs", "04_env"], str(got))
rf = sorted(f for f in os.listdir(P) if os.path.isfile(os.path.join(P, f)))
chk("A2 root files = INDEX.csv + README.md", rf == ["INDEX.csv", "README.md"], str(rf))
for d in ["01_bus_features", "02_flow_features", "03_feature_extraction", "04_doppler_selection", "05_splits", "06_v0"]:
    chk("A3 01_data/" + d, os.path.isdir(os.path.join(P, "01_data", d)))
chk("A4 03_outputs has 00_reference and 99_rerun",
    os.path.isdir(os.path.join(P, "03_outputs", "00_reference"))
    and os.path.isdir(os.path.join(P, "03_outputs", "99_rerun")))
empties = [os.path.relpath(r, P) for r, d, f in os.walk(P) if not d and not f]
chk("A5 no empty folders anywhere", not empties, str(empties[:5]))

# ---------------- B. naming ----------------
AREA = {"T1": "01_table1", "T2": "02_table2", "S321": "03_sec321_scale", "S322": "04_sec322_stats",
        "S323": "05_sec323_clinical", "S324": "06_sec324_subgroup", "ALL": "07_all_unified"}
bad = []
for r, _, fs in os.walk(os.path.join(P, "02_scripts")):
    area = os.path.basename(r)
    for f in fs:
        if not f.endswith(".py") or AREA.get(f.split("_")[0]) != area:
            bad.append("%s/%s" % (area, f))
chk("B1 script name matches its folder", not bad, str(bad))
bad = [f for r, _, fs in os.walk(P) for f in fs if "_dup" in f]
chk("B2 no *_dup leftovers", not bad, str(bad[:4]))
bad = []
for r, _, fs in os.walk(P):
    for f in fs:
        rel = os.path.relpath(os.path.join(r, f), P)
        if CJK.search(rel):
            bad.append(rel)
        elif f.endswith((".py", ".md", ".csv", ".txt", ".yml", ".yaml", ".json")):
            if CJK.search(open(os.path.join(r, f), encoding="utf-8", errors="ignore").read()):
                bad.append(rel)
chk("B3 no Chinese in names / text files", not bad, str(bad[:4]))

# ---------------- C. isolation ----------------
CN_PAPER = "\u8bba\u6587"          # paper folder name, kept as escapes to keep this file ASCII
CN_DET = "\u786e\u5b9a"            # determined folder name
CN_REPRO = "\u590d\u73b0\u5305"    # repro-pack folder name
PAT = [r"(?<![A-Za-z0-9_])[A-Za-z]:\\", r"Desktop", r"anaconda3",
       CN_PAPER, CN_DET, CN_REPRO]
leak = []
for r, _, fs in os.walk(os.path.join(P, "02_scripts")):
    for f in fs:
        if f.endswith(".py"):
            t = open(os.path.join(r, f), encoding="utf-8").read()
            for p in PAT:
                if re.search(p, t):
                    leak.append("%s ~ %s" % (f, p))
chk("C1 no external-path reference in any script", not leak, "; ".join(leak[:4]))
shim_ok = True
mkdir_ok = True
for r, _, fs in os.walk(os.path.join(P, "02_scripts")):
    area = os.path.basename(r)
    for f in fs:
        if f.endswith(".py"):
            t = open(os.path.join(r, f), encoding="utf-8").read()
            if ("parents[2]" not in t) or ("'03_outputs' / '%s'" % area not in t):
                shim_ok = False
                chk("C2 shim wrong in " + f, False)
            if "makedirs(OUTDIR, exist_ok=True)" not in t:
                mkdir_ok = False
chk("C2 every script resolves paths to 01_data + its own 03_outputs area", shim_ok)
chk("C3 every script creates its own output folder at run time", mkdir_ok)
shadow = []
for r, _, fs in os.walk(os.path.join(P, "02_scripts")):
    for f in fs:
        if f.endswith(".py"):
            body = "\n".join(open(os.path.join(r, f), encoding="utf-8").read().splitlines()[9:])
            if re.search(r"^\s*(?:DATA|OUTDIR)\s*(?:,[^=\n]*)?=", body, re.M):
                shadow.append(f)
chk("C4 no script shadows the DATA/OUTDIR path shim", not shadow, str(shadow))

# ---------------- D. inputs ----------------
bad = []
with tempfile.TemporaryDirectory() as td:
    for r, _, fs in os.walk(os.path.join(P, "02_scripts")):
        for f in fs:
            if f.endswith(".py"):
                try:
                    py_compile.compile(os.path.join(r, f), cfile=os.path.join(td, f + "c"), doraise=True)
                except Exception as e:
                    bad.append(f)
chk("D1 all scripts compile", not bad, str(bad))
names = ["01_bus_features", "02_flow_features", "03_feature_extraction", "04_doppler_selection", "05_splits", "06_v0"]
used = set()
for r, _, fs in os.walk(os.path.join(P, "02_scripts")):
    for f in fs:
        if f.endswith(".py"):
            t = open(os.path.join(r, f), encoding="utf-8").read()
            used |= {n for n in names if n in t}
missing = sorted(used - set(os.listdir(os.path.join(P, "01_data"))))
chk("D2 referenced data folders all present", not missing, str(missing))
chk("D3 data folders actually used: %d/6" % len(used), len(used) >= 5, ",".join(sorted(used)))

# ---------------- E. reference values ----------------
REF = os.path.join(P, "03_outputs", "00_reference")
d = pickle.load(open(os.path.join(REF, "S323_clinical_utility.pkl"), "rb"))
chk("E1 reference NRI = 0.4388", abs(d["nri"]["mean"] - 0.4388) < 5e-5, "%.4f" % d["nri"]["mean"])
chk("E2 reference Brier 0.1852->0.1685 (p=0.0024)",
    abs(d["brier"]["gray"] - 0.1852) < 5e-5 and abs(d["brier"]["dopp"] - 0.1685) < 5e-5 and abs(d["brier"]["p"] - 0.0024) < 5e-5,
    "%.4f -> %.4f p=%.4f" % (d["brier"]["gray"], d["brier"]["dopp"], d["brier"]["p"]))
chk("E3 reference DCA = 4-98%", [round(x * 100) for x in d["dca"]["benefit_range"]] == [4, 98], str(d["dca"]["benefit_range"]))
vv = pickle.load(open(os.path.join(REF, "T1_vfa_row.pkl"), "rb"))["B_VFA_only"]
vals = [vv[m]["auc"][0] for m in ["SVM", "LR", "RF", "XGBoost", "KNN"]]
chk("E4 reference G1+VFA row = 0.7895/0.7910/0.7991/0.8237/0.7257",
    all(abs(a - b) < 5e-5 for a, b in zip(vals, [0.7895, 0.7910, 0.7991, 0.8237, 0.7257])),
    " ".join("%.4f" % v for v in vals))
c = pickle.load(open(os.path.join(REF, "T1_canonical_main.pkl"), "rb"))
chk("E5 reference canonical fus_p3 XGBoost = 0.8274",
    abs(float(np.mean(c["auc"]["fus_p3"]["XGBoost"])) - 0.8274) < 5e-5,
    "%.4f" % float(np.mean(c["auc"]["fus_p3"]["XGBoost"])))
chk("E6 reference +D3/+D13 rows 5-model means present",
    all(m in c["auc"]["fus_p13"] for m in ["SVM", "LR", "RF", "XGBoost", "KNN"]))
sj = json.load(open(os.path.join(REF, "S322_statistical_supplement.json"), encoding="utf-8"))
y = sj["xgb_base_youden"]
chk("E7 reference Youden cells 73.8 / 78.7 / Topt 0.407",
    abs(y["Sen_Youden"][0] * 100 - 73.8) < 0.05 and abs(y["Spe_Youden"][0] * 100 - 78.7) < 0.05 and abs(y["Topt"][0] - 0.407) < 5e-4,
    "%.1f %.1f %.3f" % (y["Sen_Youden"][0] * 100, y["Spe_Youden"][0] * 100, y["Topt"][0]))
chk("E8 reference PR-AUC text contains 0.775",
    "0.775" in open(os.path.join(REF, "S322_pr_youden.txt"), encoding="utf-8").read())
rr = pickle.load(open(os.path.join(REF, "T1_official_repo_roc.pkl"), "rb"))
chk("E9 reference official-repo pkl has 5 models x aucs",
    all(m in rr.get("aucs", {}) for m in ["SVM", "LR", "RF", "XGBoost", "KNN"]), str(list(rr)[:6]))

# ---------------- F. INDEX ----------------
rows = list(csv.DictReader(open(os.path.join(P, "INDEX.csv"), encoding="utf-8")))
listed = {r["script"] for r in rows}
actual = {f for r, _, fs in os.walk(os.path.join(P, "02_scripts")) for f in fs if f.endswith(".py")}
chk("F1 INDEX.csv == scripts on disk", listed == actual,
    "missing=%s extra=%s" % (sorted(actual - listed), sorted(listed - actual)))

# ---------------- G. data integrity ----------------
pairs = [("01_data/01_bus_features/benign_1mm.csv", os.path.join(DS, "BUS_features", "benign_1mm.csv")),
         ("01_data/02_flow_features/benign_flow_density.csv", os.path.join(DS, "flow_features", "benign_flow_density.csv")),
         ("01_data/03_feature_extraction/Params.yaml", os.path.join(DS, "feature_extraction", "Params.yaml")),
         ("01_data/06_v0/group4_BT_multi_doppler_raw.csv", os.path.join(DS, "data", "prepared", "group4_BT_multi_doppler_raw.csv"))]
if DS and os.path.isdir(DS):
    bad = [a for a, b in pairs if not (os.path.exists(os.path.join(P, a)) and os.path.exists(b) and md5(os.path.join(P, a)) == md5(b))]
    chk("G1 inputs byte-identical to source dataset", not bad, str(bad))
else:
    chk("G1 inputs present", all(os.path.exists(os.path.join(P, a)) for a, _ in pairs), "source dataset not available")

# ---------------- H. live runs ----------------
runs = [("04_sec322_stats", "S322_statistical_supplement.py", "04_sec322_stats/S322_statistical_supplement.json"),
        ("05_sec323_clinical", "S323_clinical_utility.py", "05_sec323_clinical/S323_clinical_utility.pkl"),
        ("01_table1", "T1_publish_tables.py", "01_table1/T1_publish_report.txt")]
for sub, script, out in runs:
    tgt = os.path.join(P, "03_outputs", out)
    before = os.path.getmtime(tgt) if os.path.exists(tgt) else None
    r = subprocess.run([PY, "-u", script], cwd=os.path.join(P, "02_scripts", sub),
                       capture_output=True, text=True, timeout=1200)
    after = os.path.getmtime(tgt) if os.path.exists(tgt) else None
    chk("H1 %s -> %s (rewritten)" % (script, out),
        r.returncode == 0 and after is not None and after != before, "rc=%d" % r.returncode)

print()
n = sum(1 for ok, _, _ in res if ok)
print("AUDIT RESULT: %d/%d checks passed" % (n, len(res)))
for ok, name, det in res:
    if not ok:
        print("   FAILED:", name, "|", det)

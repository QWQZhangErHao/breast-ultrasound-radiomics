#!/usr/bin/env python3
"""BUSI External Validation — using paper_repro env's standard PyRadiomics."""

import os, sys, pickle, time, warnings, glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import binary_dilation
from PIL import Image
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

from radiomics import featureextractor as FEE

warnings.filterwarnings("ignore")

N_JOBS = 32
N_SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]
CV_FOLDS = 5
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

BUSI_DIR = r"C:\Users\ZhangErHao\Desktop\dataset\BUSI\Dataset_BUSI_with_GT"
OUTPUT_DIR = r"C:\Users\ZhangErHao\Desktop\重新验证后的论文数据"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSIFIERS = {
    "SVM": lambda: SVC(kernel="linear", C=1.0, probability=True, random_state=42),
    "LR": lambda: LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=10000, random_state=42),
    "RF": lambda: RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42, n_jobs=N_JOBS),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0, n_jobs=N_JOBS, tree_method="hist"),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=N_JOBS),
}

# Initialize PyRadiomics extractor — matching paper's settings
EXTRACTOR = FEE.RadiomicsFeatureExtractor(additionalInfo=False, correctMask=True)
for fc in ["firstorder", "shape2D", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
    EXTRACTOR.enableFeatureClassByName(fc)
EXTRACTOR.enableFeatureClassByName("shape", enabled=False)


# =============================================================================
# FEATURE EXTRACTION (standard PyRadiomics)
# =============================================================================
def extract_one(img_path, mask_path, label, filename, peri_pixels=5):
    """Extract intratumoral + 1mm peritumoral features using standard PyRadiomics."""
    import SimpleITK as sitk

    img = np.array(Image.open(img_path).convert('L')).astype(np.float64)
    mask = (np.array(Image.open(mask_path).convert('L')) > 0).astype(np.uint8)
    if mask.sum() == 0:
        return None
    img_sitk = sitk.GetImageFromArray(img.astype(np.float64))
    mask_sitk = sitk.GetImageFromArray(mask)

    record = {"label": label, "filename": filename}

    # Intratumoral (0mm)
    try:
        feats = EXTRACTOR.execute(img_sitk, mask_sitk)
        for k, v in feats.items():
            if k.startswith("original_"):
                record[k] = float(v) if isinstance(v, (int, float)) else 0.0
    except Exception:
        return None

    # Peritumoral (~1mm)
    struct = np.ones((peri_pixels, peri_pixels), dtype=bool)
    dilated = binary_dilation(mask > 0, structure=struct)
    peri_mask = (dilated & ~(mask > 0)).astype(np.uint8)
    if peri_mask.sum() > 10:
        peri_sitk = sitk.GetImageFromArray(peri_mask)
        try:
            pfeats = EXTRACTOR.execute(img_sitk, peri_sitk)
            for k, v in pfeats.items():
                if k.startswith("original_"):
                    record[f"peri_{k}"] = float(v) if isinstance(v, (int, float)) else 0.0
        except Exception:
            pass

    return record


def load_and_extract():
    """Load BUSI + extract features in parallel."""
    samples = []
    for cls, label in [("benign", 0), ("malignant", 1)]:
        d = os.path.join(BUSI_DIR, cls)
        if not os.path.isdir(d):
            continue
        imgs = sorted(glob.glob(os.path.join(d, "*.png")))
        imgs = [f for f in imgs if "_mask" not in os.path.basename(f)]
        for imp in imgs:
            base = os.path.splitext(imp)[0]
            msk = f"{base}_mask.png"
            if os.path.isfile(msk):
                samples.append((imp, msk, label, os.path.basename(imp)))
    print(f"  {len(samples)} images, {N_JOBS} workers...")
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(extract_one)(*s) for s in samples
    )
    return pd.DataFrame([r for r in res if r is not None])


# =============================================================================
# ML PIPELINE (same as paper)
# =============================================================================
def is_feat(c):
    return c not in ["label", "filename"] and not ("diagnostics_" in c or "Shape" in c or "shape2D" in c.lower())

def is_shape(c):
    return "shape" in c.lower() or "diagnostics" in c.lower()

def run_seed(seed, X, y):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[tr])
    lasso = LassoCV(cv=CV_FOLDS, random_state=seed, max_iter=10000, n_jobs=N_JOBS, tol=1e-4)
    lasso.fit(X_tr_s, y[tr])
    mask = lasso.coef_ != 0
    if mask.sum() == 0:
        mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True
    X_tr = scaler.transform(X[tr])[:, mask]
    X_te = scaler.transform(X[te])[:, mask]
    res = {}
    for nm, fn in CLASSIFIERS.items():
        c = fn(); c.fit(X_tr, y[tr])
        p = c.predict_proba(X_te)[:, 1]
        pr = (p >= 0.5).astype(int)
        tn, fp, fn2, tp = confusion_matrix(y[te], pr).ravel()
        res[nm] = {
            "auc": round(roc_auc_score(y[te], p), 4),
            "acc": round(accuracy_score(y[te], pr), 4),
            "sen": round(tp/(tp+fn2) if (tp+fn2)>0 else 0, 4),
            "spe": round(tn/(tn+fp) if (tn+fp)>0 else 0, 4),
            "f1": round(f1_score(y[te], pr, zero_division=0), 4),
        }
    return res


def main():
    t0 = time.time()
    print("=" * 60)
    print("BUSI EXTERNAL VALIDATION (standard PyRadiomics)")
    print("=" * 60)

    print("\n[1/3] Extracting features (standard PyRadiomics)...")
    df = load_and_extract()
    feat_cols = [c for c in df.columns if is_feat(c) and not is_shape(c)]
    print(f"  {len(df)} samples, {len(feat_cols)} features")
    nb, nm = sum(df.label==0), sum(df.label==1)
    print(f"  {nb} benign, {nm} malignant")

    X = df[feat_cols].values
    y = df.label.values

    print(f"\n[2/3] 10 seeds...")
    t1 = time.time()
    rl = Parallel(n_jobs=min(N_JOBS, 10))(delayed(run_seed)(s, X, y) for s in N_SEEDS)
    ar = {s: rl[i] for i, s in enumerate(N_SEEDS)}
    print(f"  {time.time()-t1:.0f}s")

    print(f"\n[3/3] RESULTS:")
    print(f"{'Classifier':<12} {'AUC':>10} {'Acc':>10} {'Sen':>10} {'Spe':>10} {'F1':>10}")
    print("-" * 55)
    for c in CLASSIFIERS:
        a = [ar[s][c]["auc"] for s in N_SEEDS]
        ac = [ar[s][c]["acc"] for s in N_SEEDS]
        se = [ar[s][c]["sen"] for s in N_SEEDS]
        sp = [ar[s][c]["spe"] for s in N_SEEDS]
        f = [ar[s][c]["f1"] for s in N_SEEDS]
        print(f"{c:<12} {np.mean(a):.3f}+-{np.std(a):.3f}  {np.mean(ac):.3f}+-{np.std(ac):.3f}  {np.mean(se):.3f}+-{np.std(se):.3f}  {np.mean(sp):.3f}+-{np.std(sp):.3f}  {np.mean(f):.3f}+-{np.std(f):.3f}")

    with open(os.path.join(OUTPUT_DIR, "busi_results_standard.pkl"), "wb") as f:
        pickle.dump({"results": ar, "df": df}, f)
    print(f"\nDone! {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

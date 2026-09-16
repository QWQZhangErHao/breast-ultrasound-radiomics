# -*- coding: utf-8 -*-
# --- standalone paths (self-contained package) ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
DATA = str(_ROOT / '01_data')
OUTDIR = str(_ROOT / '03_outputs' / '03_sec321_scale')
_os.makedirs(OUTDIR, exist_ok=True)
# -------------------------------------------------

"""Original script used 5 seeds, tol=1e-2 and XGB gpu_hist (needs xgboost<=2.0)."""
import os, re, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import xgboost as xgb

BUS_DIR = os.path.join(DATA, "01_bus_features")
N_SEEDS = [42, 123, 2024, 7, 999]


def sanitize_id(raw):
    m = re.search(r'(\d+)', str(raw)); return int(m.group(1)) if m else None


def is_feat(c):
    return not any(p in c.lower() for p in ['patient_name', 'file_name', 'diagnostics_'])


results = {}
for mm in [0, 1, 2, 3, 4]:
    b = pd.read_csv(os.path.join(BUS_DIR, 'benign_%dmm.csv' % mm))
    m = pd.read_csv(os.path.join(BUS_DIR, 'malignant_%dmm.csv' % mm))
    b['label'], m['label'] = 0, 1
    df = pd.concat([b, m], ignore_index=True)
    df['patient_name'] = df['patient_name'].apply(sanitize_id)
    drop = [c for c in df.columns if c not in ['patient_name', 'label'] and not is_feat(c)]
    df = df.drop(columns=drop, errors='ignore')
    feats = [c for c in df.columns if c not in ['patient_name', 'label']]
    X = df[feats].values; y = df['label'].values
    svm_a, xgb_a = [], []
    for seed in N_SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lasso = LassoCV(cv=5, random_state=seed, max_iter=5000, tol=1e-2)
        lasso.fit(Xtr, y[tr])
        mk = lasso.coef_ != 0
        if mk.sum() == 0:
            mk[np.argsort(np.abs(lasso.coef_))[-10:]] = True
        svm_c = SVC(kernel='linear', C=1.0, probability=True, random_state=seed)
        svm_c.fit(Xtr[:, mk], y[tr])
        svm_a.append(roc_auc_score(y[te], svm_c.predict_proba(Xte[:, mk])[:, 1]))
        xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
    random_state=seed, verbosity=0, tree_method='hist', device='cuda')
        xgb_c.fit(Xtr[:, mk], y[tr])
        xgb_a.append(roc_auc_score(y[te], xgb_c.predict_proba(Xte[:, mk])[:, 1]))
    results[mm] = {'svm': (np.mean(svm_a), np.std(svm_a)), 'xgb': (np.mean(xgb_a), np.std(xgb_a))}
    print('%dmm: SVM AUC=%.4f+-%.4f | XGB AUC=%.4f+-%.4f'
          % (mm, results[mm]['svm'][0], results[mm]['svm'][1],
             results[mm]['xgb'][0], results[mm]['xgb'][1]), flush=True)


# ============================================================================
# ============================================================================
def _save_results():
    import os as _os, pickle as _pk, datetime as _dt
    _skip_types = {"module", "function", "builtin_function_or_method", "type",
                   "method", "classmethod", "staticmethod", "property", "module_"}
    _keep = {}
    for _k, _v in list(globals().items()):
        if _k.startswith("_") or _k == "_save_results":
            continue
        if type(_v).__name__ in _skip_types or callable(_v):
            continue
        if isinstance(_v, (int, float, str, bool, type(None), list, tuple, dict, set)):
            _keep[_k] = _v
        elif type(_v).__name__ in ("ndarray", "DataFrame", "Series", "matrix") \
                or (hasattr(_v, "tolist") and not callable(_v)):
            _keep[_k] = _v
    _keep["_script"] = _os.path.basename(__file__)
    _keep["_saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    try:
        import sklearn as _sk, xgboost as _xg, numpy as _np, sys as _sys
        _keep["_env"] = {"sklearn": _sk.__version__, "xgboost": _xg.__version__,
                         "numpy": _np.__version__, "python": _sys.version.split()[0]}
    except Exception:
        pass
    _p = _os.path.join(OUTDIR, "S321_scale_patched.pkl")
    try:
        with open(_p, "wb") as _f:
            _pk.dump(_keep, _f)
        print("[SAVED] %s  (%.1f KB, %d vars)" % (_p, _os.path.getsize(_p) / 1024, len(_keep)))
    except Exception as _e:
        print("[SAVE-FAIL]", _e)


_save_results()

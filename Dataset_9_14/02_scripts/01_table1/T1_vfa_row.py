#!/usr/bin/env python3
# --- standalone paths (self-contained package) ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
DATA = str(_ROOT / '01_data')
OUTDIR = str(_ROOT / '03_outputs' / '01_table1')
_os.makedirs(OUTDIR, exist_ok=True)
# -------------------------------------------------

"""Detailed performance report for the 3 recommended Doppler feature sets."""
import os, warnings, re
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import xgboost as xgb
from joblib import Parallel, delayed

BASE = DATA
BUS_DIR = os.path.join(BASE, '01_bus_features')
FLOW_DIR = os.path.join(BASE, '02_flow_features')
N_JOBS = 8
N_SEEDS = [42, 123, 2024, 7, 999, 314, 271, 1618, 2048, 4096]

def sanitize_id(raw):
    m = re.search(r'(\d+)', str(raw)); return int(m.group(1)) if m else None
def is_feat(c):
    return not any(p in c.lower() for p in ['patient_name','file_name','diagnostics_'])
def load_bus(mm):
    b = pd.read_csv(os.path.join(BUS_DIR, 'benign_%dmm.csv' % mm))
    m = pd.read_csv(os.path.join(BUS_DIR, 'malignant_%dmm.csv' % mm))
    b['label'], m['label'] = 0, 1
    df = pd.concat([b, m], ignore_index=True)
    df['patient_name'] = df['patient_name'].apply(sanitize_id)
    return df

d0, d2 = load_bus(0), load_bus(2)
for t in [d0, d2]:
    drop = [c for c in t.columns if c not in ['patient_name','label'] and not is_feat(c)]
    t.drop(columns=drop, inplace=True, errors='ignore')
d2 = d2.rename(columns={c: c if c in ['patient_name','label'] else '%s_2mm' % c for c in d2.columns})
d2.drop(columns=[c for c in d2.columns if 'shape' in c.lower() and c not in ['patient_name','label']], inplace=True, errors='ignore')
g2 = pd.merge(d0, d2, on=['patient_name','label'], how='inner')

bf_b = pd.read_csv(os.path.join(FLOW_DIR, 'blood_flow_features_benign.csv'))
bf_m = pd.read_csv(os.path.join(FLOW_DIR, 'blood_flow_features_malignant.csv'))
bf_m['label'], bf_b['label'] = 1, 0
bf_all = pd.concat([bf_b, bf_m], ignore_index=True)
bf_all['patient_name'] = bf_all['image_id'].apply(sanitize_id)
bf_feats = [c for c in bf_all.columns if c.startswith('bf_')]

vfa_b = pd.read_csv(os.path.join(FLOW_DIR, 'benign_flow_density.csv'))
vfa_m = pd.read_csv(os.path.join(FLOW_DIR, 'malignant_flow_density.csv'))
vfa_m['label'], vfa_b['label'] = 1, 0
vfa_all = pd.concat([vfa_b, vfa_m], ignore_index=True)
vfa_all['patient_name'] = vfa_all['case_id'].apply(sanitize_id)
vfa_all['flow_density'] = vfa_all['flow_density'].astype(float)

all_dop = set(bf_feats + ['flow_density'])

# The 3 candidate sets
candidates = {
    'A_No_Doppler(baseline)': [],
    'B_VFA_only': ['flow_density'],
    'C_3_optimized': ['flow_density', 'bf_firstorder_Energy', 'bf_glrlm_GrayLevelNonUniformity'],
    'D_14_LASSO': ['flow_density', 'bf_firstorder_Energy', 'bf_firstorder_Entropy',
        'bf_firstorder_RobustMeanAbsoluteDeviation', 'bf_glrlm_GrayLevelNonUniformity',
        'bf_glrlm_RunLengthNonUniformity', 'bf_glrlm_RunEntropy',
        'bf_gldm_DependenceNonUniformity', 'bf_gldm_DependenceEntropy',
        'bf_glszm_GrayLevelNonUniformity', 'bf_glszm_ZoneEntropy',
        'bf_glszm_ZonePercentage', 'bf_ngtdm_Contrast'],
}

ALL_CLFS = {
    'SVM': lambda s: SVC(kernel='linear', C=1.0, probability=True, random_state=s),
    'LR': lambda s: LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=10000, random_state=s),
    'RF': lambda s: RandomForestClassifier(n_estimators=500, max_depth=5, random_state=s, n_jobs=8),
    'XGBoost': lambda s: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=s, verbosity=0, tree_method='hist', device='cuda'),
    'KNN': lambda s: KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=8),
}

def evaluate_set(set_name, dop_list):
    df = g2.copy()
    if dop_list:
        dop_sub = bf_all[['patient_name'] + [c for c in dop_list if c in bf_feats]].copy()
        if 'flow_density' in dop_list:
            dv = vfa_all[['patient_name','flow_density']].copy()
            dop_sub = pd.merge(dop_sub, dv, on='patient_name', how='right')
            dop_sub['flow_density'] = dop_sub['flow_density'].fillna(0.0)
        df = pd.merge(df, dop_sub, on='patient_name', how='inner')

    fnames = [c for c in df.columns if c not in ['patient_name','label']]
    gray_n = [c for c in fnames if c not in all_dop]
    dop_n = [c for c in fnames if c in all_dop]
    gray_i = [fnames.index(c) for c in gray_n]
    dop_i = [fnames.index(c) for c in dop_n]
    X = df[fnames].values; y = df['label'].values

    def eval_seed(seed):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
        sg = StandardScaler()
        Xgs = sg.fit_transform(X[tr][:, gray_i]); Xge = sg.transform(X[te][:, gray_i])
        lasso = LassoCV(cv=5, random_state=seed, max_iter=5000, tol=1e-2)
        lasso.fit(Xgs, y[tr])
        mask = lasso.coef_ != 0
        if mask.sum() == 0:
            mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True
        Xg_tr, Xg_te = Xgs[:, mask], Xge[:, mask]

        if dop_i:
            sd = StandardScaler()
            Xd_tr = sd.fit_transform(X[tr][:, dop_i]); Xd_te = sd.transform(X[te][:, dop_i])
            Xtr = np.concatenate([Xg_tr, Xd_tr], axis=1)
            Xte = np.concatenate([Xg_te, Xd_te], axis=1)
        else:
            Xtr, Xte = Xg_tr, Xg_te

        res = {}
        for clf_name, clf_fn in ALL_CLFS.items():
            clf = clf_fn(seed)
            clf.fit(Xtr, y[tr])
            y_proba = clf.predict_proba(Xte)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            tn, fp, fn, tp = confusion_matrix(y[te], y_pred).ravel()
            res[clf_name] = {
                'auc': roc_auc_score(y[te], y_proba),
                'acc': accuracy_score(y[te], y_pred),
                'sen': tp / (tp + fn) if (tp+fn) > 0 else 0,
                'spe': tn / (tn + fp) if (tn+fp) > 0 else 0,
                'prec': precision_score(y[te], y_pred, zero_division=0),
                'f1': f1_score(y[te], y_pred, zero_division=0),
            }
        return res

    seed_res = Parallel(n_jobs=N_JOBS)(delayed(eval_seed)(s) for s in N_SEEDS)
    summary = {}
    for clf_name in ALL_CLFS:
        vals = {m: [seed_res[s][clf_name][m] for s in range(len(N_SEEDS))] for m in ['auc','acc','sen','spe','prec','f1']}
        summary[clf_name] = {m: (np.mean(vals[m]), np.std(vals[m])) for m in vals}
    return summary

# Run all
results = {}
for sname, sfeats in candidates.items():
    print('Evaluating %s...' % sname)
    results[sname] = evaluate_set(sname, sfeats)

# Print report
print('\n' + '=' * 95)
print('FINAL RECOMMENDATION REPORT')
print('=' * 95)
print('\n%-25s %-5s | %-18s %-18s %-18s' % ('Feature Set', 'N', 'SVM AUC', 'XGBoost AUC', 'RF AUC'))
print('-' * 85)

sets_order = ['A_No_Doppler(baseline)', 'B_VFA_only', 'C_3_optimized', 'D_14_LASSO']
set_labels = ['Baseline (no Doppler)', 'VFA only (1 feat)', '3 optimized (VFA+Energy+GLRLM)', '14 LASSO features']
baseline_auc = {clf: results[sets_order[0]][clf]['auc'][0] for clf in ['SVM','XGBoost','RF']}

for si, sn in enumerate(sets_order):
    r = results[sn]
    n_feat = len(candidates[sn])
    row = '%-25s %-5d' % (set_labels[si], n_feat)
    for clf in ['SVM', 'XGBoost', 'RF']:
        m, s = r[clf]['auc']
        d = m - baseline_auc[clf]
        row += ' | %7.4f+-%.4f (%+.4f)' % (m, s, d)
    print(row)

# Detailed per-model table
for clf_name in ['SVM', 'LR', 'RF', 'XGBoost', 'KNN']:
    print('\n' + '-' * 85)
    print('Classifier: %s' % clf_name)
    print('%-30s %-12s %-12s %-12s %-12s %-12s %-12s' % ('Feature Set', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1'))
    print('-' * 85)
    for si, sn in enumerate(sets_order):
        r = results[sn][clf_name]
        row = '%-30s' % set_labels[si]
        for metric in ['auc', 'acc', 'sen', 'spe', 'prec', 'f1']:
            m, s = r[metric]
            row += ' %7.4f+-%.4f   ' % (m, s)
        print(row)

# Compute Doppler gains
print('\n' + '=' * 95)
print('DOPPLER GAIN SUMMARY (vs baseline)')
print('=' * 95)
print('\n%-30s %-20s %-20s' % ('Feature Set', 'XGBoost AUC', 'vs baseline'))
print('-' * 70)
base_xgb = results[sets_order[0]]['XGBoost']['auc'][0]
for si, sn in enumerate(sets_order):
    if si == 0:
        continue
    m, s = results[sn]['XGBoost']['auc']
    d = m - base_xgb
    gain_pct = (m - base_xgb) / base_xgb * 100
    print('%-30s %7.4f +- %.4f   Δ=%+.4f (%+.1f%%)' % (set_labels[si], m, s, d, gain_pct))

# Save
TGT = OUTDIR
import pickle
with open(os.path.join(TGT, 'T1_vfa_row.pkl'), 'wb') as f:
    pickle.dump(results, f)
print('\nSaved to %s/final_recommendation.pkl' % TGT)

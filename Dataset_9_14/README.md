# Dataset_9_14

Reproduction package for the manuscript numbers. Self-contained: every script reads `01_data`
and writes `03_outputs`, both resolved from the script location.

## Layout

```
01_data\      inputs
  01_bus_features\        benign_/malignant_{0,1,2,3,4}mm.csv,
                          selected_features_learning_{0,1,2,3,4}mm.csv
  02_flow_features\       {benign,malignant}_flow_density.csv,
                          blood_flow_features_{benign,malignant}.csv
  03_feature_extraction\  Params.yaml
  04_doppler_selection\   doppler_selection.pkl
  05_splits\              splits_base_plus3.pkl
  06_v0\                  group4_BT_multi_doppler_raw.csv, cv_records\, proba\
02_scripts\   one script per paper location; INDEX.csv maps script -> input -> output -> environment
03_outputs\   00_reference\ = artifacts of the paper's own runs
               99_rerun\     = unified-environment re-run
               <area>\       = written by the scripts when they run
04_env\       environment.yml, pip_freeze.txt, versions.txt, audit_dataset_9_14.py
```

Script names are `<AREA>_<subject>.py`: `T1` Table 1, `S321`-`S324` Sections 3.2.1-3.2.4,
`ALL` all unified numbers. An output file carries the name of the script that wrote it.

## Environment

```powershell
conda env create -f 04_env/environment.yml     # python 3.12.13, numpy 2.5.1, scikit-learn 1.9.0, xgboost 3.3.0
```

Table 1 rows and Section 3.2.2 statistics were produced with the older stack in `04_env/versions.txt`
(python 3.9.23, scikit-learn 1.2.2, xgboost 1.7.6).

## Run

```powershell
cd 02_scripts/07_all_unified
python ALL_unified_pipeline.py        # ~15 min -> 03_outputs/07_all_unified
```

Individual sections: `cd` into the folder of the script and run it; each script creates its own
output folder on the first run. Whole-package self-check (structure, isolation, reference values,
live runs):

```powershell
python 04_env/audit_dataset_9_14.py   # ~2 min, 34 checks
```

## Values to check

| source | value |
|---|---|
| `99_rerun/ALL_unified_results.json` | Table 1 G1 row 0.786 / 0.790 / 0.797 / 0.803 / 0.718 |
| `99_rerun/ALL_unified_results.json` | delta-AUC of G1+D3 vs G1: +0.0173 / +0.0226 / +0.0258 / +0.0232 |
| `00_reference/T1_vfa_row.pkl` | `B_VFA_only` 0.7895 / 0.7910 / 0.7991 / 0.8237 / 0.7257 |
| `00_reference/T1_canonical_main.pkl` | `auc.fus_p3.XGBoost` mean 0.8274 |
| `00_reference/S323_clinical_utility.pkl` | NRI 0.4388, Brier 0.1852 -> 0.1685 (p 0.0024), DCA 4-98% |
| `00_reference/S322_pr_youden.txt` | PR-AUC 0.775 +- 0.044 |
| `00_reference/S322_statistical_supplement.json` | Youden 73.8 / 78.7, T_opt 0.407 |

## Verified cold re-runs

Every script below was copied to an empty output folder and run end to end, then compared with
`03_outputs/00_reference/`.

| script | environment | compared | result |
|---|---|---|---|
| `ALL_unified_pipeline.py` | 04_env/environment.yml | 4844 numeric cells | identical |
| `T1_canonical_main.py` | xgboost 1.7.6 stack | 25 AUC cells | identical |
| `S321_cumulative_scale.py` | 04_env/environment.yml | 150 rows (mean + SD) | identical |
| `S323_clinical_utility.py` | 04_env/environment.yml | 22 numeric cells | identical |
| `S322_statistical_supplement.py` | any (reads saved splits) | 42 numeric cells | identical |
| `T1_publish_tables.py` | any (reads saved results) | report + 3 tables | written |
| `T1_vfa_row.py` | xgboost 1.7.6 stack | SVM / LR / RF / KNN | identical |
| `T1_vfa_row.py` | xgboost 1.7.6 stack | XGBoost cells | 0.8228 vs archived 0.8237 |
| `S322_pr_youden.py` | xgboost 1.7.6 stack | PR-AUC | 0.7757 vs archived 0.7754 |

XGBoost is the only source of drift: the same script returns XGBoost AUC that differ by up to
~0.005 between xgboost 1.7.6, 3.3.0 and 3.4.1, and threshold-dependent XGBoost statistics drift
more. SVM / LR / RF / KNN reproduce bit-exactly. Compare XGBoost-derived cells to 3 decimals.

## Not included

Chains whose inputs are GB-scale (radiomics extraction from images, VFA from energy maps,
deep-learning training) are not shipped. `01_data/03_feature_extraction/Params.yaml` is the
PyRadiomics configuration of the image-based chain and is not read by any script here.

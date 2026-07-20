<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Status-Research-lightgrey" alt="Status: Research">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/GPU-XGBoost%20Accel.-orange.svg" alt="GPU XGBoost">
</p>

# Breast Ultrasound Radiomics — Multimodal Doppler + Grayscale Analysis

> **Code for:** *Multimodal Ultrasound Radiomics for Early Breast Cancer: Peritumoral Scale Optimization and Quantitative Doppler Feature Analysis in 546 ≤2 cm Breast Lesions*

This repository provides a complete **five-stage nested feature selection pipeline** for multimodal ultrasound radiomics analysis. It integrates grayscale radiomic features with quantitative Doppler features (VFA, Energy, GrayLevelNonUniformity) at optimal peritumoral scales, validated across 546 small breast lesions (≤2 cm).

---

## 📋 Table of Contents

- [Key Method](#key-method)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Overview](#file-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Data](#data)
- [Citation](#citation)
- [License](#license)

---

## 🧬 Key Method

- **Five-stage nested feature selection pipeline** — all stages nested within training folds to prevent data leakage
- **10 repeated 70/30 stratified splits** for robust performance estimation
- **Peritumoral scale optimization** (0–4 mm) to identify the optimal peritumoral region
- **Multimodal fusion**: Grayscale radiomics + 3 optimized Doppler features (VFA, Energy, GrayLevelNonUniformity)
- **5 classifiers evaluated**: Logistic Regression, Random Forest, XGBoost, SVM, and LightGBM
- **Comprehensive supplementary analysis**: NRI, Brier score, calibration curves, DCA, and hyperparameter sensitivity

### 📊 Pipeline Architecture

```
Raw Ultrasound → ROI Segmentation → Feature Extraction
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            Grayscale Radiomics                    Doppler Features
            (PyRadiomics, ~1000+ features)      (VFA, Energy, GLNU)
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                Feature Selection
                        (5-stage nested LASSO + Boruta)
                                        │
                                Model Training
                        (5 classifiers × 10 repeats)
                                        │
                                Performance Evaluation
                    (AUC, NRI, DCA, Calibration, Brier)
```

---

## ⚙️ Requirements

| Dependency | Version |
|-----------|---------|
| Python | 3.9+ |
| scikit-learn | 1.2.2 |
| XGBoost | 1.7.6 (GPU acceleration recommended) |
| SciPy | 1.10.1 |
| NumPy | 1.24.3 |
| PyRadiomics | latest |
| python-docx | latest |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/QWQZhangErHao/breast-ultrasound-radiomics.git
cd breast-ultrasound-radiomics

# (Recommended) Create a conda environment
conda create -n ultrasound-radiomics python=3.9
conda activate ultrasound-radiomics

# Install dependencies
pip install scikit-learn==1.2.2 xgboost==1.7.6 scipy==1.10.1 numpy==1.24.3
pip install pyradiomics python-docx

# For GPU-accelerated XGBoost (ensure CUDA is installed)
pip install xgboost==1.7.6 --config-settings use_cuda=true
```

---

## 🚀 Usage

### Main Pipeline
```bash
python run_1mm_pipeline_optimized.py
```

### Supplementary Analyses
```bash
python supplementary_analyses.py
python pr_youden_analysis.py
```

### External Validation
```bash
python validate_busi_standard.py
```

### Reproduce Paper Results
```bash
python reproduce_paper.py
```

---

## 📁 File Overview

| File | Description |
|------|-------------|
| `run_1mm_pipeline_optimized.py` | Main pipeline: 1mm peritumoral scale, nested LASSO, 5 classifiers, 10 seeds |
| `supplementary_analyses.py` | NRI, Brier, calibration, DCA, hyperparameter sensitivity (GPU-accelerated) |
| `pr_youden_analysis.py` | PR-AUC and Youden threshold computation |
| `validate_busi_standard.py` | External validation on BUSI (Kaggle) dataset |
| `validate_pipeline.py` | Original validation pipeline with nested CV |
| `reproduce_paper.py` | Initial paper results reproduction |
| `final_update.py` | Final pipeline updates and refinements |
| `modify_draft.py` | Draft manuscript generation utilities |

---

## 🔬 Data

> **Clinical ultrasound data** is not publicly available due to **institutional ethical restrictions** imposed by the ethics committee.

For external validation, the **BUSI (Breast Ultrasound Images)** dataset is available at:
- [Breast Ultrasound Images Dataset on Kaggle](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2025multimodal,
  title={Multimodal Ultrasound Radiomics for Early Breast Cancer: Peritumoral Scale Optimization and Quantitative Doppler Feature Analysis in 546 ≤2 cm Breast Lesions},
  author={Zhang, Erhao and others},
  journal={Under Review},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

<p align="center">
  <i>For research purposes only. Not for clinical use.</i>
</p>

# Breast Ultrasound Radiomics — Multimodal Doppler + Grayscale Analysis

Code accompanying the manuscript:  
**"Multimodal Ultrasound Radiomics for Early Breast Cancer: Peritumoral Scale Optimization and Quantitative Doppler Feature Analysis in 546 ≤2 cm Breast Lesions"**

## Requirements
- Python 3.9+
- scikit-learn 1.2.2
- XGBoost 1.7.6 (GPU acceleration recommended)
- SciPy 1.10.1, NumPy 1.24.3
- PyRadiomics (for feature extraction)
- python-docx (for paper generation)

## Files

| File | Description |
|------|-------------|
| `run_1mm_pipeline_optimized.py` | Main pipeline: 1mm peritumoral scale, nested LASSO, 5 classifiers, 10 seeds |
| `supplementary_analyses.py` | NRI, Brier, calibration, DCA, hyperparameter sensitivity (GPU-accelerated) |
| `pr_youden_analysis.py` | PR-AUC and Youden threshold computation |
| `validate_busi_standard.py` | External validation on BUSI dataset |
| `validate_pipeline.py` | Original validation pipeline with nested CV |
| `reproduce_paper.py` | Initial paper results reproduction |

## Key Method
- Five-stage nested feature selection pipeline (all stages nested within training folds)
- 10 repeated 70/30 stratified splits
- Peritumoral scale optimization (0-4 mm)
- Multimodal fusion: grayscale radiomics + 3 optimized Doppler features (VFA, Energy, GrayLevelNonUniformity)

## Data
The clinical ultrasound data is not publicly available due to institutional ethical restrictions.  
The external validation dataset (BUSI) is available at:  
https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset

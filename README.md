<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Status-Research-lightgrey" alt="Status: Research">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/GPU-XGBoost%20Accel.-orange.svg" alt="GPU XGBoost">
</p>

# Breast Ultrasound Radiomics · 乳腺超声影像组学

> **Multimodal Doppler + Grayscale Analysis / 多模态多普勒 + 灰阶分析**

> **Code for:** *Multimodal Ultrasound Radiomics for Early Breast Cancer: Peritumoral Scale Optimization and Quantitative Doppler Feature Analysis in 546 ≤2 cm Breast Lesions*
>
> **配套代码：** *多模态超声影像组学用于早期乳腺癌：546 例 ≤2cm 乳腺病灶的瘤周尺度优化与定量多普勒特征分析*

---

## 📋 Table of Contents · 目录

- [Key Method / 核心方法](#-key-method)
- [Requirements / 环境要求](#-requirements)
- [Installation / 安装](#-installation)
- [Usage / 使用](#-usage)
- [File Overview / 文件概述](#-file-overview)
- [Pipeline Architecture / 流水线架构](#-pipeline-architecture)
- [Citation / 引用](#-citation)

---

## 🧬 Key Method · 核心方法

This repository provides a complete **five-stage nested feature selection pipeline** for multimodal ultrasound radiomics analysis. It integrates grayscale radiomic features with quantitative Doppler features (VFA, Energy, GrayLevelNonUniformity) at optimal peritumoral scales, validated across 546 small breast lesions (≤2 cm).

本仓库提供了一个完整的**五阶段嵌套特征选择流水线**，用于多模态超声影像组学分析。它将灰阶影像组学特征与定量多普勒特征（VFA、能量、灰度非均匀性）在最优瘤周尺度上整合，并在 546 例小乳腺癌病灶（≤2cm）中验证。

---

## 📁 File Overview · 文件概述

| File 文件 | Purpose 用途 |
|-----------|-------------|
| `final_update.py` | Final model training and evaluation / 最终模型训练与评估 |
| `reproduce_paper.py` | Reproduce paper results / 复现论文结果 |
| `run_1mm_pipeline_optimized.py` | 1mm peritumoral pipeline with optimization / 1mm 瘤周流水线（优化版） |
| `validate_busi_standard.py` | Validation on BUSI standard dataset / 在 BUSI 标准数据集上验证 |
| `validate_pipeline.py` | Pipeline validation utilities / 流水线验证工具 |
| `supplementary_analyses.py` | Supplementary experiments / 补充实验分析 |
| `pr_youden_analysis.py` | PR curve and Youden index analysis / PR 曲线与 Youden 指数分析 |
| `modify_draft.py` | Draft modification utilities / 草稿修改工具 |

---

## 🔬 Pipeline Architecture · 流水线架构

### Five-Stage Nested Feature Selection · 五阶段嵌套特征选择

```
Stage 1: Feature Extraction / 特征提取
  → Grayscale radiomics (shape, texture, wavelet)
  → Quantitative Doppler (VFA, Energy, GLNU)

Stage 2: Peritumoral Scale Optimization / 瘤周尺度优化
  → 1mm / 3mm / 5mm / 7mm / 9mm rings
  → Select optimal scale via cross-validation

Stage 3: Feature Reduction / 特征降维
  → Variance threshold → Correlation analysis → mRMR

Stage 4: Model Selection / 模型选择
  → Logistic Regression / SVM / Random Forest / XGBoost
  → 5-fold cross-validation with AUC scoring

Stage 5: Validation / 验证
  → Internal validation → External BUSI validation
```

---

## ⚙️ Requirements · 环境要求

- Python 3.9+
- PyRadiomics
- XGBoost (GPU acceleration recommended / 推荐 GPU 加速)
- scikit-learn, pandas, numpy
- matplotlib, seaborn

---

## 📦 Installation · 安装

\`\`\`bash
git clone https://github.com/QWQZhangErHao/breast-ultrasound-radiomics.git
cd breast-ultrasound-radiomics
pip install -r requirements.txt
\`\`\`

---

## 📄 Citation · 引用

If you use this code in your research, please cite the associated paper.

如果您在研究中使用此代码，请引用相关论文。

---

## 📄 License · 许可

MIT

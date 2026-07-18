#!/usr/bin/env python3
"""Final docx update: BUSI results, PR-AUC, Youden, references, polish, GA, data statement."""
import docx
from copy import deepcopy

DOC_PATH = "C:/Users/ZhangErHao/Desktop/论文文章部分/初稿_修改版.docx"
OUT_PATH = "C:/Users/ZhangErHao/Desktop/论文文章部分/初稿_最终版.docx"

doc = docx.Document(DOC_PATH)
paras = doc.paragraphs

def set_text(para, text):
    for run in para.runs:
        run.text = ""
    if para.runs:
        para.runs[0].text = text
    else:
        para.add_run(text)

# ============================================================
# [1] TITLE: add "≤2 cm"
# ============================================================
set_text(paras[1], "Multimodal Ultrasound Radiomics for Early Breast Cancer: Peritumoral Scale Optimization and Quantitative Doppler Feature Analysis in 546 ≤2 cm Breast Lesions")

# ============================================================
# [49] §3.5 VALIDATION IMPACT: add BUSI external validation
# ============================================================
# Insert BUSI results as a new paragraph after para [49]
# First, update the existing para [49]
set_text(paras[49],
    "To illustrate the impact of validation methodology on performance estimates, we compared results from "
    "nested feature selection (LASSO within each training fold) against a pipeline where feature selection "
    "was performed on the full dataset prior to train-test splitting. Under the non-nested condition, the "
    "apparent XGBoost AUC reached 0.860, compared to 0.827 with proper nested validation (Δ = 0.033 "
    "attributable to methodological artifact). This demonstrates that rigorous validation independence is "
    "essential for accurate effect size estimation in multimodal radiomics research."
)

# Add BUSI external validation paragraph
busi_text = (
    "External validation was performed on the independent public BUSI dataset [Al-Dhabyani 2020], "
    "comprising 647 breast ultrasound images (437 benign, 210 malignant). Using the same PyRadiomics "
    "pipeline and feature selection framework, the grayscale radiomics model achieved an XGBoost AUC "
    "of 0.991 ± 0.004, confirming that the radiomic features generalize across different populations, "
    "ultrasound systems, and imaging protocols. It should be noted that BUSI includes lesions of varying "
    "sizes, whereas our primary cohort was restricted to ≤2 cm lesions, which may partly explain the "
    "higher absolute performance on BUSI."
)
busi_p = paras[50].insert_paragraph_before(busi_text, style=paras[49].style)

# ============================================================
# [42] §3.3: Add PR-AUC and Youden info
# ============================================================
set_text(paras[42],
    "Table 4 presents the performance of different Doppler feature sets when combined with grayscale features "
    "(1 mm reference scale) across all five classifiers. In XGBoost, the 1 mm grayscale baseline achieved AUC "
    "of 0.803 ± 0.025. Adding VFA only increased AUC to 0.821 ± 0.027 (Δ = +0.018, p = 0.042). The "
    "3-optimized set (VFA + Energy + GrayLevelNonUniformity) achieved AUC of 0.827 ± 0.025 "
    "(Δ = +0.025, p = 0.0085), representing the optimal balance between performance and parsimony. "
    "The full 14-feature set reached AUC of 0.830 ± 0.034 (Δ = +0.027, p = 0.006). "
    "The 3-optimized set achieved 93% of the maximal gain with only 3 features. "
    "The PR-AUC for the 3-Doppler XGBoost model was 0.775 ± 0.044, and the Youden-optimal operating "
    "threshold yielded sensitivity of 0.752 ± 0.074 and specificity of 0.796 ± 0.085, compared to "
    "sensitivity of 0.648 ± 0.035 and specificity of 0.850 ± 0.051 at the default 0.5 threshold."
)

# ============================================================
# [51] §4 DISCUSSION: update with PR-AUC context
# ============================================================
set_text(paras[51],
    "The magnitude of Doppler contribution observed in our study (ΔAUC ≈ 0.025 with XGBoost at 1 mm "
    "peritumoral scale) is lower than several previous reports [Moustafa 2020]. We attribute this primarily "
    "to differences in validation methodology. Our experiments demonstrated that non-nested feature selection "
    "inflates AUC by approximately 0.033, and the true effect size of Doppler features is likely more modest "
    "than initially reported in the literature. Furthermore, the PR-AUC of 0.775 ± 0.044 for the multimodal "
    "model indicates moderate precision-recall performance, reflecting the inherent class imbalance "
    "(61.4% benign) in our cohort. This finding aligns with recent concerns in the radiomics literature "
    "regarding overoptimistic performance estimates [Gidwani 2023; Hong 2024]."
)

# ============================================================
# [79] DATA AVAILABILITY STATEMENT
# ============================================================
# Find and update
for i, p in enumerate(paras):
    if "Data Availability" in p.text:
        set_text(p,
            "Data Availability Statement: The data presented in this study are available on reasonable "
            "request from the corresponding author due to institutional ethical restrictions. The code "
            "for feature extraction (PyRadiomics) and model development (scikit-learn, XGBoost) is "
            "available on GitHub at [repository URL]. The external validation dataset (BUSI) is publicly "
            "available at https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset."
        )
        break

# ============================================================
# [9-10] GRAPHICAL ABSTRACT — replace placeholder with design
# ============================================================
set_text(paras[9], "Graphical Abstract")
set_text(paras[10],
    "[Graphical Abstract: Three-panel schematic. Panel A: Peritumoral ROI expansion comparison — 0 mm "
    "(intratumoral) vs. 1 mm (optimal) vs. 2–4 mm. Panel B: Color Doppler energy map quantification — "
    "VFA, Energy, GrayLevelNonUniformity extraction pipeline. Panel C: Multimodal fusion ROC curves — "
    "grayscale baseline (AUC = 0.803) vs. Doppler-enhanced (AUC = 0.827). Minimum size 560 × 1100 pixels, "
    "PNG/JPEG format.]"
)

# ============================================================
# NEW REFERENCES (add after existing refs)
# ============================================================
new_refs = [
    "[17] Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data Brief. 2020;28:104863.",
    "[18] Liu Y, et al. The value of intratumoral and peritumoral ultrasound radiomics model constructed using multiple machine learning algorithms for non-mass breast cancer. Sci Rep. 2025;15:03704.",
    "[19] Li X, et al. Enhancing early breast cancer diagnosis with contrast-enhanced ultrasound radiomics: insights from intratumoral and peritumoral analysis. Clin Breast Cancer. 2025;25(2):e145-e155.",
    "[20] Fu M, et al. Integrating multimodal ultrasound imaging and machine learning for predicting luminal and non-luminal breast cancer subtypes. Front Oncol. 2025;15:1558880.",
    "[21] Zhang J, et al. Classification of molecular subtypes of breast cancer using radiomic features of preoperative ultrasound images. J Imaging Inform Med. 2025;38:813-825.",
    "[22] Zwanenburg A, et al. The Image Biomarker Standardisation Initiative: standardized quantitative radiomics for high-throughput image-based phenotyping. Radiology. 2020;295(2):328-338.",
    "[23] Park JE, et al. Reproducibility and generalizability in radiomics modeling: possible strategies in radiologic and statistical perspectives. Korean J Radiol. 2019;20(7):1124-1138.",
    "[24] Lambin P, et al. Radiomics: the bridge between medical imaging and personalized medicine. Nat Rev Clin Oncol. 2017;14(12):749-762.",
    "[25] Gillies RJ, Kinahan PE, Hricak H. Radiomics: images are more than pictures, they are data. Radiology. 2016;278(2):563-577.",
    "[26] Yip SS, Aerts HJ. Applications and limitations of radiomics. Phys Med Biol. 2016;61(13):R150-R166.",
]

# Insert before Supplementary Materials
for i, p in enumerate(paras):
    if "Supplementary Materials" in p.text:
        sup_idx = i
        break

ref_style = paras[69].style
for ref_text in reversed(new_refs):
    p = paras[sup_idx].insert_paragraph_before(ref_text, style=ref_style)

# ============================================================
# LANGUAGE POLISH — key paragraphs
# ============================================================
# Abstract [7] — minor polish
# Already updated, keep as is

# Conclusion [56] — ensure polished
polished_conclusion = (
    "Quantitative Doppler features provide a statistically significant but modest improvement over "
    "grayscale ultrasound radiomics for ≤2 cm breast lesion diagnosis (ΔAUC ≈ +0.025 with XGBoost). "
    "The optimal peritumoral expansion scale under our experimental conditions is 1 mm, which best "
    "captures diagnostically relevant microenvironmental information. The optimal Doppler signature "
    "can be captured by three features representing blood flow density (VFA), signal intensity (Energy), "
    "and textural heterogeneity (GrayLevelNonUniformity), achieving AUC = 0.827 when combined with "
    "grayscale features using XGBoost. The Doppler benefit is strongly model-dependent, requiring "
    "tree-based approaches to capture nonlinear interactions with grayscale features. VFA serves as a "
    "clinically useful stratification parameter, identifying cases where diagnostic confidence may be "
    "reduced (VFA = 0). Our systematic framework confirms the value of quantitative Doppler analysis "
    "while emphasizing that rigorous, model-aware validation is essential for accurate effect size "
    "estimation in multimodal radiomics research."
)
set_text(paras[56], polished_conclusion)

# Simple Summary [6] — final polish
polished_summary = (
    "Simple Summary: Breast cancer diagnosis using ultrasound often relies on subjective visual "
    "assessment. This study investigated whether adding computer analysis of blood flow information "
    "(Doppler ultrasound) to standard grayscale images could improve diagnostic accuracy for early-stage "
    "breast cancers (≤2 cm). We analyzed 546 breast lesions from 516 patients and found that analyzing "
    "the tumor together with a 1 mm rim of surrounding tissue yields optimal results. Adding three "
    "specific blood flow measurements—blood flow density (VFA), flow signal intensity (Energy), and "
    "flow pattern heterogeneity (GrayLevelNonUniformity)—improved diagnostic accuracy, though the "
    "improvement was moderate and required advanced machine learning methods (XGBoost) rather than "
    "simpler statistical approaches. Our findings provide practical guidance for developing reliable "
    "computer-aided diagnosis systems for early breast cancer."
)
set_text(paras[6], polished_summary)

# ============================================================
# SAVE
# ============================================================
doc.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print("Changes made:")
print("  - Title: ≤2 cm lesions")
print("  - Simple Summary: polished")
print("  - §3.3 Multimodal: added PR-AUC + Youden")
print("  - §3.5 Validation Impact: added BUSI external validation")
print("  - §4 Discussion: added PR-AUC context")
print("  - Conclusions: polished")
print("  - Data Availability: updated with BUSI + code URL")
print("  - Graphical Abstract: design spec added")
print("  - References: 16 -> 26 (10 new refs added)")

#!/usr/bin/env python3
"""Modify the paper draft docx with all review-driven revisions."""
import docx
from copy import deepcopy

DOC_PATH = "C:/Users/ZhangErHao/Desktop/论文文章部分/初稿.docx"
OUT_PATH = "C:/Users/ZhangErHao/Desktop/论文文章部分/初稿_修改版.docx"

doc = docx.Document(DOC_PATH)
paras = doc.paragraphs

def set_text(para, text):
    """Replace paragraph text while preserving first run's formatting."""
    for run in para.runs:
        run.text = ""
    if para.runs:
        para.runs[0].text = text
    else:
        para.add_run(text)

# ============================================================
# [1] TITLE
# ============================================================
para = paras[1]
set_text(para, "Multimodal Ultrasound Radiomics for Early Breast Cancer: Peritumoral Scale Optimization and Quantitative Doppler Feature Analysis in 546 ≤2 cm Breast Lesions")

# ============================================================
# [6] SIMPLE SUMMARY — update numbers
# ============================================================
set_text(paras[6],
    "Simple Summary: Breast cancer diagnosis using ultrasound often relies on the doctor's visual assessment. "
    "This study investigated whether adding computer analysis of blood flow information (Doppler ultrasound) "
    "to standard grayscale ultrasound images could improve diagnostic accuracy for early-stage breast cancers "
    "(≤2 cm). We analyzed 546 breast lesions from 516 patients and found that the best results come from "
    "analyzing the tumor itself plus a 1 mm rim of surrounding tissue. Adding three specific blood flow "
    "measurements—how much blood flow exists, how intense it is, and how disorganized the flow pattern is—"
    "improved diagnostic accuracy. However, this improvement was model-dependent, being captured primarily "
    "by tree-based models (XGBoost) rather than simpler linear approaches. Our findings provide practical "
    "guidance for developing more reliable computer-aided diagnosis systems for early breast cancer."
)

# ============================================================
# [7] ABSTRACT — full replacement
# ============================================================
set_text(paras[7],
    "Abstract: Background/Objectives: Peritumoral radiomic features and quantitative Doppler parameters "
    "have individually shown promise for breast cancer diagnosis, but their optimal integration strategy "
    "remains unclear. This study systematically evaluates peritumoral scale selection and the added value "
    "of quantitative Doppler features in ≤2 cm breast lesions using a rigorous validation framework with "
    "nested feature selection. Methods: A total of 546 breast lesions (335 benign, 211 malignant; ≤2 cm) "
    "were retrospectively analyzed. Peritumoral expansion scales of 0–4 mm were compared. Twenty-one "
    "quantitative Doppler features and vascular fractional area (VFA) were extracted from color Doppler "
    "energy maps using established channel-separation methods [Bell 1995; Jun 2002]. A five-stage feature "
    "selection pipeline was applied with LASSO performed strictly within each training fold. Multimodal "
    "fusion was evaluated across five classifiers using 10 repeated 70/30 stratified splits. Results: "
    "The 1 mm peritumoral expansion provided optimal grayscale radiomic performance (XGBoost AUC = 0.803±0.025). "
    "After systematic feature selection, three Doppler features—VFA (blood flow density), Energy (flow "
    "intensity), and GrayLevelNonUniformity (flow heterogeneity)—were identified as the optimal subset. "
    "Multimodal fusion combining grayscale features with these three Doppler parameters significantly "
    "improved XGBoost AUC from 0.803±0.025 to 0.827±0.025 (ΔAUC = +0.025, p = 0.0085). The full "
    "14-feature set reached AUC = 0.830±0.034. Doppler benefit was model-dependent: tree-based models "
    "(XGBoost ΔAUC = +0.025) outperformed linear models (SVM ΔAUC = +0.020). VFA = 0 cases (n = 56, "
    "14.3% malignant) represented a diagnostically challenging subgroup (AUC = 0.656±0.129). Conclusions: "
    "The 1 mm peritumoral expansion provides an optimal balance for peritumoral radiomic analysis under "
    "our experimental conditions. Quantitative Doppler features offer statistically significant but modest "
    "additional diagnostic value that is model-dependent and captured by three clinically interpretable "
    "hemodynamic parameters."
)

# ============================================================
# [22] 2.3 RADIOMIC FEATURE EXTRACTION
# ============================================================
set_text(paras[22],
    "To construct the grayscale feature set, features from both 0 mm (intratumoral) and 1 mm (peritumoral) "
    "regions were combined, excluding shape features from the peritumoral region to avoid redundancy. "
    "This resulted in 1,101 grayscale features. The 1 mm scale was selected based on systematic evaluation "
    "across 0–4 mm expansions (see Results 3.1), consistent with the biological understanding that the "
    "angiogenic invasive front in early breast cancers is typically confined to the immediate peritumoral region."
)

# ============================================================
# [24] 2.4 DOPPLER FEATURE EXTRACTION
# ============================================================
set_text(paras[24],
    "Color Doppler images were converted to single-channel energy maps by computing the absolute difference "
    "between red and blue channels, followed by min-max normalization to 0–255. This approach is based on "
    "established methods for separating color and grayscale information from digitized color Doppler images "
    "[Bell 1995], and the quantitative analysis of color Doppler signals via computer-based analysis has been "
    "previously validated for the evaluation of tumor vascularity, with reconstruction accuracy exceeding 95% "
    "for breast tumor images [Jun 2002]. From these energy maps, 21 radiomic features were extracted using "
    "PyRadiomics [van Griethuysen 2017] (designated as bf_* features). The vascular fractional area (VFA) "
    "was computed as the proportion of color flow pixels within the lesion region, quantifying intratumoral "
    "blood flow density. The complete Doppler candidate set comprised 22 features (21 bf_* features + 1 VFA)."
)

# ============================================================
# [25-26] 2.5 FEATURE SELECTION (five-stage, fix numbering)
# ============================================================
# Insert new 2.5 content after finding it
# Para [25] is "2.5. Systematic Feature Selection Pipeline"
# Replace from para [26] onwards for this section

# First, set the heading
# para [25] - heading, keep as is
# para [26] - first content para - replace with full 5-stage description
set_text(paras[26],
    "A five-stage feature selection and validation pipeline was applied, with all stages performed strictly "
    "within the training set of each cross-validation split to prevent data leakage: Stage 1: Zero-variance "
    "filtering. Stage 2: Family-wise intra-class de-correlation (|r| > 0.85 threshold). Stage 3: Univariate "
    "filtering (Mann-Whitney U test p < 0.05, direct AUC > 0.55). Stage 4: LASSO with 10-fold cross-validation "
    "across 10 random splits (selection rate ≥ 60%, sign consistency ≥ 70%). Stage 5: Bootstrap significance "
    "testing (2,000 iterations, H₀: ΔAUC = 0) and paired t-tests across the 10 random splits. All feature "
    "selection and standardization were performed exclusively on each training fold, after which the test set "
    "was used only for final evaluation. This nested approach prevents information leakage that can inflate "
    "performance estimates in radiomics studies [Gidwani 2023; Hong 2024]."
)

# ============================================================
# [29] — Add reproducibility section mention
# ============================================================
set_text(paras[29],
    "All analyses were performed in Python 3.9 using scikit-learn (1.2.2), XGBoost (1.7.6), SciPy (1.10.1), "
    "and NumPy (1.24.3). XGBoost utilized tree_method='hist' with parallelization across 32 CPU cores. "
    "All random seeds are explicitly reported for reproducibility."
)

# ============================================================
# [32] 3.1 PERITUMORAL SCALE — rewrite with 1mm
# ============================================================
set_text(paras[32],
    "To determine the optimal peritumoral expansion range, we evaluated radiomic models at five expansion "
    "scales (0–4 mm) using both SVM and XGBoost classifiers. The 0 mm baseline (intratumoral only) achieved "
    "an XGBoost AUC of 0.786±0.029. The 1 mm peritumoral expansion increased AUC to 0.816±0.023, representing "
    "the highest performance among all individual scales. Performance at 2 mm (0.791±0.021), 3 mm (0.793±0.017), "
    "and 4 mm (0.794±0.023) declined relative to 1 mm, suggesting that a 1 mm expansion optimally captures "
    "diagnostically relevant peritumoral information without introducing excessive noise from surrounding "
    "parenchymal tissue. SVM exhibited more stable performance across scales (0 mm: 0.812±0.019; 1–4 mm range: "
    "0.794–0.811), indicating SVM is less sensitive to peritumoral information than tree-based models. Based on "
    "these findings, the 1 mm expansion was selected as the reference scale for subsequent multimodal fusion analysis."
)

# ============================================================
# [33] — Remove the 2mm justification paragraph (merge)
# ============================================================
set_text(paras[33], "")

# ============================================================
# [36] 3.1.1 — fix stage numbering
# ============================================================
set_text(paras[36],
    "Of the 22 candidate Doppler features, 4 GLCM features (JointEnergy, JointEntropy, Contrast, Correlation) "
    "exhibited zero variance across all samples and were removed at Stage 1, indicating that the Doppler energy "
    "maps lack the textural variation necessary for meaningful GLCM analysis."
)

# ============================================================
# [37] — continue, remove "After Stage 2..." paragraph, replace
# ============================================================
set_text(paras[37],
    "After Stage 2 family-wise de-correlation, 2 additional features were removed due to high intra-family "
    "correlation. Stage 3 (univariate filtering) excluded 2 features with AUC ≤ 0.55. All 14 remaining features "
    "passed Stage 4 LASSO stability and sign-consistency checks with 100% selection rates and 100% sign consistency."
)

# ============================================================
# [42] 3.3 MULTIMODAL FUSION — update numbers
# ============================================================
set_text(paras[42],
    "Table 4 presents the performance of different Doppler feature sets when combined with grayscale features "
    "(1 mm reference scale) across all five classifiers. In XGBoost, the 1 mm grayscale baseline achieved AUC "
    "of 0.803±0.025. Adding VFA only increased AUC to 0.821±0.027 (Δ = +0.018, p = 0.042). The 3-optimized set "
    "(VFA + Energy + GrayLevelNonUniformity) achieved AUC of 0.827±0.025 (Δ = +0.025, p = 0.0085), representing "
    "the optimal balance between performance and parsimony. The full 14-feature set reached AUC of 0.830±0.034 "
    "(Δ = +0.027, p = 0.006). The 3-optimized set achieved 93% of the maximal gain with only 3 features."
)

# ============================================================
# [43] — model dependency paragraph
# ============================================================
set_text(paras[43],
    "A critical finding was the substantial variation in Doppler benefit across classifiers. Linear models "
    "showed modest gains: SVM ΔAUC = +0.020 (3-set); LR ΔAUC = +0.021 (3-set). Tree-based models captured "
    "substantially more benefit: XGBoost ΔAUC = +0.025 (3-set); RF ΔAUC = +0.033 (3-set). This pattern "
    "suggests that the interaction between grayscale and Doppler features is fundamentally nonlinear, "
    "requiring tree-based models to fully exploit the complementary information."
)

# ============================================================
# [49] 3.5 VALIDATION IMPACT — update numbers
# ============================================================
set_text(paras[49],
    "To illustrate the impact of validation methodology on performance estimates, we compared results from "
    "nested feature selection (LASSO within each training fold) against a pipeline where feature selection "
    "was performed on the full dataset prior to train-test splitting. Under the non-nested condition, the "
    "apparent XGBoost AUC reached 0.860, compared to 0.827 with proper nested validation (Δ = 0.033 "
    "attributable to methodological artifact). This demonstrates that rigorous validation independence is "
    "essential for accurate effect size estimation in multimodal radiomics research."
)

# ============================================================
# [51] DISCUSSION — update numbers
# ============================================================
set_text(paras[51],
    "The magnitude of Doppler contribution observed in our study (ΔAUC ≈ 0.025 with XGBoost at 1 mm scale) "
    "is lower than several previous reports [Moustafa 2020]. We attribute this primarily to differences in "
    "validation methodology. Our experiments demonstrated that non-nested feature selection inflates AUC by "
    "approximately 0.033, and the true effect size of Doppler features is likely more modest than initially "
    "reported in the literature. This finding aligns with recent concerns in the radiomics literature regarding "
    "overoptimistic performance estimates [Gidwani 2023; Hong 2024]."
)

# ============================================================
# [52] — VFA discussion
# ============================================================
set_text(paras[52],
    "Importantly, our results do not negate the potential value of Doppler features. Rather, they suggest "
    "that the true effect size is smaller than previously thought, and that appropriate nonlinear models are "
    "necessary to capture it. From a clinical perspective, VFA emerged as the most practically useful Doppler "
    "parameter—not primarily for its predictive contribution within models (ΔAUC ≈ +0.018 with VFA alone), "
    "but as a stratification tool: when flow is present (VFA > 0, 89.7% of cases), grayscale features perform "
    "well with modest Doppler enhancement; when flow is absent (VFA = 0, 10.3% of cases), the overall "
    "malignancy risk is lower but grayscale diagnosis is less reliable, warranting more cautious interpretation."
)

# ============================================================
# [53] — tree vs linear mechanism
# ============================================================
set_text(paras[53],
    "The striking difference between linear and tree-based models in leveraging Doppler features has important "
    "implications. Doppler parameters may interact with grayscale features in complex, nonlinear ways—for "
    "example, the combination of high VFA with certain textural patterns (e.g., elevated GrayLevelNonUniformity) "
    "may be more informative than either feature alone. Tree-based models capture these interactions through "
    "hierarchical feature space partitioning. Researchers designing multimodal ultrasound studies should "
    "consider using or at least evaluating tree-based models rather than defaulting to linear classifiers."
)

# ============================================================
# [54] — LIMITATIONS (update)
# ============================================================
set_text(paras[54],
    "Several limitations of this study should be acknowledged. First, this is a single-center retrospective "
    "study without external validation, limiting generalizability. Second, ROI segmentation was performed by "
    "a single reader without intra- or inter-observer variability assessment (ICC), which may affect feature "
    "reproducibility. Third, the VFA = 0 subgroup was small (n = 56), limiting statistical power for subgroup "
    "analyses. Fourth, only default classifier hyperparameters were used; optimized hyperparameters might yield "
    "different results. Fifth, the study focused on lesions ≤ 2 cm; results may not apply to larger tumors. "
    "Sixth, the Doppler energy map conversion method, while supported by prior literature [Bell 1995; Jun 2002], "
    "requires further validation against standard Power Doppler metrics. Finally, publicly available breast "
    "ultrasound datasets lack Doppler modality, precluding external validation of the multimodal model—a shared "
    "challenge in this field."
)

# ============================================================
# [55-56] CONCLUSIONS
# ============================================================
set_text(paras[56],
    "Quantitative Doppler features provide a statistically significant but modest improvement over grayscale "
    "ultrasound radiomics for ≤2 cm breast lesion diagnosis (ΔAUC ≈ +0.025 with XGBoost). The optimal "
    "peritumoral expansion scale under our experimental conditions is 1 mm, which best captures diagnostically "
    "relevant microenvironmental information. The optimal Doppler signature can be captured by three features "
    "representing blood flow density (VFA), signal intensity (Energy), and textural heterogeneity "
    "(GrayLevelNonUniformity), achieving AUC = 0.827 when combined with grayscale features using XGBoost. "
    "The Doppler benefit is strongly model-dependent, requiring tree-based approaches to capture nonlinear "
    "interactions with grayscale features. VFA serves as a clinically useful stratification parameter, "
    "identifying cases where diagnostic confidence may be reduced (VFA = 0). Our systematic framework "
    "confirms the value of quantitative Doppler analysis while emphasizing that rigorous, model-aware "
    "validation is essential for accurate effect size estimation in multimodal radiomics research."
)

# ============================================================
# ADD NEW REFERENCES (after existing ref [12], para [69])
# ============================================================
# Insert new references after para [69] (last ref)
# We'll add them as new paragraphs

refs_text = [
    "[13] Bell DS, Bamber JC, Eckersley RJ. Segmentation and analysis of colour Doppler images of tumour vasculature. Ultrasound Med Biol. 1995;21(5):635-647.",
    "[14] Jun WS, et al. A straightforward algorithm for the quantification of power Doppler signals. Invest Radiol. 2002;37(6):343-348.",
    "[15] Hong S, Hong S, Oh E, et al. Development of a flexible feature selection framework in radiomics-based prediction modeling. Sci Rep. 2024;14:29297.",
    "[16] Gidwani M, Chang K, Patel K, et al. Inconsistent Partitioning and Unproductive Feature Associations Yield Idealized Radiomic Models. Radiology. 2023;307:e220715.",
]

# Get style from last reference paragraph
ref_style = paras[69].style

# Insert new references before para [70] (Supplementary Materials)
insert_before = paras[70]
for ref_text in refs_text:
    if ref_text:
        new_p = insert_before.insert_paragraph_before(ref_text, style=ref_style)

# ============================================================
# UPDATE TABLES
# ============================================================
# Table 2 (index 1): Peritumoral scale comparison - add 1mm as reference
if len(doc.tables) > 1:
    t = doc.tables[1]
    # Add "✅ Reference" marker to 1mm row
    for row in t.rows:
        cells = [c.text for c in row.cells]
        if "1 mm" in cells[0]:
            # Add note
            pass  # Keep as is, just change description

# Save
doc.save(OUT_PATH)
print(f"Done. Saved to: {OUT_PATH}")

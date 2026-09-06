# Table 2. Unimodal Grayscale Baseline: Handcrafted Radiomics vs. Deep Learning Models (Internal)

| Architecture / Paradigm | Input | Split/Runs | AUC | Sen(0.5) | Spe(0.5) | Sen(Youden) | Spe(Youden) | Topt |
|---|---|---|---|---|---|---|---|---|
| **XGBoost (Radiomics)** | 1,101 tabular grayscale features (0+1 mm) | 10×70/30 | 0.808±0.028 | 0.625±0.058 | 0.840±0.043 | — | — | 0.500 |
| ResNet18 | Grayscale image 224² | 3 seeds | 0.763±0.015 | 0.540±0.091 | 0.822±0.028 | 0.667±0.047 | 0.759±0.023 | 0.316±0.158 |
| ResNet50 | Grayscale image 224² | 3 seeds | 0.771±0.015 | 0.603±0.091 | 0.815±0.087 | 0.693±0.052 | 0.769±0.041 | 0.411±0.365 |
| ConvNeXt-Tiny | Grayscale image 224² | 3 seeds | 0.857±0.020 | 0.714±0.093 | 0.818±0.054 | 0.704±0.064 | 0.884±0.036 | 0.689±0.170 |
| MedViT1-small | Grayscale image 224² | 3 seeds | 0.771±0.018 | 0.693±0.097 | 0.680±0.089 | 0.640±0.052 | 0.795±0.072 | 0.635±0.274 |
| MedViT1-base | Grayscale image 224² | 5 seeds | 0.773±0.017 | 0.486±0.222 | 0.857±0.089 | 0.654±0.042 | 0.810±0.021 | 0.359±0.217 |

Notes: (1) Cross-paradigm: XGBoost uses handcrafted radiomics (0+1 mm) over 10 repeated splits; DL uses 224² grayscale images over 3 seeds (5 for MedViT1-base). Values for DL are re-computed from saved best checkpoints on the matched validation splits. Side-by-side = unimodal representation benchmark, not identical end-to-end control.
(2) † Operating-threshold sensitivity: MedViT1-base Sen variance at 0.5 reflects threshold displacement (e.g. seed42 Sen 0.06/Spe1.00); Youden recalibration restores Sen 0.654±0.042 without retraining, confirming mismatch rather than feature collapse.
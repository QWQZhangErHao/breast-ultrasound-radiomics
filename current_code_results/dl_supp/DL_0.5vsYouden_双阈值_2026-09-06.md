# DL 双阈值评估:默认 0.5 vs Youden 校准(2026-09-06)

- 评估对象:已保存 best checkpoint,在同一 70/30 验证划分(同 seed)上重算;AUC 与阈值无关。
- 结论:**异常 Sen/Spe 源于默认 0.5 阈值失配,非特征未收敛**;用 Youden 校准即可恢复平衡。

## 崩塌机制证据(MedViT1-base,internal,seed42)
- 0.5 阈值:Sen 0.06 / Spe 1.00 → ⚠0.5-COLLAPSE;
- Youden 最优阈值 T=0.11:Sen 0.63 / Spe 0.81 → **Youden rescued**;
- 5 种子 Sen 标准差 0.5 口径 ±0.222 → Youden 口径 **±0.042**。

## 内部集(internal)mean±SD:AUC / 0.5-Sen / Youden-Sen
| 模型 | AUC | 0.5 Sen | 0.5 Spe | Youden Sen | Youden Spe | 平均Topt |
|---|---|---|---|---|---|---|
| ResNet18 | 0.763±0.015 | 0.540±0.091 | 0.822±0.028 | 0.667±0.047 | 0.759±0.023 | 0.32±0.16 |
| ResNet50 | 0.771±0.015 | 0.603±0.091 | 0.815±0.087 | 0.693±0.052 | 0.769±0.041 | 0.41±0.37 |
| ConvNeXt-Tiny | 0.857±0.020 | 0.714±0.093 | 0.818±0.054 | 0.704±0.064 | 0.884±0.036 | 0.69±0.17 |
| MedViT1-small | 0.771±0.018 | 0.693±0.097 | 0.680±0.089 | 0.640±0.052 | 0.795±0.072 | 0.64±0.27 |
| MedViT1-base | 0.773±0.017 | 0.486±0.222 | 0.857±0.089 | 0.654±0.042 | 0.810±0.021 | 0.36±0.22 |

## BUSI 主要模型(Youden)
ResNet18 Sen 0.878、ResNet50 0.926、ConvNeXt-Tiny 0.921、MedViT1-small 0.799、MedViT1-base 0.914(详值见 `eval_threshold_results.json`)。

## 结论与论文写作策略
- **论文写两列**(0.5 与 Youden),并加"分类阈值与临床决策边界校准"讨论;大方差成因 = 医学小样本不平衡导致的概率分布挤压 + 0.5 工作点失配,可用 Youden 校准缓解。
- **ConvNeXt-Tiny 依旧最优**:Youden 后 Spe 升至 0.884、AUC 0.857,支持其作为推荐图像基准。
- 数据源:`论文\复现\dl\eval_threshold_results.json`、`eval_thr.log`。

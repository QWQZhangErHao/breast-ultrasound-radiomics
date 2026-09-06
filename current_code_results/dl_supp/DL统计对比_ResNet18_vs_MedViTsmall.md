# DL 对比补充统计(ResNet18 vs MedViT_small,2026-09-05)

配置:3 种子(42/123/2024)各 70/30 同划分;单次(seed42)超参、epochs 12(internal medvit run 以 seed 对应 epochs12)、AMP、224px。原始逐 seed 见 `论文\复现\dl\stats_summary.json`。

## AUC mean±SD
| 模型 | internal | BUSI |
|---|---|---|
| ResNet18(ImageNet 预训练) | 0.756±0.015 | 0.933±0.003 |
| MedViT_small(从零训练) | 0.705±0.008 | 0.877±0.006 |

## 同种子配对(MedViT − ResNet)
- internal:Δ −0.0515,paired-t p=0.021,Wilcoxon p=0.25
- BUSI:Δ −0.0564,paired-t p=0.008,Wilcoxon p=0.25

## 表述与注意(写稿必须带)
1. ResNet18 在两个数据集 paired-t 上更高(n=3 小样本;Wilcoxon 不显著→用“倾向/需更大样本”,勿写成强结论)。
2. **公平性限制**:ResNet18 为 ImageNet 预训练,MedViT 从零 → 差异主要源自预训练而非架构;公平对比须 MedViT 用其预训练权重(仓库预训练接线待补)或两模型都从零。
3. 权重已保存:`论文\复现\dl\saved\`(`*_resnet18_seed*.pt`、`*_medvit_small_seed*.pt`)。
4. 扩展(MedViT base/large、MedViTV2、MedMamba)如需,按此 3-seed 口径继续,预计数小时并可复用本统计流程。

## 更新:MedViT 补齐 ImageNet 预训练后的公平对比(3 种子,2026-09-05)
- MedViT v1 预训练权重(w1/w2/w3=small/base/large,ImageNet-1K 224)已接入(dl/pretrained),头部忽略/随机初始化后微调。
- AUC mean±SD(3 种子):
  - internal:ResNet18 0.756±0.015 | **MedViT_small(PT) 0.772±0.019**(Δ+0.016,p_t=0.37,p_w=0.50,不显著)
  - BUSI:ResNet18 0.933±0.003 | MedViT_small(PT) 0.913±0.004(Δ−0.020,p_t=0.019,p_w=0.25)
- 结论修正:补齐预训练后 MedViT 与 ResNet18 **统计相当**(internal 略高不显著、BUSI 略低但 paired-t 弱显著、Wilcoxon 均不显著,n=3)→ 原"MedViT 显著更差"主要来自"从零 vs 预训练"偏差,已消除。
- 权重/结果:dl/saved/、dl/stats_summary_pretrained.json;Excel MedViT1-small 已更新为 3-seed 预训练均值。

## 控制变量协议结果(2026-09-05,最终口径)
- 协议统一为 epochs12/lr2e-4/bs64/AMP/AdamW(wd1e-4)/同种子同增强,两者都用各自 ImageNet 预训练,仅模型骨架为变量(stats_control.json)。
- AUC mean±SD(3 种子):
  - internal:ResNet18 0.766±0.017 | MedViT_small(PT) 0.772±0.019 | Δ+0.007(p_t=0.31,p_w=0.50,ns)
  - BUSI:ResNet18 0.936±0.010 | MedViT_small(PT) 0.913±0.004 | Δ−0.022(p_t=0.093,p_w=0.25,ns)
- **控制变量结论:固定其余一切时,ResNet18 与 MedViT_small 差异不显著(两个数据集均 p>0.05)**,说明判别力主要来自预训练+数据+训练协议,单看架构差异不显著。

## 95% CI(3 种子,df=2,t=4.303)
- ResNet18 internal AUC 0.766 [0.715,0.816];busi 0.936 [0.905,0.966]
- MedViT_small(PT) internal 0.772 [0.716,0.828];busi 0.913 [0.901,0.925]
- 模型间 Δ(MedViT−ResNet):internal +0.006 [−0.014,+0.027](含 0,ns);BUSI −0.022 [−0.054,+0.009](跨 0 边界,ns)
- ⚠ n=3 的 t-CI 很宽,只能说明两模型无显著差,不能定出小效应。

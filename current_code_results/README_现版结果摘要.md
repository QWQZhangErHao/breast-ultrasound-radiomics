# 现版代码口径结果摘要(2026-09-04,决定以此为准)

> 配置:现发布代码 run_1mm 系(LASSO tol=1e-4、max_iter=1e4、LR=saga;XGB hist depth3 lr.1 n100;RF500/depth5;SVM lin C1;KNN5;种子 42..4096;10×70/30)。
> 一键复现:`Desktop\论文\复现\canonical_author_run.py` → `canonical_author_run.pkl`;后处理 `publish_canonical.py`、嵌套 `nested_author.py`、出图 `plot_canonical_figs.py`、模型 `save_best_model.py`。

## 关键数字(现版代码)
- 尺度 XGB:0mm 0.787 → **1mm 0.808(峰)** → 2mm 0.792 → 3mm 0.794 → 4mm 0.787;1mm vs 0mm Δ+0.020(p=0.046)。
- 融合 XGB:base 0.808 → +VFA 0.811 → **+3 0.827**(base→+3 Δ+0.0196,p=0.016;+VFA→+3 p=0.034)→ +13 0.827 → +21 0.831;+3→+13/+21 不显著(p=0.91/0.23)。
- 配对(全模型)见 `canonical_pair_stats.csv`(含 nominal p<.05 与 α=.005 存活标记)。
- NRI 0.279±0.143(p=0.0002,non-event .222/event .057,良性主导);Brier G .170→M .163(p=0.107 不显著);Eavg G .096 vs M .119(多模态校准更差);DCA 区间 .03–.99。
- VFA=0 亚组 AUC 0.833±0.174(仅 8 例恶性,不可靠);VFA>0 0.818±0.029。
- 嵌套 0.834 vs 非嵌套 0.844 → 膨胀 +0.010(`nested_author.txt`)。
- 单特征(独立于口径):GLNU 族 0.739 最强、Energy 0.651、VFA 0.615;4 GLCM 零方差。

## 与 1.0 论文数值的系统差异(重要)
论文 Table 的 .803/.821/.827/.830、NRI .4388、Brier 显著改善、Eavg 多模态更佳、VFA0 0.656 等,**在当前官方代码下无法复现**;现版得到的是上一段数值。→ **已决定:以"现版代码输出"为成稿口径**,论文数值需同步替换(附 `初稿 - 副本` 待用此口径更新)。

## 已保存最优模型(后续测试用)
`Desktop\论文\复现\saved_models\`:`xgb_plus3.pkl`(主,45 特征)、`xgb_plus21.pkl`(54 特征)+ `README使用说明.md`。
⚠ 全量拟合的 in-sample AUC(.996/.998)不能当结果引用;评估以嵌套/10 次 70/30 为准,独立测试另算。

## 待办(给作者)
- 用现版数字替换 1.0 / 初稿-副本 中的正文与表(尺度、融合、NRI/Brier/Eavg、VFA0、嵌套、Δ/p)。
- 论文自洽需重查的旧表述:93% 增益、Bonferroni 后 p<0.001 与 p=.0085 矛盾、特征数 1101 vs 1103、尺度表 5 vs 10 seeds。
- G3/G4/G5 组表(副本 Table2)与外部 BUSI 的“现版口径”是否也要重跑/替换——待你定。

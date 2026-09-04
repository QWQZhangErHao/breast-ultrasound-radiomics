# 已保存最优模型(现版代码口径,2026-09-04)

用于后续独立测试/外部数据的**最终推理模型**(在全部 546 例上重拟合;评估请用独立测试集,勿用文件内 AUC)。

## 文件
| 文件 | 模型 | 全量拟合信息 |
|---|---|---|
| `xgb_plus3.pkl` | XGBoost + 3 优化 Doppler(VFA,Energy,GLNU) | 主模型(简约、可解释) |
| `xgb_plus21.pkl` | XGBoost + 全部 21 bf_* Doppler | 现版数据点估计最高的候选(+21) |

每个 pkl 含:`cols`(输入特征列名,顺序),`scaler`(StandardScaler,已在全量上拟合),`mask`(LASSO tol1e-4 选择的特征布尔掩码),`clf`(已训练 XGB,hist, depth3 lr0.1 n100, seed42)。

## 推理示例
```python
import pickle, numpy as np
from sklearn.metrics import roc_auc_score

with open("xgb_plus3.pkl","rb") as f: m=pickle.load(f)
# X_new: pandas DataFrame,列名须包含/对齐 m['cols'](多余列会自动忽略,缺失则报错)
X = X_new[m['cols']].to_numpy()
proba = m['clf'].predict_proba(m['scaler'].transform(X)[:, m['mask']])[:,1]
pred  = (proba >= 0.5).astype(int)
```
- 输入列构造与训练一致:1 mm 灰度(0mm+1mm 特征,去掉 1mm 的 shape 特征)+ 相应 Doppler 列(`flow_density` 为 VFA;`bf_firstorder_Energy`、`bf_glrlm_GrayLevelNonUniformity` 为其余 2 个)。
- `m['cols']` 可直接用于从你的特征表取子集。

## 重要提醒(严谨性)
- 上面打印的"in-sample AUC ≈0.996/0.998"是**在训练同一样本上**的高拟合值,**不代表泛化能力**,不能引用为结果。
- 论文报告的判别力以 10 次 70/30 / 嵌套验证为准(现版:XGB+3 ≈0.827;XGB+21 ≈0.831)。
- 后续测试:在新/外部集上先按 `cols` 提取特征、再调用上方代码得到 proba/类别,再算 AUC/指标。

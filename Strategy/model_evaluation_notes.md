# 模型评估与交叉验证方法论记录

> 记录时间：2026-04-27
> 记录背景：针对 `IS_Train` 阶段使用 K-Fold 交叉验证（未来数据预测过去）的严谨性讨论，为了后续框架升级和回溯提供理论依据。

## 1. 现状与定位

当前框架在 `IS_Train` 阶段采用了类似于传统机器学习中的 **Out-of-Fold (OOF) 预测**（即用包含未来时间段的 Train 集去预测过去的 Val 验证集）。
- **优点**：完美阻断了**微观直接数据泄露**（预测 $T$ 日时，模型绝对没有见过 $T$ 日的 Label），并且能非常高效地利用有限的 `IS_Train` 数据进行模型早停（Early Stopping）和超参调优。
- **当前决策**：保留此做法用于 `IS_Train` 内部的训练验证。策略的核心业绩验证（回测）严格依赖于后置的 `IS_Test` 和 `OOS` 阶段。由于 `IS_Test` 和 `OOS` 严格遵循了时间推进（Time's Arrow），因此最终的业绩评估是无偏的。
- **⚠️ 绝对红线**：当前代码生成的 `IS_Train` 打分（`score_is_train`）**严禁**用于绘制模拟真实交易的回测净值曲线。

## 2. 核心问题探讨：宏观前视偏差 (Macro Look-Ahead Bias)

在量化金融中，用未来的数据（如 2022-2023）训练模型去预测过去（如 2021），即使排除了具体的某一天，依然会引入**宏观前视偏差**：
1. **结构性提前认知**：模型“提前”学习到了未来市场的风格演变、因子失效或新生规律（Structural Breaks）。这相当于赋予了历史模拟交易员“上帝视角”。
2. **序列自相关泄露**：金融数据存在时间维度的自相关性。在 Train 和 Val 切分边界处（例如昨天是 Val，今天是 Train），近期的市场情绪是高度相关的。模型容易通过紧贴 Val 之后的 Train 样本，反向推导并“拟合”出 Val 的表现。

## 3. 顶级期刊与工业界的演进思路 (Future Adjustments)

如果未来希望在 `IS_Train` 期间也能获得一条完全无偏的、可用于汇报的回测曲线，或者进一步提升模型在早停时的抗过拟合能力，可以参考以下两套国际量化顶刊标准进行框架演进：

### 思路 A：Walk-Forward Validation (WFV / 滚动推进验证)
- **学术依据**：Gu, Kelly, and Xiu (2020) *"Empirical Asset Pricing via Machine Learning"*, **Journal of Finance**.
- **核心思想**：时间箭头不可逆。训练集永远在测试集的严格左侧。
- **做法**：初始使用 $T_0$ 到 $T_1$ 训练，预测 $T_1$ 到 $T_2$；然后使用 $T_0$ 到 $T_2$ 训练，预测 $T_2$ 到 $T_3$。这是金融界评判策略业绩的**唯一黄金标准**。

### 思路 B：Purged K-Fold CV (带清洗的 K 折交叉验证)
- **学术依据**：Marcos Lopez de Prado (2018) *"Advances in Financial Machine Learning"*.
- **核心思想**：为了充分利用数据调参，允许用未来数据预测过去，但必须强行阻断 Train 和 Val 边界的序列自相关性泄露。
- **潜在代码调整方案**：未来可在 `rolling_trainer.py` 中引入 `purge_days`（清洗期），在生成 `train_mask` 时抠除掉紧挨着 `val_mask` 前后的若干个交易日。

```python
# 示例：未来可能的改进代码 (Purge 机制)
import pandas as pd
from datetime import timedelta

purge_days = 5  # 隔离期，比如 5 个交易日

val_mask = (dates >= fold.val_start) & (dates <= fold.val_end)

# Train mask 需要在 Val 前后都抠掉 5 天的安全垫，避免短期情绪自相关泄露
train_mask = (dates < (fold.val_start - timedelta(days=purge_days))) | \
             (dates > (fold.val_end + timedelta(days=purge_days)))
```

## 4. 总结
当前框架“利用 OOF 高效早停，依赖 IS_Test 严谨验证”的策略在工程上是非常务实的。此文档仅作为未来追求极致学术严谨度或排查模型极度过拟合时的**方法论武器库**。
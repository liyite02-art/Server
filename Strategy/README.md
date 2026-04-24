# 量化策略研发框架 (Strategy)

基于日频 + 分钟频数据的端到端因子挖掘与回测框架。

---

## 目录结构

```
/root/autodl-tmp/
├── min_data/                        # 原始分钟频数据 (按年/日存储的 .fea 文件)
│   └── {year}/{YYYYMMDD}.fea        #   列: StockID, time, open, high, low, price, vol, amount
├── Daily_data/                      # 原始日频数据 (每字段一个 .pkl 宽表)
│   ├── CLOSE_PRICE.pkl
│   ├── OPEN_PRICE.pkl
│   ├── HIGHEST_PRICE.pkl
│   ├── LOWEST_PRICE.pkl
│   ├── PRE_CLOSE_PRICE.pkl
│   ├── CHG_PCT.pkl                  # 涨跌幅 (百分比形式, 如 5.0 表示涨 5%)
│   ├── DEAL_AMOUNT.pkl
│   ├── TURNOVER_RATE.pkl
│   ├── TURNOVER_VALUE.pkl
│   ├── MARKET_VALUE.pkl
│   ├── LIMIT_UP_PRICE.pkl
│   ├── LIMIT_DOWN_PRICE.pkl
│   └── VOLUME.pkl                   # 由 generate_volume.py 从分钟数据合成
│
└── Strategy/                        # 框架主目录
    ├── config.py                    # ★ 全局配置 (路径/日期/参数)
    ├── outputs/                     # 所有运行产出
    │   ├── labels/                  # Label 宽表
    │   ├── factors/                 # 因子宽表
    │   ├── scores/                  # 打分宽表 + 模型文件
    │   └── bt_results/              # 回测图表与报告
    ├── data_io/
    │   ├── loader.py                # MinuteDataLoader / DailyDataLoader
    │   ├── saver.py                 # save_wide_table (统一宽表存储)
    │   └── generate_volume.py       # 合成 VOLUME.pkl
    ├── label/
    │   └── label_generator.py       # LabelGenerator (TWAP/VWAP/Close 收益率)
    ├── factor/
    │   ├── factor_base.py           # FactorBase 基类 + FactorRegistry 注册器
    │   ├── daily_factors_raw.py     # 原始因子库 (80+ 个, compute_daily_factors_panel)
    │   ├── daily_factor_library.py  # 适配器: 宽表 ↔ 长表 + shift(1) 防未来数据
    │   ├── minute_derived_factors.py# 分钟衍生因子 (JumpVariation / CR2-4 / BollPositionNorm)
    │   └── custom_factors.py        # 用户自定义新因子 (FactorBase 子类模板)
    ├── model/
    │   ├── trainer.py               # build_panel / split_panel / XGBTrainer
    │   └── scorer.py                # score_all / generate_scores
    ├── backtest/
    │   ├── quick_backtest.py        # 20 分组分层净值回测
    │   └── event_backtest.py        # 精细化事件驱动回测引擎
    └── utils/
        └── helpers.py               # 代码转换 / 交易日历 / safe_rolling
```

---

## 完整运行路径

> **工作目录统一为 `/root/autodl-tmp`，所有命令在此目录下执行。**

```
步骤 0  →  步骤 1  →  步骤 2  →  步骤 3  →  步骤 4  →  步骤 5  →  步骤 6
合成VOLUME  生成Label  计算因子   构建Panel  训练模型   生成打分   运行回测
```

---

### 步骤 0 — 合成 VOLUME.pkl（仅需运行一次）

从分钟频数据的 `vol` 列（925~1500）求和，生成每日全市场成交量宽表。

```bash
cd /root/autodl-tmp
python -m Strategy.data_io.generate_volume
```

**输出:** `Daily_data/VOLUME.pkl`

---

### 步骤 1 — 生成 Label 宽表（仅需运行一次）

计算 T 日 1430~1457 的 TWAP 买入价，Label = T+1 同期 TWAP / T 同期 TWAP - 1。

```python
import sys
sys.path.insert(0, '/root/autodl-tmp')

from Strategy.label.label_generator import LabelGenerator

lg = LabelGenerator(time_start=1430, time_end=1457, price_type="twap")
price_path, label_path = lg.generate_and_save()
print(f"价格表: {price_path}")
print(f"Label:  {label_path}")
```

**输出:**
- `outputs/labels/TWAP_1430_1457.fea`  — 基准价格宽表
- `outputs/labels/LABEL_TWAP_1430_1457.fea` — Label 宽表

> ⚠️ **Label 是预测目标，严禁作为因子输入模型！**

---

### 步骤 2 — 计算因子（仅需运行一次，耗时较长）

共三条路径，建议顺序执行：

#### 2-A  原始日频因子库（80+ 个）

```python
import sys, logging
sys.path.insert(0, '/root/autodl-tmp')
logging.basicConfig(level=logging.INFO)

from Strategy.factor.daily_factor_library import DailyFactorLibraryAdapter

adapter = DailyFactorLibraryAdapter()
saved = adapter.compute_and_save_all()
print(f"已保存 {len(saved)} 个因子")
```

> 内存提示：约需 5~7 GB，建议单独运行，全量历史约需 15~30 分钟。

#### 2-B  FactorBase 注册因子（CR2/CR3/CR4 / BollPositionNorm 等）

```python
import sys, logging
sys.path.insert(0, '/root/autodl-tmp')
logging.basicConfig(level=logging.INFO)

from Strategy.factor.factor_base import FactorRegistry
import Strategy.factor.minute_derived_factors   # 触发注册
import Strategy.factor.custom_factors           # 触发注册

FactorRegistry.compute_all()
```

#### 2-C  跳跃分解因子（可选，需遍历分钟数据，耗时较长）

```python
import sys, logging
sys.path.insert(0, '/root/autodl-tmp')
logging.basicConfig(level=logging.INFO)

from Strategy.factor.minute_derived_factors import JumpVariationFactor

jv = JumpVariationFactor()
jv.compute_and_save()  # 输出 RV / RVC / RVJ 等 13 个因子
```

**所有因子输出目录:** `outputs/factors/*.fea`

---

### 步骤 3 — 构建 Panel 并划分样本

```python
import sys
sys.path.insert(0, '/root/autodl-tmp')

from Strategy.factor.factor_base import load_all_factors
from Strategy.label.label_generator import load_label
from Strategy.model.trainer import build_panel, split_panel

# 加载
factor_dict = load_all_factors()          # 自动读取 outputs/factors/ 下所有 .fea
label_df    = load_label("TWAP_1430_1457")

# 拼接长表 Panel
panel = build_panel(factor_dict, label_df)
print(panel.shape)   # (样本数, 因子数+3)

# 按时间划分 (config.py 中已定义日期)
# Train: 2021-01-01 ~ 2023-08-01
# Val:   2023-09-01 ~ 2024-09-01
# OOS:   2024-09-01 之后
train, val, oos = split_panel(panel)
```

---

### 步骤 4 — 训练 XGBoost 模型

```python
from Strategy.model.trainer import XGBTrainer

trainer = XGBTrainer(
    params={
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "tree_method": "hist",  # 改为 "gpu_hist" 可启用 5090 GPU 加速
        "verbosity": 0,
    },
    num_boost_round=500,
    early_stopping_rounds=50,
)

trainer.train(train, val)
trainer.save_model()   # 保存到 outputs/scores/xgb_model.pkl
```

> 💡 GPU 加速：将 `"tree_method": "hist"` 改为 `"tree_method": "gpu_hist"` 即可使用 5090。

---

### 步骤 5 — 生成全市场打分

```python
from Strategy.model.scorer import generate_scores, load_scores
from Strategy.factor.factor_base import load_all_factors
from Strategy.label.label_generator import load_label

factor_dict = load_all_factors()
label_df    = load_label("TWAP_1430_1457")

score_path = generate_scores(
    trainer=trainer,
    factor_dict=factor_dict,
    label_df=label_df,
    model_name="xgb",
    label_tag="TWAP_1430_1457",
    normalize=True,               # 每日截面 Z-Score 标准化
)
print(f"打分已保存: {score_path}")

# 后续加载
score_df = load_scores("xgb", "TWAP_1430_1457")
```

**输出:** `outputs/scores/SCORE_xgb_TWAP_1430_1457.fea`

---

### 步骤 6 — 回测

#### 6-A  快速分层回测（20 组净值曲线）

```python
from Strategy.backtest.quick_backtest import run_quick_backtest
from Strategy.model.scorer import load_scores
from Strategy.label.label_generator import load_label

score_df = load_scores("xgb", "TWAP_1430_1457")
label_df = load_label("TWAP_1430_1457")

run_quick_backtest(
    score_df=score_df,
    label_df=label_df,
    n_groups=20,
    output_dir="outputs/bt_results",
)
```

**输出:** `outputs/bt_results/quantile_nav.png`

#### 6-B  精细化事件驱动回测

```python
from Strategy.backtest.event_backtest import BacktestRunner
from Strategy.model.scorer import load_scores

score_df = load_scores("xgb", "TWAP_1430_1457")

runner = BacktestRunner(
    score_df=score_df,
    top_n=50,               # 每日持仓股票数
    rebalance_freq=1,       # 调仓频率 (天)
)
result = runner.run()
result.plot(save_dir="outputs/bt_results")
```

---

## 样本划分（config.py 定义，全框架统一）

| 集合 | 起止日期 | 用途 |
|------|---------|------|
| **训练集** | 2021-01-01 ~ 2023-08-01 | 模型训练 |
| **验证集** | 2023-09-01 ~ 2024-09-01 | 早停 / 超参调优 |
| **OOS 样本外** | 2024-09-01 之后 | 最终评估，**严禁参与任何参数决策** |

---

## 防未来数据规则

| 场景 | 规则 |
|------|------|
| 所有因子 | 输出前统一 `shift(1)`，T 日因子值只用 T-1 及之前数据 |
| Label | `shift(-1)` 取次日价格，**仅作训练目标，禁止作为特征** |
| `safe_rolling()` | 工具函数内置 `shift(1)`，滚动计算自动防泄漏 |
| 模型训练 | `split_panel()` 严格按时间切分，无随机 shuffle |
| 回测 | 事件驱动引擎使用 T-1 信号驱动 T 日交易，资金 T+1 到账 |

---

## 新增自定义因子

在 `Strategy/factor/custom_factors.py` 中继承 `FactorBase`：

```python
from Strategy.factor.factor_base import FactorBase, FactorRegistry

@FactorRegistry.register
class MyNewFactor(FactorBase):
    name = "my_new_factor"

    def compute(self, daily_data):
        close = daily_data["CLOSE_PRICE"]
        # ... 计算逻辑（只用 T-1 及之前数据）...
        return result.shift(1)   # ⚠️ 必须 shift(1)
```

然后在步骤 2-B 中重新运行 `FactorRegistry.compute_all()` 即可。

---

## 硬件配置

| 资源 | 配置 |
|------|------|
| CPU | 详见 `lscpu` |
| 内存 | 64 GB+ |
| GPU | 单卡 NVIDIA RTX 5090 |
| 系统盘 | 详见 `df -h` |

XGBoost 启用 GPU：`params["tree_method"] = "gpu_hist"`

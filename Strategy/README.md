# Strategy — A-股量化选股框架

端到端机器学习选股框架，涵盖因子计算 → Label 生成 → 模型训练 → 打分 → 分层/事件驱动回测的完整流水线。

---

## 目录

1. [项目结构](#项目结构)
2. [快速开始](#快速开始)
3. [数据约定](#数据约定)
4. [模块详解](#模块详解)
   - [config.py — 全局配置](#configpy--全局配置)
   - [data\_io — 数据读写层](#data_io--数据读写层)
   - [label — Label 生成](#label--label-生成)
   - [factor — 因子体系](#factor--因子体系)
   - [model — 训练与打分](#model--训练与打分)
   - [backtest — 回测引擎](#backtest--回测引擎)
   - [utils — 通用工具](#utils--通用工具)
5. [防未来数据设计](#防未来数据设计)
6. [样本划分说明](#样本划分说明)
7. [扩展指南](#扩展指南)
8. [输出文件说明](#输出文件说明)
9. [常见问题](#常见问题)

---

## 项目结构

```
Strategy/
├── config.py                      # 全局路径、日期划分、回测参数
├── main.ipynb                     # 主流程 Notebook（编排入口）
│
├── data_io/
│   ├── loader.py                  # 分钟/日频数据加载器
│   ├── saver.py                   # 宽表持久化（Feather/PKL）
│   └── generate_volume.py         # 从分钟数据合成日频 VOLUME.pkl
│
├── label/
│   └── label_generator.py         # TWAP/VWAP/Open 多种 Label 生成
│
├── factor/
│   ├── factor_base.py             # FactorBase 抽象类 + FactorRegistry 注册器
│   ├── daily_factor_library.py    # 日频因子批量计算适配器（80+ 因子）
│   ├── daily_factors_raw.py       # 原始因子计算函数库
│   ├── minute_derived_factors.py  # 分钟数据衍生因子（Jump/CR/Boll）
│   └── custom_factors.py          # 用户自定义新增因子入口
│
├── model/
│   ├── trainer.py                 # Panel 构建 + XGBoost 训练器
│   └── scorer.py                  # 全市场打分 + T日价格 mask
│
├── backtest/
│   ├── universe.py                # 股票池过滤（新股/退市/ST/前缀）
│   ├── quick_backtest.py          # 快速分层/TopK 回测
│   ├── event_backtest.py          # 精细事件驱动回测引擎
│   └── ic_analysis.py             # IC / Rank IC 分析（与打分质量评估）
│
├── utils/
│   └── helpers.py                 # 代码转换、交易日历、safe_rolling
│
└── tools/
    └── filter_factors_sse_szse.py # 因子文件剔除非沪深列工具
```

---

## 快速开始

按 `main.ipynb` 的单元格顺序执行整个流水线：

```
Cell 0  → 生成 OPEN0935_1000 Label（可选，新标签时执行）
Cell 1  → 生成 TWAP Label（默认 14:30–14:57）
Cell 2  → 计算日频因子库（DailyFactorLibraryAdapter）
Cell 3  → 计算分钟衍生因子（Jump/CR/Boll）及自定义因子
Cell 4  → 加载所有因子 + Label，构建 Panel
Cell 5  → 训练 XGBoost 模型（Train+Val）
Cell 6  → 对全市场生成打分（Score）
Cell 7  → 快速分层/TopK 回测（quick_backtest）
Cell 8  → 精细事件驱动回测（event_backtest）
Cell 9  → 早盘 09:35 买 / 次日 10:00 卖回测（open0935_1000）
```

---

## 数据约定

### 原始数据目录结构

```
/root/autodl-tmp/
├── min_data/                      # 分钟频 K 线
│   ├── 2021/
│   │   ├── 20210104.fea           # 每日一个文件，列包含 StockID/time/open/high/low/close/vol
│   │   └── ...
│   └── 2022/ ...
│
└── Daily_data/                    # 日频数据（宽表 PKL，index=TRADE_DATE，columns=股票代码）
    ├── CLOSE_PRICE.pkl
    ├── OPEN_PRICE.pkl
    ├── HIGHEST_PRICE.pkl
    ├── LOWEST_PRICE.pkl
    ├── PRE_CLOSE_PRICE.pkl
    ├── CHG_PCT.pkl
    ├── DEAL_AMOUNT.pkl
    ├── TURNOVER_RATE.pkl
    ├── TURNOVER_VALUE.pkl
    ├── MARKET_VALUE.pkl
    ├── LIMIT_UP_PRICE.pkl
    ├── LIMIT_DOWN_PRICE.pkl
    ├── VOLUME.pkl                 # 由 generate_volume.py 从分钟数据生成
    ├── ipo_dates.pkl              # 列：TICKER_SYMBOL / INTO_DATE / OUT_DATE
    └── st_status.pkl              # 宽表，1=ST，0=正常（index=TRADE_DATE，columns=股票代码）
```

### 宽表格式规范

所有落盘的因子/Label/Score 均为"宽表"格式：
- **行索引（index）**：`TRADE_DATE`，`datetime64[ns]` 类型
- **列（columns）**：6 位纯数字股票代码字符串，如 `"000001"`
- **存储格式**：Feather（`.fea`）— 先写 `.fea.part` 再原子替换，避免写入中断损坏文件

---

## 模块详解

### `config.py` — 全局配置

所有模块通过 `from Strategy import config` 引用，确保路径和参数全局一致。

**核心配置项：**

| 配置项 | 说明 |
|--------|------|
| `TRAIN_START / TRAIN_END` | 训练集日期范围（含两端） |
| `VAL_START / VAL_END` | 验证集日期范围，`VAL_END < OOS_START` |
| `OOS_START` | 样本外起始日（含），严格禁止用于任何调参决策 |
| `MIN_DATA_DIR` | 分钟数据目录 |
| `DAILY_DATA_DIR` | 日频数据目录 |
| `FACTOR_OUTPUT_DIR` | 因子落盘目录（`outputs/factors/`） |
| `LABEL_OUTPUT_DIR` | Label 落盘目录（`outputs/labels/`） |
| `SCORE_OUTPUT_DIR` | 打分落盘目录（`outputs/scores/`） |
| `BT_RESULT_DIR` | 回测结果目录（`outputs/bt_results/`） |
| `COMMISSION_RATE` | 双向佣金（默认万一） |
| `STAMP_DUTY_RATE` | 卖出印花税（默认万五） |
| `INITIAL_CAPITAL` | 初始资金（默认 1,000 万） |

> **日期边界注意**：`VAL_END`（2024-08-31）与 `OOS_START`（2024-09-01）严格不重叠，`split_panel` 中 val 集使用 `<= VAL_END`，OOS 使用 `>= OOS_START`。

---

### `data_io` — 数据读写层

#### `loader.py`

**`MinuteDataLoader`**：逐日加载分钟 K 线文件。
- 自动过滤北交所/新三板（`BJ`/`NE` 前缀）
- 支持按日期范围批量加载（`load_date_range`）

**`DailyDataLoader`**：加载日频宽表 PKL 文件。
- `load_field(field_name)` — 加载单个字段
- `load_fields(field_names)` — 批量加载多个字段，返回 `{field: DataFrame}`
- `as_of_date` 参数 — 截断到指定日期，防止因子计算时使用未来数据

#### `saver.py`

**`save_wide_table(df, path)`**：统一的宽表持久化。
- Feather 格式：先写 `.part` 临时文件，成功后原子重命名
- 自动 `reset_index()`（将行索引转为 `TRADE_DATE` 列）

#### `generate_volume.py`

从每日分钟数据的 `vol` 字段（09:25–15:00 所有分钟）汇总为日频 `VOLUME.pkl`，供因子计算使用。

---

### `label` — Label 生成

#### `label_generator.py`

##### TWAP/VWAP Label（`LabelGenerator`）

```python
gen = LabelGenerator(time_start=1430, time_end=1457, price_type="twap")
gen.compute_and_save_price_table()    # 保存 TWAP_1430_1457.fea（T日买入锚定价）
gen.compute_and_save_label()          # 保存 LABEL_TWAP_1430_1457.fea
```

**Label 定义**：

```
label(T) = TWAP(T+1) / TWAP(T) - 1
           × adj_factor              # 除息除权调整: CLOSE(T) / PRE_CLOSE(T+1)
```

- `T` 日的 label = `T` 日 TWAP 买入、`T+1` 日 TWAP 卖出的实际收益率
- 最后一个交易日的 label 为 NaN（没有 T+1）

##### 早盘开盘 Label（`generate_and_save_open0935_1000_label`）

```python
generate_and_save_open0935_1000_label(save_price_tables=True)
```

**Label 定义**：

```
label(T) = OPEN_1000(T+1) / OPEN_0935(T) - 1
           × adj_factor              # 除息除权调整
```

**生成文件**：
| 文件名 | 内容 |
|--------|------|
| `LABEL_OPEN0935_1000.fea` | 上述 label 宽表 |
| `OPEN0935_1000.fea` | T 日 09:35 开盘价（scorer 的 T 日 mask 锚点） |
| `OPEN0935_1000_BUY0935.fea` | T 日 09:35 开盘价宽表（买入价参考） |
| `OPEN0935_1000_SELL1000.fea` | T+1 日 10:00 开盘价宽表（卖出价参考） |

##### `load_label(tag)` / `load_price(tag)`

快捷加载已保存的 Label 或价格宽表：
```python
label_df = load_label("TWAP_1430_1457")
label_df = load_label("OPEN0935_1000")    # 加载 LABEL_OPEN0935_1000.fea
price_df = load_price("TWAP_1430_1457")   # 加载 TWAP_1430_1457.fea
```

---

### `factor` — 因子体系

框架提供两套并行的因子计算路径，最终都落盘为 `outputs/factors/{name}.fea`。

#### 路径一：日频因子库（`DailyFactorLibraryAdapter`）

处理 `daily_factors_raw.py` 中定义的 80+ 个日频因子，包含：
- 动量类：N 日收益率、超额收益、MACD 等
- 波动类：历史波动率、振幅、ATR 等
- 量价类：换手率、成交量比、涨跌幅 z-score 等
- 技术指标：RSI、布林带位置、均线偏离等

```python
from Strategy.factor.daily_factor_library import DailyFactorLibraryAdapter
adapter = DailyFactorLibraryAdapter()
adapter.compute_and_save_all(start_date="2021-01-01", end_date=None)
```

**重要**：`DailyFactorLibraryAdapter` 在保存前对全量因子宽表执行 `shift(1)`，确保 `T` 日存储的因子值只含 `T-1` 及之前的信息。

#### 路径二：分钟衍生因子（`FactorRegistry` + `FactorBase`）

```python
from Strategy.factor.factor_base import FactorRegistry
import Strategy.factor.minute_derived_factors   # 注册分钟因子
import Strategy.factor.custom_factors           # 注册自定义因子
FactorRegistry.compute_all(daily_data)
```

已内置的分钟衍生因子：

| 因子名 | 说明 |
|--------|------|
| `jump_variation` | 隔夜跳空强度（基于分钟开盘价） |
| `cr2` / `cr3` / `cr4` | 中值偏离比（分钟价格中位 vs 开收盘） |
| `boll_position_norm` | 布林带位置归一化 |

#### 自定义新增因子

在 `custom_factors.py` 中继承 `FactorBase` 并用 `@FactorRegistry.register` 注册：

```python
@FactorRegistry.register
class MyFactor(FactorBase):
    name = "my_factor_20d"

    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = daily_data["CLOSE_PRICE"]
        # 只能使用 T-1 及之前数据
        return (close / close.shift(20) - 1).shift(1)   # ⚠️ 必须 shift(1)
```

> **注意**：通过 `FactorRegistry` 注册的因子，`compute()` 方法必须自行确保 `shift(1)` 防未来数据。`DailyFactorLibraryAdapter` 会对其管理的因子做整体 `shift`，不覆盖注册器路径。

加载所有因子：

```python
from Strategy.factor.factor_base import load_all_factors
factor_dict = load_all_factors()   # 加载 outputs/factors/ 下所有 .fea
```

---

### `model` — 训练与打分

#### `trainer.py`

**`build_panel(factor_dict, label_df)`**

将多个因子宽表与 Label 宽表对齐并展平为长表：
- 自动取公共日期 × 公共股票的交集
- 使用高效 `stack + concat` 路径（比 N 次 merge 快 10-100 倍）

```
输出列：TRADE_DATE | StockID | factor_1 | factor_2 | ... | label
```

**`split_panel(panel)`**

按 `config` 中的日期配置切分为训练集/验证集/OOS：
```
train:  [TRAIN_START, TRAIN_END]
val:    [VAL_START,   VAL_END]
oos:    [OOS_START,   +∞)
```

**`XGBTrainer`**

| 方法 | 说明 |
|------|------|
| `train(train_df, val_df)` | 训练 XGBoost，支持 val 早停 |
| `predict(df)` | 对 Panel 生成预测值 |
| `save_model(path)` | 保存模型到 `outputs/scores/xgb_model.pkl` |
| `load_model(path)` | 从文件加载已训练模型 |

默认超参数（可在 notebook 中覆盖）：
```python
{
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "tree_method": "hist",
}
```

#### `scorer.py`

**`generate_scores(trainer, factor_dict, label_df, model_name, label_tag)`**

端到端打分流水线：
1. 调用 `build_panel` 拼接 Panel
2. 模型预测 → 截面 Z-Score 标准化
3. 用 T 日价格 mask 屏蔽退市/停牌股票（`mask_scores_by_price`）
4. 保存为 `SCORE_{model_name}_{label_tag}.fea`

**`mask_scores_by_price(score_wide, label_tag)`**

屏蔽 T 日无执行价格的股票，防止退市/停牌股进入回测候选：
- 读取 `{label_tag}.fea`（T 日价格锚点文件）
- 对应位置为 NaN → 将 score 置为 NaN

**`load_scores(model_name, label_tag)`**

快捷加载已保存的打分，加载时自动重新应用价格 mask（幂等）。

---

### `backtest` — 回测引擎

#### `universe.py` — 股票池过滤

所有过滤条件**只使用 T 日可知信息**：

| 过滤函数 | 过滤条件 |
|----------|----------|
| `listing_age_mask` | 上市不足 `min_listing_days`（默认 20 日）的新股 |
| `out_date_mask` | 距退市日不足 `delist_buffer_days`（默认 20 日）的预退市股 |
| `st_mask` | ST/\*ST 股；`historical=True` 时历史上出现过 ST 即永久剔除（累计最大值） |
| `prefix_mask` | 前缀为 `300`（创业板）或 `688`（科创板）的股票 |

**`build_universe_filter(dates, columns, ...)`**

组合上述四个 mask，返回：
- `universe`：布尔宽表（`True` = 可交易）
- `report`：每日各维度排除股票数量统计

#### `quick_backtest.py` — 快速分层回测

无摩擦的截面收益分析工具，用于快速验证因子/模型有效性。

**执行假设**：
- 无佣金、无印花税、无滑点
- 等权持仓，使用 Label 对应的 TWAP 区间收益率

**`run_quick_backtest(score_df, label_df, ...)`**

主入口，返回并保存：
- `quantile_backtest.png`：20 组分层 NAV 曲线（含 VAL/OOS 分隔线）
- `topN_nav.png`：Top20/50/100 等 NAV 曲线对比
- `quick_backtest_universe_report.csv`：每日股票池排除明细

核心参数：

| 参数 | 说明 |
|------|------|
| `top_ks` | 列表，如 `[20, 50, 100]`，生成对应 TopK 曲线 |
| `min_listing_days` | 新股排除阈值（默认 20） |
| `delist_buffer_days` | 预退市排除缓冲（默认 20） |
| `exclude_st` | 是否排除 ST（默认 True） |
| `exclude_historical_st` | 是否排除历史出现过 ST 的股票（默认 True） |
| `excluded_prefixes` | 排除的股票代码前缀（默认 `("300", "688")`） |

> **与 `event_backtest` 的差异**：分层回测每日更换持仓，无整手约束，无执行延迟。精细回测包含费率、整手、调仓频率、涨跌停成交判断。两者结果有差距属正常现象。

#### `ic_analysis.py` — IC / Rank IC 分析

评估模型打分（Score）与下期实际收益（Label）之间的截面预测能力。

**两种相关性指标**：

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| IC (Information Coefficient) | 皮尔逊线性相关 | 衡量打分与收益的线性相关程度 |
| Rank IC | 斯皮尔曼秩相关 | 对异常值更鲁棒，更贴近实际选股排序能力 |

**`run_ic_analysis(score_df, label_df, ...)`**

一键运行入口，与 `run_quick_backtest` 共享相同 `tradeable_mask`（无重复计算）：

```python
from Strategy.backtest.ic_analysis import run_ic_analysis

ic_df, summary = run_ic_analysis(
    score_df=score_df,
    label_df=label_df,
    tradeable_mask=tradeable_mask,   # 与 quick_backtest 共用
    title="XGB | TWAP_1430_1457",
    rolling_window=20,               # IC 滚动均线窗口
)
```

分段统计（`ic_summary.csv`）字段说明：

| 字段 | 说明 |
|------|------|
| `mean` | IC / Rank IC 均值 |
| `std` | IC / Rank IC 标准差 |
| `icir` | ICIR = mean / std，衡量 IC 稳定性 |
| `win_rate` | IC > 0 的天数占比（胜率） |
| `t_stat` / `p_value` | 均值是否显著异于 0 的 t 检验 |
| `n_days` | 有效交易日数 |

**与 `run_quick_backtest` 集成**：

`run_quick_backtest` 默认 `run_ic=True`，在分层回测结束后自动调用 IC 分析：

```python
run_quick_backtest(
    score_df, label_df,
    run_ic=True,           # 默认开启，传 False 可跳过
    ic_rolling_window=20,  # IC 滚动均线窗口
)
```

#### `event_backtest.py` — 精细事件驱动回测

模拟真实交易执行的完整回测引擎。

**交易流程（每个交易日）**：
```
T 日:
  1. 调仓日 → 卖出旧仓（TWAP）→ 资金回笼
  2. 按 score_df.loc[T] 选股（T-1 收盘后信息）
  3. 检查 T 日涨停 → 无法买入则跳过
  4. 等额分配资金 → 按整手（100 股）买入（TWAP）
  5. 记录 NAV（按持仓股 TWAP 估值）
```

**`BacktestRunner` 参数**：

| 参数 | 说明 |
|------|------|
| `score_df` | 打分宽表（score.loc[T] 含 T-1 信息） |
| `top_n` | 每日持有 top N 只；与 `mirror_quantile_group` 二选一 |
| `mirror_quantile_group` | 做多第 G 组（1=最高分），与快速回测分层对应 |
| `n_quantile_groups` | 总分组数（默认 20） |
| `rebalance_freq` | 调仓频率（交易日数，默认 1=每日调仓） |
| `frictionless` | 是否忽略所有费率（用于与 quick 回测对比） |
| `twap_tag` | 执行价格对应的 Label tag（默认 `TWAP_1430_1457`） |
| `min_listing_days` | 新股排除阈值 |
| `delist_buffer_days` | 预退市排除缓冲 |
| `exclude_historical_st` | 历史 ST 排除 |
| `excluded_prefixes` | 前缀排除 |

**异常处理**：
- 卖出时无 TWAP 价格 → 加入 `_delayed_sells`，下一个交易日继续尝试
- 买入时涨停 → 跳过，记录 `ExceptionType.LIMIT_UP_CANNOT_BUY`

**`BacktestResult` 输出**：

| 文件 | 内容 |
|------|------|
| `{name}_event_nav.csv` | 每日净值 |
| `{name}_trades_all.csv` | 全量交易流水 |
| `{name}_exceptions.csv` | 异常记录（无价格/涨停无法成交等） |
| `{name}_nav.png` | NAV 曲线图（含 VAL/OOS 分隔线） |

**性能指标**：均使用**简单年化**（`日均收益 × 242`）而非复利年化，与快速回测对齐：

```
年化收益 = daily_ret.mean() × 242
年化波动 = daily_ret.std() × sqrt(242)
Sharpe   = 年化收益 / 年化波动
最大回撤 = max(1 - NAV / NAV.cummax())
```

#### 早盘开盘回测（`main.ipynb` Cell 9）

针对 `09:35 买入 / 次日 10:00 卖出` 策略的专属回测逻辑，内置于 `main.ipynb` Cell 9：

**关键逻辑**：
- 从分钟数据提取 T 日 09:35 开盘价和 T+1 日 10:00 开盘价
- 买入条件：T 日 09:35 时未封死涨停
- **连板持仓逻辑**：若 T+1 日 10:00 时股票仍封死涨停（未炸板），则继续持有，不卖出，等待后续连板机会

---

### `utils` — 通用工具

#### `helpers.py`

| 函数 | 说明 |
|------|------|
| `safe_rolling(df, window, func)` | 安全滚动计算，内置 `shift(1)` |
| `ensure_tradedate_as_index(df)` | 将 `TRADE_DATE` 列设为 DatetimeIndex 行索引 |
| `get_trade_dates()` | 从 `CLOSE_PRICE.pkl` 推断交易日历 |
| `filter_trade_dates(start, end)` | 按日期范围过滤交易日列表 |
| `strip_stock_prefix(code)` | `SZ000001` → `000001` |
| `is_sh_or_sz(code)` | 判断是否属于沪深（排除北交所） |
| `date_int_to_str(d)` | `20210104` → `"2021-01-04"` |

---

## 防未来数据设计

框架在每个环节都严格防止未来数据泄露，核心机制如下：

### 时间对齐规则

```
因子(T) = f(数据[T-1 及之前])   ← shift(1) 在因子生成阶段完成
Label(T) = return(T → T+1)       ← 训练目标，不作为特征输入
Score(T) = model(因子(T))        ← 只含 T-1 信息，T日可用于选股
交易(T)  = 按 Score(T) 买入      ← T日实际执行，时序正确
```

### `shift(1)` 的位置

| 模块 | 实现方式 |
|------|----------|
| `DailyFactorLibraryAdapter` | 全量因子宽表计算完毕后统一 `.shift(1)` 再落盘 |
| 分钟衍生因子（CR/Boll/Jump） | 各自 `compute()` 方法末尾调用 `.shift(1)` |
| 自定义因子（`custom_factors.py`） | 开发者必须在 `compute()` 末尾手动 `.shift(1)` |
| `safe_rolling()` | 内置 `.shift(1)`，调用方无需再次 shift |
| `scorer.py` | **不做二次 shift**，score.loc[T] 的时效性由因子延迟保证 |

> ⚠️ `score_df` 传入 `BacktestRunner` 时，**不应对其再额外 shift(1)**，否则会造成双重延迟（T日的 score 实际含 T-2 信息），导致性能低估。

### 股票池过滤

回测股票池过滤只使用 T 日可知信息：
- T 日 TWAP 是否有数据（`mask_scores_by_price`）
- T 日是否涨停（`LIMIT_UP_PRICE`）
- 上市天数（`ipo_dates`）
- 距退市日天数（`out_date`）
- 历史 ST 状态（`cummax` 保证不使用未来 ST 信息）
- 股票代码前缀

---

## 样本划分说明

```
2021-01-01            2023-08-31  2023-09-01            2024-08-31  2024-09-01
     |─────────────────────|           |─────────────────────|           |──────────→
           训练集（Train）                  验证集（Val）                样本外（OOS）
```

- **训练集**：`[2021-01-01, 2023-08-31]`，用于 XGBoost 迭代更新
- **验证集**：`[2023-09-01, 2024-08-31]`，用于早停（Early Stopping），**禁止用于超参调优的最终决策**
- **OOS**：`[2024-09-01, ∞)`，严格样本外，仅用于最终评估

> VAL 和 OOS 日期范围严格不重叠（`VAL_END = 2024-08-31` < `OOS_START = 2024-09-01`）。

---

## 扩展指南

### 新增因子

1. 在 `custom_factors.py` 中添加继承 `FactorBase` 的类
2. 实现 `name` 属性（全局唯一）和 `compute()` 方法
3. 在 `compute()` 末尾确保 `shift(1)`
4. 用 `@FactorRegistry.register` 装饰
5. 在 `main.ipynb` 对应 Cell 中 `import Strategy.factor.custom_factors` 后调用 `FactorRegistry.compute_all()`

### 新增 Label

1. 在 `label_generator.py` 中添加新的计算函数
2. 确保 label.loc[T] = T 日买入到 T+1 日卖出的实际收益率（考虑除息除权）
3. 保存对应的 T 日**买入价格表**（文件名须与 `label_tag` 一致），供 `scorer._load_price_mask` 使用：
   - TWAP Label：`TWAP_{start}_{end}.fea`
   - 自定义 Label（如 `MY_LABEL`）：同时保存 `MY_LABEL.fea`（T 日买入价）和 `LABEL_MY_LABEL.fea`（收益率）
4. 在 `load_label()` 的 docstring 中登记新 tag

### 更换模型

`XGBTrainer` 通过 `train() / predict() / save_model() / load_model()` 四个方法与框架交互，可按此接口实现 LightGBM / MLP 等替代模型，无需修改下游打分和回测代码。

### 调整回测参数

| 调整目标 | 修改方式 |
|----------|----------|
| 买入/卖出时间 | 修改 `LabelGenerator` 的 `time_start`/`time_end` 或使用 `generate_and_save_open0935_1000_label` |
| 持仓数量 | `BacktestRunner(top_n=...)` 或调整 `top_ks` 列表 |
| 调仓频率 | `BacktestRunner(rebalance_freq=5)` 表示每周调仓 |
| 费率参数 | `BacktestRunner(commission_rate=..., stamp_duty_rate=...)` 或修改 `config.py` |
| 股票池规则 | 调整 `min_listing_days`、`delist_buffer_days`、`excluded_prefixes` 等参数 |

---

## 输出文件说明

```
outputs/
├── labels/
│   ├── TWAP_1430_1457.fea              # T日买入价格表（scorer mask锚点）
│   ├── LABEL_TWAP_1430_1457.fea        # T→T+1 TWAP 收益率
│   ├── OPEN0935_1000.fea               # T日09:35开盘价（scorer mask锚点）
│   ├── LABEL_OPEN0935_1000.fea         # T 09:35买 / T+1 10:00卖 收益率
│   ├── OPEN0935_1000_BUY0935.fea       # T日09:35开盘价（回测价格参考）
│   └── OPEN0935_1000_SELL1000.fea      # T+1日10:00开盘价（回测价格参考）
│
├── factors/
│   ├── momentum_20d.fea                # 各因子宽表（已 shift(1)）
│   ├── volatility_20d.fea
│   └── ...（每个因子一个文件）
│
├── scores/
│   ├── xgb_model.pkl                   # 已训练的 XGBoost 模型
│   └── SCORE_xgb_TWAP_1430_1457.fea   # 打分宽表（已 mask 退市/停牌）
│
└── bt_results/
    ├── quantile_backtest.png            # 20组分层 NAV 曲线
    ├── topN_nav.png                     # Top20/50/100 NAV 曲线对比
    ├── quick_backtest_universe_report.csv
    ├── ic_series.csv                    # 每日 IC / Rank IC 时间序列
    ├── ic_summary.csv                   # Train/Val/OOS 分段 ICIR 统计
    ├── ic_analysis.png                  # IC 时序 + 累积曲线 + ICIR 柱图
    ├── {strategy}_event_nav.csv         # 精细回测每日净值
    ├── {strategy}_trades_all.csv        # 全量交易流水
    ├── {strategy}_exceptions.csv        # 异常记录
    ├── {strategy}_nav.png               # 精细回测 NAV 图
    ├── open0935_1000_quick_nav.png      # 早盘策略快速回测曲线
    ├── open0935_1000_event_topn_nav_compare.png
    └── ...
```

---

## 常见问题

**Q：`build_panel` 报错"无公共交易日或股票"**

A：因子的行索引类型与 label 不一致（常见：一侧为 `datetime.date` 对象，另一侧为 `DatetimeIndex`）。`load_all_factors` / `load_label` 内部已调用 `ensure_tradedate_as_index`，若手动加载因子，需手动调用该函数或确保统一为 `DatetimeIndex`。

**Q：因子计算速度很慢**

A：`DailyFactorLibraryAdapter` 支持分块计算（`chunk_size` 参数），并通过 `skip_existing=True` 跳过已存在的因子文件。增量更新时只需计算新增日期。

**Q：回测结果比 quick_backtest 差很多**

A：正常现象。`quick_backtest` 假设每日等权换仓、无摩擦，且 `group1` 约覆盖全池的 5% 股票；`event_backtest` 含费率（万二佣金 + 万五印花税）、整手约束、涨跌停无法成交等真实限制。若 `frictionless=True` 两者差距应大幅缩小。

**Q：如何使用已有模型文件重新打分，而不重新训练？**

```python
trainer = XGBTrainer()
trainer.load_model("outputs/scores/xgb_model.pkl")
generate_scores(trainer, factor_dict, label_df, label_tag="TWAP_1430_1457")
```

**Q：`load_scores` 每次都会重新 mask，会影响性能吗？**

A：mask 操作是幂等的（已 NaN 的位置再 mask 仍为 NaN），计算成本较低（一次 DataFrame 对齐操作）。这样设计是为了保证每次加载的打分都是最新 mask 状态，即使 mask 锚点文件被更新后重新生成，无需重新调用 `generate_scores`。

**Q：如何新增对 ETF / 期货的支持？**

A：框架目前面向 A 股正股（沪深主板）设计，若需扩展，需修改 `is_sh_or_sz` 函数的判断逻辑，以及 `universe.py` 中的股票池过滤规则（如不适用 IPO 天数限制）。其余流水线无需修改。
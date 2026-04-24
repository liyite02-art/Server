# 策略研发环境规则与数据准则

## 基本信息
- **用途**：记录 @Strategy 目录下量化策略研发的数据格式、使用规范及防未来数据准则。
- **目标**：基于分钟频、日频数据自研量化策略，确保数据的正确使用，严格杜绝未来数据。
- **核心逻辑**：所有因子的计算和信号生成，必须保证在交易时点能够获取到对应的数据。

## 服务器硬件配置
- **CPU**：Intel(R) Xeon(R) Platinum 8470Q
- **内存**：约 754 GiB 总内存
- **GPU**：单卡 NVIDIA RTX 5090 (当前可能处于无卡模式开机状态)
- **磁盘**：工作区 `/root/autodl-tmp` 挂载盘总容量约 50G

## 数据存储路径
- **分钟频数据**：`/root/autodl-tmp/min_data/` （按年份分子文件夹存放，格式为 `.fea`）
- **日频数据**：`/root/autodl-tmp/Daily_data/` （按字段名存放，格式为 `.pkl`）

## 数据格式说明

### 1. 分钟频数据 (`min_data`)
- **文件格式**：Feather (`.fea`)，通常按日或按年/日组织，例如 `20210104.fea`。
- **数据结构**：
  包含字段：
  - `date` (int64): 日期，例如 20210104
  - `time` (int64): 时间，例如 925, 931, 932
  - `StockID` (object): 股票代码，例如 SZ000001
  - `open` (float64): 分钟开盘价
  - `low` (float64): 分钟最低价
  - `high` (float64): 分钟最高价
  - `price` (float64): 分钟收盘价
  - `vol` (float64): 分钟成交量
  - `amount` (float64): 分钟成交额
- **关键时间点**：
  - `925`：代表集合竞价数据。
  - `930` - `1500`：为正常的连续交易时间。

### 2. 日频历史数据 (`Daily_data`)
- **文件格式**：Pickle (`.pkl`)，每个文件代表一个数据字段，如 `CHG_PCT.pkl`, `OPEN_PRICE.pkl` 等。
- **数据结构**：宽表（Wide DataFrame）格式。
  - `Index` (TRADE_DATE): 交易日日期索引。
  - `Columns`: 股票代码（例如 '000001', '000002'等，注意这里的代码格式可能与分钟数据中的 `SZ000001` 略有不同，需注意对其进行转换拼接）。
  - `Values` (float64): 对应的指标数值。

## ⚠️ 防未来数据红线 (CRITICAL)
在策略研发与回测中，必须严格杜绝未来数据。任何一处引入未来数据都会导致回测结果严重失真！在编写策略或因子代码时，需时刻审查以下环节：

1. **截面与时间对齐**：
   - 使用日频数据计算 T 日的交易信号时，**只能使用 T-1 日及之前的日频数据**。T 日的日频数据在 T 日盘中（9:30-15:00）是未知的（或尚未完全确定），绝不可用于 T 日的盘中交易决策。
2. **分钟数据的截断**：
   - 交易时间为 `9:30 - 15:00`。
   - 如果要在 T 日 `10:00` 产生交易信号，只能使用当天 `10:00` 及之前的分钟数据（包括 `925` 竞价数据），以及 `T-1` 日的日频数据。
   - 严禁在盘中使用当天的日频收盘数据（如当天全天的最高价、最低价或收盘价）。
3. **聚合函数的陷阱**：
   - 在计算移动平均 (MA)、标准差 (Std) 等滚动指标时，必须确保 `shift` 或取值的正确性。
   - 比如在 DataFrame 上直接使用 `df.mean()` 如果不加限制，可能会用到未来的数据，应当严格使用 `df.rolling(window).mean().shift(1)` 保证信号的滞后性。
4. **竞价数据的特殊性**：
   - 只有 `925` 数据在开盘前（9:30 前）已知。可以结合 `925` 的量价数据和 T-1 日频数据制定开盘交易计划。
5. **归一化/标准化操作**：
   - 计算截面 Z-Score 或对时间序列进行归一化时，参数（如均值和方差）只能使用历史窗口计算。绝不可以使用全局数据的均值和方差。

## 样本内外划分 (全框架统一, 定义在 config.py)
- **训练集 (Train)**: 2021-01-01 ~ 2023-08-01
- **验证集 (Validation)**: 2023-09-01 ~ 2024-09-01
- **纯样本外测试集 (OOS)**: 2024-09-01 之后 (严禁用于任何参数/特征决策)

## 框架模块结构
```
Strategy/
├── config.py               # 全局配置 (路径/样本划分/交易参数)
├── data_io/
│   ├── loader.py            # MinuteDataLoader + DailyDataLoader
│   └── saver.py             # save_wide_table (统一宽表输出)
├── label/
│   └── label_generator.py   # LabelGenerator (TWAP/VWAP/Close)
├── factor/
│   ├── factor_base.py       # FactorBase 基类 + FactorRegistry 注册器
│   └── daily_factors.py     # 具体日频因子实现
├── model/
│   ├── trainer.py           # XGBTrainer + build_panel + split_panel
│   └── scorer.py            # score_all + generate_scores
├── backtest/
│   ├── quick_backtest.py    # 20 分组分层净值回测
│   └── event_backtest.py    # 精细化事件驱动回测引擎
├── utils/
│   └── helpers.py           # 股票代码转换/交易日历/safe_rolling
└── outputs/                 # 运行产出 (labels/factors/scores/bt_results)
```

## 研发流程建议
- 新建策略脚本应统一存放在 `/root/autodl-tmp/Strategy/` 目录下。
- 数据读取应遵循增量读取或分块读取原则，避免一次性加载所有分钟数据导致内存 OOM。
- 每次特征计算完成后，必须通过随机抽查某个股票在某个特定时间点的数据，手动验证是否只用到了该时间点之前的信息。
- AI 自动生成的策略代码，使用者务必人工核对上述“防未来数据红线”。

- 添加新因子: 继承 `FactorBase`, 实现 `compute()`, 用 `@FactorRegistry.register` 装饰。
- 运行回测前需先生成 Label 宽表: `LabelGenerator().generate_and_save()`。

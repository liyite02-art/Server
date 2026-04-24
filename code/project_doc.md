# 高波趋势股日内信号策略 — 项目文档

## ⚠️ 核心交易规则（不可违反）

```
T-1 日（收盘后）：计算日频因子 → DL 模型推理 → 选出股票池 Top 300
T   日（盘中）  ：在股票池内，用分钟/逐笔信号触发买入
T+1 日（盘中）  ：以 TWAP(10:00~14:00) 卖出
```

### 严格约束

1. **股票池选择必须用 T-1 日的 DL score**
   - DL score 的特征序列包含 T-1 日的收盘价，T-1 收盘后方可计算
   - **绝对禁止**用 T 日（买入日当天）的 DL score 选池
   - 代码中：`pool = get_pool_for_date(dl, prev_date)` 而非 `get_pool_for_date(dl, buy_date)`

2. **日频因子只使用 ≤ 该日期的数据**
   - `daily_factors.py` 中所有因子使用 `rolling(backward)` / `ewm` / `shift(≥1)`
   - 禁止任何 forward-looking 的 rolling 或 shift(-N)

3. **标签（label）使用未来数据是设计意图，不是泄露**
   - `label.py` 中 `_future_rolling` 用 `shift(-1)` 取 [T+1, T+horizon] 窗口
   - 标签只用于训练，推理时不使用

4. **训练集 / OOS 标签隔离**
   - 训练 ≤ 20230731，OOS ≥ 20230901，gap ≈ 22 交易日 ≥ horizon=20 ✅

5. **分钟/逐笔信号严格因果**
   - 每日重置 EMA/KDJ（不跨日），`vol_ma5` 用 `shift(1)` 排除当前根
   - `get_first_trigger_per_stock`：每股只取首次触发（无前视选择）

6. **如果 prev_date 不存在则跳过该日，绝不回退到使用当天数据**

---

## 项目结构

```
strategy_highvol/
├── data/                  # 数据加载（daily_data, minute_data, index_data）
├── pool/                  # 股票池日频因子（daily_factors.py, concept_factors.py）
├── dl_pool/               # DL 选池模型
│   ├── label.py           # 标签构建（MFE, horizon=20）
│   ├── dataset.py         # PyTorch 时序数据集 + 截面 zscore
│   ├── model.py           # Transformer (d_model=64, n_heads=4, n_layers=2)
│   ├── train.py           # 训练循环 + early stopping
│   └── predict.py         # 推理打分
├── dl_pool_checkpoints/   # 模型权重 + 预测 CSV
│   ├── model_fold{1-8}_seed42.pt
│   ├── all_predictions.csv            # 全量 ensemble 预测（主文件）
│   ├── oos_predictions.csv            # OOS 部分（兼容）
│   └── rolling_val_predictions.csv    # 旧版单fold验证预测（保留不再使用）
├── signal/                # 日内信号层
│   ├── minute_factor_engine.py    # 分钟因子（MACD/KDJ/量能）
│   ├── minute_signal_rules.py     # 规则信号（MinuteSignalConfig）
│   ├── minute_signal_backtest.py  # 分钟信号回测框架
│   ├── tick_factor_engine.py      # 逐笔因子（polars）
│   ├── tick_label_builder.py      # 逐笔 label（T+1 TWAP）
│   ├── tick_dataset_builder.py    # 逐笔数据集构建
│   ├── minute_signal_eval/        # 信号回测产出目录
│   └── opening_analysis/          # 前5分钟特征分析（独立于backtest）
│       ├── feature_builder.py     # 构建特征面板（日频+分钟K线+逐笔）
│       ├── feature_analysis.py    # 截面分桶 + IC分析 + 可视化
│       └── results/               # 分析产出（parquet/csv/png）
├── backtest/              # 精细化回测框架（逐笔成交+LOB盘口）
│   ├── config.py          # BacktestConfig 参数配置
│   ├── portfolio.py       # 仓位管理（Position, Portfolio, 动态释放, 除权除息）
│   ├── execution.py       # 买入执行（trans信号扫描 + LOB成交模拟）
│   ├── engine.py          # 回测引擎主循环
│   ├── metrics.py         # 绩效指标 + 可视化
│   ├── run_backtest.py    # 入口脚本
│   └── results/           # 回测产出（快照/交易记录/图表）
└── notebooks/
    ├── run_dl_rolling_train.py    # DL 滚动训练（seed=42, 8 fold）
    ├── rerun_oos_inference.py     # OOS 推理（fold ensemble）
    ├── pool_analysis.py           # DL 池效果分析
    ├── run_all_backtests.py       # 信号回测 + 开盘分钟特征面板
    └── signal_backtest.ipynb      # 回测结果分析 + 分桶可视化
```

---

## 数据流

### DL 选池流水线

```
训练 (run_dl_rolling_train.py):
  OHLCV(20210101~20230901) → daily_factors(~122个) → MFE标签
  → 截面zscore → 滚动训练 8 fold (9月train + 3月val, 3月step)
  → model_fold{1-8}_seed42.pt + rolling_val_predictions.csv

推理 (rerun_oos_inference.py):
  OHLCV(20210101~20250630) → 因子 → zscore
  → 加载 8 fold 模型 → 各自全量推理 → 跨fold截面zscore均值ensemble
  → all_predictions.csv (全量, 含 segment=train/oos)
  → oos_predictions.csv (兼容)

分析 (pool_analysis.py):
  all_predictions.csv → 池效果分析
```

### 信号回测流水线

```
信号生成 (run_all_backtests.py):
  Section 1 - macd_strict:
    T-1 选池 → T日分钟K线 → MACD+KDJ双金叉+放量 → 触发买入
    → triggers_macd_strict_{train|oos}.parquet

  Section 2 - opening_ret:
    T-1 选池 → T日前5分钟(930~934) → 计算特征
    → opening_ret_panel.parquet

分析 (signal_backtest.ipynb):
  Section A: macd_strict → 指标 + 累计曲线 + 特征分桶
  Section B: opening_ret → 日频/分钟频 quintile 分桶分析
```

### 交易链路示例（以 20240225 买入日为例）

```
20240224 (T-1, 收盘后):
  ├── 特征: OHLCV ≤ 20240224 → daily_factors → 60 天序列
  ├── DL 推理 → dl_score[date=20240224]
  └── Top 300 股票池确定

20240225 (T, 买入日):
  ├── 股票池来自 dl_score[date=20240224]
  ├── macd_strict: 分钟K线扫描信号 → 触发买入
  ├── opening_ret: 前5分钟特征面板（分桶分析用）
  └── 信号触发 → 买入

20240226 (T+1, 卖出日):
  └── TWAP(10:00~14:00) 卖出
```

---

## 信号配置

### macd_strict（原 Exp2v2）

```python
MinuteSignalConfig(
    require_same_minute=True,          # 严格同根双金叉
    vol_ratio_thresh=2.0,              # 强放量
    use_prev_turnover=True,
    turnover_ratio_thresh=1.5,
    allowed_time_windows=[(930, 1130), (1430, 1450)],
    ret_from_open_min=-0.015,
    ret_from_open_max=999.0,
    pct_to_limit_min=0.066,
)
# 后过滤: kdj_j > 49 & turnover_ratio_vs_prev < 2.76
```

### opening_ret（开盘分钟特征面板）

```
面板结构: date × code × time (930~934)
特征:
  ret_vs_open         = (close - open) / open
  ret_vs_prevclose    = (close - prev_close) / prev_close
  ret_open_vs_prevclose = (open - prev_close) / prev_close  (日频)
标签:
  label_ret      = (T+1 TWAP - close) / close       (随 time 变化)
  label_ret_open = (T+1 TWAP - open) / open          (日频)
```

---

## 区间约定

| 区间 | 范围 | 说明 |
|------|------|------|
| Train | 20210101 ~ 20230731 | DL 训练 + 信号调参 |
| 空窗期 | 2023.08 | ≥ horizon=20 天 |
| OOS | 20230901 ~ 20250630 | 纯推理 + 评估 |

---

## 变更记录

### 2026-04-20: 推理统一化（方案 B）

**旧方案（可回退）：**
```
run_dl_rolling_train.py → rolling_val_predictions.csv  (样本内, 单 fold 模型推理)
rerun_oos_inference.py  → oos_predictions.csv          (OOS, 8-fold ensemble 推理)
load_dl_pools() 合并两个文件
```
问题：样本内是单 fold 打分，OOS 是 ensemble 打分，方法不一致。

**新方案：**
```
run_dl_rolling_train.py → 纯训练，产出 model + rolling_val_predictions.csv（保留但不再作为选池依据）
rerun_oos_inference.py  → 全量推理 (20210101~20250630)，8-fold ensemble
                         → all_predictions.csv (主文件, 含 segment 列)
                         → oos_predictions.csv (兼容)
load_dl_pools() 优先读 all_predictions.csv，fallback 到旧双文件模式
```

**回退方法：** 
1. 删除 `all_predictions.csv`
2. `load_dl_pools()` 会自动 fallback 到旧的 `rolling_val_predictions.csv` + `oos_predictions.csv`
3. 或直接 checkout git 历史版本

---

## 待改进 Ideas

### Idea 1: 训练集内留验证集做 early stopping

```
当前:   Train [9 months] → Val [3 months] (early stopping)
改进:   Train [8 months] → InternalVal [1 month] (early stopping) → Test [3 months] (事后评估)
```

**状态：** 📋 待实验

### Idea 2: 搭建简单精细化回测框架（⚡ 近期优先）

从 Top50 票池中验证最简单的买入策略：买入成本线设为 Open，结合逐笔成交/LOB 数据，盘口卖价低于 Open 时以 fill_rate=60% 比例成交，持续买入直到单票 300 万或总仓位超 70%。T+1 以 TWAP(10:00-14:00) 卖出。需包含仓位管理、除息除权处理（preclose vs close 区别）。

**状态：** ✅ 框架已搭建 (2026-04-21)，待运行验证

### Idea 3: 逐笔数据特征归纳（寻找规则类因子）

随机抽样若干天的 TopN 票池，按 label_ret（买入：当日 TWAP，卖出：T+1 TWAP(10:00-14:00)）排序，取 Top30/Top10 的票，分析其逐笔数据特点（成交节奏、大单分布、盘口变化等），归纳规则类因子。

数据路径：
- 买入 TWAP: `/home/user118/twap/twap_0930_1457.parquet`
- 卖出 TWAP: `/home/user118/twap/twap_1000_1400.parquet`

**状态：** 📋 待启动

### Idea 4: 特征分桶分析（优化买入信号阈值）

对候选特征进行分桶分析，检查各桶 label 是否具有单调性，据此优化买入信号阈值。

**状态：** 📋 待启动

---

## 变更记录（续）

### 2026-04-20: 卖出价格 VWAP→TWAP + 精细化回测方向

**修订内容：**

1. **卖出价格全面改为 TWAP**：框架中所有 VWAP(10:00-14:00) 卖出改为 TWAP(10:00-14:00)。原因是 VWAP 是成交量加权均价，会系统性高估收益；TWAP 更保守、更贴近均匀卖出的实际场景。
2. **买入端成交模拟**：初步回测用信号后 1 分钟 TWAP；精细化回测用逐笔 LOB 扫五档（按时间戳定点抓取，fill_rate=60%）。
3. **仓位管理两种方案**：方案 A 固定比例+时段轮动（每日最多 60% 仓位）；方案 B 动态释放+实时监控（10:00 后线性释放，14:00 全部释放）。
4. **除息除权仓位调整**：用 `adjust_ratio = close_T / preclose_{T+1}` 计算调整系数，若 ratio ≠ 1 则调整持仓股数。preclose（次日早八点出，含隔夜分股）vs close（当天收盘出，不含分股）。
5. **交易单位约束**：A 股以一手（100 股）为最小交易单位，买入股数需向下取整到 100 的整数倍（卖出可零股）。
6. **⚠️ 盘口 Vol 单位待确认**：各数据源的 Vol 可能以"手"或"股"为单位，扫五档前需确认并统一为"股"。涉及数据：tickbytick_lob、sec_lobdata、trans_fea、order_fea。
7. **新增三个近期工作方向**：精细化回测框架搭建、逐笔数据特征归纳、特征分桶分析。

详见会议纪要：`strategy_highvol/meeting_notes_20260420.md`

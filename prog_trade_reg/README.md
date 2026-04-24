# prog_trade_reg

面向「程序化交易监管」论文的 **高频计算 → 日频面板** 流水线：在逐笔成交与限价订单簿上计算微观结构指标，物化为 Parquet；连续暴露（订单流不平衡）在政策前窗口估计，与因变量一并供 DID / 事件研究使用。

---

## 1. 架构（与论文计划对齐）

### 1.1 数据流

```text
原始逐笔（trans_fea / LOB Parquet）
        ↓  单日内：LOB 按 (code,time) 聚合 → 与成交 as-of 对齐
        ↓  股票×日 聚合
派生日频层（Parquet，可断点续跑）
        ↓  合并暴露
回归面板（Polars / 统计软件读 Parquet 或再拼长表）
```

- **高频计算、低频回归**：不在全样本上反复全表扫描原始 LOB；单日任务内对原始数据尽量只读一遍，结果写入磁盘中间层（计划中的硬约束）。
- **主回归叙事（论文）**：事前连续暴露 $(\tilde{E}_i)$（代码中为 `E_oib_z`）× **施行日**之后的 `Post`，加股票与时间双向固定效应；中证 1000 / 沪深 300 等离散分组为**稳健性**，非代码默认输出。
- **制度时点**：`POLICY_NOTICE_DATE_MAIN`（发布）与 `POLICY_DATE_MAIN`（施行）在 `config` 中区分；主 `Post` 对齐 **施行日**。

### 1.2 物化目录（`DERIVED_ROOT`）

**实际路径**为 `config.DATA_ROOT` / `config.SCHEMA_VERSION`，默认：

`/home/user118/.DATA/prog_trade_reg/v1/`

（不是 `.DATA` 根目录本身；若只打开 `/home/user118/.DATA` 会看不到，须进入 `prog_trade_reg/v1/`。）

运行 `exposure` 或 `build-all` 时，终端会打印 `[paths]` 开头的绝对路径。

| 路径 | 含义 |
|------|------|
| `daily/outcomes/YYYYMMDD.parquet` | **因变量**：单日、沪深合并，`exchange` 区分市场；主键 `(exchange, code, trade_date)` |
| `meta/exposure_pre_policy.parquet` | **自变量**：每个交易日 **只读一次** `trans_fea`；**整段窗口内所有日跑完后**才一次性写出该文件（进度条走单日时磁盘上还没有此文件） |
| `meta/build_manifest.jsonl` | 构建日志（追加 JSON 行） |
| `daily/aligned_trades/` | 预留：逐笔对齐中间层 |
| `panel/` | 预留：跨日长表 |

---

## 2. 模块与 Python 文件说明

| 文件 | 职责 |
|------|------|
| `prog_trade_reg/config.py` | **单一配置源**：派生根目录、原始数据根路径、政策日、暴露窗口、批量因变量区间、`TRADE_DAYS_PKL`、成交价缩放 `TRADE_PRICE_SCALE` 等 |
| `prog_trade_reg/paths.py` | 派生路径：`outcomes_dir`、`outcome_daily_parquet_path`、`exposure_pre_policy_path`、`ensure_derived_dirs` 等 |
| `prog_trade_reg/raw_paths.py` | 原始数据路径：`trans_fea`、`lob_parquet_path`、深圳委托文件分段规则（`SZ_ORDER_SPLIT_DATE`）等 |
| `prog_trade_reg/feather_io.py` | `trans_fea`（`.fea`）统一用 `pandas.read_feather` → `polars.from_pandas`，避免压缩 IPC 在 `pl.read_ipc` 下报错 |
| `prog_trade_reg/trade_calendar.py` | 从通联风格 `trade_days_dict.pkl`（键 `trade_days`）加载交易日，并在 `[start,end]` 内筛选 |
| `prog_trade_reg/cli.py` | 命令行入口：子命令 `outcomes` / `exposure` / `outcomes-range` / `build-all` |
| `prog_trade_reg/__main__.py` | 支持 `python -m prog_trade_reg` |
| `prog_trade_reg/pipeline/outcome_daily.py` | **因变量核心**：读 `trans_fea` + 两市 LOB → 有效价差（VWES 分子分母）、`rv_mid`、`amihud` → 写单日 `daily/outcomes/` |
| `prog_trade_reg/pipeline/outcome_batch.py` | 按交易日列表批量调用 `outcome_daily`（用于 `outcomes-range` / `build-all`） |
| `prog_trade_reg/pipeline/exposure.py` | **自变量**：`compute_daily_trans_exposure_features` 单次读取 `trans_fea` 同时算 OIB、日成交笔数、总成交量等；`build_exposure_pre_policy` 对每列做时间均值与截面 z，写 `meta/exposure_pre_policy.parquet` |
| `prog_trade_reg/pipeline/build_panel.py` | **DID 长表**：合并 `daily/outcomes` 与暴露，生成 `post` 与 `post_x_E*_z`，写 `panel/panel_did_long.parquet` 与 `.csv` |
| `prog_trade_reg/pipeline/lob_agg.py` | LOB 在 `(code, time)` 上取快照（如按 `seq_num` 聚合），供 `outcome_daily` 使用 |
| `prog_trade_reg/pipeline/order_trans_agg.py` | 委托/逐笔委托相关聚合（扩展 OIB/撤单等时备用） |

---

## 3. 环境安装

```bash
cd /path/to/prog_trade_reg
pip install -e .
```

依赖见 `pyproject.toml`（Polars、PyArrow、Pandas 等）。

---

## 4. 配置（运行前必看）

编辑 `prog_trade_reg/config.py`：

- **`RAW_MDL_ROOT` / `LOB_ROOT`**：集群上 `trans_fea`、LOB Parquet 根路径。
- **`DERIVED_ROOT`**：派生数据根（含版本子目录 `v1`）。
- **`TRADE_DAYS_PKL`**：通联 `trade_days_dict.pkl` 路径；**批量任务按该日历遍历交易日**（不再使用「仅工作日」近似）。
- **`POLICY_DATE_MAIN`**：规章**施行日**（`YYYYMMDD`）；暴露窗口须**严格早于**该日。
- **`POLICY_NOTICE_DATE_MAIN`**：发布日（论文叙述/事件研究用，流水线默认不以该日切 `Post`）。
- **`EXPOSURE_PRE_START` / `EXPOSURE_PRE_END`**：连续暴露估计区间（默认至施行日前一日）。
- **`OUTCOMES_BATCH_START` / `OUTCOMES_BATCH_END`**：`outcomes-range` / `build-all` 的因变量批量区间。

---

## 5. 命令行用法

入口：`python -m prog_trade_reg <子命令>`，或安装后的 `prog-trade-reg`。

### 5.1 单日因变量

```bash
python -m prog_trade_reg outcomes 20220104
```

可选：`--exchange SZ` / `--exchange SH`（可重复传入以限定市场）。

### 5.2 政策前连续暴露（自变量，通常整段跑一次）

```bash
python -m prog_trade_reg exposure
# 或指定窗口与日历文件
python -m prog_trade_reg exposure --start 20220104 --end 20241007 --trade-days-pkl /path/to/trade_days_dict.pkl
```

### 5.3 批量因变量（区间内每个交易日）

```bash
python -m prog_trade_reg outcomes-range
# 显式区间
python -m prog_trade_reg outcomes-range --start 20220104 --end 20251231
```

默认跳过缺失原始文件的日期；若要求「缺数据即失败」：加 `--fail-on-missing`。

### 5.4 一键：暴露 + 批量因变量

```bash
python -m prog_trade_reg build-all
```

可选：`--skip-exposure`、`--skip-outcomes` 只跑其中一段；`--outcomes-start` / `--outcomes-end`、`--exposure-start` / `--exposure-end` 覆盖默认配置。

### 5.5 合并 DID 长面板（Python / Stata）

需已存在 `daily/outcomes/*.parquet` 与 `meta/exposure_pre_policy.parquet`（先跑 `exposure` 与 `outcomes-range`）。

```bash
python -m prog_trade_reg panel-did
```

写出：

- `panel/panel_did_long.parquet`（推荐 Polars / pandas）
- `panel/panel_did_long.csv`（分隔符 `|`，便于 Stata `import delimited`）

列含：因变量原列、`post`（施行日及之后为 1）、`policy_date_main`、`panel_unit`（`exchange_code`）、`vw_es_bps`，以及暴露列与 **`post_x_E_oib_z`** 等交互项（主回归可直接用 `post_x_E_oib_z` 或自行 `post * E_oib_z`）。

可选：`--policy-date YYYYMMDD`、`--exposure /path/to/exposure_pre_policy.parquet`、`--no-csv` / `--no-parquet`。

**Stata 示例**（需先把日期转成 Stata 日度）：

```stata
import delimited "panel_did_long.csv", delimiter("|") clear
gen td = daily(trade_date, "YMD")
format td %td
* 面板键：panel_unit × td；双向 FE + 聚类见论文方法章
```

---

## 6. 输出列（摘要）

**`daily/outcomes/*.parquet`**（与计划一致）：`code`, `trade_date`, `exchange`, `n_trades`, `volume_sum`, `dollar_vol`, `es_bps_num`, `es_bps_den`, `rv_mid`, `amihud`；成交量加权有效价差（基点）为 `es_bps_num / es_bps_den`。

**`meta/exposure_pre_policy.parquet`**（摘要）：`exchange`, `code`, `n_days`；OIB 与成交强度三类各含 `*_mean_raw`、winsor 后 `*_mean`、标准化 `E_*_z`——**主回归暴露**为 `E_oib_z`；`E_n_trades_z`、`E_total_vol_z` 可作控制或异质性。更多日度指标可在 `compute_daily_trans_exposure_features` 的同一 `group_by` 内扩展，无需重复读 feather。

---

## 7. 注意事项

- **算力与时间**：全样本跨年批量因变量耗时长，建议在 `screen`/`tmux` 或作业系统里跑，并关注磁盘空间。
- **交易日历**：批量逻辑依赖 `TRADE_DAYS_PKL`；请与通联（或你方 `get_trade_days`）日历一致。
- **损坏或非标准 `trans_fea`**：若某日 `.fea` 不是合法 Feather/Arrow（如 `Not an Arrow file`），默认 **`outcomes-range` 会跳过该日**（计入 `skipped_unreadable`），不中断整段任务；会先尝试 pandas Feather，再尝试 Polars IPC。修复源文件后可单独对那一日再跑 `outcomes YYYYMMDD`。**续跑**：对未完成的区间再执行 `outcomes-range --start <下一日>`（已存在的 parquet 会被覆盖重写）。
- **论文扩展**：指数成分、宏观控制、回归与事件研究 Stata/R 脚本不在本仓库核心路径内；本仓库负责**可复现的日频与暴露物化层**。

---

## 8. 相关文档

更完整的制度叙事、识别策略与公式（LaTeX）见项目外的论文数据规格计划（如 Cursor plan：`论文第一步数据规格`）；本 README 仅描述**代码架构与用法**，与之一致处以本仓库 `config` 与 `paths` 为准。

## 9. 连续暴露 × 政策（连续 DID / TWFE）方法论文献（写作引用）

以下英文文献常用于为「**事前连续测度 \(\tilde E_i\) × Post** + **股票与时间双向固定效应**」提供计量依据；请在 Google Scholar 核对卷期页码后写入 `.bib`。

1. **de Chaisemartin, C., & d’Haultfoeuille, X. (2020).** “Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects.” *American Economic Review*.（TWFE 下异质性处理效应与交互项解释的经典警示与讨论。）

2. **Callaway, B., & Sant’Anna, P. H. C. (2021).** “Difference-in-Differences with Multiple Time Periods.” *Journal of Econometrics*.（多期 DID、事件研究框架；可与连续暴露的动态图对照。）

3. **Sun, L., & Abraham, S. (2021).** “Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects.” *Journal of Econometrics*.（事件研究与异质性；用于平行趋势与动态规格表述。）

4. **Borusyak, K., Jaravel, X., & Spiess, J. (2024).** “Revisiting Event Study Designs: Robust and Efficient Estimation.” *Quarterly Journal of Economics*.（事件研究/插补思路；稳健性引用。）

5. **Rambachan, A., & Roth, J. (2023).** “A More Credible Approach to Parallel Trends.” *Review of Economic Studies*.（对平行趋势假设的敏感性分析；连续暴露主规格可与之配套。）

6. **教材与综述**：**Wooldridge, J. M.** *Econometric Analysis of Cross Section and Panel Data*（或更新版）中关于 **面板固定效应与交互项** 的章节；**Angrist & Pischke** *Mostly Harmless Econometrics* 对 **DID 与交互** 的直觉说明。中文写作可辅以 **陈强《高级计量经济学及 Stata 应用》** 等教材中 DID/面板章节，但**顶刊引用仍以英文原典为主**。

**写作提示**：正文可表述为在双向固定效应模型中加入 **政策后虚拟变量与事前连续暴露的交互项**，并引用 (1)(2) 讨论 TWFE 解释与多期拓展；(5) 用于平行趋势敏感性；(3)(4) 用于事件研究或稳健性规格。

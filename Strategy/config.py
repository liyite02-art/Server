"""
全局配置: 数据路径、样本划分、交易参数、存储规范等。
所有模块统一引用此配置，确保一致性。
"""
from pathlib import Path
import datetime as dt

# ── 根目录 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/root/autodl-tmp/Strategy")
DATA_ROOT = Path("/root/autodl-tmp")

# ── 原始数据路径 ────────────────────────────────────────────────────────
MIN_DATA_DIR = DATA_ROOT / "min_data"          # 分钟频: {year}/{YYYYMMDD}.fea
DAILY_DATA_DIR = DATA_ROOT / "Daily_data"      # 日频:   {FIELD}.pkl

# ── 产出目录 ────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LABEL_OUTPUT_DIR = OUTPUT_DIR / "labels"
FACTOR_OUTPUT_DIR = OUTPUT_DIR / "factors"
SCORE_OUTPUT_DIR = OUTPUT_DIR / "scores"
BT_RESULT_DIR = OUTPUT_DIR / "bt_results"

# pipeline_train_score_backtest.ipynb：OOS 冻结、仅用 train∪val 滚动训练时的产出目录。
# 与 main.ipynb 在 SCORE_OUTPUT_DIR 根目录下直接落盘 SCORE_xgb_*.fea、rolling_model.pkl 等区分，避免覆盖。
PIPELINE_HOLDOUT_SCORE_DIR = SCORE_OUTPUT_DIR / "pipeline_holdout"
PIPELINE_HOLDOUT_BT_DIR = BT_RESULT_DIR / "pipeline_holdout"

# ── 样本内外划分 (基于 Rolling Val CV 的 IS/OOS 三段) ─────────────────
#
# IS Train Set: 用于 Rolling Val CV 内部训练与验证 (Val 仅用于早停)
# IS Test Set:  不参与梯度更新；用 4-Fold Ensemble 评估泛化效果
# OOS:          严禁用于任何参数决策；仅当 IS Test 达标后才允许推进
#
IS_TRAIN_START = dt.date(2021, 1, 1)
IS_TRAIN_END   = dt.date(2023, 9, 30)   # IS 训练集闭区间上界 (含)

IS_TEST_START  = dt.date(2023, 10, 1)
IS_TEST_END    = dt.date(2024, 9, 1)    # IS 测试集闭区间上界 (含)

OOS_START      = dt.date(2024, 9, 1)    # 纯样本外起始 (含); 严禁用于任何参数决策

# 历史命名 / ic_analysis 分段：与上列 IS 区间一一对应（「Val」= 样本内测试集 IS Test，非滚动 CV 的 val）
TRAIN_START    = IS_TRAIN_START
TRAIN_END      = IS_TRAIN_END
VAL_START      = IS_TEST_START
VAL_END        = IS_TEST_END

# ── 滚动训练参数 ────────────────────────────────────────────────────────
ROLLING_VAL_MONTHS = 3            # 每个 Fold 验证窗口月数 (季度)
ENSEMBLE_N_FOLDS   = 4            # IS Test 推理时选取最近 N 个 Fold 做集成

# 截面神经网络 (MLP / Transformer): batch_size 表示「每个优化步合并的交易日数」。
# strategy_rules.md 要求 batch_size=1（单日截面、日期间无批量耦合）。
NN_TRAINER_BATCH_SIZE = 1

# ── Label 预处理 ────────────────────────────────────────────────────────
LABEL_WINSORIZE_SIGMA = 0.0       # 截面 Winsorize 阈值（σ 倍数）, 0 = 不做

# 日频因子落盘 (DailyFactorLibraryAdapter.compute_and_save_all)
# 特征（因子）在验证集、样本外预测与回测中仍然需要，与「标签是否用于训练」无关。
# 只要不对 OOS 做调参/选模，在 OOS 上计算因子不属于标签泄露。建议:
#   start_date = IS_TRAIN_START（或数据起点）, end_date = None（Daily_data 末交易日，含 IS_Test/OOS）
# 仅当调试或极省算力时，才可收窄为 end_date=IS_TRAIN_END，此时 is_test/oos/打分会缺行或不可用。

# ── 交易时间常量 (分钟频 time 字段为 int, 如 925, 930, 1500) ───────────
AUCTION_TIME = 925               # 集合竞价
TRADE_START = 930                # 连续交易开始
MORNING_END = 1130               # 上午收盘
AFTERNOON_START = 1301           # 下午开盘
TRADE_END = 1500                 # 收盘

# ── Label 默认参数 ──────────────────────────────────────────────────────
DEFAULT_TWAP_START = 1430
DEFAULT_TWAP_END = 1457

# ── 回测参数 ────────────────────────────────────────────────────────────
COMMISSION_RATE = 0.0001         # 双向佣金 万一
STAMP_DUTY_RATE = 0.0005         # 卖出印花税 万五
SLIPPAGE_BPS = 0.0              # 默认滑点 (bps), 0 = 无滑点
# 与分层回测 / 事件回测「对齐」时的默认本金 (1000 万)
INITIAL_CAPITAL = 10_000_000.0

# ── 分组回测参数 ────────────────────────────────────────────────────────
N_QUANTILE_GROUPS = 20           # 分层回测组数

# ── 日频数据可用字段 ────────────────────────────────────────────────────
DAILY_FIELDS = [
    "CLOSE_PRICE", "OPEN_PRICE", "HIGHEST_PRICE", "LOWEST_PRICE",
    "PRE_CLOSE_PRICE", "CHG_PCT", "DEAL_AMOUNT", "TURNOVER_RATE",
    "TURNOVER_VALUE", "MARKET_VALUE", "LIMIT_UP_PRICE", "LIMIT_DOWN_PRICE",
    "VOLUME",   # 由 generate_volume.py 从分钟数据合成
]
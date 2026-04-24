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

# ── 样本内外划分 ────────────────────────────────────────────────────────
TRAIN_START = dt.date(2021, 1, 1)
TRAIN_END = dt.date(2023, 8, 1)

VAL_START = dt.date(2023, 9, 1)
VAL_END = dt.date(2024, 9, 1)

OOS_START = dt.date(2024, 9, 1)  # 纯样本外起始，严禁用于任何参数决策

# 日频因子落盘 (DailyFactorLibraryAdapter.compute_and_save_all)
# 特征（因子）在验证集、样本外预测与回测中仍然需要，与「标签是否用于训练」无关。
# 只要不对 OOS 做调参/选模，在 OOS 上计算因子不属于标签泄露。建议:
#   start_date = TRAIN_START（或数据起点）, end_date = None（Daily_data 末交易日，含 VAL/OOS）
# 仅当调试或极省算力时，才可收窄为 end_date=TRAIN_END，此时 val/oos/打分会缺行或不可用。

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
COMMISSION_RATE = 0.0002         # 双向佣金 万二
STAMP_DUTY_RATE = 0.0005         # 卖出印花税 万五
SLIPPAGE_BPS = 0.0              # 默认滑点 (bps), 0 = 无滑点
INITIAL_CAPITAL = 10_000_000.0   # 初始资金 1000 万

# ── 分组回测参数 ────────────────────────────────────────────────────────
N_QUANTILE_GROUPS = 20           # 分层回测组数

# ── 日频数据可用字段 ────────────────────────────────────────────────────
DAILY_FIELDS = [
    "CLOSE_PRICE", "OPEN_PRICE", "HIGHEST_PRICE", "LOWEST_PRICE",
    "PRE_CLOSE_PRICE", "CHG_PCT", "DEAL_AMOUNT", "TURNOVER_RATE",
    "TURNOVER_VALUE", "MARKET_VALUE", "LIMIT_UP_PRICE", "LIMIT_DOWN_PRICE",
    "VOLUME",   # 由 generate_volume.py 从分钟数据合成
]

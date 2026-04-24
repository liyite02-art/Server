from __future__ import annotations

from pathlib import Path

# Derived artifacts (single source of truth for on-disk layout)
DATA_ROOT: Path = Path("/home/user118/.DATA/prog_trade_reg")
SCHEMA_VERSION: str = "v1"
DERIVED_ROOT: Path = DATA_ROOT / SCHEMA_VERSION

# Raw inputs (cluster paths; override with env if needed)
RAW_MDL_ROOT: Path = Path("/home/lwyxyz/2.79/ftp/mdl_fea")
LOB_ROOT: Path = Path("/home/lwyxyz/253.118/lob_data")

# Shenzhen order file naming changes at this calendar date (YYYYMMDD)
SZ_ORDER_SPLIT_DATE: str = "20230821"

# trans_fea tradePrice is stored as integer; match to LOB float yuan (see empirical check)
TRADE_PRICE_SCALE: float = 10000.0

# DID: regulation effective date (施行, YYYYMMDD). Exposure E_i must end strictly before this.
POLICY_DATE_MAIN: str = "20241008"
# Same regulation: notice / publication date (发布); use for anticipation or secondary event study
POLICY_NOTICE_DATE_MAIN: str = "20240511"

# Pre-policy window for time-invariant continuous exposure E_i (default; override in CLI)
EXPOSURE_PRE_START: str = "20220104"
EXPOSURE_PRE_END: str = "20241007"

# Batch daily outcomes: dates in [OUTCOMES_BATCH_START, OUTCOMES_BATCH_END]
OUTCOMES_BATCH_START: str = "20220104"
OUTCOMES_BATCH_END: str = "20251231"

# Tonglian ``trade_days_dict.pkl`` (dict key ``trade_days``). Required for batch date iteration.
TRADE_DAYS_PKL: Path = Path(
    "/home/lwyxyz/2.79/tonglian_data/support_data/trade_days_dict.pkl"
)

from __future__ import annotations

from pathlib import Path

from prog_trade_reg.config import DERIVED_ROOT


def meta_dir() -> Path:
    return DERIVED_ROOT / "meta"


def build_manifest_path() -> Path:
    return meta_dir() / "build_manifest.jsonl"


def outcomes_dir() -> Path:
    """Daily dependent-variable panel: ``daily/outcomes/{YYYYMMDD}.parquet`` (SZ+SH, column ``exchange``)."""
    return DERIVED_ROOT / "daily" / "outcomes"


def outcome_daily_parquet_path(trade_date: str) -> Path:
    """Single Parquet per calendar day: both exchanges, distinguished by ``exchange`` column."""
    return outcomes_dir() / f"{trade_date}.parquet"


def aligned_trades_path(trade_date: str, exchange: str) -> Path:
    ex = exchange.upper()
    if ex not in {"SZ", "SH"}:
        raise ValueError("exchange must be SZ or SH")
    return DERIVED_ROOT / "daily" / "aligned_trades" / f"{trade_date}_{ex.lower()}.parquet"


def exposure_dir() -> Path:
    return DERIVED_ROOT / "daily" / "exposure"


def exposure_pre_policy_path() -> Path:
    """Time-invariant continuous exposure (policy-window mean OIB + z-score)."""
    return meta_dir() / "exposure_pre_policy.parquet"


def panel_dir() -> Path:
    return DERIVED_ROOT / "panel"


def panel_did_long_parquet_path() -> Path:
    return panel_dir() / "panel_did_long.parquet"


def panel_did_long_csv_path() -> Path:
    return panel_dir() / "panel_did_long.csv"


def ensure_derived_dirs() -> None:
    meta_dir().mkdir(parents=True, exist_ok=True)
    outcomes_dir().mkdir(parents=True, exist_ok=True)
    exposure_dir().mkdir(parents=True, exist_ok=True)
    (DERIVED_ROOT / "daily" / "aligned_trades").mkdir(parents=True, exist_ok=True)
    panel_dir().mkdir(parents=True, exist_ok=True)

"""Backtest universe filters based on information known on trade date."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from Strategy import config

logger = logging.getLogger(__name__)


def standardize_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TRADE_DATE" in out.columns:
        out = out.set_index("TRADE_DATE")
    out.index = pd.DatetimeIndex(out.index).normalize()
    out.columns = pd.Index([str(c).zfill(6) for c in out.columns])
    return out


def load_daily_wide(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("股票池过滤数据不存在: %s", path)
        return None
    try:
        return standardize_wide(pd.read_pickle(path))
    except Exception as exc:
        logger.warning("加载股票池过滤数据失败 %s: %s", path, exc)
        return None


def load_ipo_dates(path: Optional[Path] = None) -> Optional[pd.Series]:
    path = path or (config.DAILY_DATA_DIR / "ipo_dates.pkl")
    if not path.exists():
        logger.warning("IPO 日期文件不存在: %s", path)
        return None
    try:
        df = pd.read_pickle(path)
        if not {"TICKER_SYMBOL", "INTO_DATE"}.issubset(df.columns):
            raise ValueError("ipo_dates.pkl 缺少 TICKER_SYMBOL 或 INTO_DATE 列")
        codes = df["TICKER_SYMBOL"].map(lambda x: str(x).zfill(6))
        dates = pd.to_datetime(df["INTO_DATE"], errors="coerce").dt.normalize()
        out = pd.Series(dates.values, index=codes, name="ipo_date")
        return out[~out.index.duplicated(keep="last")]
    except Exception as exc:
        logger.warning("加载 IPO 日期失败: %s", exc)
        return None


def load_out_dates(path: Optional[Path] = None) -> Optional[pd.Series]:
    path = path or (config.DAILY_DATA_DIR / "ipo_dates.pkl")
    if not path.exists():
        logger.warning("IPO 日期文件不存在: %s", path)
        return None
    try:
        df = pd.read_pickle(path)
        if not {"TICKER_SYMBOL", "OUT_DATE"}.issubset(df.columns):
            raise ValueError("ipo_dates.pkl 缺少 TICKER_SYMBOL 或 OUT_DATE 列")
        codes = df["TICKER_SYMBOL"].map(lambda x: str(x).zfill(6))
        dates = pd.to_datetime(df["OUT_DATE"], errors="coerce").dt.normalize()
        out = pd.Series(dates.values, index=codes, name="out_date")
        return out[~out.index.duplicated(keep="last")]
    except Exception as exc:
        logger.warning("加载退市日期失败: %s", exc)
        return None


def load_st_status(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    path = path or (config.DAILY_DATA_DIR / "st_status.pkl")
    return load_daily_wide(path)


def listing_age_mask(
    dates: pd.Index,
    columns: pd.Index,
    ipo_dates: Optional[pd.Series],
    min_listing_days: int = 20,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates).normalize()
    columns = pd.Index([str(c).zfill(6) for c in columns])
    if ipo_dates is None:
        return pd.DataFrame(True, index=dates, columns=columns)

    aligned = ipo_dates.reindex(columns)
    ipo_values = pd.to_datetime(aligned).to_numpy(dtype="datetime64[ns]")
    date_values = dates.to_numpy(dtype="datetime64[ns]")[:, None]
    listed_long_enough = date_values > (ipo_values + np.timedelta64(min_listing_days, "D"))
    listed_long_enough[:, pd.isna(aligned).to_numpy()] = False
    return pd.DataFrame(listed_long_enough, index=dates, columns=columns)


def out_date_mask(
    dates: pd.Index,
    columns: pd.Index,
    out_dates: Optional[pd.Series],
    delist_buffer_days: int = 20,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates).normalize()
    columns = pd.Index([str(c).zfill(6) for c in columns])
    if out_dates is None:
        return pd.DataFrame(True, index=dates, columns=columns)

    aligned = pd.to_datetime(out_dates.reindex(columns), errors="coerce")
    out_values = aligned.to_numpy(dtype="datetime64[ns]")
    date_values = dates.to_numpy(dtype="datetime64[ns]")[:, None]
    cutoff = out_values - np.timedelta64(delist_buffer_days, "D")
    ok = pd.isna(aligned).to_numpy()[None, :] | (date_values < cutoff)
    return pd.DataFrame(ok, index=dates, columns=columns)


def st_mask(
    dates: pd.Index,
    columns: pd.Index,
    st_status: Optional[pd.DataFrame],
    historical: bool = True,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates).normalize()
    columns = pd.Index([str(c).zfill(6) for c in columns])
    if st_status is None:
        return pd.DataFrame(True, index=dates, columns=columns)
    st = standardize_wide(st_status).groupby(level=0).max().sort_index()
    st_flag = st.fillna(0).ne(0)
    if historical:
        st_flag = st_flag.cummax()
    st_flag = st_flag.reindex(index=dates, method="ffill").reindex(columns=columns).fillna(False)
    return ~st_flag


def prefix_mask(
    dates: pd.Index,
    columns: pd.Index,
    excluded_prefixes: Optional[tuple[str, ...]] = ("300", "688"),
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates).normalize()
    columns = pd.Index([str(c).zfill(6) for c in columns])
    if not excluded_prefixes:
        return pd.DataFrame(True, index=dates, columns=columns)
    excluded = columns.to_series(index=columns).str.startswith(excluded_prefixes).to_numpy()
    ok = np.broadcast_to(~excluded, (len(dates), len(columns))).copy()
    return pd.DataFrame(ok, index=dates, columns=columns)


def build_universe_filter(
    dates: pd.Index,
    columns: pd.Index,
    ipo_dates: Optional[pd.Series] = None,
    out_dates: Optional[pd.Series] = None,
    st_status: Optional[pd.DataFrame] = None,
    min_listing_days: int = 20,
    delist_buffer_days: int = 20,
    exclude_st: bool = True,
    exclude_historical_st: bool = True,
    excluded_prefixes: Optional[tuple[str, ...]] = ("300", "688"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(dates).normalize()
    columns = pd.Index([str(c).zfill(6) for c in columns])
    age_ok = listing_age_mask(dates, columns, ipo_dates, min_listing_days=min_listing_days)
    delist_ok = out_date_mask(dates, columns, out_dates, delist_buffer_days=delist_buffer_days)
    st_ok = st_mask(dates, columns, st_status, historical=exclude_historical_st) if exclude_st else pd.DataFrame(True, index=dates, columns=columns)
    prefix_ok = prefix_mask(dates, columns, excluded_prefixes=excluded_prefixes)
    universe = age_ok & delist_ok & st_ok & prefix_ok

    report = pd.DataFrame(index=dates)
    report.index.name = "TRADE_DATE"
    report["excluded_new_stock"] = (~age_ok).sum(axis=1)
    report["excluded_pre_delist"] = (age_ok & ~delist_ok).sum(axis=1)
    report["excluded_st"] = (age_ok & delist_ok & ~st_ok).sum(axis=1)
    report["excluded_prefix"] = (age_ok & delist_ok & st_ok & ~prefix_ok).sum(axis=1)
    report["universe_ok"] = universe.sum(axis=1)
    return universe, report

from __future__ import annotations

from pathlib import Path

from prog_trade_reg.config import LOB_ROOT, RAW_MDL_ROOT, SZ_ORDER_SPLIT_DATE


def lob_parquet_path(exchange: str, trade_date: str) -> Path:
    """Daily LOB parquet: lob_data_sz / lob_data_sh / YYYYMMDD.parquet"""
    ex = exchange.upper()
    if ex == "SZ":
        sub = "lob_data_sz"
    elif ex == "SH":
        sub = "lob_data_sh"
    else:
        raise ValueError("exchange must be SZ or SH")
    return LOB_ROOT / sub / f"{trade_date}.parquet"


def trans_fea_path(trade_date: str) -> Path:
    """Merged trades feather: trans_fea/YYYYMMDD_trans.fea"""
    return RAW_MDL_ROOT / "trans_fea" / f"{trade_date}_trans.fea"


def order_path_sz(trade_date: str) -> Path:
    """Shenzhen orders: filename rule changes at SZ_ORDER_SPLIT_DATE."""
    if trade_date <= SZ_ORDER_SPLIT_DATE:
        return RAW_MDL_ROOT / "order_fea" / f"{trade_date}_order.fea"
    return RAW_MDL_ROOT / "order_fea" / f"{trade_date}_order_sz.fea"


def order_path_sh(trade_date: str) -> Path:
    return RAW_MDL_ROOT / "order_fea" / f"{trade_date}_order_sh2.fea"


def order_trans_sz_path(trade_date: str) -> Path:
    """Shenzhen order / trade stream (feather): order_trans_sz/YYYYMMDD.fea"""
    return RAW_MDL_ROOT / "order_trans_sz" / f"{trade_date}.fea"


def order_trans_sh_path(trade_date: str) -> Path:
    """Shanghai order / trade stream (feather)."""
    return RAW_MDL_ROOT / "order_trans_sh" / f"{trade_date}.fea"


def order_trans_sz_legacy_path(trade_date: str) -> Path:
    """Alias for order_trans_sz_path (legacy name)."""
    return order_trans_sz_path(trade_date)

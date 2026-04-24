"""
通用工具函数: 股票代码转换、交易日历、安全滚动计算等。
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import numpy as np

from Strategy import config


# ═══════════════════════════════════════════════════════════════════════
# 股票代码转换
# ═══════════════════════════════════════════════════════════════════════
_PREFIX_RE = re.compile(r"^(SZ|SH|BJ|NE)")

# 北交所前缀, 这些股票无对应数据, 需排除
_EXCLUDED_EXCHANGE_PREFIXES = ("BJ", "NE")


def strip_stock_prefix(code: str) -> str:
    """SZ000001 -> 000001, SH600000 -> 600000"""
    return _PREFIX_RE.sub("", code)


def is_sh_or_sz(code: str) -> bool:
    """判断原始 StockID 是否属于上交所 / 深交所 (排除北交所 BJ/NE)"""
    return not code.startswith(_EXCLUDED_EXCHANGE_PREFIXES)


def is_sh_or_sz_by_num(code: str) -> bool:
    """判断 6 位纯数字代码是否属于沪深 (0/3/6 开头), 排除北交所 (4/8 开头)"""
    return code[:1] in ("0", "3", "6")


def add_stock_prefix(code: str) -> str:
    """000001 -> SZ000001, 600000 -> SH600000, 8xxxxx -> BJ8xxxxx"""
    c = str(code).zfill(6)
    if c.startswith(("0", "3")):
        return f"SZ{c}"
    elif c.startswith("6"):
        return f"SH{c}"
    elif c.startswith(("4", "8")):
        return f"BJ{c}"
    return c


def standardize_stock_column(columns: pd.Index) -> pd.Index:
    """确保列名为 6 位纯数字字符串"""
    return pd.Index([str(c).zfill(6) for c in columns])


def normalize_tradedate_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    将宽表行索引 (TRADE_DATE) 统一为 ``DatetimeIndex`` (与 ``pd.Timestamp`` 可比较).

    部分 feather 读入后索引为 ``datetime.date`` 的 object 索引, 而 label 为
    ``DatetimeIndex``; 则 ``.intersection`` 无公共元素, ``build_panel`` 得到
    0 行, 进而 ``split_panel`` 报 TRADE_DATE 为 NaT。加载宽表后应始终调用本函数
    或在 ``build_panel`` 入口对 label/因子统一一次。
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out


def ensure_tradedate_as_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    将宽表规范为「TRADE_DATE 在行索引, 且为时间类型」.

    ``save_wide_table`` 落盘为 ``reset_index().to_feather``, 故 .fea 中
    **TRADE_DATE 是一列**; 必须 ``set_index('TRADE_DATE')`` 后才是可聚合的
    宽表。若仅 ``read_feather`` 而未提索引, 会保留 **默认 RangeIndex**,
    TRADE_DATE 仍留在列中, 与 label 的日期索引做 ``intersection`` 得到空/错位。

    对已从索引正确加载的表, 若列中无 ``TRADE_DATE`` 则仅做 ``normalize``。
    """
    out = df.copy()
    if "TRADE_DATE" in out.columns:
        out = out.set_index("TRADE_DATE", drop=True)
    return normalize_tradedate_index(out)


# ═══════════════════════════════════════════════════════════════════════
# 交易日历
# ═══════════════════════════════════════════════════════════════════════
_TRADE_DATES_CACHE: Optional[pd.DatetimeIndex] = None


def get_trade_dates() -> pd.DatetimeIndex:
    """从日频数据推断交易日历 (使用 CLOSE_PRICE 的 index)"""
    global _TRADE_DATES_CACHE
    if _TRADE_DATES_CACHE is None:
        close = pd.read_pickle(config.DAILY_DATA_DIR / "CLOSE_PRICE.pkl")
        _TRADE_DATES_CACHE = pd.DatetimeIndex(close.index).sort_values()
    return _TRADE_DATES_CACHE


def filter_trade_dates(
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
) -> pd.DatetimeIndex:
    """返回 [start, end] 范围内的交易日"""
    dates = get_trade_dates()
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    return dates


def date_int_to_str(d: int) -> str:
    """20210104 -> '2021-01-04'"""
    return f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"


def date_to_int(d) -> int:
    """datetime / Timestamp / str -> 20210104"""
    if isinstance(d, (dt.date, dt.datetime, pd.Timestamp)):
        return d.year * 10000 + d.month * 100 + d.day
    if isinstance(d, str):
        d = d.replace("-", "")
        return int(d)
    return int(d)


# ═══════════════════════════════════════════════════════════════════════
# 分钟频时间工具
# ═══════════════════════════════════════════════════════════════════════
def get_minute_files(
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
) -> List[Path]:
    """返回指定日期范围内的分钟数据文件路径列表 (已排序)"""
    files = []
    for year_dir in sorted(config.MIN_DATA_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        for fpath in sorted(year_dir.glob("*.fea")):
            file_date = int(fpath.stem)
            if start_date is not None and file_date < start_date:
                continue
            if end_date is not None and file_date > end_date:
                continue
            files.append(fpath)
    return files


# ═══════════════════════════════════════════════════════════════════════
# 防未来数据工具
# ═══════════════════════════════════════════════════════════════════════
def safe_rolling(
    df: pd.DataFrame,
    window: int,
    func: str = "mean",
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    安全的滚动计算: 自动 shift(1) 确保 T 日因子只用到 T-1 及之前的数据。
    ⚠️ 返回值已经做了 shift(1), 调用方无需再 shift。
    """
    roller = df.rolling(window=window, min_periods=min_periods)
    result = getattr(roller, func)()
    return result.shift(1)

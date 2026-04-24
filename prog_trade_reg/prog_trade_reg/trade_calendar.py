"""Exchange trading calendar from a Tonglian-style pickle (key ``trade_days``)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _to_yyyymmdd(x: object) -> str:
    if hasattr(x, "strftime"):
        return x.strftime("%Y%m%d")  # type: ignore[no-any-return]
    s = str(x).strip()
    if len(s) >= 10 and s[4] == "-":
        return s[:4] + s[5:7] + s[8:10]
    return s[:8]


def load_trade_days_from_pickle(path: Path) -> list[str]:
    """
    Load sorted ``YYYYMMDD`` strings from ``pd.read_pickle(path)['trade_days']``.

    Expects the same layout as ``trade_days_dict.pkl`` (通联 support_data).
    """
    if not path.is_file():
        raise FileNotFoundError(f"trade calendar pickle not found: {path}")
    obj = pd.read_pickle(path)
    if isinstance(obj, dict) and "trade_days" in obj:
        raw = obj["trade_days"]
    else:
        raise ValueError(f"pickle must be a dict with key 'trade_days', got {type(obj)}")
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    out = [_to_yyyymmdd(t) for t in raw]
    out = sorted(set(out))
    return out


def trade_days_in_range(
    start: str,
    end: str,
    all_trade_days: Iterable[str],
) -> list[str]:
    """Filter ``all_trade_days`` to ``start <= d <= end`` (string order matches calendar dates)."""
    return [d for d in all_trade_days if start <= d <= end]


def iter_dates_in_range(start: str, end: str, *, trade_days_pkl: Path) -> tuple[list[str], str]:
    """
    Return trading dates in ``[start, end]`` from the pickle calendar.

    Parameters
    ----------
    trade_days_pkl
        Path to ``trade_days_dict.pkl`` (must exist).
    """
    all_days = load_trade_days_from_pickle(trade_days_pkl)
    days = trade_days_in_range(start, end, all_days)
    return days, "trade_days_pkl"

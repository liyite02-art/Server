"""
Minute K-line factors adapted from ``Lob_tickbytick_base6_lyt.py``.

This module only uses the raw minute K-line files under ``config.MIN_DATA_DIR``.
Each trading day is filtered to the 931-1457 minute bars, each stock keeps only
the final 1457 factor snapshot, and each factor is saved as a daily-style wide
table with ``_min`` appended to the factor name.
"""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from Strategy import config
from Strategy.data_io.saver import save_wide_table
from Strategy.utils.helpers import date_to_int, get_minute_files, strip_stock_prefix

logger = logging.getLogger(__name__)

EPS = 1e-9
eps = 1e-5
TARGET_TIME = 1457


def minute_time_grid() -> List[int]:
    """A-share minute K-line grid from 09:31 to 14:57."""
    morning = list(range(931, 960)) + list(range(1000, 1060)) + list(range(1100, 1131))
    afternoon = list(range(1301, 1360)) + list(range(1400, 1458))
    return morning + afternoon


MINUTE_GRID = minute_time_grid()


def _is_sh_or_sz_code(code: str) -> bool:
    return code[:1] in ("0", "3", "6")


def _rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=2).corr(y).fillna(0.0)


def _compute_aroon(high: pd.Series, low: pd.Series, period: int = 25):
    high_idx = high.rolling(period).apply(lambda x: period - 1 - x.argmax(), raw=True)
    low_idx = low.rolling(period).apply(lambda x: period - 1 - x.argmin(), raw=True)
    aroon_up = 100 * (period - high_idx) / period
    aroon_down = 100 * (period - low_idx) / period
    return aroon_up, aroon_down


def _align_minute_kline(group: pd.DataFrame) -> pd.DataFrame:
    """Normalize one stock-day to the 237-bar 931-1457 grid."""
    g = group.sort_values("time").copy()
    g = g[(g["time"] >= 931) & (g["time"] <= TARGET_TIME)]

    if g.empty:
        return pd.DataFrame()

    g = (
        g.groupby("time", as_index=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("price", "last"),
            volume=("vol", "sum"),
            amount=("amount", "sum"),
        )
        .reindex(MINUTE_GRID)
    )
    g.index.name = "time"

    for col in ["open", "high", "low", "close"]:
        g[col] = pd.to_numeric(g[col], errors="coerce").replace(0, np.nan)
    g["close"] = g["close"].ffill()
    for col in ["open", "high", "low"]:
        g[col] = g[col].fillna(g["close"])
    g["volume"] = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0)
    g["amount"] = pd.to_numeric(g["amount"], errors="coerce").fillna(0.0)

    g = g.reset_index()
    return g


def _last_value(df: pd.DataFrame, col: str):
    if df.empty or col not in df.columns:
        return np.nan
    s = df.loc[df["time"] == TARGET_TIME, col]
    if s.empty:
        return np.nan
    return s.iloc[-1]


def _var_factors(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    open_safe = df["open"].where(df["open"] > 0)
    close_safe = df["close"].where(df["close"] > 0)
    log_r = (np.log(close_safe / open_safe) * 100).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    r2 = log_r ** 2

    work = pd.DataFrame({"intraday_log_r": log_r, "r_2": r2})
    work["RV"] = work["r_2"].cumsum()
    work["V_hat"] = (
        np.abs(work["intraday_log_r"] * work["intraday_log_r"].shift(1) * work["intraday_log_r"].shift(2))
        ** (2 / 3)
    ).fillna(0.0)
    work["IV_hat"] = work["V_hat"].cumsum()
    work["RVJ"] = np.maximum(work["RV"] - work["IV_hat"], 0.0)
    work["RVC"] = work["RV"] - work["RVJ"]

    gamma_r = np.abs(work["intraday_log_r"]).quantile(0.99)
    work["RVL"] = (work["r_2"] * (np.abs(work["intraday_log_r"]) > gamma_r).astype(int)).cumsum()
    work["RVLJ"] = np.minimum(work["RVJ"], work["RVL"])
    work["RVSJ"] = work["RVJ"] - work["RVLJ"]

    work["RSP"] = (work["r_2"] * (work["intraday_log_r"] > 0).astype(int)).cumsum()
    work["RSN"] = (work["r_2"] * (work["intraday_log_r"] < 0).astype(int)).cumsum()
    work["RVJP"] = np.maximum(work["RSP"] - 0.5 * work["IV_hat"], 0.0)
    work["RVJN"] = np.maximum(work["RSN"] - 0.5 * work["IV_hat"], 0.0)
    work["SRVJ"] = work["RVJP"] - work["RVJN"]

    work["RVLP"] = (work["r_2"] * (work["intraday_log_r"] > gamma_r).astype(int)).cumsum()
    work["RVLN"] = (work["r_2"] * (work["intraday_log_r"] < -gamma_r).astype(int)).cumsum()
    work["RVLJP"] = np.minimum(work["RVJP"], work["RVLP"])
    work["RVLJN"] = np.minimum(work["RVJN"], work["RVLN"])
    work["RVSJP"] = work["RVJP"] - work["RVLJP"]
    work["RVSJN"] = work["RVJN"] - work["RVLJN"]
    work["SRVLJ"] = work["RVLJP"] - work["RVLJN"]
    work["SRVSJ"] = work["RVSJP"] - work["RVSJN"]

    cols = ["RV", "IV_hat", "RVJ", "RVC", "RVSJ", "RSP", "RSN", "RVJP", "RVJN", "SRVJ", "RVSJP", "RVSJN", "SRVSJ"]
    out.update({f"Var_{c}_min": _last_value(work.assign(time=df["time"]), c) for c in cols})
    return out


def _osod_factors(df: pd.DataFrame) -> Dict[str, float]:
    close = df["close"]
    low = df["low"]
    high = df["high"]
    out: Dict[str, float] = {}

    low_min = low.rolling(9, min_periods=1).min()
    high_max = high.rolling(9, min_periods=1).max()
    rsv = ((close - low_min) / (high_max - low_min + eps) * 100).fillna(50.0)

    work = pd.DataFrame({"time": df["time"]})
    work["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    work["D"] = work["K"].ewm(alpha=1 / 3, adjust=False).mean()
    work["J"] = 3 * work["K"] - 2 * work["D"]

    for w in (6, 12, 24):
        ma = close.rolling(w, min_periods=1).mean()
        work[f"bias_{w}"] = (close - ma) / ma * 100

    prev_close = close.shift(1).fillna(close)
    long_strength = high - prev_close
    short_strength = prev_close - low
    up_push = high - df["open"]
    down_gravity = df["open"] - low
    for w in (7, 13, 26):
        br_den = short_strength.rolling(w, min_periods=1).sum()
        ar_den = down_gravity.rolling(w, min_periods=1).sum()
        work[f"br_{w}"] = np.where(br_den.abs() <= EPS, 0.0, long_strength.rolling(w, min_periods=1).sum() / br_den * 100)
        work[f"ar_{w}"] = np.where(ar_den.abs() <= EPS, 0.0, up_push.rolling(w, min_periods=1).sum() / ar_den * 100)

    tp = (high + low + close) / 3
    for w in (5, 10, 20):
        tp_ma = tp.rolling(w, min_periods=1).mean()
        md = (close - tp_ma).abs().rolling(w, min_periods=1).mean()
        work[f"cci_{w}"] = (tp - tp_ma) / (md + eps) * 100

    cols = ["K", "D", "J", "bias_6", "bias_12", "bias_24", "br_7", "br_13", "br_26", "ar_7", "ar_13", "ar_26", "cci_5", "cci_10", "cci_20"]
    out.update({f"Osod_{c}_min": _last_value(work, c) for c in cols})
    return out


def _energy_factors(df: pd.DataFrame) -> Dict[str, float]:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    out: Dict[str, float] = {}
    work = pd.DataFrame({"time": df["time"]})

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = high_diff.clip(lower=0)
    minus_dm = low_diff.clip(lower=0)
    dmp = np.where(plus_dm > minus_dm, plus_dm, 0.0)
    dmn = np.where(plus_dm < minus_dm, minus_dm, 0.0)

    atr = tr.ewm(span=14, adjust=False).mean()
    smoothed_dmp = pd.Series(dmp).ewm(span=14, adjust=False).mean()
    smoothed_dmn = pd.Series(dmn).ewm(span=14, adjust=False).mean()
    work["DIP"] = 100 * smoothed_dmp / atr
    work["DIN"] = 100 * smoothed_dmn / atr
    work["DX"] = 100 * (work["DIP"] - work["DIN"]).abs() / (work["DIP"] + work["DIN"] + eps)
    work["ADX"] = work["DX"].ewm(span=14, adjust=False).mean()

    low_min = low.rolling(9, min_periods=1).min()
    high_max = high.rolling(9, min_periods=1).max()
    pr = (high_max - low_min) + 0.5 * (high_max - close) + 0.5 * (low_min - close)
    close_shift = close.shift(1).fillna(close)
    si = 50 * ((close - close_shift) + 0.5 * (high_max - close) + 0.5 * (low_min - close)) / (pr + eps) * 0.3
    work["ASI"] = si.cumsum()

    minute_ret = close.pct_change(fill_method=None)
    gain = minute_ret.clip(lower=0)
    loss = -minute_ret.clip(upper=0)
    up = (minute_ret > 0).astype(float)
    down = (minute_ret < 0).astype(float)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    work["avg_gain"] = gain.rolling(14, min_periods=1).sum() / (up.rolling(14, min_periods=1).sum() + eps)
    work["avg_loss"] = loss.rolling(14, min_periods=1).sum() / (down.rolling(14, min_periods=1).sum() + eps)
    work["RSI"] = 100 - (100 / (1 + avg_gain / (avg_loss + eps)))
    work["RSII"] = work["avg_gain"] / (work["avg_gain"] + work["avg_loss"]) * 100

    tp = (high + low + close) / 3
    money_flow = tp * volume
    money_ratio = (money_flow * up).rolling(14, min_periods=1).sum() / ((money_flow * down).rolling(14, min_periods=1).sum() + eps)
    work["MFI"] = 100 - 100 / (1 + money_ratio)
    work["VR"] = (volume * up).rolling(12, min_periods=1).sum() / ((volume * down).rolling(12, min_periods=1).sum() + eps) * 100

    cols = ["DIP", "DIN", "DX", "ADX", "ASI", "avg_gain", "avg_loss", "RSI", "RSII", "MFI", "VR"]
    out.update({f"Energy_{c}_min": _last_value(work, c) for c in cols})
    return out


def _terminal_factors(df: pd.DataFrame) -> Dict[str, float]:
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
    out: Dict[str, float] = {}
    work = pd.DataFrame({"time": df["time"]})
    window = 20
    k = 2

    ma = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std()
    boll_upper = ma + std * k
    boll_lower = ma - std * k
    work["Boll_Width"] = (boll_upper - boll_lower) / close
    work["Boll_Position"] = (close - boll_lower) / (boll_upper - boll_lower)

    hl = high - low
    prev_close = close.shift(1)
    tr = pd.concat([hl, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    middle = close.ewm(span=window, adjust=False).mean()
    kc_upper = middle + atr * k
    kc_lower = middle - atr * k
    work["KC_Width"] = (kc_upper - kc_lower) / close
    work["KC_Position"] = (close - kc_lower) / (kc_upper - kc_lower)

    emad = close.ewm(span=9, adjust=False).mean() - close.ewm(span=25, adjust=False).mean()
    work["Mass"] = (hl / emad.replace(0, np.nan)).ewm(span=25, adjust=False).mean()

    don_upper = high.rolling(window, min_periods=1).max()
    don_lower = low.rolling(window, min_periods=1).min()
    don_range = don_upper - don_lower
    work["Donchian_Width"] = don_range / close
    work["Donchian_Position"] = np.where(don_range <= EPS, 0.0, (close - don_lower) / don_range)

    tmp1 = (close * 2 + high + low) / 4
    tmp2 = tmp1.rolling(20, min_periods=1).mean()
    alpha = (tmp1 - tmp2).abs() / tmp2
    xsii = np.zeros(len(close), dtype=float)
    if len(close):
        xsii[0] = close.iloc[0] if pd.notna(close.iloc[0]) else 0.0
        for i in range(1, len(close)):
            a = alpha.iloc[i] if np.isfinite(alpha.iloc[i]) else 0.0
            xsii[i] = a * close.iloc[i] + (1 - a) * xsii[i - 1]
    work["alpha_xsii"] = alpha

    midps = {
        "CR1": (2 * close + high + low) / 4,
        "CR2": (close + high + low + open_) / 4,
        "CR3": (close + high + low) / 3,
        "CR4": (high + low) / 2,
    }
    for name, midp in midps.items():
        up_price = high - midp.shift(1)
        down_price = midp.shift(1) - low
        work[name] = up_price.rolling(26, min_periods=1).sum() / (down_price.rolling(26, min_periods=1).sum() + eps) * 100

    cols = ["Boll_Width", "Boll_Position", "KC_Width", "KC_Position", "Mass", "Donchian_Width", "Donchian_Position", "alpha_xsii", "CR1", "CR2", "CR3", "CR4"]
    out.update({f"Terminal_{c}_min": _last_value(work, c) for c in cols})
    return out


def _kline_factors(df: pd.DataFrame) -> Dict[str, float]:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    out: Dict[str, float] = {}
    work = pd.DataFrame({"time": df["time"]})

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    work["macd_strength"] = np.tanh(macd / macd.abs().rolling(60, min_periods=1).max())
    work["macd_signal_gap"] = np.tanh((macd - signal) / macd.abs().rolling(30, min_periods=1).mean())
    work["trend_momentum"] = np.tanh((ema12 - ema26) * ema12.diff())

    aroon_up, aroon_down = _compute_aroon(high, low, period=25)
    work["aroon_up"] = aroon_up
    work["aroon_down"] = aroon_down
    work["aroon_diff"] = aroon_up - aroon_down

    close_ret = close.pct_change(fill_method=None)
    vol_ret = volume.pct_change(fill_method=None)
    work["pv_momentum"] = close_ret * (volume / volume.shift(1)).fillna(0.0)
    work["pv_corr"] = _rolling_corr(close_ret, vol_ret, 20)

    ema_slope = ema12.diff()
    work["long_short_energy"] = work["macd_strength"] * work["pv_corr"]

    tp = (high + low + close) / 3
    up_penetration = (high - tp.shift(1)).clip(lower=0)
    down_penetration = (tp.shift(1) - low).clip(lower=0)
    hl_range = high - low
    up_close = close.diff().gt(0).astype(float)
    for w in (7, 13, 26):
        down_mean = down_penetration.rolling(w, min_periods=1).mean()
        work[f"PPR_{w}"] = np.where(down_mean.abs() <= EPS, 0.0, up_penetration.rolling(w, min_periods=1).mean() / down_mean)
        hl_ema = hl_range.ewm(span=w, adjust=False).mean()
        work[f"CVI_{w}"] = (hl_ema - hl_ema.shift(w)) / hl_ema * 100
        work[f"PSY_{w}"] = up_close.rolling(w, min_periods=1).sum() / w * 100

    ret = close_ret.fillna(0.0)
    work["ret_ts_skew"] = ret.rolling(30, min_periods=1).skew()
    work["ret_ts_skew_std"] = work["ret_ts_skew"].rolling(30, min_periods=1).std()
    work["ret_ts_kurt"] = ret.rolling(30, min_periods=1).kurt()
    work["ret_ts_kurt_std"] = work["ret_ts_kurt"].rolling(30, min_periods=1).std()

    signed_vol = (close > close.shift(1)).astype(float) * volume - (close < close.shift(1)).astype(float) * volume
    work["signed_vol"] = signed_vol
    edit = signed_vol.rolling(20, min_periods=1).sum()
    work["EPI"] = edit / (volume.rolling(20, min_periods=1).sum() + eps)
    work["up_count"] = (close > close.shift(1)).astype(int).rolling(30, min_periods=1).sum()
    work["down_count"] = (close < close.shift(1)).astype(int).rolling(30, min_periods=1).sum()

    diff = close.diff()
    up_std = diff.where(diff > 0, 0).rolling(30, min_periods=1).std()
    down_std = diff.where(diff < 0, 0).rolling(30, min_periods=1).std()
    work["rvi"] = 100 * up_std / (up_std + down_std + eps)
    work["vol_parkinson"] = (1 / (4 * np.log(2))) * (np.log(high / low)) ** 2
    ema_range = hl_range.ewm(span=30, adjust=False).mean()
    work["chaikin_vol"] = (ema_range - ema_range.shift(30)) / (ema_range.shift(30) + eps) * 100

    cols = [
        "macd_strength", "macd_signal_gap", "trend_momentum",
        "aroon_up", "aroon_down", "aroon_diff", "pv_momentum", "pv_corr",
        "PPR_7", "PPR_13", "PPR_26", "CVI_7", "CVI_13", "CVI_26",
        "PSY_7", "PSY_13", "PSY_26", "ret_ts_skew", "ret_ts_skew_std",
        "ret_ts_kurt", "ret_ts_kurt_std", "signed_vol", "EPI",
        "up_count", "down_count", "rvi", "vol_parkinson", "chaikin_vol",
    ]
    out.update({f"Kline_{c}_min": _last_value(work, c) for c in cols})
    return out


def _klineshape_factors(df: pd.DataFrame) -> Dict[str, float]:
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    out: Dict[str, float] = {}
    work = pd.DataFrame({"time": df["time"]})
    hl = high - low
    hl_prev = hl.shift(1)
    high_prev = high.shift(1)
    low_prev = low.shift(1)
    close_prev = close.shift(1)

    work["close_location"] = np.where(hl <= EPS, 0.0, ((high - close) - (close - low)) / hl)
    work["close_loc_before"] = np.where(hl_prev <= EPS, 0.0, ((high_prev - close) - (close - low_prev)) / hl_prev)
    bullish = high - np.maximum(close, open_)
    bearish = np.minimum(close, open_) - low
    work["candle_ratio"] = np.where(bearish <= EPS, 0.0, bullish / bearish)
    work["candle_skew"] = np.where(hl <= EPS, 0.0, (bullish - bearish) / hl)
    body = close - open_
    candle_sum = bullish + bearish
    work["body_candle_ratio"] = np.where(candle_sum <= EPS, body / 1e-3, body / candle_sum)
    work["Doji"] = np.where(hl <= EPS, 0.0, body.abs() / hl)
    work["overlap_ratio"] = np.where(hl_prev <= EPS, 0.0, np.maximum(0, np.minimum(high, high_prev) - np.maximum(low, low_prev)) / hl_prev)
    work["gap_ratio"] = np.where(close_prev <= EPS, 0.0, (open_ - close_prev) / close_prev)
    work["close_position"] = np.where(hl_prev <= EPS, 0.0, (close - low_prev) / hl_prev)
    work["open_position"] = np.where(hl_prev <= EPS, 0.0, (open_ - low_prev) / hl_prev)
    ma = close.rolling(len(MINUTE_GRID), min_periods=1).mean()
    work["ma_bias"] = (close - ma) / (ma + eps)
    work["mdd"] = (high - close) / high * 100
    cum_high = high.rolling(len(MINUTE_GRID), min_periods=1).max()
    work["cum_mmd"] = (cum_high - close) / cum_high * 100

    cols = [
        "close_location", "close_loc_before", "candle_ratio", "body_candle_ratio",
        "Doji", "overlap_ratio", "candle_skew", "gap_ratio", "close_position",
        "open_position", "ma_bias", "mdd", "cum_mmd",
    ]
    out.update({f"Klineshape_{c}_min": _last_value(work, c) for c in cols})
    return out


MINUTE_FACTOR_GROUPS = (
    _var_factors,
    _osod_factors,
    _energy_factors,
    _terminal_factors,
    _kline_factors,
    _klineshape_factors,
)


def compute_minute_factors_for_stock(group: pd.DataFrame) -> pd.Series:
    """Compute all supported 1457 minute factors for one stock-day."""
    code = str(group["code"].iloc[0]).zfill(6)
    aligned = _align_minute_kline(group)
    if aligned.empty or aligned.loc[aligned["time"] == TARGET_TIME, "close"].isna().all():
        return pd.Series(dtype=float)

    factors: Dict[str, float] = {"code": code}
    for func in MINUTE_FACTOR_GROUPS:
        factors.update(func(aligned))
    return pd.Series(factors)


def compute_minute_factors_for_day(path: Path, max_stocks: Optional[int] = None) -> pd.DataFrame:
    """Compute one date's 1457 cross-section from a raw min_data feather file."""
    date_int = int(path.stem)
    df = pd.read_feather(path)
    df = df.rename(columns={"StockID": "code"})
    df["code"] = df["code"].map(strip_stock_prefix).astype(str).str.zfill(6)
    df = df[df["code"].map(_is_sh_or_sz_code)]
    df = df[(df["time"] >= 931) & (df["time"] <= TARGET_TIME)]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for idx, (_, group) in enumerate(df.groupby("code", sort=False)):
        if max_stocks is not None and idx >= max_stocks:
            break
        row = compute_minute_factors_for_stock(group)
        if not row.empty:
            rows.append(row)
    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.insert(0, "date", pd.Timestamp(str(date_int)))
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


class MinFactorAdapter:
    """Compute and save minute K-line factors as daily-style wide tables."""

    def __init__(
        self,
        minute_data_dir: Optional[Path] = None,
        factor_output_dir: Optional[Path] = None,
    ):
        self.minute_data_dir = minute_data_dir or config.MIN_DATA_DIR
        self.out_dir = factor_output_dir or config.FACTOR_OUTPUT_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _files(self, start_date=None, end_date=None) -> List[Path]:
        old_dir = config.MIN_DATA_DIR
        try:
            config.MIN_DATA_DIR = Path(self.minute_data_dir)
            return get_minute_files(
                date_to_int(start_date) if start_date is not None else None,
                date_to_int(end_date) if end_date is not None else None,
            )
        finally:
            config.MIN_DATA_DIR = old_dir

    @staticmethod
    def _default_workers() -> int:
        cpu_count = os.cpu_count() or 1
        return max(1, min(8, cpu_count))

    def _compute_day_rows(
        self,
        df: pd.DataFrame,
        *,
        max_stocks: Optional[int],
        use_multiprocess: bool,
        n_workers: Optional[int],
    ) -> List[pd.Series]:
        grouped = list(df.groupby("code", sort=False))
        if max_stocks is not None:
            grouped = grouped[:max_stocks]

        stock_groups = [group for _, group in grouped]
        if not stock_groups:
            return []

        if not use_multiprocess or len(stock_groups) <= 1:
            rows = []
            for group in stock_groups:
                row = compute_minute_factors_for_stock(group)
                if not row.empty:
                    rows.append(row)
            return rows

        workers = n_workers or self._default_workers()
        chunksize = max(1, len(stock_groups) // (workers * 4) or 1)
        rows: List[pd.Series] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for row in executor.map(compute_minute_factors_for_stock, stock_groups, chunksize=chunksize):
                if not row.empty:
                    rows.append(row)
        return rows

    def compute_panel(
        self,
        start_date=None,
        end_date=None,
        max_stocks_per_day: Optional[int] = None,
        use_multiprocess: bool = True,
        n_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        files = self._files(start_date=start_date, end_date=end_date)
        logger.info(
            "MinFactor: processing %d daily minute files (multiprocess=%s, workers=%s)",
            len(files),
            use_multiprocess,
            n_workers or self._default_workers(),
        )
        frames = []
        for path in tqdm(files, desc="MinFactor 按日计算", unit="day"):
            try:
                date_int = int(path.stem)
                df = pd.read_feather(path)
                df = df.rename(columns={"StockID": "code"})
                df["code"] = df["code"].map(strip_stock_prefix).astype(str).str.zfill(6)
                df = df[df["code"].map(_is_sh_or_sz_code)]
                df = df[(df["time"] >= 931) & (df["time"] <= TARGET_TIME)]
                if df.empty:
                    continue

                rows = self._compute_day_rows(
                    df,
                    max_stocks=max_stocks_per_day,
                    use_multiprocess=use_multiprocess,
                    n_workers=n_workers,
                )
                if not rows:
                    continue

                day_df = pd.DataFrame(rows)
                day_df.insert(0, "date", pd.Timestamp(str(date_int)))
                day_df = day_df.replace([np.inf, -np.inf], np.nan)
            except Exception as exc:
                logger.warning("跳过 %s: %s", path.name, exc)
                continue
            if not day_df.empty:
                frames.append(day_df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def compute_and_save_all(
        self,
        start_date=None,
        end_date=None,
        skip_existing: bool = False,
        shift_output: bool = False,
        max_stocks_per_day: Optional[int] = None,
        use_multiprocess: bool = True,
        n_workers: Optional[int] = None,
    ) -> Dict[str, Path]:
        """
        Compute minute factors and save one wide ``.fea`` file per factor.

        ``shift_output`` defaults to False because each row is explicitly the
        same day's 1457 snapshot requested by the caller. Set it to True only if
        the downstream workflow needs T rows to contain T-1 minute information.
        """
        panel = self.compute_panel(
            start_date=start_date,
            end_date=end_date,
            max_stocks_per_day=max_stocks_per_day,
            use_multiprocess=use_multiprocess,
            n_workers=n_workers,
        )
        if panel.empty:
            raise RuntimeError("未计算出任何分钟因子，请检查日期范围和 min_data 文件")

        factor_cols = [c for c in panel.columns if c not in ("date", "code")]
        saved: Dict[str, Path] = {}
        for factor_name in tqdm(factor_cols, desc="MinFactor 落盘", unit="factor"):
            out_path = self.out_dir / f"{factor_name}.fea"
            if skip_existing and out_path.exists():
                saved[factor_name] = out_path
                continue
            wide = panel.pivot_table(index="date", columns="code", values=factor_name, aggfunc="first").sort_index()
            wide.index.name = "TRADE_DATE"
            wide.columns = pd.Index([str(c).zfill(6) for c in wide.columns])
            if shift_output:
                wide = wide.shift(1)
            saved[factor_name] = save_wide_table(wide, out_path)

        logger.info("MinFactor 完成: %d 个因子 -> %s", len(saved), self.out_dir)
        return saved


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute minute K-line factors from min_data.")
    parser.add_argument("--start-date", default=None, help="Start date, e.g. 20210104 or 2021-01-04")
    parser.add_argument("--end-date", default=None, help="End date, e.g. 20241231 or 2024-12-31")
    parser.add_argument("--minute-data-dir", default=None, help="Override min_data directory")
    parser.add_argument("--output-dir", default=None, help="Override factor output directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip factor files that already exist")
    parser.add_argument("--shift-output", action="store_true", help="Save T rows with T-1 minute factor values")
    parser.add_argument("--max-stocks-per-day", type=int, default=None, help="Optional debug limit for each day")
    parser.add_argument("--no-multiprocess", action="store_true", help="Disable per-day per-stock multiprocessing")
    parser.add_argument("--n-workers", type=int, default=None, help="Worker count for multiprocessing")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    adapter = MinFactorAdapter(
        minute_data_dir=Path(args.minute_data_dir) if args.minute_data_dir else None,
        factor_output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    saved = adapter.compute_and_save_all(
        start_date=args.start_date,
        end_date=args.end_date,
        skip_existing=args.skip_existing,
        shift_output=args.shift_output,
        max_stocks_per_day=args.max_stocks_per_day,
        use_multiprocess=not args.no_multiprocess,
        n_workers=args.n_workers,
    )
    print(f"已保存 {len(saved)} 个分钟因子")


if __name__ == "__main__":
    main()

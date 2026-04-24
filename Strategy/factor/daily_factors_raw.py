"""
Strategy 包内日频因子库 (迁移自项目根目录 daily_factors.py)。

⚠️ 本文件为计算逻辑层，不直接对外暴露。
   请通过 DailyFactorLibraryAdapter (daily_factor_library.py) 调用，
   该适配器负责宽表/长表转换与防未来数据 shift(1)。

────────────────────────────────────────────────────────
股票池日频因子（策略文档 2.1 方式一、二、四）。

- 方式一：日频 OHLCV 直接构造（波动、量能、趋势、振幅等）。
- 方式二：与分钟模块同构的技术指标，在日 K 上重算（Osod/Energy/Terminal/Kline 的部分子集）。
- 方式四：涨跌停、连板、溢价等跨日字段（依赖 identify_limit_status / daily_info 合并列）。
- 方式五（新增）：从 Concept_daily_base1_lyt.py 迁移的个股级日频因子
  （Oscillator 扩展 / Trend 扩展 / ChannelVol 扩展 / VolPrice / Morphology）。

方式三（LOB/逐笔/概念分钟）：可在 ``compute_minute_runtime`` 中 **现场调用** 原脚本计算函数并取
``second=EOD``；亦可由 ``minute_feather.load_eod_from_fea`` 读已落盘的 .fea。二者均由
``compute_all_pool_factors`` 的 ``*_eod`` 参数并入。

板块级因子（概念板块聚合 + 技术指标映射回个股）：参见 ``concept_factors.py``。

输入列约定（单股按 date 升序的 panel 经 groupby 后传入）::

    必选: date, code, open, high, low, close, volume, amount,
          pre_close, chg_pct
    可选: turnover_rate, market_value
    跨日/规则: is_limit_up, is_limit_down, is_one_word_up,
               is_st, is_suspended, listing_days
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── 全局常量 ──
_EPS = 1e-9
_eps = 1e-5


# ══════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════

def _rolling_std(s: pd.Series, window: int) -> np.ndarray:
    mp = max(1, window // 2)
    return s.rolling(window, min_periods=mp).std().values


def _rolling_mean(s: pd.Series, window: int) -> np.ndarray:
    mp = max(1, window // 2)
    return s.rolling(window, min_periods=mp).mean().values


def _rolling_sum(s: pd.Series, window: int) -> np.ndarray:
    return s.rolling(window, min_periods=1).sum().values


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def _compute_aroon_arr(high: np.ndarray, low: np.ndarray, period: int = 25):
    """计算 Aroon Up / Down（numpy 数组版本，用于 row 函数）"""
    n = len(high)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - period + 1)
        window_h = high[start:i + 1]
        window_l = low[start:i + 1]
        wlen = len(window_h)
        days_since_high = wlen - 1 - np.argmax(window_h)
        days_since_low = wlen - 1 - np.argmin(window_l)
        aroon_up[i] = 100.0 * (wlen - days_since_high) / wlen
        aroon_down[i] = 100.0 * (wlen - days_since_low) / wlen
    return aroon_up, aroon_down


def _xsii_sequential(close_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray):
    """计算 XSII 自适应均线（顺序计算，per-stock）"""
    n = len(close_arr)
    tmp1 = (close_arr * 2.0 + high_arr + low_arr) / 4.0
    tmp2 = pd.Series(tmp1).rolling(20, min_periods=1).mean().values
    alpha = np.abs(tmp1 - tmp2) / (tmp2 + _EPS)
    xsii = np.empty(n, dtype=np.float64)
    xsii[0] = close_arr[0] if not np.isnan(close_arr[0]) else 0.0
    for i in range(1, n):
        a = alpha[i] if np.isfinite(alpha[i]) else 0.0
        xsii[i] = a * close_arr[i] + (1.0 - a) * xsii[i - 1]
    return alpha, xsii


# ══════════════════════════════════════════════════════════════════
# Row-level: 单只股票一段历史 → 最后一行的截面因子向量
# ══════════════════════════════════════════════════════════════════

def compute_daily_factors_row(df: pd.DataFrame) -> pd.Series:
    """
    单只股票一段日频历史 → 最后一行（T-1）上的截面因子向量。
    """
    if df is None or len(df) < 5:
        return pd.Series(dtype=float)

    df = df.sort_values("date").reset_index(drop=True)
    c = df["close"].astype(float).values
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    v = df["volume"].astype(float).values
    amt = df["amount"].astype(float).values
    ret = df["chg_pct"].astype(float).values / 100.0
    n = len(df)

    tr = (
        df["turnover_rate"].astype(float).values
        if "turnover_rate" in df.columns
        else np.zeros(n)
    )
    mv = (
        df["market_value"].astype(float).values
        if "market_value" in df.columns
        else np.zeros(n)
    )

    fac: Dict[str, Any] = {}

    # ── 方式一：波动 ──
    for w in (5, 10, 20):
        fac[f"ret_std_{w}d"] = float(_rolling_std(pd.Series(ret), w)[-1])
    _c_safe = np.where(c > 0, c, 1.0)
    _amp_raw = (h - l) / _c_safe
    _amp_raw[c <= 0] = np.nan
    amp_series = pd.Series(_amp_raw)
    for w in (5, 10, 20):
        fac[f"amplitude_mean_{w}d"] = float(
            amp_series.rolling(w, min_periods=max(1, w // 2)).mean().iloc[-1]
        )
    if n >= 20:
        log_hl = np.log(np.where(h > 0, h, np.nan)) - np.log(np.where(l > 0, l, np.nan))
        tail = log_hl[max(0, n - 20):]
        valid = tail[np.isfinite(tail)]
        if len(valid) > 0:
            ph = np.mean(valid ** 2)
            fac["parkinson_vol_20d"] = float(np.sqrt(ph / (4 * np.log(2))))
        else:
            fac["parkinson_vol_20d"] = np.nan
    else:
        fac["parkinson_vol_20d"] = np.nan

    # ── 方式一：量能 ──
    for w in (5, 10, 20):
        fac[f"avg_amount_{w}d"] = float(_rolling_mean(pd.Series(amt), w)[-1])
        fac[f"avg_volume_{w}d"] = float(_rolling_mean(pd.Series(v), w)[-1])
        fac[f"avg_turnover_{w}d"] = float(_rolling_mean(pd.Series(tr), w)[-1])

    avg5 = _rolling_mean(pd.Series(amt), 5)[-1]
    avg20 = _rolling_mean(pd.Series(amt), 20)[-1]
    fac["amount_ratio_5_20"] = float(avg5 / avg20) if avg20 and avg20 > 0 else np.nan

    tr5 = _rolling_mean(pd.Series(tr), 5)[-1]
    tr20 = _rolling_mean(pd.Series(tr), 20)[-1]
    fac["turnover_ratio_5_20"] = float(tr5 / tr20) if tr20 and tr20 > 0 else np.nan

    # ── 方式一：趋势 ──
    for w in (3, 5, 10, 20):
        if n >= w and c[-w] > 0:
            fac[f"cum_return_{w}d"] = float(c[-1] / c[-w] - 1)
        else:
            fac[f"cum_return_{w}d"] = np.nan

    ma5 = float(_rolling_mean(pd.Series(c), 5)[-1])
    ma10 = float(_rolling_mean(pd.Series(c), 10)[-1])
    ma20 = float(_rolling_mean(pd.Series(c), 20)[-1])
    fac["ma5"], fac["ma10"], fac["ma20"] = ma5, ma10, ma20
    if not (np.isnan(ma5) or np.isnan(ma10) or np.isnan(ma20)):
        fac["ma_bull_align"] = int(ma5 > ma10 > ma20)
    else:
        fac["ma_bull_align"] = 0
    fac["price_above_ma_count"] = int(c[-1] > ma5) + int(c[-1] > ma10) + int(c[-1] > ma20)
    for w in (5, 10, 20):
        fac[f"momentum_{w}d"] = float(c[-1] / c[-w]) if n >= w and c[-w] > 0 else np.nan

    fac["market_value"] = float(mv[-1])
    fac["log_market_value"] = float(np.log(mv[-1])) if mv[-1] > 0 else np.nan

    # ── 方式二：RSI ──
    gains = np.where(ret > 0, ret, 0.0)
    losses = np.where(ret < 0, -ret, 0.0)
    for w in (6, 12, 24):
        if n >= w + 1:
            ag = _ema(gains, w)[-1]
            al = _ema(losses, w)[-1]
            fac[f"rsi_{w}"] = float(100 * ag / (ag + al)) if (ag + al) > 0 else 50.0
        else:
            fac[f"rsi_{w}"] = np.nan

    # ── 方式二：ATR ──
    tr_val = None
    if n >= 2:
        tr_val = np.maximum(
            h[1:] - l[1:],
            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
        )
        tr_val = np.concatenate([[h[0] - l[0]], tr_val])
        for w in (5, 14, 20):
            atr = float(_rolling_mean(pd.Series(tr_val), w)[-1])
            fac[f"atr_{w}"] = atr
            fac[f"atr_ratio_{w}"] = float(atr / c[-1]) if c[-1] > 0 else np.nan

    # ── 方式二：MACD ──
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    fac["macd_dif"] = float(dif[-1])
    fac["macd_dea"] = float(dea[-1])
    fac["macd_hist"] = float(2 * (dif[-1] - dea[-1]))
    fac["macd_cross"] = int(dif[-1] > dea[-1]) - int(dif[-1] < dea[-1])

    # ── 方式二：KDJ ──
    if n >= 9:
        low9 = pd.Series(l).rolling(9).min().values
        high9 = pd.Series(h).rolling(9).max().values
        _hl_range_kdj = high9 - low9
        _hl_safe = np.where(_hl_range_kdj > 0, _hl_range_kdj, 1.0)
        rsv = np.where(_hl_range_kdj > 0, (c - low9) / _hl_safe * 100, 50.0)
        k_arr = _ema(rsv, 3)
        d_arr = _ema(k_arr, 3)
        j_arr = 3 * k_arr - 2 * d_arr
        fac["kdj_k"] = float(k_arr[-1])
        fac["kdj_d"] = float(d_arr[-1])
        fac["kdj_j"] = float(j_arr[-1])

    # ── 方式二：Bollinger ──
    if n >= 20:
        bb_mid = float(_rolling_mean(pd.Series(c), 20)[-1])
        bb_std = float(_rolling_std(pd.Series(c), 20)[-1])
        fac["bb_width"] = float(2 * bb_std / bb_mid) if bb_mid > 0 else np.nan
        fac["bb_position"] = float((c[-1] - bb_mid) / (2 * bb_std)) if bb_std > 0 else 0.0

    # ── 方式二：CCI ──
    tp = (h + l + c) / 3
    for w in (5, 14, 20):
        tp_s = pd.Series(tp)
        tp_ma = tp_s.rolling(w, min_periods=max(1, w // 2)).mean().values
        md = (
            tp_s.rolling(w, min_periods=max(1, w // 2))
            .apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            .values
        )
        fac[f"cci_{w}"] = (
            float((tp[-1] - tp_ma[-1]) / (0.015 * md[-1])) if md[-1] > 0 else 0.0
        )

    # ── 方式二：BIAS ──
    for w in (6, 12, 24):
        ma_w = float(_rolling_mean(pd.Series(c), w)[-1])
        fac[f"bias_{w}"] = float((c[-1] - ma_w) / ma_w * 100) if ma_w > 0 else 0.0

    # ── 方式二：ADX ──
    if n >= 15 and tr_val is not None:
        high_s = pd.Series(h)
        low_s = pd.Series(l)
        high_diff = high_s.diff()
        low_diff = -low_s.diff()
        plus_dm = high_diff.clip(lower=0)
        minus_dm = low_diff.clip(lower=0)
        mask_p = plus_dm > minus_dm
        mask_n = plus_dm < minus_dm
        dmp = np.where(mask_p, plus_dm, 0.0)
        dmn = np.where(mask_n, minus_dm, 0.0)
        tr_s = pd.Series(tr_val)
        atr14 = tr_s.ewm(span=14, adjust=False).mean()
        sdmp = pd.Series(dmp).ewm(span=14, adjust=False).mean()
        sdmn = pd.Series(dmn).ewm(span=14, adjust=False).mean()
        dip_s = 100 * sdmp / (atr14 + 1e-12)
        din_s = 100 * sdmn / (atr14 + 1e-12)
        dx = 100 * (dip_s - din_s).abs() / (dip_s + din_s + 1e-12)
        adx_s = dx.ewm(span=14, adjust=False).mean()
        fac["dip_d"] = float(dip_s.iloc[-1])
        fac["din_d"] = float(din_s.iloc[-1])
        fac["adx_d"] = float(adx_s.iloc[-1])

    # ══════════════════════════════════════════════════════════════
    # 方式五（新增）：从 Concept_daily_base1_lyt 迁移的个股级因子
    # ══════════════════════════════════════════════════════════════

    prev_c = np.concatenate([[c[0]], c[:-1]])  # shift(1).fillna(c)

    # ── Oscillator: BR / AR ──
    long_str = h - prev_c
    short_str = prev_c - l
    up_push = h - o
    down_grav = o - l
    _long_s = pd.Series(long_str)
    _short_s = pd.Series(short_str)
    _up_push_s = pd.Series(up_push)
    _down_grav_s = pd.Series(down_grav)
    for w in (7, 13, 26):
        ls = float(_rolling_sum(_long_s, w)[-1])
        ss = float(_rolling_sum(_short_s, w)[-1])
        fac[f"br_{w}"] = float(ls / ss * 100) if abs(ss) > _EPS else 0.0
        us = float(_rolling_sum(_up_push_s, w)[-1])
        dg = float(_rolling_sum(_down_grav_s, w)[-1])
        fac[f"ar_{w}"] = float(us / dg * 100) if abs(dg) > _EPS else 0.0

    # ── Oscillator: RSII ──
    up_flag = (ret > 0).astype(float)
    down_flag = (ret < 0).astype(float)
    gain14_sum = float(_rolling_sum(pd.Series(gains), 14)[-1])
    up_cnt14 = float(_rolling_sum(pd.Series(up_flag), 14)[-1])
    loss14_sum = float(_rolling_sum(pd.Series(losses), 14)[-1])
    down_cnt14 = float(_rolling_sum(pd.Series(down_flag), 14)[-1])
    avg_gain_rsii = gain14_sum / (up_cnt14 + _eps)
    avg_loss_rsii = loss14_sum / (down_cnt14 + _eps)
    fac["rsii"] = float(avg_gain_rsii / (avg_gain_rsii + avg_loss_rsii + _EPS) * 100)

    # ── Oscillator: PSY ──
    close_s = pd.Series(c)
    for w in (7, 13, 26):
        psy_val = close_s.diff().gt(0).rolling(w, min_periods=1).sum().iloc[-1] / w * 100
        fac[f"psy_{w}"] = float(psy_val)

    # ── Trend: MACD 派生 ──
    macd_raw = dif  # ema12 - ema26 (numpy)
    signal = dea     # ema(dif, 9)
    hist_arr = macd_raw - signal
    fac["hist_slope"] = float(hist_arr[-1] - hist_arr[-6]) if n >= 6 else np.nan
    _macd_abs_max = pd.Series(np.abs(macd_raw)).rolling(60, min_periods=1).max().values[-1]
    fac["macd_strength"] = float(np.tanh(macd_raw[-1] / (_macd_abs_max + _EPS)))
    _macd_abs_mean = pd.Series(np.abs(macd_raw)).rolling(30, min_periods=1).mean().values[-1]
    fac["macd_signal_gap"] = float(np.tanh((macd_raw[-1] - signal[-1]) / (_macd_abs_mean + _EPS)))
    ema12_diff = np.diff(ema12, prepend=ema12[0])
    fac["trend_momentum"] = float(np.tanh((ema12[-1] - ema26[-1]) * ema12_diff[-1]))
    fac["ema_slope"] = float(ema12_diff[-1])

    # ── Trend: Aroon ──
    aroon_up, aroon_down = _compute_aroon_arr(h, l, period=25)
    fac["aroon_up"] = float(aroon_up[-1])
    fac["aroon_down"] = float(aroon_down[-1])
    fac["aroon_diff"] = float(aroon_up[-1] - aroon_down[-1])

    # ── Trend: ASI ──
    low9 = pd.Series(l).rolling(9, min_periods=1).min().values
    high9 = pd.Series(h).rolling(9, min_periods=1).max().values
    pr_arr = (high9 - low9) + 0.5 * (high9 - c) + 0.5 * (low9 - c)
    close_shift1 = np.concatenate([[c[0]], c[:-1]])
    si_arr = 50 * ((c - close_shift1) + 0.5 * (high9 - c) + 0.5 * (low9 - c)) / (pr_arr + _eps) * 0.3
    fac["pr_swing"] = float(pr_arr[-1])
    fac["si_swing"] = float(si_arr[-1])
    fac["asi_swing"] = float(np.nansum(si_arr))

    # ── Trend: BBI / DMA ──
    for w in (3, 6, 12, 24):
        locals()[f"ma_bbi_{w}"] = _rolling_mean(pd.Series(c), w)[-1]
    fac["bbi"] = float(
        (locals()["ma_bbi_3"] + locals()["ma_bbi_6"] +
         locals()["ma_bbi_12"] + locals()["ma_bbi_24"]) / 4)
    fac["dma_10_20"] = float(_rolling_mean(pd.Series(c), 10)[-1] - _rolling_mean(pd.Series(c), 20)[-1])

    # ── ChannelVol: Keltner Channel ──
    if tr_val is not None:
        kc_mid = pd.Series(c).ewm(span=20, adjust=False).mean().values[-1]
        kc_atr = pd.Series(tr_val).ewm(span=20, adjust=False).mean().values[-1]
        kc_upper = kc_mid + kc_atr * 2
        kc_lower = kc_mid - kc_atr * 2
        fac["kc_width"] = float((kc_upper - kc_lower) / c[-1]) if c[-1] > 0 else np.nan
        kc_range = kc_upper - kc_lower
        fac["kc_position"] = float((c[-1] - kc_lower) / kc_range) if kc_range > _EPS else 0.0

    # ── ChannelVol: Donchian Channel ──
    don_upper = pd.Series(h).rolling(20, min_periods=1).max().values[-1]
    don_lower = pd.Series(l).rolling(20, min_periods=1).min().values[-1]
    don_range = don_upper - don_lower
    fac["donchian_width"] = float(don_range / c[-1]) if c[-1] > 0 else np.nan
    fac["donchian_position"] = float((c[-1] - don_lower) / don_range) if don_range > _EPS else 0.0

    # ── ChannelVol: 波动率指标 ──
    if h[-1] > 0 and l[-1] > 0:
        fac["vol_parkinson_d"] = float((1 / (4 * np.log(2))) * (np.log(h[-1] / l[-1])) ** 2)
    else:
        fac["vol_parkinson_d"] = np.nan
    hl_arr = h - l
    ema_hl = pd.Series(hl_arr).ewm(span=30, adjust=False).mean().values
    if n > 30:
        fac["chaikin_vol"] = float(
            (ema_hl[-1] - ema_hl[-31]) / (ema_hl[-31] + _eps) * 100)
    else:
        fac["chaikin_vol"] = np.nan
    c_diff = np.diff(c, prepend=c[0])
    up_vol_arr = np.where(c_diff > 0, c_diff, 0.0)
    down_vol_arr = np.where(c_diff < 0, c_diff, 0.0)
    up_std_val = float(pd.Series(up_vol_arr).rolling(30, min_periods=1).std().iloc[-1])
    down_std_val = float(pd.Series(down_vol_arr).rolling(30, min_periods=1).std().iloc[-1])
    fac["rvi"] = float(100 * up_std_val / (up_std_val + down_std_val + _eps))

    ret_s = pd.Series(ret).fillna(0)
    if n >= 10:
        fac["ret_skew_30"] = float(ret_s.rolling(30, min_periods=10).apply(
            lambda x: pd.Series(x).skew(), raw=False).iloc[-1])
        fac["ret_skew_std"] = float(pd.Series(
            ret_s.rolling(30, min_periods=10).apply(
                lambda x: pd.Series(x).skew(), raw=False)
        ).rolling(30, min_periods=1).std().iloc[-1])
        fac["ret_kurt_30"] = float(ret_s.rolling(30, min_periods=10).apply(
            lambda x: pd.Series(x).kurtosis(), raw=False).iloc[-1])
        fac["ret_kurt_std"] = float(pd.Series(
            ret_s.rolling(30, min_periods=10).apply(
                lambda x: pd.Series(x).kurtosis(), raw=False)
        ).rolling(30, min_periods=1).std().iloc[-1])
    else:
        fac["ret_skew_30"] = fac["ret_skew_std"] = np.nan
        fac["ret_kurt_30"] = fac["ret_kurt_std"] = np.nan

    # ── VolPrice: OBV / MFI / VR ──
    direction = np.sign(np.diff(c, prepend=c[0]))
    fac["obv"] = float(np.sum(direction * v))
    # MFI
    mf = tp * v
    up_mf_sum = float(pd.Series(mf * up_flag).rolling(14, min_periods=1).sum().iloc[-1])
    down_mf_sum = float(pd.Series(mf * down_flag).rolling(14, min_periods=1).sum().iloc[-1])
    mf_ratio = up_mf_sum / (down_mf_sum + _eps)
    fac["mfi"] = float(100 - 100 / (1 + mf_ratio))
    # VR
    up_vol_sum = float(pd.Series(v * up_flag).rolling(12, min_periods=1).sum().iloc[-1])
    down_vol_sum = float(pd.Series(v * down_flag).rolling(12, min_periods=1).sum().iloc[-1])
    fac["vr_12"] = float(up_vol_sum / (down_vol_sum + _eps) * 100)

    # ── VolPrice: PV 动量与相关性 ──
    vol_shifted = np.concatenate([[v[0]], v[:-1]])
    fac["pv_momentum"] = float(ret[-1] * (v[-1] / (vol_shifted[-1] + _eps)))
    if n >= 10:
        ret_pct = pd.Series(ret)
        vol_pct = pd.Series(v).pct_change(fill_method=None)
        fac["pv_corr"] = float(ret_pct.rolling(20, min_periods=10).corr(vol_pct).iloc[-1])
    else:
        fac["pv_corr"] = np.nan

    # Trend Energy / Long-Short Energy
    fac["trend_energy"] = float(ema12_diff[-1] * v[-1])
    macd_strength_val = fac.get("macd_strength", 0.0)
    pv_corr_val = fac.get("pv_corr", 0.0)
    if np.isnan(pv_corr_val):
        pv_corr_val = 0.0
    fac["long_short_energy"] = float(macd_strength_val * pv_corr_val)

    # ── VolPrice: VMACD ──
    vol_ma12 = _rolling_mean(pd.Series(v), 12)[-1]
    vol_ma26 = _rolling_mean(pd.Series(v), 26)[-1]
    v_dif = vol_ma12 - vol_ma26
    v_signal = pd.Series(
        _rolling_mean(pd.Series(v), 12) - _rolling_mean(pd.Series(v), 26)
    ).rolling(9, min_periods=1).mean().values[-1]
    fac["vmacd"] = float(v_dif - v_signal)

    # ── VolPrice: EDIT / EPI ──
    signed_vol_edit = (c > prev_c).astype(float) * v - (c < prev_c).astype(float) * v
    fac["edit_20"] = float(pd.Series(signed_vol_edit).rolling(20, min_periods=1).sum().iloc[-1])
    vol_sum_20 = float(pd.Series(v).rolling(20, min_periods=1).sum().iloc[-1])
    fac["epi_20"] = float(fac["edit_20"] / (vol_sum_20 + _eps))

    # ── VolPrice: Up / Down Count ──
    up_tick = (c > prev_c).astype(float)
    down_tick = (c < prev_c).astype(float)
    fac["up_count_30"] = float(pd.Series(up_tick).rolling(30, min_periods=1).sum().iloc[-1])
    fac["down_count_30"] = float(pd.Series(down_tick).rolling(30, min_periods=1).sum().iloc[-1])

    # ── VolPrice: EM / EMV / MAEMV ──
    A = (h + l) / 2
    B = np.concatenate([[(h[0] + l[0]) / 2], (h[:-1] + l[:-1]) / 2])
    C = h - l
    em = np.where(v <= _EPS, 0.0, (A - B) * C / v)
    emv_s = pd.Series(em).rolling(14, min_periods=1).sum()
    fac["em"] = float(em[-1])
    fac["emv"] = float(emv_s.iloc[-1])
    fac["maemv"] = float(emv_s.rolling(9, min_periods=1).mean().iloc[-1])

    # ── VolPrice: CR1 ──
    mid1 = (2 * c + h + l) / 4
    mid1_shift = np.concatenate([[mid1[0]], mid1[:-1]])
    cr_up = h - mid1_shift
    cr_down = mid1_shift - l
    cr_up_sum = float(_rolling_sum(pd.Series(cr_up), 26)[-1])
    cr_down_sum = float(_rolling_sum(pd.Series(cr_down), 26)[-1])
    fac["cr1"] = float(cr_up_sum / (cr_down_sum + _eps) * 100)

    # ── VolPrice: Mass Index ──
    ema9_mass = _ema(c, 9)
    ema25_mass = _ema(c, 25)
    emad_mass = ema9_mass - ema25_mass
    emad_safe = np.where(np.abs(emad_mass) > _EPS, emad_mass, np.nan)
    hl_over_emad = (h - l) / emad_safe
    fac["mass_idx"] = float(pd.Series(hl_over_emad).ewm(span=25, adjust=False).mean().iloc[-1])

    # ── Morphology: K 线形态 ──
    hl_range = h - l
    fac["close_location"] = float(
        ((h[-1] - c[-1]) - (c[-1] - l[-1])) / hl_range[-1]
    ) if hl_range[-1] > _EPS else 0.0

    if n >= 2 and hl_range[-2] > _EPS:
        fac["close_loc_before"] = float(
            ((h[-2] - c[-1]) - (c[-1] - l[-2])) / hl_range[-2])
    else:
        fac["close_loc_before"] = 0.0

    bull_candle = h[-1] - max(c[-1], o[-1])
    bear_candle = min(c[-1], o[-1]) - l[-1]
    fac["candle_ratio"] = float(bull_candle / bear_candle) if bear_candle > _EPS else 0.0
    fac["candle_skew"] = float(
        (bull_candle - bear_candle) / hl_range[-1]
    ) if hl_range[-1] > _EPS else 0.0
    body = c[-1] - o[-1]
    fac["body_size"] = float(body)
    candle_sum = bull_candle + bear_candle
    fac["body_candle_ratio"] = float(body / candle_sum) if candle_sum > _EPS else float(body / 1e-3)
    fac["doji"] = float(abs(body) / hl_range[-1]) if hl_range[-1] > _EPS else 0.0

    # ── Morphology: 穿透与跨日形态 ──
    if n >= 2:
        hl_prev = hl_range[-2]
        fac["overlap_ratio"] = float(
            max(0, min(h[-1], h[-2]) - max(l[-1], l[-2])) / hl_prev
        ) if hl_prev > _EPS else 0.0
        fac["gap_ratio"] = float((o[-1] - c[-2]) / c[-2]) if c[-2] > _EPS else 0.0
        fac["close_position"] = float((c[-1] - l[-2]) / hl_prev) if hl_prev > _EPS else 0.0
        fac["open_position"] = float((o[-1] - l[-2]) / hl_prev) if hl_prev > _EPS else 0.0
    else:
        fac["overlap_ratio"] = fac["gap_ratio"] = fac["close_position"] = fac["open_position"] = 0.0

    # PPR / CVI
    tp_shift1 = np.concatenate([[tp[0]], tp[:-1]])
    up_pen = np.maximum(h - tp_shift1, 0)
    down_pen = np.maximum(tp_shift1 - l, 0)
    for w in (7, 13):
        up_pen_ma = float(_rolling_mean(pd.Series(up_pen), w)[-1])
        down_pen_ma = float(_rolling_mean(pd.Series(down_pen), w)[-1])
        fac[f"ppr_{w}"] = float(up_pen_ma / down_pen_ma) if abs(down_pen_ma) > _EPS else 0.0
        hl_ema = pd.Series(hl_arr).ewm(span=w, adjust=False).mean()
        if n > w:
            fac[f"cvi_{w}"] = float((hl_ema.iloc[-1] - hl_ema.iloc[-w - 1]) / (hl_ema.iloc[-1] + _EPS) * 100)
        else:
            fac[f"cvi_{w}"] = np.nan

    # MA bias (60), MDD, Cum MDD
    ma60 = _rolling_mean(pd.Series(c), 60)[-1]
    fac["ma_bias_60"] = float((c[-1] - ma60) / (ma60 + _eps))
    fac["mdd_daily"] = float((h[-1] - c[-1]) / (h[-1] + _EPS) * 100)
    cum_high_60 = pd.Series(h).rolling(60, min_periods=1).max().values[-1]
    fac["cum_mmd_60"] = float((cum_high_60 - c[-1]) / (cum_high_60 + _EPS) * 100)

    # ── Morphology: XSII 自适应均线 ──
    alpha_xsii, xsii_vals = _xsii_sequential(c, h, l)
    fac["alpha_xsii"] = float(alpha_xsii[-1])
    fac["xsii"] = float(xsii_vals[-1])

    # ── 结束 ──
    fac["date"] = df["date"].iloc[-1]
    fac["code"] = df["code"].iloc[-1]
    return pd.Series(fac)


# ══════════════════════════════════════════════════════════════════
# 涨跌停与状态类跨日因子
# ══════════════════════════════════════════════════════════════════

def compute_cross_day_factors_row(df: pd.DataFrame) -> pd.Series:
    """涨跌停与状态类跨日因子（最后一行对应日）。"""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    lu = df["is_limit_up"].values if "is_limit_up" in df.columns else np.zeros(n)
    ld = df["is_limit_down"].values if "is_limit_down" in df.columns else np.zeros(n)
    owu = df["is_one_word_up"].values if "is_one_word_up" in df.columns else np.zeros(n)

    fac: Dict[str, Any] = {}
    for w in (3, 5, 10, 20):
        sl = lu[-w:] if n >= w else lu
        fac[f"limit_up_count_{w}d"] = int(np.sum(sl))
        sl2 = ld[-w:] if n >= w else ld
        fac[f"limit_down_count_{w}d"] = int(np.sum(sl2))

    consec = 0
    for i in range(n - 1, -1, -1):
        if lu[i] > 0:
            consec += 1
        else:
            break
    fac["consecutive_limit_up"] = consec

    for w in (5, 10):
        sl = owu[-w:] if n >= w else owu
        fac[f"one_word_up_count_{w}d"] = int(np.sum(sl))

    lu10 = np.sum(lu[-10:]) if n >= 10 else np.sum(lu)
    owu10 = np.sum(owu[-10:]) if n >= 10 else np.sum(owu)
    fac["seal_proxy_ow_per_lu_10d"] = float(owu10 / lu10) if lu10 > 0 else np.nan

    c = df["close"].astype(float).values
    o = df["open"].astype(float).values
    premiums = []
    lookback = min(20, n - 1)
    for i in range(n - lookback, n - 1):
        if lu[i] > 0 and i + 1 < n and c[i] > 0:
            premiums.append((o[i + 1] - c[i]) / c[i])
    fac["limit_up_premium_mean"] = float(np.mean(premiums)) if premiums else np.nan
    fac["limit_up_premium_median"] = float(np.median(premiums)) if premiums else np.nan

    def _flag_int(v, default=0) -> int:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    fac["is_limit_up_today"] = _flag_int(lu[-1])
    fac["is_limit_down_today"] = _flag_int(ld[-1])
    fac["is_one_word_up_today"] = _flag_int(owu[-1])

    if "is_st" in df.columns:
        fac["is_st"] = _flag_int(df["is_st"].iloc[-1])
    if "is_suspended" in df.columns:
        fac["is_suspended"] = _flag_int(df["is_suspended"].iloc[-1])
    if "listing_days" in df.columns:
        ldv = df["listing_days"].iloc[-1]
        fac["listing_days"] = float(ldv) if pd.notna(ldv) else np.nan

    return pd.Series(fac)


# ══════════════════════════════════════════════════════════════════
# 编排函数：全市场 T-1 截面因子表
# ══════════════════════════════════════════════════════════════════

def compute_all_pool_factors(
    ohlcv_df: pd.DataFrame,
    limit_df: Optional[pd.DataFrame] = None,
    info_df: Optional[pd.DataFrame] = None,
    *,
    concept_eod: Optional[pd.DataFrame] = None,
    lob_eod: Optional[pd.DataFrame] = None,
    trans_eod: Optional[pd.DataFrame] = None,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    全市场 T-1 截面因子表，index 为 6 位 code。

    Parameters
    ----------
    ohlcv_df
        多日全市场日频行情（已含或可合并 limit/info）。
    limit_df, info_df
        可选；若提供则 merge is_limit_up 等列。
    concept_eod, lob_eod, trans_eod
        已由外部流水线生成的分钟因子表在 ``second=EOD`` 的切片，index=code。
    """
    df = ohlcv_df.copy()
    if limit_df is not None and len(limit_df):
        df = df.merge(
            limit_df[
                ["date", "code", "is_limit_up", "is_limit_down", "is_one_word_up"]
            ],
            on=["date", "code"],
            how="left",
        )
    if info_df is not None and len(info_df):
        df = df.merge(
            info_df[["date", "code", "is_st", "is_suspended", "listing_days"]],
            on=["date", "code"],
            how="left",
        )

    row_dict: Dict[str, pd.Series] = {}
    for code, grp in df.groupby("code", sort=False):
        grp = grp.sort_values("date")
        if len(grp) < 5:
            continue
        if len(grp) > lookback_days:
            grp = grp.iloc[-lookback_days:]
        daily_part = compute_daily_factors_row(grp)
        if daily_part.empty:
            continue
        if "is_limit_up" in grp.columns:
            cross_part = compute_cross_day_factors_row(grp)
            daily_part = pd.concat([daily_part, cross_part])
        key = str(code).replace(".", "")
        digits = "".join(ch for ch in key if ch.isdigit())
        key6 = digits[-6:].zfill(6) if len(digits) >= 6 else str(code)
        row_dict[key6] = daily_part

    if not row_dict:
        return pd.DataFrame()

    result = pd.DataFrame.from_dict(row_dict, orient="index")
    result.index.name = "code"

    extras = [concept_eod, lob_eod, trans_eod]
    for ext in extras:
        if ext is not None and len(ext):
            ext = ext.copy()
            ext.index = ext.index.astype(str).str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)
            result = result.join(ext, how="left")

    return result


# ══════════════════════════════════════════════════════════════════
# Panel-level: 向量化全市场全历史因子面板（训练 / 回测用）
# ══════════════════════════════════════════════════════════════════

def compute_daily_factors_panel(ohlcv_df: pd.DataFrame) -> tuple:
    """
    向量化计算全市场全历史的日频因子面板（训练 / 回测用）。

    与 ``compute_daily_factors_row`` 计算相同因子，但以 groupby + transform
    一次性完成所有股票，避免逐股票循环，速度快 10~50×。

    Parameters
    ----------
    ohlcv_df : 日频行情长表，须含 date, code, open, high, low, close,
               volume, amount, chg_pct。可选 turnover_rate, market_value。

    Returns
    -------
    (df_with_factors, factor_cols) : 原始列 + 因子列, 以及因子列名列表。
    """
    df = ohlcv_df.sort_values(["code", "date"]).reset_index(drop=True).copy()

    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    amt = df["amount"].astype(float)
    ret = df["chg_pct"].astype(float) / 100.0
    tr = df["turnover_rate"].astype(float) if "turnover_rate" in df.columns else pd.Series(0.0, index=df.index)
    mv = df["market_value"].astype(float) if "market_value" in df.columns else pd.Series(0.0, index=df.index)

    c_safe = c.replace(0, np.nan)
    amp = (h - l) / c_safe

    # 临时列
    df["_ret"] = ret
    df["_amp"] = amp
    df["_amt"] = amt
    df["_vol"] = v
    df["_tr"] = tr

    def _grp_rolling(col, w, func="mean"):
        def _fn(s):
            mp = max(1, w // 2)
            if func == "mean":
                return s.rolling(w, min_periods=mp).mean()
            elif func == "std":
                return s.rolling(w, min_periods=mp).std()
        return df.groupby("code", sort=False)[col].transform(_fn)

    def _grp_rolling_sum(col, w):
        return df.groupby("code", sort=False)[col].transform(
            lambda s: s.rolling(w, min_periods=1).sum())

    def _grp_ewm(col, span):
        return df.groupby("code", sort=False)[col].transform(
            lambda s: s.ewm(span=span, adjust=False).mean())

    def _grp_shift(col, periods=1, fill=None):
        shifted = df.groupby("code", sort=False)[col].shift(periods)
        if fill is not None:
            shifted = shifted.fillna(fill)
        return shifted

    def _grp_diff(col, periods=1):
        return df.groupby("code", sort=False)[col].diff(periods)

    def _grp_cumsum(col):
        return df.groupby("code", sort=False)[col].cumsum()

    # ══════════════════════════════════════════════════════════════
    # 原有因子 (方式一 + 方式二)
    # ══════════════════════════════════════════════════════════════

    # ── 波动 & 量能 ──
    for w in (5, 10, 20):
        df[f"ret_std_{w}d"] = _grp_rolling("_ret", w, "std")
        df[f"amplitude_mean_{w}d"] = _grp_rolling("_amp", w, "mean")
        df[f"avg_amount_{w}d"] = _grp_rolling("_amt", w, "mean")
        df[f"avg_volume_{w}d"] = _grp_rolling("_vol", w, "mean")
        df[f"avg_turnover_{w}d"] = _grp_rolling("_tr", w, "mean")

    df["amount_ratio_5_20"] = df["avg_amount_5d"] / df["avg_amount_20d"].replace(0, np.nan)
    df["turnover_ratio_5_20"] = df["avg_turnover_5d"] / df["avg_turnover_20d"].replace(0, np.nan)

    # ── 趋势（累计收益） ──
    for w in (3, 5, 10, 20):
        shifted = _grp_shift("close", w)
        df[f"cum_return_{w}d"] = c / shifted.replace(0, np.nan) - 1.0

    # ── 均线 ──
    for w in (5, 10, 20):
        df[f"ma{w}"] = df.groupby("code", sort=False)["close"].transform(
            lambda s: s.rolling(w, min_periods=max(1, w // 2)).mean()
        )
    df["ma_bull_align"] = ((df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])).astype(float)
    df["price_above_ma_count"] = (
        (c > df["ma5"]).astype(float)
        + (c > df["ma10"]).astype(float)
        + (c > df["ma20"]).astype(float)
    )

    # ── 动量 ──
    for w in (5, 10, 20):
        shifted = _grp_shift("close", w)
        df[f"momentum_{w}d"] = c / shifted.replace(0, np.nan)

    # ── 市值 ──
    df["log_market_value"] = np.log(mv.replace(0, np.nan))

    # ── RSI ──
    df["_gain"] = ret.clip(lower=0)
    df["_loss"] = (-ret).clip(lower=0)
    for w in (6, 12, 24):
        ag = _grp_ewm("_gain", w)
        al = _grp_ewm("_loss", w)
        df[f"rsi_{w}"] = 100 * ag / (ag + al).replace(0, np.nan)

    # ── ATR ──
    prev_c = _grp_shift("close", 1)
    tr_val = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["_tr_val"] = tr_val
    for w in (5, 14, 20):
        df[f"atr_{w}"] = df.groupby("code", sort=False)["_tr_val"].transform(
            lambda s: s.rolling(w, min_periods=max(1, w // 2)).mean()
        )
        df[f"atr_ratio_{w}"] = df[f"atr_{w}"] / c_safe

    # ── MACD ──
    ema12 = _grp_ewm("close", 12)
    ema26 = _grp_ewm("close", 26)
    dif = ema12 - ema26
    df["macd_dif"] = dif
    df["_dif"] = dif
    df["macd_dea"] = _grp_ewm("_dif", 9)
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # ── Bollinger ──
    bb_mid = df["ma20"]
    bb_std = df.groupby("code", sort=False)["close"].transform(
        lambda s: s.rolling(20, min_periods=10).std()
    )
    df["bb_width"] = 2 * bb_std / bb_mid.replace(0, np.nan)
    df["bb_position"] = (c - bb_mid) / (2 * bb_std).replace(0, np.nan)

    # ── 碎片整理①: 方式一/二 计算完毕, 合并内部 Block ──
    df = df.copy()

    # ══════════════════════════════════════════════════════════════
    # 新增因子 (方式五)：从 Concept_daily_base1_lyt 迁移的个股级因子
    # ══════════════════════════════════════════════════════════════

    # 常用中间变量
    _prev_c = _grp_shift("close", 1, fill=c)
    df["_up_flag"] = (ret > 0).astype(float)
    df["_down_flag"] = (ret < 0).astype(float)
    df["_hl"] = h - l  # 日内振幅

    # ── Oscillator: BR / AR ──
    df["_long_str"] = h - _prev_c
    df["_short_str"] = _prev_c - l
    df["_up_push"] = h - o
    df["_down_grav"] = o - l
    for w in (7, 13, 26):
        ls = _grp_rolling_sum("_long_str", w)
        ss = _grp_rolling_sum("_short_str", w)
        df[f"br_{w}"] = np.where(ss.abs() <= _EPS, 0.0, ls / ss * 100)
        us = _grp_rolling_sum("_up_push", w)
        dg = _grp_rolling_sum("_down_grav", w)
        df[f"ar_{w}"] = np.where(dg.abs() <= _EPS, 0.0, us / dg * 100)

    # ── Oscillator: RSII ──
    gain14_sum = _grp_rolling_sum("_gain", 14)
    up_cnt14 = _grp_rolling_sum("_up_flag", 14)
    loss14_sum = _grp_rolling_sum("_loss", 14)
    down_cnt14 = _grp_rolling_sum("_down_flag", 14)
    _avg_gain = gain14_sum / (up_cnt14 + _eps)
    _avg_loss = loss14_sum / (down_cnt14 + _eps)
    df["rsii"] = _avg_gain / (_avg_gain + _avg_loss + _EPS) * 100

    # ── Oscillator: PSY ──
    df["_close_up"] = _grp_diff("close", 1).gt(0).astype(float)
    for w in (7, 13, 26):
        df[f"psy_{w}"] = _grp_rolling_sum("_close_up", w) / w * 100

    # ── Trend: MACD 派生 ──
    df["hist_slope"] = _grp_diff("macd_hist", 5)
    df["_dif_abs"] = dif.abs()
    _macd_abs_max = df.groupby("code", sort=False)["_dif_abs"].transform(
        lambda s: s.rolling(60, min_periods=1).max())
    df["macd_strength"] = np.tanh(dif / (_macd_abs_max + _EPS))
    _macd_abs_mean = df.groupby("code", sort=False)["_dif_abs"].transform(
        lambda s: s.rolling(30, min_periods=1).mean())
    df["macd_signal_gap"] = np.tanh((dif - df["macd_dea"]) / (_macd_abs_mean + _EPS))
    df["_ema12"] = ema12
    _ema12_diff = _grp_diff("_ema12", 1)
    df["trend_momentum"] = np.tanh((ema12 - ema26) * _ema12_diff)
    df["ema_slope"] = _ema12_diff

    # ── Trend: Aroon ──
    _aroon_p = 25
    df["aroon_up"] = df.groupby("code", sort=False)["high"].transform(
        lambda s: s.rolling(_aroon_p, min_periods=1).apply(
            lambda x: 100.0 * (np.argmax(x) + 1) / len(x), raw=True))
    df["aroon_down"] = df.groupby("code", sort=False)["low"].transform(
        lambda s: s.rolling(_aroon_p, min_periods=1).apply(
            lambda x: 100.0 * (np.argmin(x) + 1) / len(x), raw=True))
    df["aroon_diff"] = df["aroon_up"] - df["aroon_down"]

    # ── Trend: ASI ──
    _low9 = df.groupby("code", sort=False)["low"].transform(
        lambda s: s.rolling(9, min_periods=1).min())
    _high9 = df.groupby("code", sort=False)["high"].transform(
        lambda s: s.rolling(9, min_periods=1).max())
    _pr = (_high9 - _low9) + 0.5 * (_high9 - c) + 0.5 * (_low9 - c)
    _c_shift1 = _grp_shift("close", 1, fill=c)
    _si = 50 * ((c - _c_shift1) + 0.5 * (_high9 - c) + 0.5 * (_low9 - c)) / (_pr + _eps) * 0.3
    df["pr_swing"] = _pr
    df["si_swing"] = _si
    df["_si_tmp"] = _si
    df["asi_swing"] = _grp_cumsum("_si_tmp")

    # ── Trend: BBI / DMA ──
    for w in (3, 6, 12, 24):
        df[f"_ma_bbi_{w}"] = df.groupby("code", sort=False)["close"].transform(
            lambda s: s.rolling(w, min_periods=1).mean())
    df["bbi"] = (df["_ma_bbi_3"] + df["_ma_bbi_6"] + df["_ma_bbi_12"] + df["_ma_bbi_24"]) / 4
    _ma10_dma = df.groupby("code", sort=False)["close"].transform(
        lambda s: s.rolling(10, min_periods=1).mean())
    _ma20_dma = df.groupby("code", sort=False)["close"].transform(
        lambda s: s.rolling(20, min_periods=1).mean())
    df["dma_10_20"] = _ma10_dma - _ma20_dma

    # ── ChannelVol: Keltner Channel ──
    _kc_middle = _grp_ewm("close", 20)
    _kc_atr = _grp_ewm("_tr_val", 20)
    _kc_upper = _kc_middle + _kc_atr * 2
    _kc_lower = _kc_middle - _kc_atr * 2
    _kc_range = _kc_upper - _kc_lower
    df["kc_width"] = _kc_range / c_safe
    df["kc_position"] = (c - _kc_lower) / _kc_range.replace(0, np.nan)

    # ── ChannelVol: Donchian Channel ──
    _don_upper = df.groupby("code", sort=False)["high"].transform(
        lambda s: s.rolling(20, min_periods=1).max())
    _don_lower = df.groupby("code", sort=False)["low"].transform(
        lambda s: s.rolling(20, min_periods=1).min())
    _don_range = _don_upper - _don_lower
    df["donchian_width"] = _don_range / c_safe
    df["donchian_position"] = np.where(
        _don_range <= _EPS, 0.0, (c - _don_lower) / _don_range)

    # ── ChannelVol: 波动率指标 ──
    _h_safe = h.replace(0, np.nan)
    _l_safe = l.replace(0, np.nan)
    df["vol_parkinson_d"] = (1 / (4 * np.log(2))) * (np.log(_h_safe / _l_safe)) ** 2

    _ema_hl = _grp_ewm("_hl", 30)
    df["_ema_hl"] = _ema_hl
    _ema_hl_shift = _grp_shift("_ema_hl", 30)
    df["chaikin_vol"] = (_ema_hl - _ema_hl_shift) / (_ema_hl_shift + _eps) * 100

    df["_c_diff"] = _grp_diff("close", 1)
    df["_up_vol_rvi"] = df["_c_diff"].where(df["_c_diff"] > 0, 0.0)
    df["_down_vol_rvi"] = df["_c_diff"].where(df["_c_diff"] < 0, 0.0)
    _up_std = df.groupby("code", sort=False)["_up_vol_rvi"].transform(
        lambda s: s.rolling(30, min_periods=1).std())
    _down_std = df.groupby("code", sort=False)["_down_vol_rvi"].transform(
        lambda s: s.rolling(30, min_periods=1).std())
    df["rvi"] = 100 * _up_std / (_up_std + _down_std + _eps)

    df["_ret_filled"] = ret.fillna(0)
    df["ret_skew_30"] = df.groupby("code", sort=False)["_ret_filled"].transform(
        lambda s: s.rolling(30, min_periods=10).apply(
            lambda x: pd.Series(x).skew(), raw=False))
    df["ret_skew_std"] = df.groupby("code", sort=False)["ret_skew_30"].transform(
        lambda s: s.rolling(30, min_periods=1).std())
    df["ret_kurt_30"] = df.groupby("code", sort=False)["_ret_filled"].transform(
        lambda s: s.rolling(30, min_periods=10).apply(
            lambda x: pd.Series(x).kurtosis(), raw=False))
    df["ret_kurt_std"] = df.groupby("code", sort=False)["ret_kurt_30"].transform(
        lambda s: s.rolling(30, min_periods=1).std())

    # ── 碎片整理②: Oscillator/Trend/ChannelVol 计算完毕, 合并内部 Block ──
    df = df.copy()

    # ── VolPrice: OBV / MFI / VR ──
    _direction = _grp_diff("close", 1).apply(np.sign)
    df["_signed_vol_obv"] = _direction * v
    df["obv"] = _grp_cumsum("_signed_vol_obv")

    _tp = (h + l + c) / 3
    df["_mf"] = _tp * v
    df["_up_mf"] = df["_mf"] * df["_up_flag"]
    df["_down_mf"] = df["_mf"] * df["_down_flag"]
    _up_mf_sum = _grp_rolling_sum("_up_mf", 14)
    _down_mf_sum = _grp_rolling_sum("_down_mf", 14)
    _mf_ratio = _up_mf_sum / (_down_mf_sum + _eps)
    df["mfi"] = 100 - 100 / (1 + _mf_ratio)

    df["_up_vol_vr"] = v * df["_up_flag"]
    df["_down_vol_vr"] = v * df["_down_flag"]
    _up_vol_sum = _grp_rolling_sum("_up_vol_vr", 12)
    _down_vol_sum = _grp_rolling_sum("_down_vol_vr", 12)
    df["vr_12"] = _up_vol_sum / (_down_vol_sum + _eps) * 100

    # ── VolPrice: PV 动量与相关性 ──
    _v_shift1 = _grp_shift("volume", 1, fill=v)
    df["pv_momentum"] = ret * (v / _v_shift1.replace(0, np.nan)).fillna(0)

    df["_vol_pct"] = df.groupby("code", sort=False)["volume"].pct_change(
        fill_method=None
    )
    # 向量化 rolling correlation: corr = (E[xy]-E[x]E[y]) / (std_x * std_y)
    df["_ret_x_volpct"] = df["_ret_filled"] * df["_vol_pct"]
    _w_corr = 20
    _mp_corr = 10
    _e_xy = df.groupby("code", sort=False)["_ret_x_volpct"].transform(
        lambda s: s.rolling(_w_corr, min_periods=_mp_corr).mean())
    _e_x = df.groupby("code", sort=False)["_ret_filled"].transform(
        lambda s: s.rolling(_w_corr, min_periods=_mp_corr).mean())
    _e_y = df.groupby("code", sort=False)["_vol_pct"].transform(
        lambda s: s.rolling(_w_corr, min_periods=_mp_corr).mean())
    _std_x = df.groupby("code", sort=False)["_ret_filled"].transform(
        lambda s: s.rolling(_w_corr, min_periods=_mp_corr).std())
    _std_y = df.groupby("code", sort=False)["_vol_pct"].transform(
        lambda s: s.rolling(_w_corr, min_periods=_mp_corr).std())
    df["pv_corr"] = (_e_xy - _e_x * _e_y) / (_std_x * _std_y + _EPS)

    # Trend Energy / Long-Short Energy
    df["trend_energy"] = _ema12_diff * v
    df["long_short_energy"] = df["macd_strength"] * df["pv_corr"]

    # ── VolPrice: VMACD ──
    df["_vol_ma12"] = df.groupby("code", sort=False)["volume"].transform(
        lambda s: s.rolling(12, min_periods=1).mean())
    df["_vol_ma26"] = df.groupby("code", sort=False)["volume"].transform(
        lambda s: s.rolling(26, min_periods=1).mean())
    df["_v_dif"] = df["_vol_ma12"] - df["_vol_ma26"]
    _v_signal = df.groupby("code", sort=False)["_v_dif"].transform(
        lambda s: s.rolling(9, min_periods=1).mean())
    df["vmacd"] = df["_v_dif"] - _v_signal

    # ── VolPrice: EDIT / EPI ──
    _c_up = (c > _prev_c).astype(float)
    _c_down = (c < _prev_c).astype(float)
    df["_signed_vol_edit"] = _c_up * v - _c_down * v
    df["edit_20"] = _grp_rolling_sum("_signed_vol_edit", 20)
    df["_vol_sum_20"] = _grp_rolling_sum("_vol", 20)
    df["epi_20"] = df["edit_20"] / (df["_vol_sum_20"] + _eps)

    # ── VolPrice: Up / Down Count ──
    df["_up_tick"] = _c_up
    df["_down_tick"] = _c_down
    df["up_count_30"] = _grp_rolling_sum("_up_tick", 30)
    df["down_count_30"] = _grp_rolling_sum("_down_tick", 30)

    # ── VolPrice: EM / EMV / MAEMV ──
    _A = (h + l) / 2
    _h_shift1 = _grp_shift("high", 1, fill=h)
    _l_shift1 = _grp_shift("low", 1, fill=l)
    _B = (_h_shift1 + _l_shift1) / 2
    _C = h - l
    df["_em"] = np.where(v <= _EPS, 0.0, (_A - _B) * _C / v)
    df["em"] = df["_em"]
    df["emv"] = df.groupby("code", sort=False)["_em"].transform(
        lambda s: s.rolling(14, min_periods=1).sum())
    df["maemv"] = df.groupby("code", sort=False)["emv"].transform(
        lambda s: s.rolling(9, min_periods=1).mean())

    # ── VolPrice: CR1 ──
    _mid1 = (2 * c + h + l) / 4
    df["_mid1"] = _mid1
    _mid1_shift = _grp_shift("_mid1", 1, fill=_mid1)
    df["_cr_up"] = h - _mid1_shift
    df["_cr_down"] = _mid1_shift - l
    _cr_up_sum = _grp_rolling_sum("_cr_up", 26)
    _cr_down_sum = _grp_rolling_sum("_cr_down", 26)
    df["cr1"] = _cr_up_sum / (_cr_down_sum + _eps) * 100

    # ── VolPrice: Mass Index ──
    _ema9 = _grp_ewm("close", 9)
    _ema25 = _grp_ewm("close", 25)
    _emad = _ema9 - _ema25
    df["_hl_emad"] = (h - l) / _emad.replace(0, np.nan)
    df["mass_idx"] = _grp_ewm("_hl_emad", 25)

    # ── 碎片整理③: VolPrice 计算完毕, 合并内部 Block ──
    df = df.copy()

    # ── Morphology: K 线形态 ──
    _hl_range = h - l
    df["close_location"] = np.where(
        _hl_range <= _EPS, 0.0, ((h - c) - (c - l)) / _hl_range)
    _hl_shift1 = df.assign(_hlr=_hl_range).groupby("code", sort=False)["_hlr"].shift(1)
    _h_s1 = _grp_shift("high", 1)
    _l_s1 = _grp_shift("low", 1)
    df["close_loc_before"] = np.where(
        _hl_shift1 <= _EPS, 0.0, ((_h_s1 - c) - (c - _l_s1)) / _hl_shift1)
    _bull_candle = h - np.maximum(c, o)
    _bear_candle = np.minimum(c, o) - l
    df["candle_ratio"] = np.where(
        _bear_candle <= _EPS, 0.0, _bull_candle / _bear_candle)
    df["candle_skew"] = np.where(
        _hl_range <= _EPS, 0.0, (_bull_candle - _bear_candle) / _hl_range)
    _body = c - o
    df["body_size"] = _body
    _candle_sum = _bull_candle + _bear_candle
    df["body_candle_ratio"] = np.where(
        _candle_sum <= _EPS, _body / 1e-3, _body / _candle_sum)
    df["doji"] = np.where(
        _hl_range <= _EPS, 0.0, np.abs(_body) / _hl_range)

    # ── Morphology: 穿透与跨日形态 ──
    _c_s1 = _grp_shift("close", 1)
    df["overlap_ratio"] = np.where(
        _hl_shift1 <= _EPS, 0.0,
        np.maximum(0, np.minimum(h, _h_s1) - np.maximum(l, _l_s1)) / _hl_shift1)
    df["gap_ratio"] = np.where(
        _c_s1 <= _EPS, 0.0, (o - _c_s1) / _c_s1)
    df["close_position"] = np.where(
        _hl_shift1 <= _EPS, 0.0, (c - _l_s1) / _hl_shift1)
    df["open_position"] = np.where(
        _hl_shift1 <= _EPS, 0.0, (o - _l_s1) / _hl_shift1)

    # PPR / CVI
    _tp_morph = (h + l + c) / 3
    df["_tp_morph"] = _tp_morph
    _tp_shift1 = _grp_shift("_tp_morph", 1, fill=_tp_morph)
    df["_up_pen"] = (h - _tp_shift1).clip(lower=0)
    df["_down_pen"] = (_tp_shift1 - l).clip(lower=0)
    for w in (7, 13):
        _up_pen_ma = df.groupby("code", sort=False)["_up_pen"].transform(
            lambda s: s.rolling(w, min_periods=1).mean())
        _down_pen_ma = df.groupby("code", sort=False)["_down_pen"].transform(
            lambda s: s.rolling(w, min_periods=1).mean())
        df[f"ppr_{w}"] = np.where(
            _down_pen_ma.abs() <= _EPS, 0.0, _up_pen_ma / _down_pen_ma)
        _hl_ema_w = _grp_ewm("_hl", w)
        df[f"_hl_ema_{w}"] = _hl_ema_w
        _hl_ema_shift_w = _grp_shift(f"_hl_ema_{w}", w)
        df[f"cvi_{w}"] = (_hl_ema_w - _hl_ema_shift_w) / (_hl_ema_w + _EPS) * 100

    # MA bias (60), MDD, Cum MDD
    _ma60 = df.groupby("code", sort=False)["close"].transform(
        lambda s: s.rolling(60, min_periods=1).mean())
    df["ma_bias_60"] = (c - _ma60) / (_ma60 + _eps)
    df["mdd_daily"] = (h - c) / (h + _EPS) * 100
    _cum_high_60 = df.groupby("code", sort=False)["high"].transform(
        lambda s: s.rolling(60, min_periods=1).max())
    df["cum_mmd_60"] = (_cum_high_60 - c) / (_cum_high_60 + _EPS) * 100

    # ── Morphology: XSII 自适应均线 ──
    def _xsii_grp(g):
        alpha, xsii_val = _xsii_sequential(
            g["close"].values.astype(float),
            g["high"].values.astype(float),
            g["low"].values.astype(float),
        )
        return pd.DataFrame(
            {"alpha_xsii": alpha, "xsii": xsii_val}, index=g.index)

    _xsii_res = df.groupby("code", sort=False)[
        ["close", "high", "low"]
    ].apply(_xsii_grp)
    if isinstance(_xsii_res.index, pd.MultiIndex):
        _xsii_res = _xsii_res.droplevel(0)
    df["alpha_xsii"] = _xsii_res["alpha_xsii"].reindex(df.index)
    df["xsii"] = _xsii_res["xsii"].reindex(df.index)

    # ══════════════════════════════════════════════════════════════
    # 前沿波动率特征（作为 DL 模型输入，非 label）
    # 来源: volatility.py (Yang-Zhang, Overnight, Garman-Klass, Rogers-Satchell)
    # ══════════════════════════════════════════════════════════════
    try:
        from strategy_highvol.dl_pool.volatility import (
            yang_zhang_vol, rogers_satchell_vol, overnight_vol, garman_klass_vol,
        )
        if all(c in df.columns for c in ["open", "high", "low", "close", "pre_close", "code", "date"]):
            df["yang_zhang_vol_20d"] = yang_zhang_vol(df, window=20)
            df["overnight_vol_20d"] = overnight_vol(df, window=20)
            df["garman_klass_vol_20d"] = garman_klass_vol(df, window=20)
            df["rogers_satchell_vol_20d"] = rogers_satchell_vol(df, window=20)
            # 短期波动率（5日）用于捕捉近期波动变化
            df["yang_zhang_vol_5d"] = yang_zhang_vol(df, window=5)
            df["overnight_vol_5d"] = overnight_vol(df, window=5)
            # 波动率变化比: 短期/长期，>1 表示近期波动放大
            yz_20 = df["yang_zhang_vol_20d"].replace(0, np.nan)
            df["vol_expansion_ratio"] = df["yang_zhang_vol_5d"] / yz_20
    except ImportError:
        pass  # volatility 模块不可用时跳过

    # ══════════════════════════════════════════════════════════════
    # 清理临时列
    # ══════════════════════════════════════════════════════════════
    temp_cols = [col for col in df.columns if col.startswith("_")]
    df = df.drop(columns=temp_cols, errors="ignore")

    # ── 因子列名 ──
    base_cols = set(ohlcv_df.columns)
    factor_cols = [col for col in df.columns if col not in base_cols]

    return df, factor_cols

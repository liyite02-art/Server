"""
分钟频衍生日频因子: 提取 Lob_tickbytick_base6_lyt.py 中与 daily_factors.py 不重复的因子。

已提取因子:
1. JumpVariationFactor  — 日内跳跃分解 (RV/RVC/RVJ/RVSJ 等 13 个)
   来源: process_stock_group_var
   论文: "New Evidence of the Marginal Predictive Content of Small and Large Jumps"
   原理: 将日内分钟对数收益率分解为连续成分 (RVC) 和跳跃成分 (RVJ 等)

2. ExtraCRFactor         — CR2/CR3/CR4 三种中间价变体
   来源: process_stock_group_terminal
   相比 daily_factors.py 已有的 CR1 (midp = (2c+h+l)/4), 新增:
     CR2: midp = (c+h+l+o)/4
     CR3: midp = (c+h+l)/3
     CR4: midp = (h+l)/2

3. BollPositionNormFactor — [0,1] 归一化 Bollinger 位置
   来源: process_stock_group_terminal
   daily_factors.py 的 bb_position = (close - mid) / (2*std), 居中在 0
   本因子 = (close - lower) / (upper - lower), 归一化到 [0,1]

⚠️ 防未来数据:
- JumpVariationFactor 使用 T 日分钟数据, compute_all() 输出已 shift(1)
- ExtraCRFactor 和 BollPositionNormFactor 通过 safe_rolling + shift(1) 保证无未来数据
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from Strategy import config
from Strategy.data_io.loader import MinuteDataLoader, DailyDataLoader
from Strategy.data_io.saver import save_wide_table
from Strategy.factor.factor_base import FactorBase, FactorRegistry
from Strategy.utils.helpers import get_minute_files, strip_stock_prefix

logger = logging.getLogger(__name__)
_EPS = 1e-9
_eps = 1e-5


# ══════════════════════════════════════════════════════════════════
# 辅助: 跳跃分解 (单日)
# ══════════════════════════════════════════════════════════════════
def _jump_variation_single_day(df: pd.DataFrame) -> pd.Series:
    """
    输入: 单日单股分钟 OHLCV (已过滤 930-1500, 包含 925)
    输出: 各跳跃分解指标的日终值

    使用分钟 K 线 log(close/open)*100 作为每根 K 线的对数收益率。
    """
    if len(df) < 5:
        return pd.Series(dtype=float)

    df = df.sort_values("time").reset_index(drop=True)
    open_ = df["open"].astype(float).values
    close_ = df["price"].astype(float).values  # 分钟收盘价

    # 防止 0 价格
    open_safe = np.where(open_ > 0, open_, np.nan)
    close_safe = np.where(close_ > 0, close_, np.nan)
    log_r = np.log(close_safe / open_safe) * 100.0
    log_r = np.where(np.isfinite(log_r), log_r, 0.0)

    n = len(log_r)
    r2 = log_r ** 2

    # Tripower variation (integrated volatility estimate)
    v_hat = np.full(n, 0.0)
    for i in range(2, n):
        v_hat[i] = (abs(log_r[i]) * abs(log_r[i - 1]) * abs(log_r[i - 2])) ** (2.0 / 3.0)

    # Cumulative sums (take EOD = final value)
    RV = r2.sum()
    IV_hat = v_hat.sum()
    RVJ = max(RV - IV_hat, 0.0)
    RVC = RV - RVJ

    # Jump threshold: 99th percentile of |r|
    gamma_r = np.quantile(np.abs(log_r), 0.99)

    # Large / Small jumps
    RVL = (r2 * (np.abs(log_r) > gamma_r)).sum()
    RVLJ = min(RVJ, RVL)
    RVSJ = RVJ - RVLJ

    # Positive / Negative semivariance
    RSP = (r2 * (log_r > 0)).sum()
    RSN = (r2 * (log_r < 0)).sum()

    # Positive / Negative jump variation
    RVJP = max(RSP - 0.5 * IV_hat, 0.0)
    RVJN = max(RSN - 0.5 * IV_hat, 0.0)
    SRVJ = RVJP - RVJN

    # Large positive / negative jumps
    RVLP = (r2 * (log_r > gamma_r)).sum()
    RVLN = (r2 * (log_r < -gamma_r)).sum()
    RVLJP = min(RVJP, RVLP)
    RVLJN = min(RVJN, RVLN)
    RVSJP = RVJP - RVLJP
    RVSJN = RVJN - RVLJN
    SRVLJ = RVLJP - RVLJN
    SRVSJ = RVSJP - RVSJN

    return pd.Series({
        "RV": RV, "RVC": RVC, "RVJ": RVJ, "RVSJ": RVSJ,
        "RSP": RSP, "RSN": RSN, "RVJP": RVJP, "RVJN": RVJN, "SRVJ": SRVJ,
        "RVSJP": RVSJP, "RVSJN": RVSJN, "SRVLJ": SRVLJ, "SRVSJ": SRVSJ,
    })


# ══════════════════════════════════════════════════════════════════
# Factor 1: 跳跃分解 (需要分钟数据逐日迭代)
# ══════════════════════════════════════════════════════════════════
class JumpVariationFactor:
    """
    日内跳跃分解因子 (从分钟 K 线计算, 非 FactorBase 子类)。

    该因子需要逐日加载分钟数据, 计算完成后对所有列 shift(1) 防未来数据。

    输出因子名: RV, RVC, RVJ, RVSJ, RSP, RSN, RVJP, RVJN, SRVJ,
                RVSJP, RVSJN, SRVLJ, SRVSJ
    """

    FACTOR_NAMES = [
        "RV", "RVC", "RVJ", "RVSJ",
        "RSP", "RSN", "RVJP", "RVJN", "SRVJ",
        "RVSJP", "RVSJN", "SRVLJ", "SRVSJ",
    ]

    def __init__(self):
        self._loader = MinuteDataLoader()

    def compute_and_save(
        self,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        逐日计算跳跃分解因子, 汇总为宽表后 shift(1) 保存。

        ⚠️ 因子值在 shift(1) 后, T 日的值代表 T-1 日的跳跃分解信息。
        """
        out = output_dir or config.FACTOR_OUTPUT_DIR
        files = get_minute_files(start_date, end_date)
        logger.info("JumpVariation: 开始计算 %d 个交易日", len(files))

        rows: dict = {}
        for fpath in tqdm(files, desc="JumpVariation 按日计算", unit="day"):
            date_int = int(fpath.stem)
            try:
                df = pd.read_feather(fpath)
                df["StockID"] = df["StockID"].map(strip_stock_prefix)
                # 过滤交易时间 (含竞价 925, 到收盘 1500)
                df = df[(df["time"] >= 925) & (df["time"] <= 1500)]
                date_key = pd.Timestamp(
                    year=date_int // 10000,
                    month=(date_int % 10000) // 100,
                    day=date_int % 100,
                )
                day_results = {}
                for stock, grp in df.groupby("StockID"):
                    s = _jump_variation_single_day(grp)
                    if not s.empty:
                        day_results[stock] = s
                rows[date_key] = pd.DataFrame(day_results).T
            except Exception as e:
                logger.warning("跳过 %s: %s", fpath.name, e)
                continue

        if not rows:
            raise RuntimeError("无有效数据")

        saved_paths = []
        for fname in self.FACTOR_NAMES:
            factor_rows = {}
            for date_key, day_df in rows.items():
                if fname in day_df.columns:
                    factor_rows[date_key] = day_df[fname]
            wide = pd.DataFrame(factor_rows).T.sort_index()
            wide.index.name = "TRADE_DATE"
            wide.columns = pd.Index([str(c).zfill(6) for c in wide.columns])

            # ⚠️ shift(1): T 日因子只用 T-1 及之前分钟数据
            wide = wide.shift(1)

            path = save_wide_table(wide, out / f"{fname}.fea")
            saved_paths.append(path)
            logger.info("  [%s] 已保存: %s, shape=%s", fname, path, wide.shape)

        return saved_paths


# ══════════════════════════════════════════════════════════════════
# Factor 2: CR2 / CR3 / CR4 (日频 OHLCV, FactorBase 子类)
# ══════════════════════════════════════════════════════════════════
# CR1: midp = (2*close + high + low) / 4  ← 已在 daily_factors.py 中
# CR2: midp = (close + high + low + open) / 4  ← 本文件
# CR3: midp = (close + high + low) / 3         ← 本文件
# CR4: midp = (high + low) / 2                 ← 本文件

@FactorRegistry.register
class CR3Factor(FactorBase):
    """CR3: midp = (close + high + low) / 3, 窗口 26 天"""
    name = "CR3"

    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        c = daily_data["CLOSE_PRICE"]
        h = daily_data["HIGHEST_PRICE"]
        l = daily_data["LOWEST_PRICE"]
        midp = (c + h + l) / 3
        up = (h - midp.shift(1)).clip(lower=0)
        dn = (midp.shift(1) - l).clip(lower=0)
        cr3 = (up.rolling(26, min_periods=1).sum() /
               (dn.rolling(26, min_periods=1).sum() + _eps) * 100)
        return cr3.shift(1)


@FactorRegistry.register
class CR4Factor(FactorBase):
    """CR4: midp = (high + low) / 2, 窗口 26 天"""
    name = "CR4"

    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        h = daily_data["HIGHEST_PRICE"]
        l = daily_data["LOWEST_PRICE"]
        midp = (h + l) / 2
        up = (h - midp.shift(1)).clip(lower=0)
        dn = (midp.shift(1) - l).clip(lower=0)
        cr4 = (up.rolling(26, min_periods=1).sum() /
               (dn.rolling(26, min_periods=1).sum() + _eps) * 100)
        return cr4.shift(1)


@FactorRegistry.register
class CR2Factor(FactorBase):
    """CR2: midp = (close + high + low + open) / 4, 窗口 26 天"""
    name = "CR2"

    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        c = daily_data["CLOSE_PRICE"]
        h = daily_data["HIGHEST_PRICE"]
        l = daily_data["LOWEST_PRICE"]
        o = daily_data["OPEN_PRICE"]
        midp = (c + h + l + o) / 4
        up = (h - midp.shift(1)).clip(lower=0)
        dn = (midp.shift(1) - l).clip(lower=0)
        cr2 = (up.rolling(26, min_periods=1).sum() /
               (dn.rolling(26, min_periods=1).sum() + _eps) * 100)
        return cr2.shift(1)


# ══════════════════════════════════════════════════════════════════
# Factor 3: Bollinger Position 归一化到 [0,1]
# ══════════════════════════════════════════════════════════════════
@FactorRegistry.register
class BollPositionNormFactor(FactorBase):
    """
    Bollinger Band Position 归一化到 [0,1]。

    daily_factors.py 已有 bb_position = (close - mid) / (2*std), 中心在 0。
    本因子使用 Lob 文件中的定义:
        boll_pos_norm = (close - lower) / (upper - lower)
    取值范围约 [0, 1], 更直观地表示价格在通道内的相对位置。

    窗口 20 天, 2 倍标准差。
    """
    name = "boll_pos_norm"

    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        c = daily_data["CLOSE_PRICE"]
        window = 20
        k = 2
        mid = c.rolling(window, min_periods=max(1, window // 2)).mean()
        std = c.rolling(window, min_periods=max(1, window // 2)).std()
        upper = mid + k * std
        lower = mid - k * std
        boll_range = upper - lower
        pos = (c - lower) / boll_range.replace(0, np.nan)
        return pos.shift(1)

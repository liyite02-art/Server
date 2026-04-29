"""
分钟频衍生日频因子。

当前本文件仅保留必须直接读取 ``min_data/*.fea`` 的分钟逻辑。
任何仅依赖日频宽表即可计算的因子，已统一迁入 ``daily_factor.py``。

已保留因子:
1. JumpVariationFactor  — 日内跳跃分解 (RV/RVC/RVJ/RVSJ 等 13 个)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from Strategy import config
from Strategy.data_io.loader import MinuteDataLoader
from Strategy.data_io.saver import save_wide_table
from Strategy.utils.helpers import get_minute_files, strip_stock_prefix

logger = logging.getLogger(__name__)
_EPS = 1e-9


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

"""
custom_factors.py — 用户自定义新增因子。

在此文件中继承 FactorBase 并用 @FactorRegistry.register 装饰，
即可自动接入框架的因子计算 / 保存 / 打分流水线。

已有因子体系概览
────────────────────────────────────────────────────────────────
daily_factors_raw.py        原始因子库 (80+ 个), 通过 DailyFactorLibraryAdapter 批量计算
minute_derived_factors.py   分钟数据衍生日频因子 (JumpVariation / CR2-4 / BollPositionNorm)
────────────────────────────────────────────────────────────────

新增因子模板::

    @FactorRegistry.register
    class MyFactor(FactorBase):
        name = "my_factor"

        def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
            close = daily_data["CLOSE_PRICE"]
            # ... 计算逻辑 (只能使用 T-1 及之前数据) ...
            return result.shift(1)   # ⚠️ 必须 shift(1) 防未来数据

调用方式::

    from Strategy.factor.factor_base import FactorRegistry
    import Strategy.factor.custom_factors  # 触发注册
    FactorRegistry.compute_all(daily_data, factor_names=["my_factor"])
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from Strategy.factor.factor_base import FactorBase, FactorRegistry
from Strategy.utils.helpers import safe_rolling


# ── 示例因子 1: 过去 N 日收益率动量 ──────────────────────────────────────
@FactorRegistry.register
class MomentumFactor(FactorBase):
    """
    动量因子: 过去 20 日累计收益率。
    factor(T) = close(T-1) / close(T-21) - 1
    """
    name = "momentum_20d"

    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = daily_data["CLOSE_PRICE"]
        ret = close / close.shift(20) - 1
        return ret.shift(1)


# ── 示例因子 2: 过去 N 日波动率 ──────────────────────────────────────────
@FactorRegistry.register
class VolatilityFactor(FactorBase):
    """
    波动率因子: 过去 20 日收益率标准差。
    safe_rolling 内部已自动 shift(1)。
    """
    name = "volatility_20d"

    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = daily_data["CLOSE_PRICE"]
        daily_ret = close.pct_change(fill_method=None)
        return safe_rolling(daily_ret, window=20, func="std")


# ── 示例因子 3: 换手率均值 ────────────────────────────────────────────────
@FactorRegistry.register
class TurnoverMeanFactor(FactorBase):
    """换手率因子: 过去 10 日平均换手率。"""
    name = "turnover_mean_10d"

    def compute(self, daily_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        turnover = daily_data["TURNOVER_RATE"]
        return safe_rolling(turnover, window=10, func="mean")

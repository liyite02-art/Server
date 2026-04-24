"""
Label 生成模块: 预计算基准价格 (TWAP/VWAP/Close) 并生成收益率 Label。

⚠️ 防未来数据:
- TWAP 仅使用指定时间窗口内的分钟数据
- Label 为"未来收益率", 仅供训练目标使用, 绝不可作为因子输入
- 因子与 Label 的时间对齐: 因子用 T-1 及之前数据, Label 用 T 日收益 (T->T+1)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from Strategy import config
from Strategy.data_io.loader import MinuteDataLoader
from Strategy.data_io.saver import save_wide_table
from Strategy.utils.helpers import get_minute_files, date_to_int

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    可配置的 Label 生成器。

    Parameters
    ----------
    time_start : int
        TWAP/VWAP 计算起始时间, 默认 1430
    time_end : int
        TWAP/VWAP 计算结束时间, 默认 1457
    price_type : str
        'twap' | 'vwap' | 'close'
    """

    def __init__(
        self,
        time_start: int = config.DEFAULT_TWAP_START,
        time_end: int = config.DEFAULT_TWAP_END,
        price_type: str = "twap",
    ):
        self.time_start = time_start
        self.time_end = time_end
        self.price_type = price_type.lower()
        self._loader = MinuteDataLoader()

        self._tag = f"{self.price_type.upper()}_{self.time_start}_{self.time_end}"

    # ─── 单日基准价格 ───────────────────────────────────────────────
    def _compute_day_price(self, date: int) -> pd.Series:
        """计算单日每只股票的基准价格 (TWAP / VWAP / Close)"""
        df = self._loader.load_single_day(date)
        mask = (df["time"] >= self.time_start) & (df["time"] <= self.time_end)
        sub = df.loc[mask]

        if sub.empty:
            return pd.Series(dtype=float)

        if self.price_type == "twap":
            return sub.groupby("StockID")["price"].mean()
        elif self.price_type == "vwap":
            g = sub.groupby("StockID")
            vwap = g["amount"].sum() / g["vol"].sum()
            return vwap.replace([np.inf, -np.inf], np.nan)
        elif self.price_type == "close":
            return sub.groupby("StockID")["price"].last()
        else:
            raise ValueError(f"不支持的 price_type: {self.price_type}")

    # ─── 批量计算基准价格宽表 ───────────────────────────────────────
    def compute_price_table(
        self,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        遍历所有分钟数据文件, 计算每日基准价格, 输出宽表。

        Returns
        -------
        pd.DataFrame
            index=TRADE_DATE (datetime), columns=股票代码, values=基准价格
        """
        files = get_minute_files(start_date, end_date)
        logger.info("开始计算 %s, 共 %d 个交易日", self._tag, len(files))

        rows = {}
        for fpath in tqdm(files, desc=f"LabelGenerator [{self._tag}]", unit="day"):
            date_int = int(fpath.stem)
            try:
                price_series = self._compute_day_price(date_int)
                date_key = pd.Timestamp(
                    year=date_int // 10000,
                    month=(date_int % 10000) // 100,
                    day=date_int % 100,
                )
                rows[date_key] = price_series
            except FileNotFoundError:
                logger.warning("文件缺失, 跳过: %s", fpath)
                continue

        price_df = pd.DataFrame(rows).T.sort_index()
        price_df.index.name = "TRADE_DATE"
        logger.info("基准价格表计算完成: shape=%s", price_df.shape)
        return price_df

    # ─── 计算 Label (未来收益率) ────────────────────────────────────
    def compute_label(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        label(T) = price(T+1) / price(T) - 1

        ⚠️ 这是"未来收益率", 仅作为训练目标!
        使用 shift(-1) 取次日价格, 最后一行为 NaN。
        """
        label = price_df.shift(-1) / price_df - 1
        label.index.name = "TRADE_DATE"
        return label

    # ─── 一键生成并保存 ────────────────────────────────────────────
    def generate_and_save(
        self,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """
        一键计算基准价格 + Label, 并保存到 outputs/labels/

        Returns
        -------
        (price_path, label_path)
        """
        out = output_dir or config.LABEL_OUTPUT_DIR

        price_df = self.compute_price_table(start_date, end_date)
        price_path = save_wide_table(price_df, out / f"{self._tag}.fea")
        logger.info("基准价格已保存: %s", price_path)

        label_df = self.compute_label(price_df)
        label_path = save_wide_table(label_df, out / f"LABEL_{self._tag}.fea")
        logger.info("Label 已保存: %s", label_path)

        return price_path, label_path


def load_label(tag: str = "TWAP_1430_1457") -> pd.DataFrame:
    """快捷加载已保存的 Label 宽表"""
    path = config.LABEL_OUTPUT_DIR / f"LABEL_{tag}.fea"
    df = pd.read_feather(path)
    df = df.set_index("TRADE_DATE")
    return df


def load_price(tag: str = "TWAP_1430_1457") -> pd.DataFrame:
    """快捷加载已保存的基准价格宽表"""
    path = config.LABEL_OUTPUT_DIR / f"{tag}.fea"
    df = pd.read_feather(path)
    df = df.set_index("TRADE_DATE")
    return df

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

    # ─── 计算 Label (未来收益率, 含除息除权调整) ───────────────────
    def compute_label(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算除息除权调整后的未来收益率。

        公式:
            adj_label(T) = TWAP(T+1) / TWAP(T) * CLOSE(T) / PRE_CLOSE(T+1) - 1

        其中:
          - TWAP(T+1) / TWAP(T)      : 原始 TWAP 收益率 (未复权, 除权日会虚假下跌)
          - CLOSE(T) / PRE_CLOSE(T+1): 除权因子:
              PRE_CLOSE 是 T+1 日早上公布的参考价 (已考虑隔夜分红送股),
              CLOSE 是 T 日实际收盘价 (未考虑次日分股);
              普通交易日 CLOSE ≈ PRE_CLOSE, 因子 ≈ 1; 分股日因子放大,
              从而消除 TWAP 收益率中的虚假下跌。

        ⚠️ 这是"未来收益率", 仅作为训练目标!
        使用 shift(-1) 取次日价格, 最后一行为 NaN。
        若 Daily_data 中缺少复权价格文件, 则回退到未复权公式并输出警告。
        """
        close_path     = config.DAILY_DATA_DIR / "CLOSE_PRICE.pkl"
        pre_close_path = config.DAILY_DATA_DIR / "PRE_CLOSE_PRICE.pkl"

        if close_path.exists() and pre_close_path.exists():
            try:
                close_df = pd.read_pickle(close_path)
                close_df.index = pd.DatetimeIndex(close_df.index)
                close_df.columns = pd.Index([str(c).zfill(6) for c in close_df.columns])

                pre_close_df = pd.read_pickle(pre_close_path)
                pre_close_df.index = pd.DatetimeIndex(pre_close_df.index)
                pre_close_df.columns = pd.Index([str(c).zfill(6) for c in pre_close_df.columns])

                # 对齐到 price_df 的日期和股票列
                common_stocks = (
                    price_df.columns
                    .intersection(close_df.columns)
                    .intersection(pre_close_df.columns)
                )
                close_aligned     = close_df.reindex(index=price_df.index, columns=common_stocks)
                pre_close_aligned = pre_close_df.reindex(index=price_df.index, columns=common_stocks)

                # adj_factor(T) = CLOSE(T) / PRE_CLOSE(T+1)
                # PRE_CLOSE(T+1) → shift(-1) 将 T+1 行的数据对齐到 T 的索引
                adj_factor = close_aligned / pre_close_aligned.shift(-1)
                # price_df 中有而 close/pre_close 无的股票, 用 1.0 填充 (不调整)
                adj_factor = adj_factor.reindex(columns=price_df.columns).fillna(1.0)

                # adj_label(T) = (TWAP(T+1) / TWAP(T)) * adj_factor(T) - 1
                raw_return = price_df.shift(-1) / price_df
                label = raw_return.multiply(adj_factor) - 1
                logger.info(
                    "除权调整已应用 (CLOSE / PRE_CLOSE): 共 %d 支股票参与调整",
                    len(common_stocks),
                )
            except Exception as exc:
                logger.warning(
                    "除权调整失败 (%s), 回退到未复权公式: %s", exc.__class__.__name__, exc
                )
                label = price_df.shift(-1) / price_df - 1
        else:
            missing = [p.name for p in (close_path, pre_close_path) if not p.exists()]
            logger.warning(
                "缺少文件 %s, 无法进行除权调整, 使用未复权 Label。"
                "建议确认 Daily_data 中存在 CLOSE_PRICE.pkl 和 PRE_CLOSE_PRICE.pkl",
                missing,
            )
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

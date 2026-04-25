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
from Strategy.utils.helpers import get_minute_files

logger = logging.getLogger(__name__)

# 分钟线约定买入/卖出时点 (与 main.ipynb 早盘回测一致)
OPEN0935_1000_TAG = "OPEN0935_1000"
OPEN0935_TIME = 935
OPEN1000_TIME = 1000


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


def _compute_day_open_at_time(loader: MinuteDataLoader, date_int: int, time_key: int) -> pd.Series:
    """单日指定分钟的 open 价 (按 StockID 聚合, 与 MinuteDataLoader 一致已过滤北交所)."""
    df = loader.load_single_day(date_int)
    sub = df[df["time"] == time_key]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("StockID")["open"].first()


def compute_open_wide_table(
    time_key: int,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
) -> pd.DataFrame:
    """
    从分钟数据提取每个交易日、每只股票在指定 time 的 open 宽表。

    Parameters
    ----------
    time_key : int
        如 935 表示 09:35, 1000 表示 10:00
    """
    loader = MinuteDataLoader()
    files = get_minute_files(start_date, end_date)
    logger.info("开始计算分钟 open 宽表 time=%s, 共 %d 个交易日", time_key, len(files))

    rows: dict = {}
    for fpath in tqdm(files, desc=f"Open@{time_key}", unit="day"):
        date_int = int(fpath.stem)
        try:
            s = _compute_day_open_at_time(loader, date_int, time_key)
            date_key = pd.Timestamp(
                year=date_int // 10000,
                month=(date_int % 10000) // 100,
                day=date_int % 100,
            )
            rows[date_key] = s
        except FileNotFoundError:
            logger.warning("分钟文件缺失, 跳过: %s", fpath)
            continue

    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "TRADE_DATE"
    out.columns = pd.Index([str(c).zfill(6) for c in out.columns])
    logger.info("open 宽表计算完成 time=%s, shape=%s", time_key, out.shape)
    return out


def compute_label_open0935_1000(
    buy_open: pd.DataFrame,
    sell_open: pd.DataFrame,
) -> pd.DataFrame:
    """
    T 日 09:35 open 买入, T+1 日 10:00 open 卖出的收益率 Label。

    未复权收益:
        raw(T) = OPEN_1000(T+1) / OPEN_0935(T) - 1

    除息除权调整与 ``compute_label`` 一致 (对齐到 buy_open 的 index/columns):
        adj(T) = raw_ratio(T) * CLOSE(T) / PRE_CLOSE(T+1) - 1
        其中 raw_ratio(T) = OPEN_1000(T+1) / OPEN_0935(T)

    最后一行因无 T+1 卖出价而为 NaN。
    """
    buy_open = buy_open.sort_index()
    sell_open = sell_open.sort_index()
    common_idx = buy_open.index.intersection(sell_open.index).sort_values()
    common_cols = buy_open.columns.intersection(sell_open.columns)
    buy_a = buy_open.reindex(index=common_idx, columns=common_cols)
    sell_a = sell_open.reindex(index=common_idx, columns=common_cols)
    raw_ratio = sell_a.shift(-1) / buy_a

    close_path = config.DAILY_DATA_DIR / "CLOSE_PRICE.pkl"
    pre_close_path = config.DAILY_DATA_DIR / "PRE_CLOSE_PRICE.pkl"

    if close_path.exists() and pre_close_path.exists():
        try:
            close_df = pd.read_pickle(close_path)
            close_df.index = pd.DatetimeIndex(close_df.index)
            close_df.columns = pd.Index([str(c).zfill(6) for c in close_df.columns])

            pre_close_df = pd.read_pickle(pre_close_path)
            pre_close_df.index = pd.DatetimeIndex(pre_close_df.index)
            pre_close_df.columns = pd.Index([str(c).zfill(6) for c in pre_close_df.columns])

            common_stocks = (
                buy_a.columns.intersection(close_df.columns).intersection(pre_close_df.columns)
            )
            close_aligned = close_df.reindex(index=buy_a.index, columns=common_stocks)
            pre_close_aligned = pre_close_df.reindex(index=buy_a.index, columns=common_stocks)
            adj_factor = close_aligned / pre_close_aligned.shift(-1)
            adj_factor = adj_factor.reindex(columns=buy_a.columns).fillna(1.0)

            label = raw_ratio.multiply(adj_factor) - 1
            logger.info(
                "OPEN0935_1000 Label 除权调整已应用 (CLOSE / PRE_CLOSE), 共 %d 支股票列对齐",
                len(common_stocks),
            )
        except Exception as exc:
            logger.warning(
                "OPEN0935_1000 除权调整失败 (%s), 回退未复权: %s",
                exc.__class__.__name__,
                exc,
            )
            label = raw_ratio - 1
    else:
        missing = [p.name for p in (close_path, pre_close_path) if not p.exists()]
        logger.warning(
            "缺少文件 %s, OPEN0935_1000 使用未复权 Label",
            missing,
        )
        label = raw_ratio - 1

    label.index.name = "TRADE_DATE"
    return label


def generate_and_save_open0935_1000_label(
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    output_dir: Optional[Path] = None,
    save_price_tables: bool = False,
) -> Path:
    """
    生成并保存 ``LABEL_OPEN0935_1000.fea``，供训练/回测复用。

    另可选保存两张价格锚点宽表 (与 LABEL 同目录):
    - ``OPEN0935_1000_BUY0935.fea`` : T 日 09:35 open
    - ``OPEN0935_1000_SELL1000.fea`` : T 日 10:00 open

    加载 Label: ``load_label("OPEN0935_1000")``
    """
    out = output_dir or config.LABEL_OUTPUT_DIR
    buy_open = compute_open_wide_table(OPEN0935_TIME, start_date, end_date)
    sell_open = compute_open_wide_table(OPEN1000_TIME, start_date, end_date)
    label = compute_label_open0935_1000(buy_open, sell_open)

    label_path = save_wide_table(label, out / f"LABEL_{OPEN0935_1000_TAG}.fea")
    logger.info("OPEN0935_1000 Label 已保存: %s, shape=%s", label_path, label.shape)

    # 保存买入价格表作为 scorer 的 T 日可交易 mask 锚点。
    # scorer._load_current_price_mask 会查找 f"{label_tag}.fea"
    # 即 OPEN0935_1000.fea; 此处将 09:35 open 宽表保存为该文件名。
    buy_mask_path = save_wide_table(buy_open, out / f"{OPEN0935_1000_TAG}.fea")
    logger.info("OPEN0935_1000 买入价格 mask 已保存: %s", buy_mask_path)

    if save_price_tables:
        buy_path = save_wide_table(buy_open, out / f"{OPEN0935_1000_TAG}_BUY0935.fea")
        sell_path = save_wide_table(sell_open, out / f"{OPEN0935_1000_TAG}_SELL1000.fea")
        logger.info("价格锚点已保存: %s, %s", buy_path, sell_path)

    return Path(label_path)


def load_label(tag: str = "TWAP_1430_1457") -> pd.DataFrame:
    """
    快捷加载已保存的 Label 宽表。

    tag 示例
    --------
    - ``"TWAP_1430_1457"`` -> ``LABEL_TWAP_1430_1457.fea``
    - ``"OPEN0935_1000"`` -> ``LABEL_OPEN0935_1000.fea`` (T 日 09:35 open 买, T+1 日 10:00 open 卖)
    """
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

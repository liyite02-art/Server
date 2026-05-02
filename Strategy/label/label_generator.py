"""
Label 生成模块: 预计算基准价格 (TWAP/VWAP/Close) 并生成收益率 Label。

⚠️ 防未来数据:
- TWAP 仅使用指定时间窗口内的分钟数据
- Label 为"未来收益率", 仅供训练目标使用, 绝不可作为因子输入
- 因子与 Label 的时间对齐（按 ``build_panel`` 行 ``TRADE_DATE=T``）:
  - **TWAP/VWAP 类**: Label 通常为 **T→T+1**（详见各类 TWAP Label 公式）。
  - **CLOSE_PRECLOSE**: 宽表行 ``TRADE_DATE=T`` 上 ``label(T)=CLOSE(T+1)/PRE_CLOSE(T+1)-1``（T+1 日涨跌幅，``PRE_CLOSE`` 含除权除息）；因子 **T-1 收盘后** 可得。
- ``OPEN0935_1000`` / ``OPEN930_1000``：分钟 open 持有期 label（买价 time=935 或首根连续竞价 931/930，卖价 1000）；涨跌停剔除见 ``_compute_label_open_buy_next_day_sell``
  （买入涨停、**T+1 卖出时点价** 跌停）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import json

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from Strategy import config
from Strategy.data_io.loader import MinuteDataLoader
from Strategy.data_io.saver import save_wide_table
from Strategy.utils.helpers import coerce_wide_values_dtype
from Strategy.utils.helpers import get_minute_files

logger = logging.getLogger(__name__)

# 分钟线约定买入/卖出时点 (与 main.ipynb 早盘回测一致)
OPEN0935_1000_TAG = "OPEN0935_1000"
OPEN0935_TIME = 935
# T 日「开盘即买入」→ T+1 日 10:00 卖出；因子为 T-1 收盘后可得。
# 注意: 本仓库 min_data 在 925 集合竞价后，**无独立 time=930 行**，首根连续竞价 K 线多为 **931**（与 0935/1000 能命中同理）。
# 标的名仍用 OPEN930_1000 以保持与已保存 .fea 文件名一致；买价取 time=931 的 open。
OPEN930_1000_TAG = "OPEN930_1000"
# 首根连续竞价分钟: 数据为 931；若你方分钟库提供 930 行，可改为 930 或依赖下方 _compute_day_open_at_time 回退
OPEN0930_TIME = 931
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
    """单日指定分钟的 open 价 (按 StockID 聚合, 与 MinuteDataLoader 一致已过滤北交所).

    若 ``time_key`` 无行情行（常见于数据源不设独立 930 bar），则对 **930** 自动尝试 **931** 作为首根连续竞价 K 线。
    """
    df = loader.load_single_day(date_int)
    sub = df[df["time"] == time_key]
    if sub.empty and time_key == 930:
        sub = df[df["time"] == 931]
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
    limit_up_df: Optional[pd.DataFrame] = None,
    limit_down_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """见 :func:`_compute_label_open_buy_next_day_sell`。"""
    return _compute_label_open_buy_next_day_sell(
        buy_open,
        sell_open,
        log_tag="OPEN0935_1000",
        limit_up_df=limit_up_df,
        limit_down_df=limit_down_df,
    )


def compute_label_open930_1000(
    buy_open: pd.DataFrame,
    sell_open: pd.DataFrame,
    limit_up_df: Optional[pd.DataFrame] = None,
    limit_down_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """T 日 **连续竞价首根 K 线**（本仓库为 time=**931** 的 open）买入, T+1 日 10:00 open 卖出。除权与涨跌停剔除见 ``_compute_label_open_buy_next_day_sell``。"""
    return _compute_label_open_buy_next_day_sell(
        buy_open,
        sell_open,
        log_tag="OPEN930_1000",
        limit_up_df=limit_up_df,
        limit_down_df=limit_down_df,
    )


def _compute_label_open_buy_next_day_sell(
    buy_open: pd.DataFrame,
    sell_open: pd.DataFrame,
    *,
    log_tag: str,
    limit_up_df: Optional[pd.DataFrame] = None,
    limit_down_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    T 日「买入价」宽表 × T+1 日「卖出价」宽表的持有期收益。

    未复权:
        raw_ratio(T) = SELL_OPEN(T+1) / BUY_OPEN(T)

    除权调整 (与 ``LabelGenerator.compute_label`` 一致):
        label(T) = raw_ratio(T) * CLOSE(T) / PRE_CLOSE(T+1) - 1

    时间对齐（因子 **T-1 收盘后** 可得，决策在 **T 及之后** 执行时再看行情）:
      - 宽表行 ``TRADE_DATE=T`` 的 label = **T 日收盘价买入、T+1 日收盘价卖出** 的已实现除权收益（见 ``CLOSE_PRECLOSE``：
        ``buy_open`` 与 ``sell_open`` 均为收盘价宽表时，``SELL(T+1)/BUY(T)-1``）。
      - 分钟线策略（如 **09:35 买、次日 10:00 卖**）：买入价为 ``BUY_OPEN(T)``，卖出价为 ``SELL_OPEN(T+1)``。

    若提供 ``limit_up_df`` / ``limit_down_df``（``Daily_data`` 宽表），则剔除:
      - **T 日买入价涨停**: ``BUY_PRICE(T)`` 与 ``LIMIT_UP_PRICE(T)`` 一致；
      - **T+1 日卖出价跌停**: ``SELL_PRICE(T+1)`` 与 ``LIMIT_DOWN_PRICE(T+1)`` 一致（行 ``T`` 上用 ``sell.shift(-1)`` 对齐）。

    最后一行无 T+1 卖出价为 NaN。
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
                "%s Label 除权调整已应用 (CLOSE / PRE_CLOSE), 共 %d 支股票列对齐",
                log_tag,
                len(common_stocks),
            )
        except Exception as exc:
            logger.warning(
                "%s 除权调整失败 (%s), 回退未复权: %s",
                log_tag,
                exc.__class__.__name__,
                exc,
            )
            label = raw_ratio - 1
    else:
        missing = [p.name for p in (close_path, pre_close_path) if not p.exists()]
        logger.warning(
            "缺少文件 %s, %s 使用未复权 Label",
            missing,
            log_tag,
        )
        label = raw_ratio - 1

    if limit_up_df is not None and limit_down_df is not None:
        limit_up_df = _standardize_daily_wide(limit_up_df)
        limit_down_df = _standardize_daily_wide(limit_down_df)
        lu = limit_up_df.reindex(index=common_idx, columns=common_cols).astype(float)
        ld = limit_down_df.reindex(index=common_idx, columns=common_cols).astype(float)
        mask_buy_limit_up = _limit_touch(buy_a, lu)
        sell_tp1 = sell_a.shift(-1)
        ld_tp1 = ld.shift(-1)
        mask_sell1000_limit_down = _limit_touch(sell_tp1, ld_tp1)
        bad = mask_buy_limit_up | mask_sell1000_limit_down
        label = label.where(~bad)
        logger.info(
            "%s 涨跌停剔除: 买入价涨停=%s | T+1卖出价跌停=%s",
            log_tag,
            int(mask_buy_limit_up.sum().sum()),
            int(mask_sell1000_limit_down.sum().sum()),
        )

    label.index.name = "TRADE_DATE"
    return label


def generate_and_save_open0935_1000_label(
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    output_dir: Optional[Path] = None,
    save_price_tables: bool = False,
    use_limit_price_tables: bool = True,
) -> Path:
    """
    生成并保存 ``LABEL_OPEN0935_1000.fea``，供训练/回测复用。

    另可选保存两张价格锚点宽表 (与 LABEL 同目录):
    - ``OPEN0935_1000_BUY0935.fea`` : T 日 09:35 open
    - ``OPEN0935_1000_SELL1000.fea`` : T 日 10:00 open

    默认使用 ``LIMIT_UP_PRICE.pkl`` / ``LIMIT_DOWN_PRICE.pkl``：剔除 T 日买入价涨停、T+1 日 10:00 跌停样本；
    若 ``use_limit_price_tables=False`` 则不剔除。

    加载 Label: ``load_label("OPEN0935_1000")``
    """
    out = output_dir or config.LABEL_OUTPUT_DIR
    buy_open = compute_open_wide_table(OPEN0935_TIME, start_date, end_date)
    sell_open = compute_open_wide_table(OPEN1000_TIME, start_date, end_date)

    limit_up_df: Optional[pd.DataFrame] = None
    limit_down_df: Optional[pd.DataFrame] = None
    if use_limit_price_tables:
        lim_up_path = config.DAILY_DATA_DIR / "LIMIT_UP_PRICE.pkl"
        lim_dn_path = config.DAILY_DATA_DIR / "LIMIT_DOWN_PRICE.pkl"
        if not lim_up_path.exists() or not lim_dn_path.exists():
            raise FileNotFoundError(
                f"OPEN0935_1000 涨跌停剔除需要 {lim_up_path} 与 {lim_dn_path} "
                f"(或设置 use_limit_price_tables=False)"
            )
        limit_up_df = _standardize_daily_wide(pd.read_pickle(lim_up_path))
        limit_down_df = _standardize_daily_wide(pd.read_pickle(lim_dn_path))
        limit_up_df = _filter_wide_by_yyyymmdd(limit_up_df, start_date, end_date)
        limit_down_df = _filter_wide_by_yyyymmdd(limit_down_df, start_date, end_date)

    label = compute_label_open0935_1000(buy_open, sell_open, limit_up_df, limit_down_df)

    label_path = save_wide_table(label, out / f"LABEL_{OPEN0935_1000_TAG}.fea")
    logger.info("OPEN0935_1000 Label 已保存: %s, shape=%s", label_path, label.shape)

    # 保存买入价格表作为 scorer 的 T 日可交易 mask 锚点。
    # scorer._load_price_mask 会查找 f"{label_tag}.fea"
    # 即 OPEN0935_1000.fea; 此处将 09:35 open 宽表保存为该文件名。
    buy_mask_path = save_wide_table(buy_open, out / f"{OPEN0935_1000_TAG}.fea")
    logger.info("OPEN0935_1000 买入价格 mask 已保存: %s", buy_mask_path)

    if save_price_tables:
        buy_path = save_wide_table(buy_open, out / f"{OPEN0935_1000_TAG}_BUY0935.fea")
        sell_path = save_wide_table(sell_open, out / f"{OPEN0935_1000_TAG}_SELL1000.fea")
        logger.info("价格锚点已保存: %s, %s", buy_path, sell_path)

    return Path(label_path)


def generate_and_save_open930_1000_label(
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    output_dir: Optional[Path] = None,
    save_price_tables: bool = False,
    use_limit_price_tables: bool = True,
) -> Path:
    """
    生成并保存 ``LABEL_OPEN930_1000.fea``：T 日 **连续竞价首根 K 线（本仓库 time=931）** 的 open 买入，T+1 日 **10:00** open 卖出。

    **与因子时间对齐**：因子在 **T-1 日收盘后** 计算完成；本 label 刻画 **开盘后不久买入 → T+1 日 10:00 卖出**
    的除权收益，宽表行 ``TRADE_DATE=T`` 与 ``build_panel`` 一致（``shift(-1)`` 仅跨 **一个交易日**）。

    默认使用 ``LIMIT_UP_PRICE.pkl`` / ``LIMIT_DOWN_PRICE.pkl``：剔除 T 日买入价涨停、T+1 日 10:00 跌停样本。

    另保存 ``OPEN930_1000.fea``（T 日买入价锚点，对应 ``OPEN0930_TIME``，一般为 **931** 行 open，供 ``mask_scores_by_price``）。

    加载: ``load_label("OPEN930_1000")``
    """
    out = output_dir or config.LABEL_OUTPUT_DIR
    buy_open = compute_open_wide_table(OPEN0930_TIME, start_date, end_date)
    sell_open = compute_open_wide_table(OPEN1000_TIME, start_date, end_date)

    limit_up_df: Optional[pd.DataFrame] = None
    limit_down_df: Optional[pd.DataFrame] = None
    if use_limit_price_tables:
        lim_up_path = config.DAILY_DATA_DIR / "LIMIT_UP_PRICE.pkl"
        lim_dn_path = config.DAILY_DATA_DIR / "LIMIT_DOWN_PRICE.pkl"
        if not lim_up_path.exists() or not lim_dn_path.exists():
            raise FileNotFoundError(
                f"OPEN930_1000 涨跌停剔除需要 {lim_up_path} 与 {lim_dn_path} "
                f"(或设置 use_limit_price_tables=False)"
            )
        limit_up_df = _standardize_daily_wide(pd.read_pickle(lim_up_path))
        limit_down_df = _standardize_daily_wide(pd.read_pickle(lim_dn_path))
        limit_up_df = _filter_wide_by_yyyymmdd(limit_up_df, start_date, end_date)
        limit_down_df = _filter_wide_by_yyyymmdd(limit_down_df, start_date, end_date)

    if buy_open.shape[1] == 0 or sell_open.shape[1] == 0:
        raise ValueError(
            "OPEN930_1000：分钟线 open 宽表股票列为空 (buy_open=%s sell_open=%s)。"
            "请确认 %s 下存在对应交易日的 .fea，且分钟 `time` 在买入侧含 **931 或 930**（首根连续竞价 K 线）、卖出侧含 **1000**，"
            "且 loader 未过滤掉全部行。"
            % (buy_open.shape, sell_open.shape, config.MIN_DATA_DIR)
        )

    label = compute_label_open930_1000(buy_open, sell_open, limit_up_df, limit_down_df)

    if label.shape[1] == 0:
        raise ValueError(
            "OPEN930_1000：Label 计算结果无任何股票列，请检查 buy_open/sell_open 与涨跌停掩膜对齐。"
        )

    label_path = save_wide_table(label, out / f"LABEL_{OPEN930_1000_TAG}.fea")
    logger.info("OPEN930_1000 Label 已保存: %s, shape=%s", label_path, label.shape)

    buy_mask_path = save_wide_table(buy_open, out / f"{OPEN930_1000_TAG}.fea")
    logger.info("OPEN930_1000 买入价格 mask 已保存: %s", buy_mask_path)

    if save_price_tables:
        buy_path = save_wide_table(buy_open, out / f"{OPEN930_1000_TAG}_BUY0930.fea")
        sell_path = save_wide_table(sell_open, out / f"{OPEN930_1000_TAG}_SELL1000.fea")
        logger.info("价格锚点已保存: %s, %s", buy_path, sell_path)

    return Path(label_path)


# ─── CLOSE_PRECLOSE: T 收盘买入 → T+1 收盘卖出（日频收盘价宽表）────────────────

DAILY_CLOSE_PRECLOSE_TAG = "CLOSE_PRECLOSE"
# 修改 CLOSE_PRECLOSE 公式时递增，便于与磁盘 LABEL_CLOSE_PRECLOSE.meta.json 对照
CLOSE_PRECLOSE_DEFINITION_VERSION = 3


def _standardize_daily_wide(df: pd.DataFrame) -> pd.DataFrame:
    """与 ``quick_backtest._standardize_wide`` 一致: TRADE_DATE 索引 + 6 位代码列。"""
    out = df.copy()
    if "TRADE_DATE" in out.columns:
        out = out.set_index("TRADE_DATE")
    out.index = pd.DatetimeIndex(out.index)
    out.columns = pd.Index([str(c).zfill(6) for c in out.columns])
    return out.sort_index()


def _limit_touch(price: pd.DataFrame, limit_px: pd.DataFrame) -> pd.DataFrame:
    """
    价格触及涨停价或跌停价 (相对误差 < 1e-4)。
    与 ``Strategy.backtest.quick_backtest._limit_hit`` 一致。
    """
    lu = limit_px.replace(0, np.nan)
    rel = (price - lu).abs() / lu
    hit = rel < 1e-4
    return hit.fillna(False)


def compute_label_close_preclose(
    close_df: pd.DataFrame,
    pre_close_df: pd.DataFrame,
    limit_up_df: Optional[pd.DataFrame] = None,
    limit_down_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    **CLOSE_PRECLOSE**：宽表行 ``TRADE_DATE=T`` 定义为 **T+1 交易日** 的官方口径涨跌幅::

        label(T) = CLOSE(T+1) / PRE_CLOSE(T+1) - 1

    其中 ``PRE_CLOSE(T+1)`` 为 T+1 日开盘前公布的昨收参考价，**已反映除权除息**，与行情软件「当日涨跌幅」分母一致，
    无需再叠乘 ``CLOSE/PRE_CLOSE`` 形式的额外复权因子。

    **经济含义**（与因子 **T-1 收盘后** 可得、**T 收盘买入 → T+1 收盘卖出** 对齐）：持仓跨越 T 收盘至 T+1 收盘时，
    T+1 当日盈亏等价于 **收盘价相对当日开盘参考价** 的收益；除权缺口已由 ``PRE_CLOSE(T+1)`` 吸收。

    **涨跌停**（需 ``LIMIT_*``；与 ``quick_backtest._limit_hit`` 相对误差 < 1e-4）:
      - **PRE_CLOSE 涨停 / 昨收涨停**（``PRE_CLOSE(T+1)`` 由 **T 日收盘** 决定）: ``CLOSE(T)`` 封涨停 → ``CLOSE(T)≈LIMIT_UP(T)``；
      - **CLOSE 跌停**: **卖出日** ``CLOSE(T+1)`` 封跌停；**建仓日** ``CLOSE(T)`` 封跌停亦剔除（收盘跌停流动性异常）。
    """
    close_df = _standardize_daily_wide(close_df)
    pre_close_df = _standardize_daily_wide(pre_close_df)

    common_idx = close_df.index.intersection(pre_close_df.index).sort_values()
    common_cols = close_df.columns.intersection(pre_close_df.columns)

    use_limits = limit_up_df is not None and limit_down_df is not None
    if use_limits:
        limit_up_df = _standardize_daily_wide(limit_up_df)
        limit_down_df = _standardize_daily_wide(limit_down_df)
        common_idx = common_idx.intersection(limit_up_df.index).intersection(limit_down_df.index).sort_values()
        common_cols = common_cols.intersection(limit_up_df.columns).intersection(limit_down_df.columns)

    common_cols = common_cols.sort_values()
    c = close_df.reindex(index=common_idx, columns=common_cols).astype(float)
    p = pre_close_df.reindex(index=common_idx, columns=common_cols).astype(float)

    c_tp1 = c.shift(-1)
    pc_tp1 = p.shift(-1)
    label = c_tp1 / pc_tp1 - 1

    if use_limits:
        lu = limit_up_df.reindex(index=common_idx, columns=common_cols).astype(float)
        ld = limit_down_df.reindex(index=common_idx, columns=common_cols).astype(float)
        mask_preclose_limit_up = _limit_touch(c, lu)
        mask_close_tp1_limit_down = _limit_touch(c_tp1, ld.shift(-1))
        mask_close_t_limit_down = _limit_touch(c, ld)
        bad = mask_preclose_limit_up | mask_close_tp1_limit_down | mask_close_t_limit_down
        label = label.where(~bad)
        logger.info(
            "CLOSE_PRECLOSE 涨跌停剔除: PRE_CLOSE侧涨停(=T收盘涨停)=%s | T+1收盘跌停=%s | T收盘跌停=%s",
            int(mask_preclose_limit_up.sum().sum()),
            int(mask_close_tp1_limit_down.sum().sum()),
            int(mask_close_t_limit_down.sum().sum()),
        )

    label.index.name = "TRADE_DATE"
    return label


def _filter_wide_by_yyyymmdd(
    df: pd.DataFrame,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
) -> pd.DataFrame:
    """按 index 上的交易日筛 [start_date, end_date] (int YYYYMMDD)。"""
    out = df
    if start_date is not None:
        t0 = pd.Timestamp(str(int(start_date)))
        out = out[out.index >= t0]
    if end_date is not None:
        t1 = pd.Timestamp(str(int(end_date)))
        out = out[out.index <= t1]
    return out


def generate_and_save_close_preclose_label(
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    output_dir: Optional[Path] = None,
    use_limit_price_tables: bool = True,
) -> Path:
    """
    生成 ``LABEL_CLOSE_PRECLOSE.fea``：``CLOSE(T+1)/PRE_CLOSE(T+1)-1``（T+1 日官方涨跌幅，含除权除息），
    并保存 ``CLOSE_PRECLOSE.fea``（T 日收盘价作买入锚点，供 ``mask_scores_by_price``）。

    需要 ``CLOSE_PRICE.pkl`` 与 ``PRE_CLOSE_PRICE.pkl``；涨跌停剔除默认需要 ``LIMIT_UP_PRICE`` / ``LIMIT_DOWN_PRICE``
    （PRE_CLOSE/昨收侧涨停、T+1 收盘跌停、T 收盘跌停，见 :func:`compute_label_close_preclose`）。

    加载: ``load_label("CLOSE_PRECLOSE")``
    """
    close_path = config.DAILY_DATA_DIR / "CLOSE_PRICE.pkl"
    pre_path = config.DAILY_DATA_DIR / "PRE_CLOSE_PRICE.pkl"
    lim_up_path = config.DAILY_DATA_DIR / "LIMIT_UP_PRICE.pkl"
    lim_dn_path = config.DAILY_DATA_DIR / "LIMIT_DOWN_PRICE.pkl"

    if not close_path.exists() or not pre_path.exists():
        raise FileNotFoundError(
            f"需要日频文件: {close_path} 与 {pre_path}"
        )

    close_df = _standardize_daily_wide(pd.read_pickle(close_path))
    pre_close_df = _standardize_daily_wide(pd.read_pickle(pre_path))

    limit_up_df: Optional[pd.DataFrame] = None
    limit_down_df: Optional[pd.DataFrame] = None
    if use_limit_price_tables:
        if not lim_up_path.exists() or not lim_dn_path.exists():
            raise FileNotFoundError(
                f"CLOSE_PRECLOSE 涨跌停剔除需要 {lim_up_path} 与 {lim_dn_path} "
                f"(或设置 use_limit_price_tables=False)"
            )
        limit_up_df = _standardize_daily_wide(pd.read_pickle(lim_up_path))
        limit_down_df = _standardize_daily_wide(pd.read_pickle(lim_dn_path))

    close_df = _filter_wide_by_yyyymmdd(close_df, start_date, end_date)
    pre_close_df = _filter_wide_by_yyyymmdd(pre_close_df, start_date, end_date)
    if limit_up_df is not None:
        limit_up_df = _filter_wide_by_yyyymmdd(limit_up_df, start_date, end_date)
    if limit_down_df is not None:
        limit_down_df = _filter_wide_by_yyyymmdd(limit_down_df, start_date, end_date)

    label = compute_label_close_preclose(close_df, pre_close_df, limit_up_df, limit_down_df)

    out = output_dir or config.LABEL_OUTPUT_DIR
    label_path = save_wide_table(label, out / f"LABEL_{DAILY_CLOSE_PRECLOSE_TAG}.fea")
    logger.info(
        "CLOSE_PRECLOSE (CLOSE(T+1)/PRE_CLOSE(T+1)-1) Label 已保存: %s, shape=%s",
        label_path,
        label.shape,
    )

    mask_wide = close_df.reindex(index=label.index, columns=label.columns)
    mask_path = save_wide_table(mask_wide, out / f"{DAILY_CLOSE_PRECLOSE_TAG}.fea")
    logger.info("CLOSE_PRECLOSE 买入价 mask 已保存: %s", mask_path)

    meta_path = out / "LABEL_CLOSE_PRECLOSE.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "tag": "CLOSE_PRECLOSE",
                "definition_version": CLOSE_PRECLOSE_DEFINITION_VERSION,
                "formula": "label(TRADE_DATE=T) = CLOSE(T+1)/PRE_CLOSE(T+1) - 1",
                "align_note": "旧版 CLOSE(T)/PRE_CLOSE(T)-1 在行T上≈T-1收盘→T收盘；若训练仍像该口径，请确认已重新运行 generate 覆盖 .fea",
                "label_fea": str(Path(label_path).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("CLOSE_PRECLOSE 元数据: %s", meta_path)

    return Path(label_path)


def load_label(
    tag: str = "TWAP_1430_1457",
    dtype: Optional[Union[str, np.dtype]] = None,
) -> pd.DataFrame:
    """
    快捷加载已保存的 Label 宽表。

    tag 示例
    --------
    - ``"TWAP_1430_1457"`` -> ``LABEL_TWAP_1430_1457.fea``
    - ``"OPEN0935_1000"`` -> ``LABEL_OPEN0935_1000.fea``
      (T 日 09:35 open 买, T+1 日 10:00 open 卖；可选剔除 T 日开盘涨停、T+1 日 10:00 跌停,
      见 ``generate_and_save_open0935_1000_label`` 与 ``LIMIT_*``)
    - ``"OPEN930_1000"`` -> ``LABEL_OPEN930_1000.fea``
      (T 日 09:30 open 买, T+1 日 10:00 open 卖；同上剔除逻辑；因子 T-1 收盘后可得)
    - ``"CLOSE_PRECLOSE"`` -> ``LABEL_CLOSE_PRECLOSE.fea``
      (``CLOSE(T+1)/PRE_CLOSE(T+1)-1``，T+1 日涨跌幅口径；涨跌停见 ``compute_label_close_preclose``)
    """
    path = config.LABEL_OUTPUT_DIR / f"LABEL_{tag}.fea"
    df = pd.read_feather(path)
    df = coerce_wide_values_dtype(df, dtype)
    df = df.set_index("TRADE_DATE")
    return df


def load_price(tag: str = "TWAP_1430_1457") -> pd.DataFrame:
    """快捷加载已保存的基准价格宽表"""
    path = config.LABEL_OUTPUT_DIR / f"{tag}.fea"
    df = pd.read_feather(path)
    df = df.set_index("TRADE_DATE")
    return df

"""
数据载入模块: 分钟频 / 日频数据的统一加载接口。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from Strategy import config
from Strategy.utils.helpers import (
    strip_stock_prefix,
    is_sh_or_sz,
    is_sh_or_sz_by_num,
    standardize_stock_column,
    get_minute_files,
    date_to_int,
)


class MinuteDataLoader:
    """分钟频数据加载器 (Feather 格式, 按日存储)"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or config.MIN_DATA_DIR

    def load_single_day(self, date: int) -> pd.DataFrame:
        """
        加载单日分钟数据。自动完成:
        1. 过滤掉北交所 (BJ/NE) 股票
        2. StockID 去前缀 (SZ000001 -> 000001)
        3. 返回标准 DataFrame
        """
        year = str(date)[:4]
        fpath = self.data_dir / year / f"{date}.fea"
        if not fpath.exists():
            raise FileNotFoundError(f"分钟数据文件不存在: {fpath}")

        df = pd.read_feather(fpath)
        df = df[df["StockID"].map(is_sh_or_sz)]
        df["StockID"] = df["StockID"].map(strip_stock_prefix)
        return df

    def load_date_range(
        self,
        start: int,
        end: int,
        time_filter: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        按日期范围逐日加载并拼接。

        Parameters
        ----------
        start, end : int
            日期范围, 如 20210104, 20231231
        time_filter : tuple, optional
            (start_time, end_time), 如 (1430, 1457), 加载时即过滤减少内存占用

        Returns
        -------
        pd.DataFrame  拼接后的长表
        """
        files = get_minute_files(start, end)
        chunks = []
        for fpath in files:
            df = pd.read_feather(fpath)
            df = df[df["StockID"].map(is_sh_or_sz)]
            df["StockID"] = df["StockID"].map(strip_stock_prefix)
            if time_filter is not None:
                df = df[(df["time"] >= time_filter[0]) & (df["time"] <= time_filter[1])]
            chunks.append(df)

        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    def load_single_day_pivot(
        self,
        date: int,
        time_start: int,
        time_end: int,
        value_col: str = "price",
        agg: str = "mean",
    ) -> pd.Series:
        """
        加载单日数据, 筛选时间窗口, 按股票聚合为 Series。
        用于高效计算 TWAP/VWAP 等, 避免加载全量后再 pivot。
        """
        df = self.load_single_day(date)
        mask = (df["time"] >= time_start) & (df["time"] <= time_end)
        sub = df.loc[mask]
        if agg == "mean":
            return sub.groupby("StockID")[value_col].mean()
        elif agg == "sum":
            return sub.groupby("StockID")[value_col].sum()
        else:
            return sub.groupby("StockID")[value_col].agg(agg)


class DailyDataLoader:
    """日频数据加载器 (Pickle 宽表格式)"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or config.DAILY_DATA_DIR

    def load_field(
        self,
        field_name: str,
        as_of_date: Optional[Union[dt.date, str, int]] = None,
    ) -> pd.DataFrame:
        """
        加载单个日频字段的宽表。

        Parameters
        ----------
        field_name : str
            字段名, 如 'CLOSE_PRICE'
        as_of_date : optional
            ⚠️ 防未来数据: 若指定, 自动截断到该日期及之前的数据

        Returns
        -------
        pd.DataFrame  (index=TRADE_DATE, columns=股票代码)
        """
        fpath = self.data_dir / f"{field_name}.pkl"
        if not fpath.exists():
            raise FileNotFoundError(f"日频数据文件不存在: {fpath}")

        df = pd.read_pickle(fpath)
        df.columns = standardize_stock_column(df.columns)
        sh_sz_cols = [c for c in df.columns if is_sh_or_sz_by_num(c)]
        df = df[sh_sz_cols]

        if as_of_date is not None:
            cutoff = pd.Timestamp(self._normalize_date(as_of_date))
            df = df.loc[pd.DatetimeIndex(df.index) <= cutoff]

        return df

    def load_fields(
        self,
        field_names: List[str],
        as_of_date: Optional[Union[dt.date, str, int]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """批量加载多个日频字段, 返回 {field_name: DataFrame} 字典"""
        return {name: self.load_field(name, as_of_date) for name in field_names}

    @staticmethod
    def _normalize_date(d) -> dt.date:
        if isinstance(d, dt.date):
            return d
        if isinstance(d, pd.Timestamp):
            return d.date()
        if isinstance(d, int):
            return dt.date(d // 10000, (d % 10000) // 100, d % 100)
        if isinstance(d, str):
            return pd.Timestamp(d).date()
        raise TypeError(f"无法识别的日期格式: {type(d)}")

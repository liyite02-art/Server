"""
生成 VOLUME.pkl: 每日全市场成交量汇总 (分钟频 vol 列 925-1500 之和)。

运行方式:
    cd /root/autodl-tmp
    python -m Strategy.data_io.generate_volume

输出:
    /root/autodl-tmp/Daily_data/VOLUME.pkl
    宽表格式: index=TRADE_DATE, columns=股票代码 (6位), values=当日总成交量(股)
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from Strategy import config
from Strategy.utils.helpers import get_minute_files, strip_stock_prefix

logger = logging.getLogger(__name__)


def generate_volume_pkl(
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    output_path: Optional[Path] = None,
    time_start: int = 925,
    time_end: int = 1500,
) -> Tuple[Path, List[int]]:
    """
    遍历分钟数据文件, 对每日每只股票的 vol 列 (time 在 [time_start, time_end] 内) 求和,
    拼接为宽表并保存为 VOLUME.pkl。

    原始数据损坏 (如 "Not an Arrow file") 的日期会被跳过, 并在返回值中记录。

    Parameters
    ----------
    start_date, end_date : int, optional
        日期范围, 如 20210104, 20261231
    output_path : Path, optional
        输出路径, 默认保存到 Daily_data/VOLUME.pkl
    time_start, time_end : int
        时间过滤范围, 默认 925-1500

    Returns
    -------
    (Path, List[int])
        保存路径 及 读取失败的日期列表 (YYYYMMDD 整数格式)
    """
    out_path = output_path or (config.DAILY_DATA_DIR / "VOLUME.pkl")
    files = get_minute_files(start_date, end_date)
    logger.info("开始生成 VOLUME.pkl: %d 个交易日 (time %d~%d)", len(files), time_start, time_end)

    rows: dict = {}
    bad_dates: List[int] = []

    for fpath in tqdm(files, desc="生成 VOLUME (按日汇总)", unit="day"):
        date_int = int(fpath.stem)
        try:
            df = pd.read_feather(fpath)
            df["StockID"] = df["StockID"].map(strip_stock_prefix)
            mask = (df["time"] >= time_start) & (df["time"] <= time_end)
            vol_series = df.loc[mask].groupby("StockID")["vol"].sum()
            date_key = pd.Timestamp(
                year=date_int // 10000,
                month=(date_int % 10000) // 100,
                day=date_int % 100,
            )
            rows[date_key] = vol_series
            del df
            gc.collect()
        except Exception as e:
            logger.warning("跳过损坏文件 %s: %s", fpath.name, e)
            bad_dates.append(date_int)
            continue

    if not rows:
        raise RuntimeError("未能读取任何分钟数据文件, 请检查路径")

    vol_wide = pd.DataFrame(rows).T.sort_index()
    vol_wide.index.name = "TRADE_DATE"

    # 列名标准化为 6 位纯数字字符串
    vol_wide.columns = pd.Index([str(c).zfill(6) for c in vol_wide.columns])

    vol_wide.to_pickle(out_path)
    logger.info("VOLUME.pkl 已保存: %s, shape=%s", out_path, vol_wide.shape)

    if bad_dates:
        logger.warning(
            "以下 %d 个日期的原始数据文件损坏, 已跳过: %s",
            len(bad_dates),
            bad_dates,
        )
    else:
        logger.info("所有文件读取正常, 无损坏日期。")

    return out_path, bad_dates


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    _, bad = generate_volume_pkl()
    if bad:
        print(f"\n[损坏日期列表] 共 {len(bad)} 个: {bad}")

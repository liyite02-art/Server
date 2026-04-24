"""
数据保存模块: 统一宽表格式的序列化输出。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from Strategy.utils.helpers import standardize_stock_column


def save_wide_table(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str = "fea",
    index_name: str = "TRADE_DATE",
) -> Path:
    """
    将宽表保存到指定路径, 确保格式规范:
    - index 命名为 TRADE_DATE
    - columns 为 6 位纯数字股票代码 (str)

    Parameters
    ----------
    df : pd.DataFrame
        待保存的宽表
    path : str or Path
        输出路径
    fmt : str
        'fea' (feather) 或 'pkl' (pickle)
    index_name : str
        索引列名称

    Returns
    -------
    Path  实际保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out.columns = standardize_stock_column(out.columns)
    out.index.name = index_name

    if fmt == "fea":
        # 先写临时文件再原子替换，避免进程中断时留下损坏的 .fea（仅头几字节合法、读报 Not an Arrow file）
        tmp_path = path.with_suffix(path.suffix + ".part")
        try:
            out.reset_index().to_feather(tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise
    elif fmt == "pkl":
        out.to_pickle(path)
    else:
        raise ValueError(f"不支持的格式: {fmt}, 请使用 'fea' 或 'pkl'")

    return path

"""
模型打分模块: 使用训练好的模型对全市场股票生成每日标准化打分。

输出: 标准宽表 (index=TRADE_DATE, columns=股票代码, values=打分)

支持任意实现 predict(df) -> np.ndarray 接口的模型:
- XGBTrainer
- TransformerTrainer
- 其他自定义模型
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from Strategy import config
from Strategy.data_io.saver import save_wide_table
from Strategy.model.trainer import build_panel

logger = logging.getLogger(__name__)


def _load_current_price_mask(label_tag: str, index: pd.Index, columns: pd.Index) -> Optional[pd.DataFrame]:
    """T 日当前可交易价格 mask; 不使用未来 label 非空性。"""
    price_path = config.LABEL_OUTPUT_DIR / f"{label_tag}.fea"
    if not price_path.exists():
        logger.warning("价格表不存在, 无法按 T 日价格 mask score: %s", price_path)
        return None
    price_df = pd.read_feather(price_path).set_index("TRADE_DATE")
    price_df.index = pd.DatetimeIndex(price_df.index)
    price_df.columns = pd.Index([str(c).zfill(6) for c in price_df.columns])
    return price_df.reindex(index=pd.DatetimeIndex(index), columns=columns).notna()


def mask_scores_by_current_price(
    score_wide: pd.DataFrame,
    label_tag: str = "TWAP_1430_1457",
) -> pd.DataFrame:
    """剔除 T 日没有执行价格的股票, 避免退市/停牌股票进入后续回测候选。"""
    out = score_wide.copy()
    out.index = pd.DatetimeIndex(out.index)
    out.columns = pd.Index([str(c).zfill(6) for c in out.columns])
    mask = _load_current_price_mask(label_tag, out.index, out.columns)
    if mask is None:
        return out
    before = int(out.notna().sum().sum())
    out = out.where(mask)
    after = int(out.notna().sum().sum())
    logger.info("score 已按 T 日价格 mask: before=%d after=%d removed=%d", before, after, before - after)
    return out


def score_all(
    trainer: Any,
    panel: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    对 Panel 中所有日期的股票生成打分, 输出为宽表。

    Parameters
    ----------
    trainer : XGBTrainer | TransformerTrainer | Any
        已训练好的模型, 需实现 predict(df) -> np.ndarray
    panel : pd.DataFrame
        含因子列的 Panel 长表
    normalize : bool
        是否在每日截面内做 Z-Score 标准化

    Returns
    -------
    pd.DataFrame
        标准宽表 (index=TRADE_DATE, columns=股票代码, values=打分)
    """
    raw_scores = trainer.predict(panel)
    panel = panel.copy()
    panel["score"] = raw_scores

    if normalize:
        panel["score"] = panel.groupby("TRADE_DATE")["score"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    score_wide = panel.pivot(index="TRADE_DATE", columns="StockID", values="score")
    score_wide.index.name = "TRADE_DATE"
    return score_wide


def generate_scores(
    trainer: Any,
    factor_dict: Dict[str, pd.DataFrame],
    label_df: pd.DataFrame,
    model_name: str = "xgb",
    label_tag: str = "TWAP_1430_1457",
    normalize: bool = True,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    端到端打分流水线: 拼接 Panel -> 预测 -> 保存宽表。

    Parameters
    ----------
    trainer : XGBTrainer | TransformerTrainer | Any
        已训练好的模型, 需实现 predict(df) -> np.ndarray
    factor_dict : dict
        {factor_name: wide_df}
    label_df : pd.DataFrame
        Label 宽表 (仅用于对齐日期和股票, 打分时不使用 label 值)
    model_name : str
        模型标识
    label_tag : str
        Label 标识
    normalize : bool
        是否截面标准化
    output_dir : Path, optional
        输出目录

    Returns
    -------
    Path  保存路径
    """
    out = output_dir or config.SCORE_OUTPUT_DIR

    panel = build_panel(factor_dict, label_df)
    score_wide = score_all(trainer, panel, normalize=normalize)
    score_wide = mask_scores_by_current_price(score_wide, label_tag=label_tag)

    fname = f"SCORE_{model_name}_{label_tag}.fea"
    path = save_wide_table(score_wide, out / fname)
    logger.info("打分已保存: %s, shape=%s", path, score_wide.shape)
    return path


def load_scores(
    model_name: str = "xgb",
    label_tag: str = "TWAP_1430_1457",
) -> pd.DataFrame:
    """快捷加载已保存的打分宽表"""
    fname = f"SCORE_{model_name}_{label_tag}.fea"
    path = config.SCORE_OUTPUT_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"打分文件不存在: {path}")
    df = pd.read_feather(path)
    df = df.set_index("TRADE_DATE")
    df = mask_scores_by_current_price(df, label_tag=label_tag)
    return df

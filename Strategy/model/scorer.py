"""
模型打分模块: 生成每日全市场标准化打分宽表。

支持两种调用路径:
  1. 单模型打分 (score_all):
       直接使用一个已训练的模型对整个 Panel 预测

  2. IS Test 集成打分 (generate_is_test_scores):
       通过 RollingTrainer.predict_is_test() 执行 4-Fold Ensemble

输出格式: 标准宽表 (index=TRADE_DATE, columns=股票代码, values=打分)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from Strategy import config
from Strategy.data_io.saver import save_wide_table
from Strategy.model.trainer import build_panel

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 辅助: 按 T 日可交易价格 mask 剔除停牌/退市股
# ═══════════════════════════════════════════════════════════════════════

def _load_price_mask(label_tag: str, index: pd.Index, columns: pd.Index) -> Optional[pd.DataFrame]:
    """加载 T 日交易价格表, 返回可交易 bool mask。"""
    price_path = config.LABEL_OUTPUT_DIR / f"{label_tag}.fea"
    if not price_path.exists():
        logger.warning("价格表不存在，无法按 T 日价格 mask: %s", price_path)
        return None
    price_df = pd.read_feather(price_path).set_index("TRADE_DATE")
    price_df.index = pd.DatetimeIndex(price_df.index)
    price_df.columns = pd.Index([str(c).zfill(6) for c in price_df.columns])
    return price_df.reindex(index=pd.DatetimeIndex(index), columns=columns).notna()


def mask_scores_by_price(
    score_wide: pd.DataFrame,
    label_tag: str = "TWAP_1430_1457",
) -> pd.DataFrame:
    """剔除 T 日无执行价格的股票, 避免停牌/退市股进入回测候选池。"""
    out = score_wide.copy()
    out.index = pd.DatetimeIndex(out.index)
    out.columns = pd.Index([str(c).zfill(6) for c in out.columns])
    mask = _load_price_mask(label_tag, out.index, out.columns)
    if mask is None:
        return out
    before = int(out.notna().sum().sum())
    out = out.where(mask)
    after = int(out.notna().sum().sum())
    logger.info("price mask: before=%d after=%d removed=%d", before, after, before - after)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 打分函数
# ═══════════════════════════════════════════════════════════════════════

def score_all(
    trainer: Any,
    panel: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    对 Panel 中所有日期的股票生成打分, 输出宽表。

    适用于: 单个已训练模型 (XGBTrainer / TransformerTrainer) 的全量推理。
    IS Test 集成推理请使用 generate_is_test_scores()。

    Parameters
    ----------
    trainer : Any
        已训练好的模型, 需实现 predict(df) -> np.ndarray
    panel : pd.DataFrame
        含因子列的 Panel 长表
    normalize : bool
        是否在每日截面内做 Z-Score 标准化

    Returns
    -------
    pd.DataFrame
        打分宽表 (index=TRADE_DATE, columns=股票代码)
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
    端到端打分流水线 (单模型): 拼接 Panel → 预测 → 价格 mask → 保存宽表。

    Parameters
    ----------
    trainer : Any
        已训练好的模型
    factor_dict : dict
        {factor_name: wide_df}
    label_df : pd.DataFrame
        Label 宽表 (仅用于对齐日期和股票, 打分时不使用 label 值)
    model_name : str
        模型标识, 用于命名输出文件
    label_tag : str
        Label 标识
    normalize : bool
        是否截面标准化
    output_dir : Path, optional
        输出目录, 默认 config.SCORE_OUTPUT_DIR

    Returns
    -------
    Path  保存路径
    """
    out = Path(output_dir or config.SCORE_OUTPUT_DIR)
    panel = build_panel(factor_dict, label_df)
    score_wide = score_all(trainer, panel, normalize=normalize)
    score_wide = mask_scores_by_price(score_wide, label_tag=label_tag)
    fname = f"SCORE_{model_name}_{label_tag}.fea"
    path = save_wide_table(score_wide, out / fname)
    logger.info("打分已保存: %s, shape=%s", path, score_wide.shape)
    return path


def generate_is_test_scores(
    rolling_trainer: Any,
    factor_dict: Dict[str, pd.DataFrame],
    label_df: pd.DataFrame,
    model_name: str = "rolling",
    label_tag: str = "TWAP_1430_1457",
    normalize: bool = True,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    IS Test Set 4-Fold Ensemble 打分流水线。

    流程:
      1. 构建 IS Test Panel (仅保留 IS_TEST_START ~ IS_TEST_END 的行)
      2. 调用 RollingTrainer.predict_is_test() 执行 4-Fold Ensemble
      3. 价格 mask → 保存宽表

    Parameters
    ----------
    rolling_trainer : RollingTrainer
        已完成 train_all_folds() 的 RollingTrainer 实例
    factor_dict : dict
        {factor_name: wide_df}
    label_df : pd.DataFrame
        Label 宽表 (对齐用)
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
    from Strategy.model.trainer import split_panel

    out = Path(output_dir or config.SCORE_OUTPUT_DIR)
    panel = build_panel(factor_dict, label_df)

    # 仅保留 IS Test 部分
    _, is_test_panel, _ = split_panel(panel)
    if len(is_test_panel) == 0:
        raise ValueError(
            "IS Test Panel 为空: 请检查 factor_dict / label_df 的日期覆盖是否包含 IS Test 区间 "
            f"[{config.IS_TEST_START}, {config.IS_TEST_END}]"
        )

    score_wide = rolling_trainer.predict_is_test(is_test_panel, normalize=normalize)
    score_wide = mask_scores_by_price(score_wide, label_tag=label_tag)

    fname = f"SCORE_{model_name}_IS_TEST_{label_tag}.fea"
    path = save_wide_table(score_wide, out / fname)
    logger.info("IS Test 集成打分已保存: %s, shape=%s", path, score_wide.shape)
    return path


def load_scores(
    model_name: str = "rolling",
    label_tag: str = "TWAP_1430_1457",
    is_test: bool = True,
) -> pd.DataFrame:
    """
    快捷加载已保存的打分宽表。

    Parameters
    ----------
    model_name : str
        模型标识
    label_tag : str
        Label 标识
    is_test : bool
        True → 加载 IS Test 集成打分 (SCORE_{model}_IS_TEST_{label}.fea)
        False → 加载普通打分 (SCORE_{model}_{label}.fea)
    """
    if is_test:
        fname = f"SCORE_{model_name}_IS_TEST_{label_tag}.fea"
    else:
        fname = f"SCORE_{model_name}_{label_tag}.fea"

    path = config.SCORE_OUTPUT_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"打分文件不存在: {path}")
    df = pd.read_feather(path).set_index("TRADE_DATE")
    df = mask_scores_by_price(df, label_tag=label_tag)
    return df


def load_score_feather(
    path: Union[str, Path],
    label_tag: Optional[str] = None,
    apply_price_mask: bool = True,
) -> pd.DataFrame:
    """
    读取 ``save_wide_table`` 落盘的 Feather 宽表（含 ``TRADE_DATE`` 列）。

    Parameters
    ----------
    path
        ``.fea`` 文件路径
    label_tag
        若 ``apply_price_mask=True``，按该 tag 加载 ``{LABEL_OUTPUT_DIR}/{label_tag}.fea`` 做可交易 mask
    apply_price_mask
        是否再次套用 ``mask_scores_by_price``（与训练后落盘时一致）
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"打分文件不存在: {path}")
    df = pd.read_feather(path).set_index("TRADE_DATE")
    df.index = pd.DatetimeIndex(df.index)
    if apply_price_mask and label_tag:
        df = mask_scores_by_price(df, label_tag=label_tag)
    return df


def load_is_test_scores_from_disk(
    label_tag: str,
    score_dir: Optional[Path] = None,
    *,
    apply_price_mask: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    加载 ``main.ipynb`` 在 ``config.SCORE_OUTPUT_DIR`` 下保存的 IS Test 打分::

        SCORE_xgb_{label_tag}_is_test.fea
        SCORE_transformer_{label_tag}_is_test.fea
        SCORE_mlp_{label_tag}_is_test.fea
        SCORE_ensemble_{label_tag}_is_test.fea

    若某个模型文件不存在则跳过该键 (不打断仅训练了部分模型的流程)。

    Returns
    -------
    dict
        键为 ``xgb`` / ``transformer`` / ``mlp`` / ``ensemble`` 中存在的子集
    """
    d = Path(score_dir or config.SCORE_OUTPUT_DIR)
    keys = ("xgb", "transformer", "mlp", "ensemble")
    out: Dict[str, pd.DataFrame] = {}
    for k in keys:
        p = d / f"SCORE_{k}_{label_tag}_is_test.fea"
        if not p.exists():
            logger.info("IS Test 打分文件缺失, 跳过: %s", p.name)
            continue
        out[k] = load_score_feather(
            p,
            label_tag=label_tag if apply_price_mask else None,
            apply_price_mask=apply_price_mask,
        )
    return out
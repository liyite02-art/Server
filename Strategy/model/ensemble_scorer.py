"""
跨模型种类集成打分模块。

职责划分:
  - RollingTrainer.predict_is_test()   → 同一模型种类内的 4-Fold Ensemble
  - 本模块 (ensemble_scorer.py)        → 不同模型种类之间的集成
      例如: XGB 4-Fold Score + Transformer 4-Fold Score → 最终信号

典型使用流程:
  # 1. 各模型独立生成 IS Test 打分宽表
  xgb_score = load_scores("xgb_rolling", label_tag="TWAP_1430_1457", is_test=True)
  tfm_score  = load_scores("transformer_rolling", label_tag="TWAP_1430_1457", is_test=True)

  # 2. 按相关性筛选 + 等权平均
  final_score = ensemble_scores(
      {"xgb": xgb_score, "transformer": tfm_score},
      label_tag="TWAP_1430_1457",
  )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Strategy import config
from Strategy.data_io.saver import save_wide_table

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 标准化检查
# ═══════════════════════════════════════════════════════════════════════

def _ensure_normalized(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """检查并确保打分已截面标准化 (mean≈0, std≈1)。"""
    sample_means = df.mean(axis=1).dropna()
    sample_stds  = df.std(axis=1).dropna()
    if sample_means.abs().mean() > 0.5 or abs(sample_stds.mean() - 1.0) > 0.5:
        logger.warning(
            "[%s] 打分可能未标准化 (截面 mean=%.3f, std=%.3f)，自动做 Z-Score",
            name, sample_means.abs().mean(), sample_stds.mean(),
        )
        df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 截面相关性矩阵
# ═══════════════════════════════════════════════════════════════════════

def compute_score_correlation(score_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    计算多个模型打分的截面平均 Pearson 相关性矩阵。

    对每个交易日计算各模型打分的两两 Pearson 相关系数, 取日均值。

    Returns
    -------
    pd.DataFrame
        (n_models × n_models) 相关性矩阵
    """
    names = sorted(score_dfs.keys())
    n = len(names)

    # 对齐日期和股票
    common_dates, common_stocks = None, None
    for df in score_dfs.values():
        idx  = pd.DatetimeIndex(df.index)
        cols = df.columns
        common_dates  = idx  if common_dates  is None else common_dates.intersection(idx)
        common_stocks = cols if common_stocks is None else common_stocks.intersection(cols)

    common_dates = common_dates.sort_values()
    aligned = {
        name: score_dfs[name].reindex(index=common_dates, columns=common_stocks)
        for name in names
    }

    corr_accum = np.zeros((n, n))
    n_days = 0

    for dt in common_dates:
        vectors = {}
        for name in names:
            v = aligned[name].loc[dt].dropna()
            vectors[name] = v

        common_stk = None
        for v in vectors.values():
            common_stk = v.index if common_stk is None else common_stk.intersection(v.index)
        if common_stk is None or len(common_stk) < 20:
            continue

        vals = np.array([vectors[name].reindex(common_stk).values for name in names])
        day_corr = np.corrcoef(vals)
        if not np.isnan(day_corr).any():
            corr_accum += day_corr
            n_days += 1

    if n_days == 0:
        logger.warning("无有效交易日计算相关性")
        return pd.DataFrame(np.nan, index=names, columns=names)

    avg_corr = corr_accum / n_days
    result = pd.DataFrame(avg_corr, index=names, columns=names)
    logger.info("模型打分相关性矩阵 (%d 日均):\n%s", n_days, result.round(3).to_string())
    return result


# ═══════════════════════════════════════════════════════════════════════
# 模型筛选
# ═══════════════════════════════════════════════════════════════════════

def select_ensemble_models(
    score_dfs: Dict[str, pd.DataFrame],
    ic_summaries: Optional[Dict[str, pd.DataFrame]] = None,
    min_icir: float = 0.0,
    max_pairwise_corr: float = 0.85,
) -> List[str]:
    """
    筛选适合集成的跨模型种类子集。

    注意: 此处是跨模型种类的筛选 (如 XGB vs Transformer)。
    同一种类内的 4-Fold Ensemble 由 RollingTrainer.predict_is_test() 负责。

    Parameters
    ----------
    score_dfs : dict
        {model_name: score_wide_df}，已完成 4-Fold Ensemble 的打分宽表
    ic_summaries : dict, optional
        {model_name: ic_summary_df}，含 icir 列
    min_icir : float
        最低 ICIR 阈值, 低于此值的模型被排除
    max_pairwise_corr : float
        两两相关性上限; 超过此值只保留 ICIR 更高的

    Returns
    -------
    List[str]  入选的模型名称列表
    """
    candidates = list(score_dfs.keys())

    # 1. ICIR 筛选
    if ic_summaries and min_icir > 0:
        filtered = []
        for name in candidates:
            if name not in ic_summaries:
                filtered.append(name)
                continue
            try:
                icir_val = ic_summaries[name].loc[("rank_ic", "IS_Test"), "icir"]
                if icir_val >= min_icir:
                    filtered.append(name)
                else:
                    logger.info("排除 %s: ICIR=%.4f < %.4f", name, icir_val, min_icir)
            except KeyError:
                filtered.append(name)
        candidates = filtered

    if len(candidates) <= 1:
        return candidates

    # 2. 相关性去冗余 (贪心: ICIR 高的优先, 与已选模型相关性 > 阈值则跳过)
    corr_matrix = compute_score_correlation({k: score_dfs[k] for k in candidates})

    icir_order = {}
    for name in candidates:
        if ic_summaries and name in ic_summaries:
            try:
                icir_order[name] = ic_summaries[name].loc[("rank_ic", "IS_Test"), "icir"]
            except KeyError:
                icir_order[name] = 0.0
        else:
            icir_order[name] = 0.0

    sorted_cands = sorted(candidates, key=lambda x: icir_order.get(x, 0), reverse=True)
    selected = [sorted_cands[0]]
    for name in sorted_cands[1:]:
        too_correlated = any(
            abs(corr_matrix.loc[name, sel]) > max_pairwise_corr
            for sel in selected
            if name in corr_matrix.index and sel in corr_matrix.columns
        )
        if too_correlated:
            logger.info("排除 %s: 与已选模型相关性超阈值 %.2f", name, max_pairwise_corr)
        else:
            selected.append(name)

    logger.info("跨模型集成入选: %s (%d / %d)", selected, len(selected), len(score_dfs))
    return selected


# ═══════════════════════════════════════════════════════════════════════
# 集成打分
# ═══════════════════════════════════════════════════════════════════════

def ensemble_scores(
    score_dfs: Dict[str, pd.DataFrame],
    selected_models: Optional[List[str]] = None,
    label_tag: str = "TWAP_1430_1457",
    save: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    将多个模型种类的标准化打分取简单平均, 得到跨模型集成打分。

    Parameters
    ----------
    score_dfs : dict
        {model_name: score_wide_df}
        每个 df 应为已完成 4-Fold Ensemble 后的打分宽表 (截面标准化)
    selected_models : list, optional
        入选模型名, None = 使用所有
    label_tag : str
        Label 标识, 用于保存文件名
    save : bool
        是否保存集成打分到磁盘
    output_dir : Path, optional
        输出目录

    Returns
    -------
    pd.DataFrame
        集成打分宽表 (index=TRADE_DATE, columns=股票代码)
    """
    models = selected_models or list(score_dfs.keys())
    if len(models) == 0:
        raise ValueError("无模型可集成")

    logger.info("跨模型集成: %d 个模型 → %s", len(models), models)

    # 确保截面标准化
    normalized = {name: _ensure_normalized(score_dfs[name], name) for name in models}

    # 对齐日期和股票
    common_dates, common_stocks = None, None
    for df in normalized.values():
        idx  = pd.DatetimeIndex(df.index)
        cols = df.columns
        common_dates  = idx  if common_dates  is None else common_dates.intersection(idx)
        common_stocks = cols if common_stocks is None else common_stocks.intersection(cols)

    common_dates = common_dates.sort_values()
    aligned = [
        normalized[name].reindex(index=common_dates, columns=common_stocks)
        for name in models
    ]

    # 等权平均
    stacked    = np.stack([df.values for df in aligned], axis=0)  # (n_models, dates, stocks)
    avg_scores = np.nanmean(stacked, axis=0)                       # (dates, stocks)

    ensemble_df = pd.DataFrame(avg_scores, index=common_dates, columns=common_stocks)
    ensemble_df.index.name = "TRADE_DATE"

    # 再做一次截面标准化
    ensemble_df = ensemble_df.sub(ensemble_df.mean(axis=1), axis=0)
    ensemble_df = ensemble_df.div(ensemble_df.std(axis=1) + 1e-8, axis=0)

    logger.info("跨模型集成打分: %d dates × %d stocks", *ensemble_df.shape)

    if save:
        out = Path(output_dir or config.SCORE_OUTPUT_DIR)
        fname = f"SCORE_ensemble_{label_tag}.fea"
        save_wide_table(ensemble_df, out / fname)
        logger.info("跨模型集成打分已保存: %s", out / fname)

    return ensemble_df
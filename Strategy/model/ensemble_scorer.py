"""
集成打分模块: 将多个模型的打分进行筛选与合并。

流程:
  1. 各模型独立生成标准化打分宽表 (截面 Z-Score, mean=0, std=1)
  2. 计算两两打分截面相关性矩阵
  3. 筛选: IC / Rank IC 高 & 两两相关性 < max_corr
  4. 简单平均 → 集成打分

使用:
  score_dfs = {"xgb": load_scores("xgb", ...), "transformer": load_scores("transformer", ...)}
  ensemble = ensemble_scores(score_dfs)
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


def _ensure_normalized(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """检查并确保打分已截面标准化。"""
    sample_means = df.mean(axis=1).dropna()
    sample_stds  = df.std(axis=1).dropna()
    mean_of_means = sample_means.abs().mean()
    mean_of_stds  = sample_stds.mean()

    if mean_of_means > 0.5 or abs(mean_of_stds - 1.0) > 0.5:
        logger.warning(
            "[%s] 打分可能未标准化 (截面 mean=%.3f, std=%.3f), 自动做 Z-Score",
            name, mean_of_means, mean_of_stds,
        )
        df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)
    return df


def compute_score_correlation(
    score_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    计算多个模型打分的截面平均相关性矩阵。

    对每个交易日计算各模型打分的两两 Pearson 相关系数, 取日均值。

    Returns
    -------
    pd.DataFrame
        (n_models × n_models) 相关性矩阵
    """
    names = sorted(score_dfs.keys())
    n = len(names)

    # 对齐日期和股票
    common_dates = None
    common_stocks = None
    for df in score_dfs.values():
        idx = pd.DatetimeIndex(df.index)
        cols = df.columns
        common_dates  = idx if common_dates is None else common_dates.intersection(idx)
        common_stocks = cols if common_stocks is None else common_stocks.intersection(cols)

    common_dates = common_dates.sort_values()
    aligned = {name: score_dfs[name].reindex(index=common_dates, columns=common_stocks)
               for name in names}

    # 逐日计算两两相关, 取均值
    corr_accum = np.zeros((n, n))
    n_days = 0

    for dt in common_dates:
        vectors = {}
        for name in names:
            v = aligned[name].loc[dt].dropna()
            vectors[name] = v

        # 取公共股票
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
    logger.info("模型打分相关性矩阵 (%d 日平均):\n%s", n_days, result.to_string())
    return result


def select_ensemble_models(
    score_dfs: Dict[str, pd.DataFrame],
    ic_summaries: Optional[Dict[str, pd.DataFrame]] = None,
    min_oos_icir: float = 0.0,
    max_pairwise_corr: float = 0.8,
) -> List[str]:
    """
    筛选适合集成的模型子集。

    Parameters
    ----------
    score_dfs : dict
        {model_name: score_wide_df}
    ic_summaries : dict, optional
        {model_name: ic_summary_df}, 来自 ic_analysis.compute_ic_summary
    min_oos_icir : float
        OOS 段最低 ICIR 阈值, 低于此值的模型被排除
    max_pairwise_corr : float
        两两相关性上限, 超过此值的模型只保留 ICIR 更高的那个

    Returns
    -------
    List[str]
        入选的模型名称列表
    """
    candidates = list(score_dfs.keys())

    # 1. 按 ICIR 筛选
    if ic_summaries and min_oos_icir > 0:
        filtered = []
        for name in candidates:
            if name not in ic_summaries:
                filtered.append(name)  # 无 IC 信息的模型默认入选
                continue
            summary = ic_summaries[name]
            try:
                oos_icir = summary.loc[("rank_ic", "OOS"), "icir"]
                if oos_icir >= min_oos_icir:
                    filtered.append(name)
                else:
                    logger.info("排除 %s: OOS ICIR=%.4f < %.4f", name, oos_icir, min_oos_icir)
            except KeyError:
                filtered.append(name)
        candidates = filtered

    if len(candidates) <= 1:
        return candidates

    # 2. 按相关性去冗余 (贪心: 按 ICIR 降序逐个加入, 与已选模型相关性 > 阈值则跳过)
    corr_matrix = compute_score_correlation(
        {k: score_dfs[k] for k in candidates}
    )

    # 排序: ICIR 高的优先
    icir_order = {}
    for name in candidates:
        if ic_summaries and name in ic_summaries:
            try:
                icir_order[name] = ic_summaries[name].loc[("rank_ic", "Overall"), "icir"]
            except KeyError:
                icir_order[name] = 0.0
        else:
            icir_order[name] = 0.0

    sorted_candidates = sorted(candidates, key=lambda x: icir_order.get(x, 0), reverse=True)

    selected = [sorted_candidates[0]]
    for name in sorted_candidates[1:]:
        # 检查与所有已选模型的相关性
        too_correlated = False
        for sel in selected:
            try:
                corr_val = corr_matrix.loc[name, sel]
                if abs(corr_val) > max_pairwise_corr:
                    logger.info("排除 %s: 与 %s 相关性 %.3f > %.3f",
                                name, sel, corr_val, max_pairwise_corr)
                    too_correlated = True
                    break
            except KeyError:
                pass
        if not too_correlated:
            selected.append(name)

    logger.info("集成模型入选: %s (共 %d / %d)", selected, len(selected), len(score_dfs))
    return selected


def ensemble_scores(
    score_dfs: Dict[str, pd.DataFrame],
    selected_models: Optional[List[str]] = None,
    label_tag: str = "TWAP_1430_1457",
    save: bool = True,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    将多个模型的标准化打分取简单平均, 得到集成打分。

    Parameters
    ----------
    score_dfs : dict
        {model_name: score_wide_df}, 每个 df 应已截面标准化
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
        集成打分宽表
    """
    models = selected_models or list(score_dfs.keys())
    if len(models) == 0:
        raise ValueError("无模型可集成")

    logger.info("集成 %d 个模型: %s", len(models), models)

    # 确保标准化
    normalized = {name: _ensure_normalized(score_dfs[name], name) for name in models}

    # 对齐日期和股票
    common_dates = None
    common_stocks = None
    for df in normalized.values():
        idx = pd.DatetimeIndex(df.index)
        cols = df.columns
        common_dates  = idx if common_dates is None else common_dates.intersection(idx)
        common_stocks = cols if common_stocks is None else common_stocks.intersection(cols)

    common_dates = common_dates.sort_values()
    aligned = [normalized[name].reindex(index=common_dates, columns=common_stocks)
               for name in models]

    # 简单平均
    stacked = np.stack([df.values for df in aligned], axis=0)  # (n_models, n_dates, n_stocks)
    avg_scores = np.nanmean(stacked, axis=0)                   # (n_dates, n_stocks)

    ensemble_df = pd.DataFrame(avg_scores, index=common_dates, columns=common_stocks)
    ensemble_df.index.name = "TRADE_DATE"

    # 再做一次截面标准化
    ensemble_df = ensemble_df.sub(ensemble_df.mean(axis=1), axis=0)
    ensemble_df = ensemble_df.div(ensemble_df.std(axis=1) + 1e-8, axis=0)

    logger.info("集成打分: %d dates × %d stocks", *ensemble_df.shape)

    if save:
        out = Path(output_dir or config.SCORE_OUTPUT_DIR)
        fname = f"SCORE_ensemble_{label_tag}.fea"
        save_wide_table(ensemble_df, out / fname)
        logger.info("集成打分已保存: %s", out / fname)

    return ensemble_df

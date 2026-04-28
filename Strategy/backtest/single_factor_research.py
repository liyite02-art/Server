"""
单因子研究: 与 ``pipeline_train_score_backtest`` 中模型打分使用同一套
``compute_daily_ic`` / ``run_ic_analysis`` / ``run_quick_backtest``,
便于把「基线因子」与 XGB/Transformer/集成 的 IC、分层回测图对照。

典型流程
--------
1. ``ic_scan_dataframe`` —— 对全部 .fea 因子做截面 IC, 得到总表 ``factor_ic_scan.csv``（无图, 快）
2. 按 OOS 均值 / ICIR 排序, 对关心的若干因子调用 ``export_factor_report`` 生成
   ``ic_val_to_end/`` 与 ``quick_val_to_end/``（与 pipeline 子目录结构一致）
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from Strategy import config
from Strategy.backtest.ic_analysis import compute_daily_ic, compute_ic_summary, run_ic_analysis
from Strategy.backtest.quick_backtest import build_tradeable_mask, run_quick_backtest
from Strategy.label.label_generator import load_label
from Strategy.utils.helpers import ensure_tradedate_as_index

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^\w\-.]+")


def sanitize_dir_name(name: str) -> str:
    """用于子目录名 (因子名里可能有特殊字符)。"""
    s = _UNSAFE.sub("_", str(name).strip())
    return s[:120] or "factor"


def _standardize_wide_like_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_tradedate_as_index(df)
    if "TRADE_DATE" in out.columns:
        out = out.set_index("TRADE_DATE")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out.index = pd.DatetimeIndex(out.index)
    out.columns = pd.Index([str(c).zfill(6) for c in out.columns])
    return out.sort_index()


def build_reference_tradeable_mask(
    reference_score: pd.DataFrame,
    twap_tag: str = "OPEN0935_1000",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    与 ``run_quick_backtest`` 一致: 用参考打分宽表 (任意列覆盖度足够的表即可)
    与 Label 同维度股票池, 建可交易 bool mask, 使单因子 IC 与组合模型 IC 口径一致。

    常用: 用 pipeline 产出的 ``SCORE_xgb_*.fea`` 或全样本 panel 上某一列展成的表。
    """
    score = _standardize_wide_like_pipeline(reference_score)
    m, _ = build_tradeable_mask(score, **kwargs)
    return m


def _summary_to_flat_row(
    factor_name: str, summary: pd.DataFrame, prefix: str = ""
) -> Dict[str, Any]:
    row: Dict[str, Any] = {prefix + "factor": factor_name}
    if summary.empty:
        return row
    for idx in summary.index:
        if isinstance(idx, tuple) and len(idx) == 2:
            metric, seg = idx[0], idx[1]
        else:
            continue
        srow = summary.loc[idx]
        for c in srow.index:
            key = f"{prefix}{metric}_{str(seg)}_{c}"
            v = srow[c]
            row[key] = float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v
    return row


def ic_one_factor(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    tradeable_mask: Optional[pd.DataFrame] = None,
    min_stocks: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """单因子: 日度 IC 序列 + Train/Val/OOS 分段表。"""
    s = _standardize_wide_like_pipeline(score_df)
    l = _standardize_wide_like_pipeline(label_df)
    ic_df = compute_daily_ic(
        s, l, tradeable_mask=tradeable_mask, min_stocks=min_stocks
    )
    summ = compute_ic_summary(ic_df)
    return ic_df, summ


def ic_scan_dataframe(
    label_df: pd.DataFrame,
    factors: Dict[str, pd.DataFrame],
    tradeable_mask: Optional[pd.DataFrame] = None,
    min_stocks: int = 10,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    批量单因子, 只算 IC 汇总, 不画图。

    Returns
    -------
    pd.DataFrame
        每行一个因子, 列为 ``ic_OOS_mean`` 等 (由 ``compute_ic_summary`` 展平) + ``factor`` 列
    """
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **kw):
            return x

    rows: List[Dict[str, Any]] = []
    it = factors.items()
    if show_progress:
        it = tqdm(list(it), desc="IC scan (factors)")

    for name, fdf in it:
        try:
            _, summ = ic_one_factor(
                fdf, label_df,
                tradeable_mask=tradeable_mask,
                min_stocks=min_stocks,
            )
        except Exception as e:
            logger.warning("IC scan 跳过 [%s]: %s", name, e)
            continue
        row = _summary_to_flat_row(name, summ, prefix="")
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if "factor" not in out.columns and "ic_factor" not in out.columns:
        out.insert(0, "factor", [r.get("factor") for r in rows])
    return out


def load_all_factors_glob() -> Dict[str, pd.DataFrame]:
    """``outputs/factors`` 下全部 ``.fea``。"""
    from Strategy.factor.factor_base import load_all_factors
    return load_all_factors()


def export_factor_report_discrete_ic_then_bt(
    factor_name: str,
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    out_root: Path,
    tradeable_mask: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> None:
    """先只做 IC 目录, 再 quick 且 ``run_ic=False`` (与 pipeline 一致)。"""
    kwargs.setdefault("val_start", config.IS_TEST_START)
    kwargs.setdefault("end_date", None)
    kwargs.setdefault("twap_tag", "OPEN0935_1000")
    base = out_root / sanitize_dir_name(factor_name)
    base.mkdir(parents=True, exist_ok=True)
    s = _standardize_wide_like_pipeline(score_df)
    l = _standardize_wide_like_pipeline(label_df)
    t0 = pd.Timestamp(kwargs.get("val_start", config.IS_TEST_START))

    run_ic_analysis(
        s, l,
        tradeable_mask=tradeable_mask,
        title=f"{factor_name} | {kwargs.get('twap_tag')} | IC",
        output_dir=base / "ic_val_to_end",
        min_stocks=10,
        rolling_window=20,
    )
    run_quick_backtest(
        s, l,
        n_groups=kwargs.get("n_groups", config.N_QUANTILE_GROUPS),
        title=f"{factor_name} | {kwargs.get('twap_tag')} | quick",
        output_dir=base / "quick_val_to_end",
        start_date=t0,
        end_date=kwargs.get("end_date"),
        top_ks=kwargs.get("top_ks", (20, 50, 100)),
        tail_ks=kwargs.get("tail_ks", (20, 50, 100)),
        twap_tag=kwargs.get("twap_tag", "OPEN0935_1000"),
        run_ic=False,
    )


def run_ic_scan_to_csv(
    out_csv: Union[str, Path],
    label_tag: str = "OPEN0935_1000",
    reference_score: Optional[pd.DataFrame] = None,
    factors: Optional[Dict[str, pd.DataFrame]] = None,
    **scan_kw: Any,
) -> pd.DataFrame:
    """
    便利: 读 Label + 因子, 做 IC 扫描, 存 CSV, 返回表。

    ``reference_score`` 可传 ``pd.read_feather`` 的 pipeline 打分 (如 xgb 全区间),
    以建立与主流程一致的可交易池 mask; 为 None 则不用 mask (仅用 score/label 非空)。
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    label_df = load_label(label_tag)
    fac = factors or load_all_factors_glob()
    mask = None
    if reference_score is not None:
        mask = build_reference_tradeable_mask(reference_score)
    df = ic_scan_dataframe(label_df, fac, tradeable_mask=mask, **scan_kw)
    df.to_csv(out_csv, index=False)
    logger.info("IC 扫描表: %d 因子, 已写 %s", len(df), out_csv)
    return df


__all__ = [
    "sanitize_dir_name",
    "build_reference_tradeable_mask",
    "ic_one_factor",
    "ic_scan_dataframe",
    "load_all_factors_glob",
    "export_factor_report_discrete_ic_then_bt",
    "run_ic_scan_to_csv",
]

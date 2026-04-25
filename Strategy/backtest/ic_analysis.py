"""
IC / Rank IC 分析模块

计算模型打分（Score）与下期实际收益（Label）的截面相关性：
- IC  (Information Coefficient) : 皮尔逊线性相关
- Rank IC                       : 斯皮尔曼秩相关

输出：
  ic_series.csv   — 每日 IC / Rank IC 时间序列
  ic_summary.csv  — Train / Val / OOS 分段统计（均值 / ICIR / 胜率）
  ic_analysis.png — IC 时序 + 累积曲线 + 分段 ICIR 柱图
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Strategy import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 核心计算
# ═══════════════════════════════════════════════════════════════════════

def compute_daily_ic(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    tradeable_mask: Optional[pd.DataFrame] = None,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    逐日计算 IC 与 Rank IC。

    Parameters
    ----------
    score_df : pd.DataFrame
        打分宽表 (index=TRADE_DATE, columns=StockID)
    label_df : pd.DataFrame
        收益率宽表，与 score_df 对齐
    tradeable_mask : pd.DataFrame, optional
        可交易 bool 宽表；为 None 时仅用 score 非空过滤
    min_stocks : int
        每日有效股票数阈值，不足时跳过（返回 NaN）

    Returns
    -------
    pd.DataFrame
        index=TRADE_DATE, columns=['ic', 'rank_ic', 'n_stocks']
    """
    score = score_df.copy()
    label = label_df.copy()

    if "TRADE_DATE" in score.columns:
        score = score.set_index("TRADE_DATE")
    if "TRADE_DATE" in label.columns:
        label = label.set_index("TRADE_DATE")

    score.index = pd.DatetimeIndex(score.index)
    label.index = pd.DatetimeIndex(label.index)
    score.columns = pd.Index([str(c).zfill(6) for c in score.columns])
    label.columns = pd.Index([str(c).zfill(6) for c in label.columns])

    common_dates  = score.index.intersection(label.index).sort_values()
    common_stocks = score.columns.intersection(label.columns)
    score = score.loc[common_dates, common_stocks]
    label = label.loc[common_dates, common_stocks]

    if tradeable_mask is not None:
        mask = tradeable_mask.reindex(index=common_dates, columns=common_stocks).fillna(False)
    else:
        mask = score.notna()

    records = []
    for date in common_dates:
        s = score.loc[date]
        r = label.loc[date]
        m = mask.loc[date].astype(bool)

        valid = m & s.notna() & r.notna()
        n = int(valid.sum())

        if n < min_stocks:
            records.append({"TRADE_DATE": date, "ic": np.nan, "rank_ic": np.nan, "n_stocks": n})
            continue

        s_v = s.loc[valid].values.astype(float)
        r_v = r.loc[valid].values.astype(float)

        ic_val,      _ = stats.pearsonr(s_v, r_v)
        rank_ic_val, _ = stats.spearmanr(s_v, r_v)

        records.append({"TRADE_DATE": date, "ic": ic_val, "rank_ic": rank_ic_val, "n_stocks": n})

    ic_df = pd.DataFrame(records).set_index("TRADE_DATE")
    ic_df.index = pd.DatetimeIndex(ic_df.index)
    return ic_df


# ═══════════════════════════════════════════════════════════════════════
# 分段统计
# ═══════════════════════════════════════════════════════════════════════

def _segment_stats(series: pd.Series, name: str) -> dict:
    """对单条 IC 序列计算统计指标（均值、标准差、ICIR、IC胜率）。"""
    s = series.dropna()
    if len(s) == 0:
        return {
            "segment": name,
            "n_days": 0,
            "mean": np.nan,
            "std": np.nan,
            "icir": np.nan,
            "ic_win_rate": np.nan,
        }
    mean = s.mean()
    std  = s.std()
    icir = mean / std if std > 0 else np.nan
    ic_win_rate = float((s > 0).mean())
    return {
        "segment": name,
        "n_days": len(s),
        "mean": mean,
        "std": std,
        "icir": icir,
        "ic_win_rate": ic_win_rate,
    }


def compute_ic_summary(
    ic_df: pd.DataFrame,
    train_start=None,
    train_end=None,
    val_start=None,
    val_end=None,
    oos_start=None,
) -> pd.DataFrame:
    """
    对 IC / Rank IC 按 Train / Val / OOS / Overall 四段统计。

    Returns
    -------
    pd.DataFrame
        multi-index: (metric, segment)
        columns: n_days / mean / std / icir / win_rate
    """
    train_start = pd.Timestamp(train_start or config.TRAIN_START)
    train_end   = pd.Timestamp(train_end   or config.TRAIN_END)
    val_start   = pd.Timestamp(val_start   or config.VAL_START)
    val_end     = pd.Timestamp(val_end     or config.VAL_END)
    oos_start   = pd.Timestamp(oos_start   or config.OOS_START)

    idx = pd.DatetimeIndex(ic_df.index)

    def _slice(start, end=None):
        cond = idx >= start
        if end is not None:
            cond &= idx <= end
        return ic_df.loc[cond]

    slices = {
        "Train":   _slice(train_start, train_end),
        "Val":     _slice(val_start,   val_end),
        "OOS":     _slice(oos_start),
        "Overall": ic_df,
    }

    rows = []
    for metric in ["ic", "rank_ic"]:
        for seg_name, seg_df in slices.items():
            if metric not in seg_df.columns:
                continue
            row = _segment_stats(seg_df[metric], seg_name)
            row["metric"] = metric
            rows.append(row)

    return pd.DataFrame(rows).set_index(["metric", "segment"])


# ═══════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════

def plot_ic_analysis(
    ic_df: pd.DataFrame,
    summary: pd.DataFrame,
    title: str = "IC / Rank IC Analysis",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
    rolling_window: int = 20,
    oos_start=None,
    val_start=None,
) -> Path:
    """
    绘制 IC 分析图（2×2 布局）：
      左上: IC 时序（原始柱 + 滚动均线 + 整体均值水平线）
      右上: Rank IC 时序
      左下: IC / Rank IC 累积曲线
      右下: 各段 ICIR 对比柱状图
    """
    oos_ts = pd.Timestamp(oos_start or config.OOS_START)
    val_ts = pd.Timestamp(val_start or config.VAL_START)

    ic_s   = ic_df["ic"].dropna()
    ric_s  = ic_df["rank_ic"].dropna()

    C_IC    = "#2196F3"
    C_RIC   = "#FF5722"
    C_ROLL  = "#0D47A1"
    C_RROLL = "#BF360C"

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

    ax_ic   = fig.add_subplot(gs[0, 0])
    ax_ric  = fig.add_subplot(gs[0, 1])
    ax_cum  = fig.add_subplot(gs[1, 0])
    ax_icir = fig.add_subplot(gs[1, 1])

    def _add_vlines(ax, dates, yref):
        for ts, label_txt, color in [
            (val_ts, "Val", "#555555"),
            (oos_ts, "OOS", "#000000"),
        ]:
            if len(dates) > 0 and dates.min() < ts < dates.max():
                xpos = dates.searchsorted(ts)
                ax.axvline(xpos, color=color, ls="--", lw=0.9, alpha=0.7)
                ax.text(xpos + 0.5, yref, label_txt, fontsize=7, va="top", color=color)

    def _plot_ic_series(ax, series, roll_c, raw_c, metric_label):
        x     = np.arange(len(series))
        dates = pd.DatetimeIndex(series.index)
        roll  = series.rolling(rolling_window, min_periods=5).mean()
        ax.bar(x, series.values, color=raw_c, alpha=0.30, width=1.0, label=metric_label)
        ax.plot(x, roll.values, color=roll_c, lw=1.5, label=f"{rolling_window}d MA")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axhline(series.mean(), color=roll_c, lw=1.0, ls=":",
                   label=f"Mean={series.mean():.4f}")
        _add_vlines(ax, dates, yref=series.max() * 0.92)
        ax.set_title(f"{metric_label} Time Series", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        step = max(len(x) // 10, 1)
        ticks = list(range(0, len(x), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([series.index[i].strftime("%Y%m%d") for i in ticks],
                           rotation=30, fontsize=6)

    _plot_ic_series(ax_ic,  ic_s,  C_ROLL,  C_IC,  "IC")
    _plot_ic_series(ax_ric, ric_s, C_RROLL, C_RIC, "Rank IC")

    # ── 累积曲线 ────────────────────────────────────────────────────────
    all_dates = pd.DatetimeIndex(ic_s.index.union(ric_s.index)).sort_values()
    x_all = np.arange(len(all_dates))
    ax_cum.plot(x_all, ic_s.cumsum().reindex(all_dates).values,
                color=C_IC,  lw=1.5, label="Cum IC")
    ax_cum.plot(x_all, ric_s.cumsum().reindex(all_dates).values,
                color=C_RIC, lw=1.5, label="Cum Rank IC", ls="--")
    ax_cum.axhline(0, color="gray", lw=0.5)
    _add_vlines(ax_cum, all_dates, yref=0)
    ax_cum.set_title("Cumulative IC / Rank IC", fontsize=10)
    ax_cum.set_ylabel("Cumulative IC", fontsize=9)
    ax_cum.legend(fontsize=8)
    step = max(len(x_all) // 10, 1)
    ticks = list(range(0, len(x_all), step))
    ax_cum.set_xticks(ticks)
    ax_cum.set_xticklabels([all_dates[i].strftime("%Y%m%d") for i in ticks],
                           rotation=30, fontsize=6)

    # ── 分段 ICIR 柱状图 ─────────────────────────────────────────────
    segments = ["Train", "Val", "OOS", "Overall"]
    ic_icirs, ric_icirs = [], []
    for seg in segments:
        for lst, key in [(ic_icirs, "ic"), (ric_icirs, "rank_ic")]:
            try:
                lst.append(summary.loc[(key, seg), "icir"])
            except KeyError:
                lst.append(np.nan)

    x_bar = np.arange(len(segments))
    w = 0.35
    bars_ic  = ax_icir.bar(x_bar - w / 2, ic_icirs,  width=w, color=C_IC,  label="ICIR",      alpha=0.85)
    bars_ric = ax_icir.bar(x_bar + w / 2, ric_icirs, width=w, color=C_RIC, label="Rank ICIR", alpha=0.85)
    ax_icir.axhline(0,    color="gray", lw=0.5)
    ax_icir.axhline( 0.5, color=C_IC,  lw=0.8, ls=":", alpha=0.5)
    ax_icir.axhline(-0.5, color=C_IC,  lw=0.8, ls=":", alpha=0.5)

    for bars, vals in [(bars_ic, ic_icirs), (bars_ric, ric_icirs)]:
        for bar, val in zip(bars, vals):
            if not (np.isnan(val) if isinstance(val, float) else False):
                offset = 0.02 if val >= 0 else -0.05
                ax_icir.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7,
                )

    ax_icir.set_xticks(x_bar)
    ax_icir.set_xticklabels(segments, fontsize=9)
    ax_icir.set_title("ICIR by Segment", fontsize=10)
    ax_icir.set_ylabel("ICIR", fontsize=9)
    ax_icir.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = Path(save_path or (config.BT_RESULT_DIR / "ic_analysis.png"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("IC 分析图已保存: %s", save_path)
    return save_path


# ═══════════════════════════════════════════════════════════════════════
# 一键运行入口
# ═══════════════════════════════════════════════════════════════════════

def run_ic_analysis(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    tradeable_mask: Optional[pd.DataFrame] = None,
    title: str = "IC / Rank IC Analysis",
    output_dir: Optional[Path] = None,
    min_stocks: int = 10,
    rolling_window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    一键运行 IC / Rank IC 分析，输出 CSV + 图表。

    Parameters
    ----------
    score_df : pd.DataFrame
        打分宽表
    label_df : pd.DataFrame
        Label（收益率）宽表
    tradeable_mask : pd.DataFrame, optional
        可交易 bool 宽表；与 run_quick_backtest 共享同一 mask，无重复计算
    title : str
        图表标题
    output_dir : Path, optional
        输出目录，默认 config.BT_RESULT_DIR
    min_stocks : int
        每日最小有效股票数，不足时该日 IC 记 NaN
    rolling_window : int
        IC 图中滚动均线的窗口大小（交易日）

    Returns
    -------
    ic_df : pd.DataFrame
        每日 IC / Rank IC（index=TRADE_DATE, columns=[ic, rank_ic, n_stocks]）
    summary : pd.DataFrame
        分段统计（multi-index: metric × segment）
    """
    out = Path(output_dir or config.BT_RESULT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 逐日计算
    ic_df = compute_daily_ic(score_df, label_df,
                             tradeable_mask=tradeable_mask,
                             min_stocks=min_stocks)

    # 2. 分段统计
    summary = compute_ic_summary(ic_df)

    # 3. 日志
    for metric in ["ic", "rank_ic"]:
        for seg in ["Train", "Val", "OOS", "Overall"]:
            try:
                row = summary.loc[(metric, seg)]
                logger.info(
                    "  [%s][%s] mean=%.4f  std=%.4f  ICIR=%.4f  IC_WinRate=%.2f%%  n=%d",
                    metric.upper(), seg,
                    row["mean"], row["std"], row["icir"],
                    row["ic_win_rate"] * 100, int(row["n_days"]),
                )
            except KeyError:
                pass

    # 4. 保存 CSV
    ic_df.to_csv(out / "ic_series.csv")
    logger.info("IC 时间序列已保存: %s", out / "ic_series.csv")
    summary.to_csv(out / "ic_summary.csv")
    logger.info("IC 分段统计已保存: %s", out / "ic_summary.csv")

    # 5. 绘图
    plot_ic_analysis(ic_df, summary,
                     title=title,
                     save_path=out / "ic_analysis.png",
                     rolling_window=rolling_window)

    return ic_df, summary

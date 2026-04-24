"""
快速分层回测模块: 按打分截面排序分 N 组, 计算等权组合净值。

执行假设:
- 无滑点成交, 使用 Label 设计时的基准价格 (TWAP) 作为买卖价
- 单期收益 = 组内股票等权平均收益率
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Strategy import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 分组回测核心逻辑
# ═══════════════════════════════════════════════════════════════════════
def quantile_backtest(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    n_groups: int = config.N_QUANTILE_GROUPS,
) -> pd.DataFrame:
    """
    每日截面按打分排序, 等分为 n_groups 组, 计算每组等权平均收益率。

    Parameters
    ----------
    score_df : pd.DataFrame
        打分宽表 (index=TRADE_DATE, columns=StockID)
    label_df : pd.DataFrame
        Label (收益率) 宽表, 与 score_df 对齐
    n_groups : int
        分组数

    Returns
    -------
    pd.DataFrame
        index=TRADE_DATE, columns=['group1', 'group2', ..., 'group{n}']
        每列为该组当日等权平均收益率
    """
    common_dates = score_df.index.intersection(label_df.index).sort_values()
    common_stocks = score_df.columns.intersection(label_df.columns)

    scores = score_df.loc[common_dates, common_stocks]
    labels = label_df.loc[common_dates, common_stocks]

    group_returns = {}

    for date in common_dates:
        s = scores.loc[date].dropna()
        r = labels.loc[date].reindex(s.index).dropna()
        valid = s.index.intersection(r.index)
        s = s.loc[valid]
        r = r.loc[valid]

        if len(valid) < n_groups:
            continue

        ranks = s.rank(method="first", ascending=False)
        group_size = len(valid) / n_groups
        group_ret = {}
        for g in range(1, n_groups + 1):
            mask = (ranks > (g - 1) * group_size) & (ranks <= g * group_size)
            if mask.sum() > 0:
                group_ret[f"group{g}"] = r.loc[mask].mean()
            else:
                group_ret[f"group{g}"] = np.nan

        group_returns[date] = group_ret

    ret_df = pd.DataFrame(group_returns).T.sort_index()
    ret_df.index.name = "TRADE_DATE"
    return ret_df


# ═══════════════════════════════════════════════════════════════════════
# 绩效统计
# ═══════════════════════════════════════════════════════════════════════
def calc_performance(ret_series: pd.Series, annual_days: int = 242) -> dict:
    """计算年化收益、年化波动、最大回撤、夏普比率"""
    total_ret = (1 + ret_series).prod() - 1
    n_days = len(ret_series)
    ann_ret = (1 + total_ret) ** (annual_days / max(n_days, 1)) - 1
    ann_vol = ret_series.std() * np.sqrt(annual_days)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + ret_series).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "total_ret": total_ret,
        "n_days": n_days,
    }


# ═══════════════════════════════════════════════════════════════════════
# 可视化 (匹配示例图风格)
# ═══════════════════════════════════════════════════════════════════════
def plot_quantile_nav(
    ret_df: pd.DataFrame,
    title: str = "factor | label",
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 6),
) -> Path:
    """
    绘制 N 组累计收益曲线, 风格匹配示例图。

    - Y 轴: 累计收益率 (从 0 开始)
    - X 轴: YYYYMMDD 格式日期
    - 图例: groupN, 年化收益%, 年化波动%, 夏普
    """
    cum_ret = (1 + ret_df).cumprod() - 1

    group_cols = [c for c in cum_ret.columns if c.startswith("group")]

    x_labels = []
    for d in cum_ret.index:
        if isinstance(d, pd.Timestamp):
            x_labels.append(d.strftime("%Y%m%d"))
        else:
            x_labels.append(str(d))

    fig, ax = plt.subplots(figsize=figsize)

    n = len(group_cols)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, col in enumerate(group_cols):
        perf = calc_performance(ret_df[col])
        label_text = (
            f"{col}, {perf['ann_ret']:.2%}, "
            f"{perf['max_drawdown']:.2%}, {perf['sharpe']:.2f}"
        )
        color = default_colors[i % len(default_colors)]
        ax.plot(range(len(cum_ret)), cum_ret[col].values, label=label_text, color=color, lw=1.0)

    tick_step = max(len(cum_ret) // 15, 1)
    tick_positions = list(range(0, len(cum_ret), tick_step))
    tick_labels = [x_labels[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Net Value", fontsize=10)
    ax.axhline(0.0, color="gray", ls="-", lw=0.5)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()

    if save_path is None:
        save_path = config.BT_RESULT_DIR / "quantile_backtest.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("回测图已保存: %s", save_path)
    return save_path


# ═══════════════════════════════════════════════════════════════════════
# 一键运行入口
# ═══════════════════════════════════════════════════════════════════════
def run_quick_backtest(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    n_groups: int = config.N_QUANTILE_GROUPS,
    title: str = "factor | label",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    一键运行分层回测: 分组 -> 统计 -> 绘图

    Returns
    -------
    pd.DataFrame  各组每日收益率 (同 quantile_backtest 输出)
    """
    ret_df = quantile_backtest(score_df, label_df, n_groups)
    logger.info("分层回测完成: %d 交易日, %d 组", len(ret_df), n_groups)

    for col in ret_df.columns:
        perf = calc_performance(ret_df[col])
        logger.info("  %s: Ann=%.2f%% MDD=%.2f%% SR=%.2f",
                     col, perf["ann_ret"] * 100, perf["max_drawdown"] * 100, perf["sharpe"])

    plot_quantile_nav(ret_df, title=title, save_path=save_path)
    return ret_df

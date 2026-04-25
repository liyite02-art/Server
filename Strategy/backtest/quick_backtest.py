"""
快速分层回测模块: 按打分截面排序分 N 组, 计算等权组合净值。

执行假设:
- 无滑点、无佣金印花税, 使用 Label 设计时的 TWAP 区间收益
- 单期收益 = 组内股票**收益率的简单平均** (横截面算术平均), 不是真实资金曲线

与 ``event_backtest`` 的差异 (为何曲线往往更好看):
- group1 约含 len(有效股票)/N 只 (如 20 组时约前 5%), 而 ``BacktestRunner(top_n=50)``
  只持有 50 只, 二者持仓集合与权重并不相同
- 分层回测忽略整手、涨跌停无法成交、现金占用等执行摩擦
- 事件回测有调仓频率、费用; 若 ``rebalance_freq>1``, 持仓在调仓间隔内不随截面更新,
  与「每日按新 group1 换仓」的隐含假设也不一致

要对齐 group1 选股, 请使用 ``BacktestRunner(mirror_quantile_group=1, n_quantile_groups=20)``。

``top50`` / ``top100`` 列为按当日打分排序后, 等权持有前 K 只 (K=min(配置, 当日有效股票数)) 的截面平均收益,
与 ``group1`` (按分位切段) 的持仓集合在股票总数非整除分组时可能略有差异, 便于与固定持仓数量的回测对比。
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
from Strategy.backtest.universe import (
    listing_age_mask,
    load_ipo_dates,
    load_out_dates,
    load_st_status,
    out_date_mask,
    prefix_mask,
    st_mask,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 可交易股票池: 只能使用 T 日可知信息
# ═══════════════════════════════════════════════════════════════════════
def _standardize_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TRADE_DATE" in out.columns:
        out = out.set_index("TRADE_DATE")
    out.index = pd.DatetimeIndex(out.index)
    out.columns = pd.Index([str(c).zfill(6) for c in out.columns])
    return out


def _load_twap_price(tag: str = "TWAP_1430_1457") -> Optional[pd.DataFrame]:
    path = config.LABEL_OUTPUT_DIR / f"{tag}.fea"
    if not path.exists():
        logger.warning("TWAP 价格表不存在, quick 回测将仅按 score 非空建股票池: %s", path)
        return None
    return _standardize_wide(pd.read_feather(path))


def _load_daily_wide(field_name: str) -> Optional[pd.DataFrame]:
    path = config.DAILY_DATA_DIR / f"{field_name}.pkl"
    if not path.exists():
        logger.warning("日频数据文件不存在: %s", path)
        return None
    try:
        return _standardize_wide(pd.read_pickle(path))
    except Exception as exc:
        logger.warning("加载 %s 失败: %s", field_name, exc)
        return None


def _limit_hit(twap: pd.DataFrame, limit_price: pd.DataFrame) -> pd.DataFrame:
    return (twap - limit_price).abs() / limit_price.replace(0, np.nan) < 1e-4


def build_tradeable_mask(
    score_df: pd.DataFrame,
    price_df: Optional[pd.DataFrame] = None,
    limit_up_df: Optional[pd.DataFrame] = None,
    ipo_dates: Optional[pd.Series] = None,
    out_dates: Optional[pd.Series] = None,
    st_status: Optional[pd.DataFrame] = None,
    exclude_limit_up: bool = True,
    min_listing_days: int = 20,
    delist_buffer_days: int = 20,
    exclude_st: bool = True,
    exclude_historical_st: bool = True,
    excluded_prefixes: Optional[tuple[str, ...]] = ("300", "688"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建 T 日可知的 quick 回测股票池 mask, 并输出逐日报告。

    - price_df 非空: 仅保留 T 日 TWAP 有价格的股票。
    - limit_up_df 非空且 exclude_limit_up=True: 剔除 T 日 TWAP 封涨停、无法买入的股票。
    - ipo_dates/out_dates/st_status 非空: 剔除新股、退市前缓冲期股票和截至 T 日曾经 ST 的股票。
    - excluded_prefixes: 剔除指定代码前缀, 默认剔除 300/688。
    - 不使用 label 的非空性决定股票池, 避免未来可得性筛选。
    """
    scores = _standardize_wide(score_df)
    score_valid = scores.notna()
    age_ok = listing_age_mask(
        scores.index,
        scores.columns,
        ipo_dates=ipo_dates,
        min_listing_days=min_listing_days,
    )
    delist_ok = out_date_mask(
        scores.index,
        scores.columns,
        out_dates=out_dates,
        delist_buffer_days=delist_buffer_days,
    )
    st_ok = st_mask(scores.index, scores.columns, st_status, historical=exclude_historical_st) if exclude_st else pd.DataFrame(
        True, index=scores.index, columns=scores.columns
    )
    prefix_ok = prefix_mask(scores.index, scores.columns, excluded_prefixes=excluded_prefixes)
    universe_ok = age_ok & delist_ok & st_ok & prefix_ok

    if price_df is None:
        base = score_valid
        tradeable = score_valid & universe_ok
        new_stock = base & ~age_ok
        pre_delist = base & age_ok & ~delist_ok
        st_excluded = base & age_ok & delist_ok & ~st_ok
        prefix_excluded = base & age_ok & delist_ok & st_ok & ~prefix_ok
        report = pd.DataFrame(index=scores.index)
        report["score_nonnull"] = score_valid.sum(axis=1)
        report["excluded_no_twap_price"] = 0
        report["excluded_limit_up_cannot_buy"] = 0
        report["excluded_new_stock"] = new_stock.sum(axis=1)
        report["excluded_pre_delist"] = pre_delist.sum(axis=1)
        report["excluded_st"] = st_excluded.sum(axis=1)
        report["excluded_prefix"] = prefix_excluded.sum(axis=1)
        report["tradeable"] = tradeable.sum(axis=1)
        report.index.name = "TRADE_DATE"
        return tradeable, report

    prices = _standardize_wide(price_df).reindex(index=scores.index, columns=scores.columns)
    has_price = prices.notna()
    base = score_valid & has_price
    no_price = score_valid & ~has_price
    new_stock = base & ~age_ok
    pre_delist = base & age_ok & ~delist_ok
    st_excluded = base & age_ok & delist_ok & ~st_ok
    prefix_excluded = base & age_ok & delist_ok & st_ok & ~prefix_ok

    limit_up = pd.DataFrame(False, index=scores.index, columns=scores.columns)
    if exclude_limit_up and limit_up_df is not None:
        lu = _standardize_wide(limit_up_df).reindex(index=scores.index, columns=scores.columns)
        limit_up = base & age_ok & delist_ok & st_ok & prefix_ok & _limit_hit(prices, lu)
    tradeable = base & age_ok & delist_ok & st_ok & prefix_ok & ~limit_up

    report = pd.DataFrame(index=scores.index)
    report["score_nonnull"] = score_valid.sum(axis=1)
    report["excluded_no_twap_price"] = no_price.sum(axis=1)
    report["excluded_limit_up_cannot_buy"] = limit_up.sum(axis=1)
    report["excluded_new_stock"] = new_stock.sum(axis=1)
    report["excluded_pre_delist"] = pre_delist.sum(axis=1)
    report["excluded_st"] = st_excluded.sum(axis=1)
    report["excluded_prefix"] = prefix_excluded.sum(axis=1)
    report["tradeable"] = tradeable.sum(axis=1)
    report.index.name = "TRADE_DATE"
    return tradeable, report


# ═══════════════════════════════════════════════════════════════════════
# 分组回测核心逻辑
# ═══════════════════════════════════════════════════════════════════════
def quantile_backtest(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    n_groups: int = config.N_QUANTILE_GROUPS,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    tradeable_mask: Optional[pd.DataFrame] = None,
    return_diagnostics: bool = False,
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
    start_date : pd.Timestamp, optional
        回测起始日 (含). 默认 None 表示不限制.
        ⚠️ 强烈建议设置为 config.OOS_START 或 config.VAL_START,
           避免将训练集日期（样本内）纳入统计，导致虚高收益。
    end_date : pd.Timestamp, optional
        回测截止日 (含). 默认 None 表示不限制.

    Returns
    -------
    pd.DataFrame
        index=TRADE_DATE, columns=['group1', 'group2', ..., 'group{n}']
        每列为该组当日等权平均收益率
    """
    common_dates = score_df.index.intersection(label_df.index).sort_values()
    if start_date is not None:
        common_dates = common_dates[common_dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        common_dates = common_dates[common_dates <= pd.Timestamp(end_date)]
    common_stocks = score_df.columns.intersection(label_df.columns)

    scores = score_df.loc[common_dates, common_stocks]
    labels = label_df.loc[common_dates, common_stocks]
    if tradeable_mask is not None:
        tradeable = tradeable_mask.reindex(index=common_dates, columns=common_stocks).fillna(False)
    else:
        tradeable = scores.notna()

    group_returns = {}
    diagnostics = {}

    for date in common_dates:
        s = scores.loc[date].dropna()
        tmask = tradeable.loc[date].reindex(s.index).fillna(False).astype(bool)
        s = s.loc[tmask]
        r = labels.loc[date].reindex(s.index)

        diagnostics[date] = {
            "group_universe_size": len(s),
            "group_label_missing": int(r.isna().sum()),
        }

        if len(s) < n_groups:
            diagnostics[date]["group_skipped"] = 1
            continue

        ranks = s.rank(method="first", ascending=False)
        group_size = len(s) / n_groups
        group_ret = {}
        for g in range(1, n_groups + 1):
            mask = (ranks > (g - 1) * group_size) & (ranks <= g * group_size)
            if mask.sum() > 0:
                group_ret[f"group{g}"] = r.loc[mask].dropna().mean()
            else:
                group_ret[f"group{g}"] = np.nan

        diagnostics[date]["group_skipped"] = 0
        group_returns[date] = group_ret

    ret_df = pd.DataFrame(group_returns).T.sort_index().dropna(how="all")
    ret_df.index.name = "TRADE_DATE"
    diag_df = pd.DataFrame(diagnostics).T.sort_index()
    diag_df.index.name = "TRADE_DATE"
    if return_diagnostics:
        return ret_df, diag_df
    return ret_df


def topk_equal_weight_returns(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    ks: tuple[int, ...] = (50, 100),
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    tradeable_mask: Optional[pd.DataFrame] = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    每日截面按打分降序, 取前 K 只有效股票, 计算等权平均收益率 (列名为 top{K})。

    与 ``quantile_backtest`` 使用相同的日期、股票对齐方式; 仅要求当日有效股票数 >= 1。
    """
    common_dates = score_df.index.intersection(label_df.index).sort_values()
    if start_date is not None:
        common_dates = common_dates[common_dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        common_dates = common_dates[common_dates <= pd.Timestamp(end_date)]
    common_stocks = score_df.columns.intersection(label_df.columns)

    scores = score_df.loc[common_dates, common_stocks]
    labels = label_df.loc[common_dates, common_stocks]
    if tradeable_mask is not None:
        tradeable = tradeable_mask.reindex(index=common_dates, columns=common_stocks).fillna(False)
    else:
        tradeable = scores.notna()

    rows = {}
    diagnostics = {}

    for date in common_dates:
        s = scores.loc[date].dropna()
        tmask = tradeable.loc[date].reindex(s.index).fillna(False).astype(bool)
        s = s.loc[tmask]
        r = labels.loc[date].reindex(s.index)

        if len(s) < 1:
            continue

        ranks = s.rank(method="first", ascending=False)
        row: dict = {}
        diag_row: dict = {}
        for k in ks:
            col = f"top{k}"
            take = min(k, len(s))
            mask = ranks <= take
            selected_ret = r.loc[mask]
            row[col] = selected_ret.dropna().mean() if mask.any() else np.nan
            diag_row[f"{col}_selected"] = int(mask.sum())
            diag_row[f"{col}_label_missing"] = int(selected_ret.isna().sum())
        rows[date] = row
        diagnostics[date] = diag_row

    out = pd.DataFrame(rows).T.sort_index().dropna(how="all")
    out.index.name = "TRADE_DATE"
    diag_df = pd.DataFrame(diagnostics).T.sort_index()
    diag_df.index.name = "TRADE_DATE"
    if return_diagnostics:
        return out, diag_df
    return out


# ═══════════════════════════════════════════════════════════════════════
# 绩效统计
# ═══════════════════════════════════════════════════════════════════════
def calc_performance(ret_series: pd.Series, annual_days: int = 242) -> dict:
    """计算单利年化收益、年化波动、最大回撤、夏普比率"""
    total_ret = (1 + ret_series).prod() - 1
    n_days = len(ret_series)
    ann_ret = ret_series.mean() * annual_days
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
    split_date=None,
) -> Path:
    """
    绘制 N 组累计收益曲线, 可选含 topK 曲线, 风格匹配示例图。

    - Y 轴: 累计收益率 (从 0 开始)
    - X 轴: YYYYMMDD 格式日期
    - 图例: groupN / topK, 年化收益%, 最大回撤%, 夏普
    """
    cum_ret = (1 + ret_df).cumprod() - 1

    def _group_sort_key(name: str) -> tuple[int, str]:
        if name.startswith("group") and name[5:].isdigit():
            return (0, int(name[5:]))
        return (1, name)

    group_cols = sorted(
        [c for c in cum_ret.columns if c.startswith("group")],
        key=_group_sort_key,
    )
    top_cols = [c for c in cum_ret.columns if c.startswith("top") and c[3:].isdigit()]

    x_labels = []
    for d in cum_ret.index:
        if isinstance(d, pd.Timestamp):
            x_labels.append(d.strftime("%Y%m%d"))
        else:
            x_labels.append(str(d))

    fig, ax = plt.subplots(figsize=figsize)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, col in enumerate(group_cols):
        perf = calc_performance(ret_df[col].dropna())
        label_text = (
            f"{col}, {perf['ann_ret']:.2%}, "
            f"{perf['max_drawdown']:.2%}, {perf['sharpe']:.2f}"
        )
        color = default_colors[i % len(default_colors)]
        ax.plot(range(len(cum_ret)), cum_ret[col].values, label=label_text, color=color, lw=1.0)

    top_style = {
        "top50": {"ls": "--", "lw": 1.4, "color": "#1f77b4"},
        "top100": {"ls": "--", "lw": 1.4, "color": "#d62728"},
    }
    for col in sorted(top_cols, key=lambda x: int(x[3:]) if x[3:].isdigit() else 0):
        st = top_style.get(col, {"ls": "--", "lw": 1.3, "color": "#333333"})
        perf = calc_performance(ret_df[col].dropna())
        label_text = (
            f"{col}, {perf['ann_ret']:.2%}, "
            f"{perf['max_drawdown']:.2%}, {perf['sharpe']:.2f}"
        )
        ax.plot(
            range(len(cum_ret)),
            cum_ret[col].values,
            label=label_text,
            color=st["color"],
            ls=st["ls"],
            lw=st["lw"],
        )

    split_ts = pd.Timestamp(split_date or config.OOS_START)
    split_pos = pd.DatetimeIndex(pd.to_datetime(cum_ret.index)).searchsorted(split_ts)
    if 0 < split_pos < len(cum_ret):
        ax.axvline(split_pos - 0.5, color="black", ls="--", lw=0.8, alpha=0.75)
        ax.text(
            split_pos,
            float(np.nanmax(cum_ret.values)),
            "OOS",
            fontsize=8,
            va="top",
            ha="left",
            color="black",
        )

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
    output_dir: Optional[Path] = None,
    start_date=None,
    end_date=None,
    top_ks: tuple[int, ...] = (50, 100),
    price_df: Optional[pd.DataFrame] = None,
    limit_up_df: Optional[pd.DataFrame] = None,
    ipo_dates: Optional[pd.Series] = None,
    out_dates: Optional[pd.Series] = None,
    st_status: Optional[pd.DataFrame] = None,
    twap_tag: str = "TWAP_1430_1457",
    exclude_limit_up: bool = True,
    min_listing_days: int = 20,
    delist_buffer_days: int = 20,
    exclude_st: bool = True,
    excluded_prefixes: Optional[tuple[str, ...]] = ("300", "688"),
) -> pd.DataFrame:
    """
    一键运行分层回测: 分组 + topK 等权 -> 统计 -> 绘图

    Parameters
    ----------
    save_path
        净值图保存路径 (完整文件名). 若与 ``output_dir`` 同时给出, 以 ``save_path`` 为准.
    output_dir
        结果目录; 图保存为 ``{output_dir}/quantile_backtest.png``.
        未传 ``save_path`` 且未传 ``output_dir`` 时, 使用 ``config.BT_RESULT_DIR``.
    start_date : date-like, optional
        回测起始日. 强烈建议传入 config.OOS_START 或 config.VAL_START 以
        避免样本内日期拉高回测表现.
    end_date : date-like, optional
        回测截止日.
    top_ks : tuple[int, ...]
        额外计算 topK 等权日收益并与分组列合并绘图; 默认 ``(50, 100)``. 传空元组 ``()`` 可关闭.
    price_df / limit_up_df
        T 日可知的价格/涨停宽表, 用于构建可交易股票池。默认自动加载当前 TWAP 价格和涨停价。
    ipo_dates / out_dates / st_status
        用于剔除新股、退市前缓冲期股票和截至 T 日曾经 ST 的股票; 默认自动加载
        ``Daily_data/ipo_dates.pkl`` 与 ``Daily_data/st_status.pkl``.
    excluded_prefixes
        需要剔除的股票代码前缀, 默认 ``("300", "688")``.

    Returns
    -------
    pd.DataFrame
        各组每日收益率列 ``group*``; 若启用 topK, 另含 ``top{k}`` 列 (与分组行对齐).
    """
    if save_path is None and output_dir is not None:
        save_path = Path(output_dir) / "quantile_backtest.png"

    if price_df is None:
        price_df = _load_twap_price(twap_tag)
    if limit_up_df is None and exclude_limit_up:
        limit_up_df = _load_daily_wide("LIMIT_UP_PRICE")
    if ipo_dates is None:
        ipo_dates = load_ipo_dates()
    if out_dates is None:
        out_dates = load_out_dates()
    if st_status is None and exclude_st:
        st_status = load_st_status()
    tradeable_mask, universe_report = build_tradeable_mask(
        score_df,
        price_df=price_df,
        limit_up_df=limit_up_df,
        ipo_dates=ipo_dates,
        out_dates=out_dates,
        st_status=st_status,
        exclude_limit_up=exclude_limit_up,
        min_listing_days=min_listing_days,
        delist_buffer_days=delist_buffer_days,
        exclude_st=exclude_st,
        excluded_prefixes=excluded_prefixes,
    )

    ret_df, group_diag = quantile_backtest(
        score_df, label_df, n_groups,
        start_date=start_date, end_date=end_date,
        tradeable_mask=tradeable_mask,
        return_diagnostics=True,
    )
    if top_ks:
        top_df, top_diag = topk_equal_weight_returns(
            score_df, label_df, ks=top_ks,
            start_date=start_date, end_date=end_date,
            tradeable_mask=tradeable_mask,
            return_diagnostics=True,
        )
        ret_df = ret_df.join(top_df.reindex(ret_df.index), how="left")
    else:
        top_diag = pd.DataFrame()

    report_dates = universe_report.index
    if start_date is not None:
        report_dates = report_dates[report_dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        report_dates = report_dates[report_dates <= pd.Timestamp(end_date)]
    universe_report = universe_report.loc[report_dates]
    report = universe_report.join(group_diag, how="left")
    if not top_diag.empty:
        report = report.join(top_diag, how="left")

    logger.info("分层回测完成: %d 交易日, %d 组%s",
                len(ret_df), n_groups,
                f", topK={top_ks}" if top_ks else "")
    logger.info(
        "quick 股票池报告: 平均 score=%d, 可交易=%d, 无TWAP=%d, 涨停无法买入=%d, 新股=%d, 退市缓冲=%d, ST=%d, 前缀剔除=%d, group缺label=%d",
        int(report["score_nonnull"].mean()) if not report.empty else 0,
        int(report["tradeable"].mean()) if not report.empty else 0,
        int(report["excluded_no_twap_price"].mean()) if not report.empty else 0,
        int(report["excluded_limit_up_cannot_buy"].mean()) if not report.empty else 0,
        int(report["excluded_new_stock"].mean()) if not report.empty else 0,
        int(report["excluded_pre_delist"].mean()) if not report.empty else 0,
        int(report["excluded_st"].mean()) if not report.empty else 0,
        int(report["excluded_prefix"].mean()) if not report.empty else 0,
        int(report["group_label_missing"].mean()) if "group_label_missing" in report else 0,
    )

    for col in ret_df.columns:
        perf = calc_performance(ret_df[col].dropna())
        logger.info("  %s: Ann=%.2f%% MDD=%.2f%% SR=%.2f",
                     col, perf["ann_ret"] * 100, perf["max_drawdown"] * 100, perf["sharpe"])

    if save_path is None:
        save_path = config.BT_RESULT_DIR / "quantile_backtest.png"
    report_path = Path(save_path).with_name("quick_backtest_universe_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path)
    logger.info("quick 股票池/缺失报告已保存: %s", report_path)

    plot_quantile_nav(ret_df, title=title, save_path=save_path)
    return ret_df

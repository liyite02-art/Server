"""
精细化事件驱动回测引擎。

交易流水线 (严格遵循):
    score_df.loc[T] 已 shift(1), 仅含 T-1 收盘后信息
    T 日先卖旧仓 (TWAP) -> 资金回笼 -> 按 score_df.loc[T] 选股买入 (TWAP)

⚠️ 防未来数据:
- 选股信号严格基于 T-1 日收盘后可获取的信息 (因子计算时已 shift(1))
- 涨跌停判断使用 LIMIT_UP_PRICE / LIMIT_DOWN_PRICE 日频数据

使用示例::

    runner = BacktestRunner(
        score_df=score_df,
        mirror_quantile_group=1,   # 做多第 1 组 (最高分)
        n_quantile_groups=20,
        rebalance_freq=1,
        frictionless=True,
    )
    result = runner.run(start_date=config.VAL_START, end_date=None)
    result.plot(save_dir="outputs/bt_results")
    result.save_details(config.BT_RESULT_DIR)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Strategy import config
from Strategy.backtest.universe import load_ipo_dates, load_out_dates, load_st_status

logger = logging.getLogger(__name__)


# region agent log
def _agent_debug_log(hypothesis_id: str, message: str, data: dict):
    import json
    import os
    import time

    payload = {
        "sessionId": "da6e13",
        "runId": "pre-delist-fix",
        "hypothesisId": hypothesis_id,
        "location": "Strategy/backtest/event_backtest.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        os.makedirs("/root/autodl-tmp/.cursor", exist_ok=True)
        with open("/root/autodl-tmp/.cursor/debug-da6e13.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
# endregion


# ─── 内部辅助: 加载日频宽表 ────────────────────────────────────────────
def _load_daily_wide(field_name: str) -> Optional[pd.DataFrame]:
    """从 config.DAILY_DATA_DIR 加载日频宽表 pkl, 失败返回 None。"""
    path = config.DAILY_DATA_DIR / f"{field_name}.pkl"
    if not path.exists():
        logger.warning("日频数据文件不存在: %s", path)
        return None
    try:
        df = pd.read_pickle(path)
        df.index = pd.DatetimeIndex(df.index)
        df.columns = pd.Index([str(c).zfill(6) for c in df.columns])
        return df
    except Exception as exc:
        logger.warning("加载 %s 失败: %s", field_name, exc)
        return None


def _load_twap(tag: str = "TWAP_1430_1457") -> pd.DataFrame:
    """加载 LabelGenerator 输出的 TWAP 基准价格宽表。"""
    path = config.LABEL_OUTPUT_DIR / f"{tag}.fea"
    if not path.exists():
        raise FileNotFoundError(
            f"TWAP 价格表不存在: {path}\n"
            "请先运行 LabelGenerator.generate_and_save() 生成价格表。"
        )
    df = pd.read_feather(path)
    df = df.set_index("TRADE_DATE")
    df.index = pd.DatetimeIndex(df.index)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 数据类型定义
# ═══════════════════════════════════════════════════════════════════════
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExceptionType(Enum):
    LIMIT_UP_CANNOT_BUY = "涨停无法买入"
    LIMIT_DOWN_CANNOT_SELL = "跌停无法卖出"
    INSUFFICIENT_CASH = "可用资金不足"
    NO_PRICE_DATA = "无价格数据"


@dataclass
class Order:
    date: pd.Timestamp
    stock: str
    side: OrderSide
    target_shares: int
    anchor_price: float
    actual_price: float = 0.0
    filled_shares: int = 0
    commission: float = 0.0
    stamp_duty: float = 0.0
    pnl: float = 0.0
    filled: bool = False
    exception: Optional[ExceptionType] = None


@dataclass
class Position:
    stock: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp


@dataclass
class ExceptionRecord:
    date: pd.Timestamp
    stock: str
    exception_type: ExceptionType
    detail: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Portfolio: 持仓与资金管理
# ═══════════════════════════════════════════════════════════════════════
class Portfolio:
    """
    管理持仓、可用资金、冻结资金。

    T+1 资金时序:
    每日先执行全部卖出 -> 更新可用资金 -> 再执行买入
    """

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.nav_history: List[dict] = []

    @property
    def position_value(self) -> float:
        return sum(p.shares * p.entry_price for p in self.positions.values())

    @property
    def total_nav(self) -> float:
        return self.cash + self.position_value

    def has_position(self, stock: str) -> bool:
        return stock in self.positions and self.positions[stock].shares > 0

    def add_position(self, stock: str, shares: int, price: float, date: pd.Timestamp):
        if stock in self.positions:
            old = self.positions[stock]
            total_shares = old.shares + shares
            avg_price = (old.shares * old.entry_price + shares * price) / total_shares
            self.positions[stock] = Position(stock, total_shares, avg_price, date)
        else:
            self.positions[stock] = Position(stock, shares, price, date)

    def remove_position(self, stock: str, shares: Optional[int] = None) -> Position:
        if stock not in self.positions:
            raise KeyError(f"无持仓: {stock}")
        pos = self.positions[stock]
        if shares is None or shares >= pos.shares:
            del self.positions[stock]
            return pos
        else:
            pos.shares -= shares
            return Position(stock, shares, pos.entry_price, pos.entry_date)

    def record_nav(
        self,
        date: pd.Timestamp,
        price_dict: Dict[str, float],
        extra_positions: Optional[Dict[str, Position]] = None,
    ):
        """记录当日 NAV (使用当日 TWAP 估算持仓市值)"""
        pos_val = 0.0
        all_positions = dict(self.positions)
        if extra_positions:
            all_positions.update(extra_positions)
        for stock, pos in all_positions.items():
            px = price_dict.get(stock, pos.entry_price)
            pos_val += pos.shares * px
        self.nav_history.append({
            "TRADE_DATE": date,
            "cash": self.cash,
            "position_value": pos_val,
            "nav": self.cash + pos_val,
            "n_positions": len(all_positions),
            "n_delayed_sells": len(extra_positions or {}),
        })


# ═══════════════════════════════════════════════════════════════════════
# TradeEngine: 撮合引擎
# ═══════════════════════════════════════════════════════════════════════
class TradeEngine:
    """
    撮合引擎: 处理滑点、佣金、印花税、涨跌停判断。
    """

    def __init__(
        self,
        commission_rate: float = config.COMMISSION_RATE,
        stamp_duty_rate: float = config.STAMP_DUTY_RATE,
        slippage_bps: float = config.SLIPPAGE_BPS,
        slippage_fixed: float = 0.0,
    ):
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage_bps = slippage_bps
        self.slippage_fixed = slippage_fixed

    def calc_buy_price(self, anchor_price: float) -> float:
        """买入成交价 = 锚定价 + 滑点"""
        slippage = anchor_price * self.slippage_bps / 10000 + self.slippage_fixed
        return anchor_price + slippage

    def calc_sell_price(self, anchor_price: float) -> float:
        """卖出成交价 = 锚定价 - 滑点"""
        slippage = anchor_price * self.slippage_bps / 10000 + self.slippage_fixed
        return anchor_price - slippage

    def calc_commission(self, amount: float) -> float:
        return amount * self.commission_rate

    def calc_stamp_duty(self, amount: float) -> float:
        return amount * self.stamp_duty_rate

    def is_limit_up(
        self,
        stock: str,
        twap_price: float,
        limit_up_price: float,
    ) -> bool:
        """判断是否封死涨停 (TWAP == 涨停价 视为封死)"""
        if np.isnan(limit_up_price) or np.isnan(twap_price):
            return False
        return abs(twap_price - limit_up_price) / limit_up_price < 1e-4

    def is_limit_down(
        self,
        stock: str,
        twap_price: float,
        limit_down_price: float,
    ) -> bool:
        """判断是否封死跌停 (TWAP == 跌停价 视为封死)"""
        if np.isnan(limit_down_price) or np.isnan(twap_price):
            return False
        return abs(twap_price - limit_down_price) / limit_down_price < 1e-4


# ═══════════════════════════════════════════════════════════════════════
# ExceptionTracker & TradeLogger
# ═══════════════════════════════════════════════════════════════════════
class ExceptionTracker:
    """记录异常交易事件"""

    def __init__(self):
        self.records: List[ExceptionRecord] = []

    def log(self, date: pd.Timestamp, stock: str, etype: ExceptionType, detail: str = ""):
        self.records.append(ExceptionRecord(date, stock, etype, detail))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["date", "stock", "type", "detail"])
        return pd.DataFrame([
            {"date": r.date, "stock": r.stock, "type": r.exception_type.value, "detail": r.detail}
            for r in self.records
        ])


class TradeLogger:
    """记录每笔订单明细"""

    def __init__(self):
        self.orders: List[Order] = []

    def log(self, order: Order):
        self.orders.append(order)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": o.date, "stock": o.stock, "side": o.side.value,
                "target_shares": o.target_shares, "filled_shares": o.filled_shares,
                "anchor_price": o.anchor_price, "actual_price": o.actual_price,
                "commission": o.commission, "stamp_duty": o.stamp_duty,
                "pnl": o.pnl, "filled": o.filled,
                "exception": o.exception.value if o.exception else None,
            }
            for o in self.orders
        ])


# ═══════════════════════════════════════════════════════════════════════
# BacktestResult: 回测结果容器
# ═══════════════════════════════════════════════════════════════════════
class BacktestResult:
    """
    回测结果容器, 包含 NAV 序列、逐笔成交、异常事件。

    提供方法:
        plot(save_dir)       -- 绘制 NAV 曲线并保存图片
        save_details(outdir) -- 输出 trades_all.csv / daily_trade_summary.csv / exceptions.csv
    """

    def __init__(
        self,
        nav_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        exceptions_df: pd.DataFrame,
        initial_capital: float,
        strategy_name: str = "event_backtest",
    ):
        self.nav_df = nav_df
        self.trades_df = trades_df
        self.exceptions_df = exceptions_df
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name

    # ── 绩效统计 ──────────────────────────────────────────────────
    def _perf(self, annual_days: int = 242) -> dict:
        if self.nav_df.empty:
            return {}
        nav = self.nav_df["nav"]
        ret = nav.pct_change().dropna()
        total_ret = nav.iloc[-1] / self.initial_capital - 1
        ann_ret = ret.mean() * annual_days
        ann_vol = ret.std() * np.sqrt(annual_days)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cum = (1 + ret).cumprod()
        max_dd = ((cum / cum.cummax()) - 1).min()
        return dict(total_ret=total_ret, ann_ret=ann_ret, ann_vol=ann_vol,
                    sharpe=sharpe, max_dd=max_dd, n_days=len(ret))

    # ── 绘图 ──────────────────────────────────────────────────────
    def plot(
        self,
        save_dir: Optional[str | Path] = None,
        split_date=None,
    ) -> Path:
        """绘制 NAV 曲线; 保存到 save_dir/event_backtest_nav.png。"""
        out = Path(save_dir or config.BT_RESULT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / f"{self.strategy_name}_nav.png"

        nav = self.nav_df.set_index("TRADE_DATE")["nav"] if "TRADE_DATE" in self.nav_df.columns else self.nav_df["nav"]
        cum = nav / self.initial_capital  # 净值从 1 开始

        perf = self._perf()
        title = (
            f"{self.strategy_name} NAV  "
            f"AnnRet={perf.get('ann_ret', 0):.2%}  "
            f"MaxDD={perf.get('max_dd', 0):.2%}  "
            f"Sharpe={perf.get('sharpe', 0):.2f}"
        )

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(cum.values, lw=1.2, color="steelblue", label="Portfolio NAV")
        split_ts = pd.Timestamp(split_date or config.OOS_START)
        split_pos = pd.DatetimeIndex(pd.to_datetime(cum.index)).searchsorted(split_ts)
        if 0 < split_pos < len(cum):
            ax.axvline(split_pos - 0.5, color="black", ls="--", lw=0.8, alpha=0.75)
            ax.text(
                split_pos,
                float(cum.max()),
                "OOS",
                fontsize=8,
                va="top",
                ha="left",
                color="black",
            )
        x_idx = list(range(0, len(cum), max(len(cum) // 12, 1)))
        ax.set_xticks(x_idx)
        ax.set_xticklabels(
            [str(cum.index[i])[:10] if hasattr(cum.index[i], '__str__') else str(cum.index[i]) for i in x_idx],
            rotation=45, fontsize=7,
        )
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Net Value")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("NAV 曲线已保存: %s", save_path)
        return save_path

    # ── 保存明细 ──────────────────────────────────────────────────
    def save_details(self, output_dir: Optional[str | Path] = None) -> Path:
        """保存逐笔成交 (trades_all.csv)、按日汇总 (daily_trade_summary.csv)、异常事件 (exceptions.csv)。"""
        out = Path(output_dir or config.BT_RESULT_DIR)
        out.mkdir(parents=True, exist_ok=True)

        prefix = "" if self.strategy_name == "event_backtest" else f"{self.strategy_name}_"

        # NAV
        self.nav_df.to_csv(out / f"{prefix}event_nav.csv", index=False)

        # 逐笔成交
        self.trades_df.to_csv(out / f"{prefix}trades_all.csv", index=False)

        # 按日汇总
        if not self.trades_df.empty:
            daily = (
                self.trades_df.groupby("date")
                .agg(
                    n_trades=("date", "count"),
                    buy_amount=("filled_shares", lambda x: x[self.trades_df.loc[x.index, "side"] == "BUY"].sum()),
                    sell_amount=("filled_shares", lambda x: x[self.trades_df.loc[x.index, "side"] == "SELL"].sum()),
                    total_commission=("commission", "sum"),
                    total_stamp_duty=("stamp_duty", "sum"),
                    realized_pnl=("pnl", "sum"),
                )
                .reset_index()
            )
            daily.to_csv(out / f"{prefix}daily_trade_summary.csv", index=False)

        # 异常事件
        self.exceptions_df.to_csv(out / f"{prefix}exceptions.csv", index=False)

        logger.info("回测明细已保存至: %s", out)
        return out


# ═══════════════════════════════════════════════════════════════════════
# BacktestRunner: 主循环调度
# ═══════════════════════════════════════════════════════════════════════
class BacktestRunner:
    """
    精细化回测主控器。

    Parameters
    ----------
    score_df : pd.DataFrame
        打分宽表 (index=TRADE_DATE, columns=StockID)。
        ⚠️ 该表必须已经过 shift(1), 即 score_df.loc[T] 仅含 T-1 收盘后信息。
    mirror_quantile_group : int
        做多第几组 (1 = 最高分组, n_quantile_groups = 最低分组)。
        与 quick_backtest 的分层逻辑一一对应。
    top_n : int, optional
        若设置, 每个调仓日按分数买入前 top_n 只股票; 优先于 mirror_quantile_group。
    n_quantile_groups : int
        截面分组总数, 默认 20。
    rebalance_freq : int
        调仓频率 (交易日数), 默认 1 = 每日调仓。
    initial_capital : float
        初始资金, 默认取 config.INITIAL_CAPITAL。
    frictionless : bool
        True 时佣金/印花税/滑点全部清零, 便于与 quick_backtest 对齐。
    commission_rate / stamp_duty_rate / slippage_bps
        frictionless=False 时生效的费率参数。
    twap_tag : str
        TWAP 价格表标签, 默认 "TWAP_1430_1457"。
    """

    def __init__(
        self,
        score_df: pd.DataFrame,
        mirror_quantile_group: int = 1,
        top_n: Optional[int] = None,
        n_quantile_groups: int = config.N_QUANTILE_GROUPS,
        rebalance_freq: int = 1,
        initial_capital: float = config.INITIAL_CAPITAL,
        frictionless: bool = False,
        commission_rate: float = config.COMMISSION_RATE,
        stamp_duty_rate: float = config.STAMP_DUTY_RATE,
        slippage_bps: float = config.SLIPPAGE_BPS,
        twap_tag: str = "TWAP_1430_1457",
        min_listing_days: int = 20,
        delist_buffer_days: int = 20,
        exclude_st: bool = True,
        exclude_historical_st: bool = True,
        excluded_prefixes: Optional[tuple[str, ...]] = ("300", "688"),
    ):
        if not (1 <= mirror_quantile_group <= n_quantile_groups):
            raise ValueError(
                f"mirror_quantile_group={mirror_quantile_group} 超出范围 [1, {n_quantile_groups}]"
            )
        if rebalance_freq < 1:
            raise ValueError("rebalance_freq 必须 >= 1")
        if top_n is not None and top_n < 1:
            raise ValueError("top_n 必须 >= 1")

        self.score_df = score_df
        self.mirror_quantile_group = mirror_quantile_group
        self.top_n = top_n
        self.n_quantile_groups = n_quantile_groups
        self.rebalance_freq = rebalance_freq
        self.min_listing_days = min_listing_days
        self.delist_buffer_days = delist_buffer_days
        self.exclude_st = exclude_st
        self.exclude_historical_st = exclude_historical_st
        self.excluded_prefixes = excluded_prefixes or ()

        # 自动加载价格/涨跌停数据
        self.twap_df = _load_twap(twap_tag)
        self.limit_up_df   = _load_daily_wide("LIMIT_UP_PRICE")
        self.limit_down_df = _load_daily_wide("LIMIT_DOWN_PRICE")
        self.ipo_dates = load_ipo_dates()
        self.out_dates = load_out_dates()
        self.st_status_df = load_st_status() if exclude_st else None
        self.historical_st_df = None
        if self.st_status_df is not None and self.exclude_historical_st:
            self.historical_st_df = self.st_status_df.groupby(level=0).max().sort_index().fillna(0).ne(0).cummax()

        # region agent log
        try:
            info = pd.read_pickle(config.DAILY_DATA_DIR / "ipo_dates.pkl")
            row = info.loc[info["TICKER_SYMBOL"].astype(str).str.zfill(6) == "600225"].tail(1)
            listing_info = row.iloc[0].to_dict() if len(row) else {}
        except Exception as exc:
            listing_info = {"error": str(exc)}
        probe_dates = [pd.Timestamp("2025-02-27"), pd.Timestamp("2025-05-07")]
        probe = {}
        for d in probe_dates:
            st_value = None
            if self.st_status_df is not None:
                try:
                    st_value = self.st_status_df.loc[d, "600225"]
                except KeyError:
                    st_value = None
            historical_st_value = None
            if self.historical_st_df is not None:
                try:
                    historical_st_value = self.historical_st_df.loc[d, "600225"]
                except KeyError:
                    historical_st_value = None
            score_value = None
            try:
                score_value = self.score_df.loc[d, "600225"]
            except KeyError:
                score_value = None
            probe[str(d.date())] = {
                "score": score_value,
                "twap": self._get_price(d, "600225"),
                "limit_up": self._get_limit_up(d, "600225"),
                "limit_down": self._get_limit_down(d, "600225"),
                "st_status": st_value,
                "historical_st_status": historical_st_value,
            }
        _agent_debug_log(
            "D1,D2,D3,D4",
            "600225 listing and data probe at runner init",
            {"listing_info": listing_info, "probe": probe},
        )
        # endregion

        # frictionless 模式清零所有费率
        if frictionless:
            commission_rate = stamp_duty_rate = slippage_bps = 0.0

        self.portfolio = Portfolio(initial_capital)
        self.engine = TradeEngine(commission_rate, stamp_duty_rate, slippage_bps)
        self.exception_tracker = ExceptionTracker()
        self.trade_logger = TradeLogger()

        self._delayed_sells: Dict[str, Position] = {}

    def _get_price(self, date: pd.Timestamp, stock: str) -> float:
        try:
            v = self.twap_df.loc[date, stock]
            return float(v) if pd.notna(v) else np.nan
        except KeyError:
            return np.nan

    def _get_limit_up(self, date: pd.Timestamp, stock: str) -> float:
        if self.limit_up_df is None:
            return np.nan
        try:
            v = self.limit_up_df.loc[date, stock]
            return float(v) if pd.notna(v) else np.nan
        except KeyError:
            return np.nan

    def _get_limit_down(self, date: pd.Timestamp, stock: str) -> float:
        if self.limit_down_df is None:
            return np.nan
        try:
            v = self.limit_down_df.loc[date, stock]
            return float(v) if pd.notna(v) else np.nan
        except KeyError:
            return np.nan

    def _is_universe_eligible(self, date: pd.Timestamp, stock: str) -> bool:
        """回测股票池: 剔除新股、退市前缓冲期、截至 T 日曾经 ST 以及当前无法买入的代码前缀。"""
        stock = str(stock).zfill(6)
        if self.excluded_prefixes and stock.startswith(self.excluded_prefixes):
            return False

        if self.ipo_dates is not None:
            try:
                ipo_date = pd.Timestamp(self.ipo_dates.loc[stock])
            except KeyError:
                return False
            if pd.isna(ipo_date) or pd.Timestamp(date).normalize() <= ipo_date + pd.Timedelta(days=self.min_listing_days):
                return False

        if self.out_dates is not None:
            try:
                out_date = pd.Timestamp(self.out_dates.loc[stock])
            except KeyError:
                out_date = pd.NaT
            if pd.notna(out_date) and pd.Timestamp(date).normalize() >= out_date - pd.Timedelta(days=self.delist_buffer_days):
                return False

        if self.exclude_st and self.st_status_df is not None:
            try:
                st_value = self.st_status_df.loc[pd.Timestamp(date).normalize(), stock]
            except KeyError:
                st_value = 0
            historical_st_value = False
            if self.historical_st_df is not None:
                try:
                    historical_st_value = bool(self.historical_st_df.loc[pd.Timestamp(date).normalize(), stock])
                except KeyError:
                    historical_st_value = False
            # region agent log
            if stock == "600225" and pd.Timestamp("2025-02-20") <= pd.Timestamp(date).normalize() <= pd.Timestamp("2025-03-03"):
                _agent_debug_log(
                    "D2,D3,H1,H2",
                    "600225 universe eligibility probe",
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "stock": stock,
                        "ipo_date": str(self.ipo_dates.loc[stock]) if self.ipo_dates is not None and stock in self.ipo_dates.index else None,
                        "st_value": st_value,
                        "historical_st_value": historical_st_value,
                        "exclude_st": self.exclude_st,
                        "exclude_historical_st": self.exclude_historical_st,
                    },
                )
            # endregion
            if historical_st_value:
                return False
            if pd.notna(st_value) and float(st_value) != 0.0:
                return False
        return True

    def _generate_target(self, trade_date: pd.Timestamp) -> List[str]:
        """
        基于 trade_date 的打分, 按分位组选股。

        score_df.loc[T] 已含 shift(1), 仅包含 T-1 收盘后信息。
        若 top_n 不为空, 取分数最高的 top_n 只; 否则按分位组取第 mirror_quantile_group 组。
        """
        if trade_date not in self.score_df.index:
            return []
        scores = self.score_df.loc[trade_date].dropna().sort_values(ascending=False)
        tradable = []
        for stock in scores.index:
            if not self._is_universe_eligible(trade_date, stock):
                continue
            twap = self._get_price(trade_date, stock)
            if np.isnan(twap):
                continue
            lu = self._get_limit_up(trade_date, stock)
            if self.engine.is_limit_up(stock, twap, lu):
                continue
            tradable.append(stock)
        scores = scores.loc[tradable]
        n = len(scores)
        if self.top_n is not None:
            targets = list(scores.iloc[: min(self.top_n, n)].index)
            # region agent log
            if "600225" in scores.index or pd.Timestamp(trade_date).normalize() in {pd.Timestamp("2025-02-27"), pd.Timestamp("2025-05-07")}:
                _agent_debug_log(
                    "D1,D2,D3",
                    "600225 target generation probe",
                    {
                        "date": str(pd.Timestamp(trade_date).date()),
                        "stock": "600225",
                        "score_exists_after_universe_filter": "600225" in scores.index,
                        "score_value": scores.get("600225", None),
                        "rank_after_filter": int(scores.index.get_loc("600225")) + 1 if "600225" in scores.index else None,
                        "in_targets": "600225" in targets,
                        "target_count": len(targets),
                        "twap": self._get_price(trade_date, "600225"),
                        "st_value": self.st_status_df.loc[pd.Timestamp(trade_date).normalize(), "600225"]
                        if self.st_status_df is not None
                        and pd.Timestamp(trade_date).normalize() in self.st_status_df.index
                        and "600225" in self.st_status_df.columns
                        else None,
                    },
                )
            # endregion
            return targets
        if n < self.n_quantile_groups:
            return []
        g = self.mirror_quantile_group
        G = self.n_quantile_groups
        start = int((g - 1) / G * n)
        end   = int(g / G * n)
        return list(scores.iloc[start:end].index)

    def _execute_sells(self, trade_date: pd.Timestamp, sell_all: bool = True):
        """执行卖出; 非调仓日仅继续尝试此前延迟卖出的仓位。"""
        stocks_to_sell = list(self._delayed_sells.keys())
        if sell_all:
            stocks_to_sell += list(self.portfolio.positions.keys())
        stocks_to_sell = list(set(stocks_to_sell))

        for stock in stocks_to_sell:
            if stock in self._delayed_sells:
                pos = self._delayed_sells[stock]
            elif self.portfolio.has_position(stock):
                pos = self.portfolio.positions[stock]
            else:
                continue

            twap = self._get_price(trade_date, stock)
            # region agent log
            if stock == "600225":
                _agent_debug_log(
                    "D1,D2,D3",
                    "600225 sell attempt probe",
                    {
                        "date": str(pd.Timestamp(trade_date).date()),
                        "twap": twap,
                        "is_delayed_sell": stock in self._delayed_sells,
                        "has_position": self.portfolio.has_position(stock),
                    },
                )
            # endregion
            if np.isnan(twap):
                self.exception_tracker.log(
                    trade_date, stock, ExceptionType.NO_PRICE_DATA,
                    f"卖出时无 TWAP 价格, 延后"
                )
                if stock not in self._delayed_sells:
                    self._delayed_sells[stock] = self.portfolio.remove_position(stock)
                continue

            ld = self._get_limit_down(trade_date, stock)
            if self.engine.is_limit_down(stock, twap, ld):
                self.exception_tracker.log(
                    trade_date, stock, ExceptionType.LIMIT_DOWN_CANNOT_SELL,
                    f"跌停封死, TWAP={twap:.2f}, LD={ld:.2f}, 延后卖出"
                )
                if stock not in self._delayed_sells:
                    self._delayed_sells[stock] = self.portfolio.remove_position(stock)
                continue

            sell_price = self.engine.calc_sell_price(twap)
            shares = pos.shares
            amount = sell_price * shares
            commission = self.engine.calc_commission(amount)
            stamp_duty = self.engine.calc_stamp_duty(amount)
            net_proceeds = amount - commission - stamp_duty
            pnl = (sell_price - pos.entry_price) * shares - commission - stamp_duty

            self.portfolio.cash += net_proceeds

            if stock in self._delayed_sells:
                del self._delayed_sells[stock]
            elif self.portfolio.has_position(stock):
                self.portfolio.remove_position(stock)

            order = Order(
                date=trade_date, stock=stock, side=OrderSide.SELL,
                target_shares=shares, anchor_price=twap, actual_price=sell_price,
                filled_shares=shares, commission=commission, stamp_duty=stamp_duty,
                pnl=pnl, filled=True,
            )
            self.trade_logger.log(order)

    def _execute_buys(self, trade_date: pd.Timestamp, targets: List[str]):
        """执行买入 (可用资金约束 + 涨停检查)"""
        if not targets:
            return

        valid_targets = []
        for stock in targets:
            if stock in self._delayed_sells or self.portfolio.has_position(stock):
                continue
            twap = self._get_price(trade_date, stock)
            # region agent log
            if stock == "600225":
                _agent_debug_log(
                    "D1,D2,D3",
                    "600225 buy precheck probe",
                    {
                        "date": str(pd.Timestamp(trade_date).date()),
                        "twap": twap,
                        "limit_up": self._get_limit_up(trade_date, stock),
                        "st_value": self.st_status_df.loc[pd.Timestamp(trade_date).normalize(), stock]
                        if self.st_status_df is not None
                        and pd.Timestamp(trade_date).normalize() in self.st_status_df.index
                        and stock in self.st_status_df.columns
                        else None,
                    },
                )
            # endregion
            if np.isnan(twap):
                self.exception_tracker.log(
                    trade_date, stock, ExceptionType.NO_PRICE_DATA,
                    "买入时无 TWAP 价格"
                )
                continue

            lu = self._get_limit_up(trade_date, stock)
            if self.engine.is_limit_up(stock, twap, lu):
                self.exception_tracker.log(
                    trade_date, stock, ExceptionType.LIMIT_UP_CANNOT_BUY,
                    f"涨停封死, TWAP={twap:.2f}, LU={lu:.2f}"
                )
                continue
            valid_targets.append((stock, twap))

        if not valid_targets:
            return

        per_stock_cash = self.portfolio.cash / len(valid_targets)

        for stock, twap in valid_targets:
            buy_price = self.engine.calc_buy_price(twap)
            shares = int(per_stock_cash / (buy_price * (1 + self.engine.commission_rate)) / 100) * 100
            if shares <= 0:
                self.exception_tracker.log(
                    trade_date, stock, ExceptionType.INSUFFICIENT_CASH,
                    f"可用资金不足, cash_per_stock={per_stock_cash:.2f}, price={buy_price:.2f}"
                )
                continue

            amount = buy_price * shares
            commission = self.engine.calc_commission(amount)
            total_cost = amount + commission

            if total_cost > self.portfolio.cash:
                shares = int(self.portfolio.cash / (buy_price * (1 + self.engine.commission_rate)) / 100) * 100
                if shares <= 0:
                    self.exception_tracker.log(
                        trade_date, stock, ExceptionType.INSUFFICIENT_CASH,
                        f"资金不足, 需要={total_cost:.2f}, 可用={self.portfolio.cash:.2f}"
                    )
                    continue
                amount = buy_price * shares
                commission = self.engine.calc_commission(amount)
                total_cost = amount + commission

            self.portfolio.cash -= total_cost
            self.portfolio.add_position(stock, shares, buy_price, trade_date)

            order = Order(
                date=trade_date, stock=stock, side=OrderSide.BUY,
                target_shares=shares, anchor_price=twap, actual_price=buy_price,
                filled_shares=shares, commission=commission, stamp_duty=0.0,
                pnl=0.0, filled=True,
            )
            self.trade_logger.log(order)

    def run(
        self,
        start_date=None,
        end_date=None,
    ) -> "BacktestResult":
        """
        运行回测主循环。

        Parameters
        ----------
        start_date : str | date | Timestamp | None
            回测起始日 (含), None 表示从 twap_df 第一个交易日开始。
        end_date : str | date | Timestamp | None
            回测终止日 (含), None 表示到 twap_df 最后一个交易日。

        Returns
        -------
        BacktestResult
            包含 NAV 序列、逐笔成交记录、异常事件记录。
        """
        def _to_ts(d):
            return None if d is None else pd.Timestamp(d)

        ts_start = _to_ts(start_date)
        ts_end   = _to_ts(end_date)

        trade_dates = self.twap_df.index.sort_values()
        if ts_start is not None:
            trade_dates = trade_dates[trade_dates >= ts_start]
        if ts_end is not None:
            trade_dates = trade_dates[trade_dates <= ts_end]
        trade_dates = list(trade_dates)

        logger.info(
            "回测开始: %s ~ %s, %d 交易日, mode=%s, rebalance_freq=%d, 初始资金=%.0f",
            trade_dates[0].strftime("%Y-%m-%d") if trade_dates else "N/A",
            trade_dates[-1].strftime("%Y-%m-%d") if trade_dates else "N/A",
            len(trade_dates),
            f"top{self.top_n}" if self.top_n is not None else f"group={self.mirror_quantile_group}/{self.n_quantile_groups}",
            self.rebalance_freq,
            self.portfolio.initial_capital,
        )

        for i, trade_date in enumerate(trade_dates):
            is_rebalance_day = i % self.rebalance_freq == 0

            # 1. 调仓日卖出旧仓; 非调仓日只处理此前未能卖出的仓位
            self._execute_sells(trade_date, sell_all=is_rebalance_day)

            # 2. 生成买入目标 (仅调仓日)
            # score_df 存储时已 shift(1): score_df.loc[T] = 基于 T-1 收盘数据的信号
            if is_rebalance_day:
                targets = self._generate_target(trade_date)
            else:
                targets = []  # 非调仓日: 不开新仓, 已有仓位继续持有

            # 3. 执行买入
            self._execute_buys(trade_date, targets)

            # 4. 记录当日 NAV
            price_dict = {}
            for stock in set(self.portfolio.positions) | set(self._delayed_sells):
                px = self._get_price(trade_date, stock)
                if not np.isnan(px):
                    price_dict[stock] = px
            self.portfolio.record_nav(
                trade_date,
                price_dict,
                extra_positions=self._delayed_sells,
            )

            if (i + 1) % 50 == 0:
                logger.info(
                    "  Day %d/%d [%s] NAV=%.0f Cash=%.0f Pos=%d",
                    i + 1, len(trade_dates),
                    trade_date.strftime("%Y-%m-%d"),
                    self.portfolio.nav_history[-1]["nav"],
                    self.portfolio.cash,
                    len(self.portfolio.positions),
                )

        nav_df = pd.DataFrame(self.portfolio.nav_history)
        final_nav = nav_df["nav"].iloc[-1] if len(nav_df) > 0 else 0
        logger.info(
            "回测完成: 最终NAV=%.0f, 总收益=%.2f%%, 交易笔数=%d, 异常事件=%d",
            final_nav,
            (final_nav / self.portfolio.initial_capital - 1) * 100 if len(nav_df) > 0 else 0,
            len(self.trade_logger.orders),
            len(self.exception_tracker.records),
        )

        return BacktestResult(
            nav_df=nav_df,
            trades_df=self.trade_logger.to_dataframe(),
            exceptions_df=self.exception_tracker.to_dataframe(),
            initial_capital=self.portfolio.initial_capital,
            strategy_name=f"top{self.top_n}" if self.top_n is not None else "event_backtest",
        )

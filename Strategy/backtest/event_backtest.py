"""
精细化事件驱动回测引擎。

交易流水线 (严格遵循):
    T-1 日收盘后: 模型打分 -> 选股 -> 生成目标持仓
    T   日盘中:   先卖出旧持仓 (TWAP) -> 资金回笼 -> 再买入新目标 (TWAP)
    T+1 日盘中:   卖出 T 日持仓 -> 资金回笼 -> 买入新目标
    ... 循环

⚠️ 防未来数据:
- 选股信号严格基于 T-1 日收盘后可获取的信息
- 涨跌停判断使用当日 TWAP 区间分钟数据, 不使用日频收盘数据
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Strategy import config

logger = logging.getLogger(__name__)


def aggregate_trades_daily(trades: pd.DataFrame) -> pd.DataFrame:
    """
    将逐笔订单汇总为「按交易日」的买卖笔数、成交额、佣金、印花税、卖出侧 pnl。

    用于与 quantile_backtest 对比时核对: 事件回测在哪些天成交了多少、费用多少。
    """
    cols = [
        "TRADE_DATE",
        "n_buy",
        "n_sell",
        "buy_notional",
        "sell_notional",
        "commission",
        "stamp_duty",
        "pnl_on_sells",
    ]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=cols)

    t = trades.copy()
    t["TRADE_DATE"] = pd.to_datetime(t["date"], errors="coerce").dt.normalize()
    t = t.dropna(subset=["TRADE_DATE"])
    t["notional"] = t["filled_shares"].astype(float) * t["actual_price"].astype(float)
    t["buy_notional"] = np.where(t["side"] == "BUY", t["notional"], 0.0)
    t["sell_notional"] = np.where(t["side"] == "SELL", t["notional"], 0.0)
    t["n_buy"] = (t["side"] == "BUY").astype(int)
    t["n_sell"] = (t["side"] == "SELL").astype(int)
    t["pnl_sell"] = np.where(t["side"] == "SELL", t["pnl"].astype(float), 0.0)

    g = t.groupby("TRADE_DATE", sort=True).agg(
        n_buy=("n_buy", "sum"),
        n_sell=("n_sell", "sum"),
        buy_notional=("buy_notional", "sum"),
        sell_notional=("sell_notional", "sum"),
        commission=("commission", "sum"),
        stamp_duty=("stamp_duty", "sum"),
        pnl_on_sells=("pnl_sell", "sum"),
    ).reset_index()
    return g


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

    def record_nav(self, date: pd.Timestamp, price_dict: Dict[str, float]):
        """记录当日 NAV (使用当日 TWAP 估算持仓市值)"""
        pos_val = 0.0
        for stock, pos in self.positions.items():
            px = price_dict.get(stock, pos.entry_price)
            pos_val += pos.shares * px
        self.nav_history.append({
            "TRADE_DATE": date,
            "cash": self.cash,
            "position_value": pos_val,
            "nav": self.cash + pos_val,
            "n_positions": len(self.positions),
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
# BacktestResult: run() 的返回值，携带 plot() 方法
# ═══════════════════════════════════════════════════════════════════════
class BacktestResult:
    """
    回测结果容器。

    Attributes
    ----------
    nav : pd.DataFrame
        每日 NAV 记录 (含 cash / position_value / nav / n_positions)
    trades : pd.DataFrame
        逐笔订单明细
    exceptions : pd.DataFrame
        异常事件记录
    """

    def __init__(
        self,
        nav: pd.DataFrame,
        trades: pd.DataFrame,
        exceptions: pd.DataFrame,
        initial_capital: float,
    ):
        self.nav = nav
        self.trades = trades
        self.exceptions = exceptions
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    def plot(
        self,
        save_dir: Optional[Path] = None,
        save_path: Optional[Path] = None,
        figsize: tuple = (14, 6),
        title: str = "Event Backtest — NAV",
    ) -> Path:
        """
        绘制净值曲线并保存。

        Parameters
        ----------
        save_dir  : 输出目录; 图保存为 {save_dir}/nav_curve.png
        save_path : 完整文件路径 (与 save_dir 同时提供时以 save_path 优先)
        """
        if save_path is None:
            out_dir = Path(save_dir) if save_dir is not None else config.BT_RESULT_DIR
            save_path = out_dir / "nav_curve.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        nav_df = self.nav.copy()
        if "TRADE_DATE" in nav_df.columns:
            nav_df = nav_df.set_index("TRADE_DATE")
        nav_df.index = pd.to_datetime(nav_df.index)

        cum_ret = nav_df["nav"] / self.initial_capital - 1

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(nav_df.index, cum_ret.values, lw=1.2, label="Strategy")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("NAV 曲线已保存: %s", save_path)
        return save_path

    def summary(self, annual_days: int = 242) -> dict:
        """输出关键绩效指标"""
        nav_df = self.nav.copy()
        if "TRADE_DATE" in nav_df.columns:
            nav_df = nav_df.set_index("TRADE_DATE")
        ret = nav_df["nav"].pct_change().dropna()
        total = nav_df["nav"].iloc[-1] / self.initial_capital - 1
        ann = (1 + total) ** (annual_days / max(len(ret), 1)) - 1
        vol = ret.std() * np.sqrt(annual_days)
        sharpe = ann / vol if vol > 0 else 0.0
        cum = (1 + ret).cumprod()
        mdd = ((cum / cum.cummax()) - 1).min()
        return {"total_return": total, "ann_return": ann, "ann_vol": vol,
                "sharpe": sharpe, "max_drawdown": mdd, "n_trades": len(self.trades)}

    def save_details(self, output_dir: Optional[Path] = None) -> Path:
        """
        导出回测明细到 CSV:

        - ``nav.csv``              每日 NAV
        - ``trades_all.csv``       逐笔订单 (每笔一行)
        - ``exceptions.csv``       涨跌停、缺价等异常
        - ``daily_trade_summary.csv`` 按日汇总的买卖笔数与金额、费用
        """
        out = Path(output_dir or config.BT_RESULT_DIR)
        out.mkdir(parents=True, exist_ok=True)

        self.nav.to_csv(out / "nav.csv", index=False)
        self.trades.to_csv(out / "trades_all.csv", index=False)
        self.exceptions.to_csv(out / "exceptions.csv", index=False)
        aggregate_trades_daily(self.trades).to_csv(
            out / "daily_trade_summary.csv", index=False
        )
        logger.info(
            "回测明细已写入 %s: nav.csv, trades_all.csv, exceptions.csv, daily_trade_summary.csv",
            out,
        )
        return out


# ═══════════════════════════════════════════════════════════════════════
# BacktestRunner: 主循环调度
# ═══════════════════════════════════════════════════════════════════════
def _load_daily_wide(key: str) -> Optional[pd.DataFrame]:
    """尝试从 Daily_data/ 读宽表, 不存在时返回 None。"""
    path = config.DAILY_DATA_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    df = pd.read_pickle(path)
    df.index = pd.to_datetime(df.index)
    df.columns = pd.Index([str(c).zfill(6) for c in df.columns])
    return df


class BacktestRunner:
    """
    精细化回测主控器。

    Parameters
    ----------
    score_df : pd.DataFrame
        打分宽表 (index=TRADE_DATE, columns=StockID)
    twap_df : pd.DataFrame, optional
        TWAP 基准价格宽表; 不传时自动从 outputs/labels/TWAP_1430_1457.fea 加载
    limit_up_df : pd.DataFrame, optional
        涨停价宽表; 不传时自动从 Daily_data/LIMIT_UP_PRICE.pkl 加载
    limit_down_df : pd.DataFrame, optional
        跌停价宽表; 不传时自动从 Daily_data/LIMIT_DOWN_PRICE.pkl 加载
    top_n : int
        每日选取打分前 N 只股票买入 (与 ``mirror_quantile_group`` 二选一)
    mirror_quantile_group : int, optional
        若设为 ``1`` 且 ``n_quantile_groups=20``, 选股规则与 ``quantile_backtest``
        的 **group1** 一致 (截面分位第一层, 约前 5% 股票数, 等权含义在分层回测里
        是「组内收益平均」而非真实组合)。不设时沿用 ``top_n``。
    n_quantile_groups : int
        与快速分层回测 ``n_groups`` 对齐, 默认 ``config.N_QUANTILE_GROUPS``
    rebalance_freq : int
        调仓频率 (天); 1=每日调仓, 与 ``quantile_backtest`` 的「按日截面」一致
    initial_capital : float
        初始资金 (与分层图对比时建议与 ``config.INITIAL_CAPITAL`` 一致, 默认 100 万)
    frictionless : bool
        True 时佣金、印花税、滑点均为 0, 与 ``run_quick_backtest`` 的无摩擦假设对齐;
        实盘检验请设 False。
    commission_rate : float
    stamp_duty_rate : float
    slippage_bps : float
    """

    def __init__(
        self,
        score_df: pd.DataFrame,
        twap_df: Optional[pd.DataFrame] = None,
        limit_up_df: Optional[pd.DataFrame] = None,
        limit_down_df: Optional[pd.DataFrame] = None,
        top_n: int = 50,
        mirror_quantile_group: Optional[int] = None,
        n_quantile_groups: int = config.N_QUANTILE_GROUPS,
        rebalance_freq: int = 1,
        initial_capital: float = config.INITIAL_CAPITAL,
        frictionless: bool = False,
        commission_rate: float = config.COMMISSION_RATE,
        stamp_duty_rate: float = config.STAMP_DUTY_RATE,
        slippage_bps: float = config.SLIPPAGE_BPS,
    ):
        self.score_df = score_df
        self.top_n = top_n
        self.mirror_quantile_group = mirror_quantile_group
        self.n_quantile_groups = max(1, int(n_quantile_groups))
        self.rebalance_freq = max(1, int(rebalance_freq))
        self.frictionless = bool(frictionless)
        if self.frictionless:
            commission_rate = 0.0
            stamp_duty_rate = 0.0
            slippage_bps = 0.0

        # ── 自动加载价格/涨跌停数据 ──────────────────────────────────
        if twap_df is None:
            from Strategy.label.label_generator import load_price
            twap_df = load_price("TWAP_1430_1457")
            logger.info("已自动加载 TWAP_1430_1457 价格表, shape=%s", twap_df.shape)
        if limit_up_df is None:
            limit_up_df = _load_daily_wide("LIMIT_UP_PRICE")
            if limit_up_df is None:
                logger.warning("LIMIT_UP_PRICE.pkl 不存在, 涨停判断将跳过")
                limit_up_df = pd.DataFrame()
        if limit_down_df is None:
            limit_down_df = _load_daily_wide("LIMIT_DOWN_PRICE")
            if limit_down_df is None:
                logger.warning("LIMIT_DOWN_PRICE.pkl 不存在, 跌停判断将跳过")
                limit_down_df = pd.DataFrame()

        self.twap_df = twap_df
        self.limit_up_df = limit_up_df
        self.limit_down_df = limit_down_df

        self.portfolio = Portfolio(initial_capital)
        self.engine = TradeEngine(commission_rate, stamp_duty_rate, slippage_bps)
        self.exception_tracker = ExceptionTracker()
        self.trade_logger = TradeLogger()

        self._delayed_sells: Dict[str, Position] = {}

    def _get_price(self, date: pd.Timestamp, stock: str) -> float:
        try:
            return self.twap_df.loc[date, stock]
        except KeyError:
            return np.nan

    def _get_limit_up(self, date: pd.Timestamp, stock: str) -> float:
        try:
            return self.limit_up_df.loc[date, stock]
        except KeyError:
            return np.nan

    def _get_limit_down(self, date: pd.Timestamp, stock: str) -> float:
        try:
            return self.limit_down_df.loc[date, stock]
        except KeyError:
            return np.nan

    def _generate_target(self, signal_date: pd.Timestamp) -> List[str]:
        """
        基于 signal_date 当日 score 生成买入目标。

        ``score_df[T]`` 已是 T-1 日因子可得信息下的打分 (因子宽表做过 shift(1)),
        与 ``quantile_backtest`` 在日期 T 上用的截面一致。
        """
        if signal_date not in self.score_df.index:
            return []
        scores = self.score_df.loc[signal_date].dropna().sort_values(ascending=False)
        if self.mirror_quantile_group is not None:
            g = int(self.mirror_quantile_group)
            ng = self.n_quantile_groups
            if g < 1 or g > ng:
                raise ValueError(f"mirror_quantile_group 须在 [1, {ng}] 内, 当前 {g}")
            ranks = scores.rank(method="first", ascending=False)
            gs = len(scores) / ng
            mask = (ranks > (g - 1) * gs) & (ranks <= g * gs)
            return list(scores.loc[mask].index)
        return list(scores.index[: self.top_n])

    def _execute_sells(self, trade_date: pd.Timestamp):
        """执行所有卖出 (含延迟卖出的跌停股)"""
        stocks_to_sell = list(self.portfolio.positions.keys()) + list(self._delayed_sells.keys())
        stocks_to_sell = list(set(stocks_to_sell))

        for stock in stocks_to_sell:
            if stock in self._delayed_sells:
                pos = self._delayed_sells[stock]
            elif self.portfolio.has_position(stock):
                pos = self.portfolio.positions[stock]
            else:
                continue

            twap = self._get_price(trade_date, stock)
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
            twap = self._get_price(trade_date, stock)
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
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> "BacktestResult":
        """
        运行回测主循环。

        T-1 日产出信号 -> T 日执行交易 (先卖后买)；
        ``rebalance_freq`` 控制每隔几天才调一次仓。

        Returns
        -------
        BacktestResult  含 .nav / .trades / .exceptions 及 .plot() / .summary()
        """
        trade_dates = self.twap_df.index.sort_values()
        if start_date is not None:
            trade_dates = trade_dates[trade_dates >= pd.Timestamp(start_date)]
        if end_date is not None:
            trade_dates = trade_dates[trade_dates <= pd.Timestamp(end_date)]
        trade_dates = list(trade_dates)

        if not trade_dates:
            raise ValueError("回测日期范围内无可用交易日")

        sel = (
            f"mirror_group{self.mirror_quantile_group}/{self.n_quantile_groups}"
            if self.mirror_quantile_group is not None
            else f"top_{self.top_n}"
        )
        logger.info(
            "回测开始: %s ~ %s, %d 交易日, %s, 调仓=%d天, 本金=%.0f, frictionless=%s",
            trade_dates[0].strftime("%Y-%m-%d"),
            trade_dates[-1].strftime("%Y-%m-%d"),
            len(trade_dates),
            sel,
            self.rebalance_freq,
            self.portfolio.initial_capital,
            self.frictionless,
        )

        rebalance_counter = 0
        for i, trade_date in enumerate(trade_dates):
            # ⚠️ score_df[T] 已经是用 T-1 日因子打出的分 (因子文件做了 shift(1)),
            # 代表"T-1 日收盘后可知的信号 → 预测 T→T+1 收益"。
            # 因此在 T 日交易时, 直接取 score_df.loc[trade_date] 即可,
            # 无需再往前取 trade_dates[i-1], 否则信号会落后 2 天。
            signal_date = trade_date

            # 1. 执行卖出 (含跌停延后) — 只在调仓日才换仓
            do_rebalance = (rebalance_counter % self.rebalance_freq == 0)
            if do_rebalance:
                self._execute_sells(trade_date)
            rebalance_counter += 1

            # 2. 生成买入目标 (基于当日 score, 即 T-1 日因子信号)
            targets: List[str] = []
            if do_rebalance:
                targets = self._generate_target(signal_date)

            # 3. 执行买入
            if do_rebalance:
                self._execute_buys(trade_date, targets)

            # 4. 记录当日 NAV
            price_dict = {}
            for stock in self.portfolio.positions:
                px = self._get_price(trade_date, stock)
                if not np.isnan(px):
                    price_dict[stock] = px
            self.portfolio.record_nav(trade_date, price_dict)

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
        logger.info(
            "回测完成: 最终NAV=%.0f, 总收益=%.2f%%, 交易笔数=%d, 异常事件=%d",
            nav_df["nav"].iloc[-1] if len(nav_df) > 0 else 0,
            ((nav_df["nav"].iloc[-1] / self.portfolio.initial_capital - 1) * 100) if len(nav_df) > 0 else 0,
            len(self.trade_logger.orders),
            len(self.exception_tracker.records),
        )
        return BacktestResult(
            nav=nav_df,
            trades=self.trade_logger.to_dataframe(),
            exceptions=self.exception_tracker.to_dataframe(),
            initial_capital=self.portfolio.initial_capital,
        )

    def get_results(self) -> dict:
        """汇总回测结果"""
        nav_df = pd.DataFrame(self.portfolio.nav_history)
        return {
            "nav": nav_df,
            "trades": self.trade_logger.to_dataframe(),
            "exceptions": self.exception_tracker.to_dataframe(),
        }

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """保存回测结果到文件"""
        out = output_dir or config.BT_RESULT_DIR
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        results["nav"].to_csv(out / "nav.csv", index=False)
        results["trades"].to_csv(out / "trades.csv", index=False)
        results["exceptions"].to_csv(out / "exceptions.csv", index=False)
        aggregate_trades_daily(results["trades"]).to_csv(
            out / "daily_trade_summary.csv", index=False
        )

        logger.info("回测结果已保存: %s", out)
        return out

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

import pandas as pd
import numpy as np

from Strategy import config

logger = logging.getLogger(__name__)


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
# BacktestRunner: 主循环调度
# ═══════════════════════════════════════════════════════════════════════
class BacktestRunner:
    """
    精细化回测主控器。

    Parameters
    ----------
    score_df : pd.DataFrame
        打分宽表 (index=TRADE_DATE, columns=StockID)
    twap_df : pd.DataFrame
        TWAP 基准价格宽表
    limit_up_df : pd.DataFrame
        涨停价宽表
    limit_down_df : pd.DataFrame
        跌停价宽表
    top_n : int
        每日选取打分前 N 只股票买入
    initial_capital : float
        初始资金
    commission_rate : float
    stamp_duty_rate : float
    slippage_bps : float
    """

    def __init__(
        self,
        score_df: pd.DataFrame,
        twap_df: pd.DataFrame,
        limit_up_df: pd.DataFrame,
        limit_down_df: pd.DataFrame,
        top_n: int = 50,
        initial_capital: float = config.INITIAL_CAPITAL,
        commission_rate: float = config.COMMISSION_RATE,
        stamp_duty_rate: float = config.STAMP_DUTY_RATE,
        slippage_bps: float = config.SLIPPAGE_BPS,
    ):
        self.score_df = score_df
        self.twap_df = twap_df
        self.limit_up_df = limit_up_df
        self.limit_down_df = limit_down_df
        self.top_n = top_n

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
        基于 signal_date (T-1) 的打分生成买入目标。
        ⚠️ 只使用 signal_date 及之前的数据, 信号基于 T-1 收盘。
        """
        if signal_date not in self.score_df.index:
            return []
        scores = self.score_df.loc[signal_date].dropna().sort_values(ascending=False)
        return list(scores.index[:self.top_n])

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
    ) -> pd.DataFrame:
        """
        运行回测主循环。

        T-1 日产出信号 -> T 日执行交易 (先卖后买)

        Returns
        -------
        pd.DataFrame  每日 NAV 记录
        """
        trade_dates = self.twap_df.index.sort_values()
        if start_date is not None:
            trade_dates = trade_dates[trade_dates >= start_date]
        if end_date is not None:
            trade_dates = trade_dates[trade_dates <= end_date]
        trade_dates = list(trade_dates)

        logger.info(
            "回测开始: %s ~ %s, %d 交易日, top_%d, 初始资金=%.0f",
            trade_dates[0].strftime("%Y-%m-%d") if trade_dates else "N/A",
            trade_dates[-1].strftime("%Y-%m-%d") if trade_dates else "N/A",
            len(trade_dates), self.top_n, self.portfolio.initial_capital,
        )

        for i, trade_date in enumerate(trade_dates):
            signal_date = trade_dates[i - 1] if i > 0 else None

            # 1. 执行卖出 (含跌停延后)
            self._execute_sells(trade_date)

            # 2. 生成买入目标 (基于 T-1 信号)
            targets = []
            if signal_date is not None:
                targets = self._generate_target(signal_date)

            # 3. 执行买入
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
        return nav_df

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

        logger.info("回测结果已保存: %s", out)
        return out

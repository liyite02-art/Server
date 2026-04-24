"""
Backtesting engine for RL-mined factors.

Data logic (same as env):
  - Factor operates on (n_days*237, n_stocks) minute-level mmap data
  - Final factor value = last minute (second=145600000) of each day
  - TS operators need full 237 bars;  CS / element-wise only the last bar

Output:
  - 20-group cumulative net-value curves (matching the provided example figure)
  - Per-group: annualised return, annualised volatility, Sharpe ratio
  - IC / RankIC / ICIR summary
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..data.mmap_dataset import MMapDataset
from ..data.label_loader import LabelLoader
from ..operators.registry import OperatorRegistry

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 243


@dataclass
class BacktestResult:
    dates: List[str]
    group_nav: np.ndarray       # (n_days, n_groups) cumulative return
    group_daily_ret: np.ndarray # (n_days, n_groups) daily return
    ic: float
    rank_ic: float
    ic_ir: float
    ann_ret: np.ndarray         # (n_groups,) annualised return
    ann_vol: np.ndarray         # (n_groups,) annualised volatility
    sharpe: np.ndarray          # (n_groups,) Sharpe ratio


# ------------------------------------------------------------------ #
#                       helper functions                               #
# ------------------------------------------------------------------ #

def _factor_to_daily(factor: torch.Tensor, n_minutes: int) -> torch.Tensor:
    """Extract last minute (145600000) of each day → (n_days, n_stocks)."""
    n_rows = factor.shape[0]
    n_days = n_rows // n_minutes
    trimmed = factor[: n_days * n_minutes]
    reshaped = trimmed.reshape(n_days, n_minutes, -1)
    return reshaped[:, -1, :]


def _daily_rank_ic_and_ic(factor: torch.Tensor,
                          label: torch.Tensor) -> Tuple[float, float, float]:
    """Return (pearson_ic_mean, rank_ic_mean, ic_ir)."""
    n = min(factor.shape[0], label.shape[0])
    factor, label = factor[:n], label[:n]

    rank_ics, pearson_ics = [], []
    for d in range(n):
        fd, ld = factor[d], label[d]
        v = ~(torch.isnan(fd) | torch.isnan(ld) | torch.isinf(fd))
        if v.sum() < 30:
            continue
        fv, lv = fd[v], ld[v]
        # RankIC
        fr = fv.argsort().argsort().float(); lr = lv.argsort().argsort().float()
        fr -= fr.mean(); lr -= lr.mean()
        rank_ics.append(
            ((fr * lr).sum() / (torch.sqrt((fr**2).sum() * (lr**2).sum()) + 1e-9)).item()
        )
        # Pearson
        fc = fv - fv.mean(); lc = lv - lv.mean()
        pearson_ics.append(
            ((fc * lc).sum() / (torch.sqrt((fc**2).sum() * (lc**2).sum()) + 1e-9)).item()
        )

    if not rank_ics:
        return 0.0, 0.0, 0.0
    rm = float(np.mean(rank_ics))
    rs = float(np.std(rank_ics) + 1e-9)
    pm = float(np.mean(pearson_ics))
    return pm, rm, rm / rs


def _group_returns(factor: torch.Tensor, label: torch.Tensor,
                   n_groups: int) -> Tuple[np.ndarray, List[int]]:
    """Per-day group returns → (n_valid_days, n_groups), valid_day_indices."""
    n = min(factor.shape[0], label.shape[0])
    factor, label = factor[:n], label[:n]

    ret_list: List[np.ndarray] = []
    valid_idx: List[int] = []

    for d in range(n):
        fd, ld = factor[d], label[d]
        v = ~(torch.isnan(fd) | torch.isnan(ld) | torch.isinf(fd))
        if v.sum() < 2 * n_groups:
            continue
        fv, lv = fd[v], ld[v]
        nn = fv.shape[0]
        order = fv.argsort()
        gs = nn // n_groups
        day_ret = []
        for g in range(n_groups):
            s = g * gs
            e = (g + 1) * gs if g < n_groups - 1 else nn
            day_ret.append(lv[order[s:e]].mean().item())
        ret_list.append(np.array(day_ret, dtype=np.float64))
        valid_idx.append(d)

    if not ret_list:
        return np.zeros((0, n_groups)), []
    return np.vstack(ret_list), valid_idx


# ------------------------------------------------------------------ #
#                       FactorBacktester                               #
# ------------------------------------------------------------------ #

class FactorBacktester:
    """Build factor from expression, run group backtest, plot results."""

    def __init__(
        self,
        dataset: MMapDataset,
        label_loader: LabelLoader,
        op_registry: OperatorRegistry,
        feature_names: List[str],
        n_groups: int = 20,
        label_shift: int = 1,
        device: str = 'cpu',
    ):
        self.dataset = dataset
        self.label_loader = label_loader
        self.op_registry = op_registry
        self.feature_names = feature_names
        self.n_groups = n_groups
        self.label_shift = max(0, int(label_shift))
        self.device = device

    def _align_factor_label(self, factor_daily: torch.Tensor,
                            label_daily: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align factor and label with positive shift:
          label_shift=1 => factor[t] vs label[t+1]
        """
        n = min(factor_daily.shape[0], label_daily.shape[0])
        factor_daily = factor_daily[:n]
        label_daily = label_daily[:n]

        if self.label_shift == 0:
            return factor_daily, label_daily
        if n <= self.label_shift:
            return factor_daily[:0], label_daily[:0]
        return factor_daily[:-self.label_shift], label_daily[self.label_shift:]

    # ---------- expression parsing ---------- #

    def _parse_expression(self, expr: str) -> List[Tuple[int, int, int]]:
        steps: List[Tuple[int, int, int]] = []
        if not expr:
            return steps
        for i, seg in enumerate(s for s in expr.split(';') if s):
            rhs = seg.split('=', 1)[1].strip()
            if i == 0:
                feat_name, trans_name = self._parse_feat_expr(rhs)
                feat_idx = self.feature_names.index(feat_name)
                trans_idx = self._get_transform_index(trans_name)
                op_idx = 0
            else:
                op_name, rest = rhs.split('(', 1)
                rest = rest.rstrip(')')
                if not rest.startswith('temp,'):
                    raise ValueError(f"Unexpected format: {seg}")
                feat_part = rest[len('temp,'):]
                feat_name, trans_name = self._parse_feat_expr(feat_part)
                feat_idx = self.feature_names.index(feat_name)
                trans_idx = self._get_transform_index(trans_name)
                op_idx = self.op_registry.binary_op_names.index(op_name)
            steps.append((op_idx, feat_idx, trans_idx))
        return steps

    def _parse_feat_expr(self, s: str) -> Tuple[str, str]:
        s = s.strip()
        if '(' not in s:
            return s, 'identity'
        t_name, inner = s.split('(', 1)
        return inner.rstrip(')').strip(), t_name.strip()

    def _get_transform_index(self, trans_name: str) -> int:
        for i, (name, _, _) in enumerate(self.op_registry.transforms):
            if name == trans_name:
                return i
        raise ValueError(f"Transform '{trans_name}' not in registry")

    # ---------- factor reconstruction ---------- #

    def build_factor_from_expression(self, expression: str,
                                     date_start: str,
                                     date_end: str) -> torch.Tensor:
        """Reconstruct minute-level factor tensor from expression string."""
        steps = self._parse_expression(expression)
        if not steps:
            raise ValueError("Empty expression")

        used = sorted({fi for _, fi, _ in steps})
        feat_t: Dict[int, torch.Tensor] = {}
        for fi in used:
            feat_t[fi] = self.dataset.load_field(
                self.feature_names[fi], date_start, date_end, to_torch=True
            ).to(self.device)

        temp: Optional[torch.Tensor] = None
        for i, (op_idx, fi, ti) in enumerate(steps):
            x = self.op_registry.apply_transform(ti, feat_t[fi])
            if i == 0:
                temp = x
            else:
                m = min(temp.shape[0], x.shape[0])
                temp = self.op_registry.apply_binary_op(op_idx, temp[-m:], x[-m:])
        return temp

    # ---------- main backtest ---------- #

    def backtest_expression(self, expression: str,
                            date_start: str,
                            date_end: str) -> BacktestResult:
        logger.info("Backtesting: %s", expression)

        factor_min = self.build_factor_from_expression(expression, date_start, date_end)
        factor_daily = _factor_to_daily(factor_min, self.dataset.n_minutes)

        label_daily = self.label_loader.get(date_start, date_end).to(self.device)
        factor_daily, label_daily = self._align_factor_label(factor_daily, label_daily)

        ic, rank_ic, ic_ir = _daily_rank_ic_and_ic(factor_daily, label_daily)

        group_ret, valid_idx = _group_returns(factor_daily, label_daily, self.n_groups)
        if group_ret.shape[0] == 0:
            empty = np.zeros((0, self.n_groups))
            return BacktestResult([], empty, empty, ic, rank_ic, ic_ir,
                                  np.zeros(self.n_groups), np.zeros(self.n_groups),
                                  np.zeros(self.n_groups))

        cum_ret = np.cumsum(group_ret, axis=0)
        all_dates = self.dataset.get_dates_in_range(date_start, date_end)
        if self.label_shift > 0:
            all_dates = all_dates[:-self.label_shift]
        dates = [all_dates[i] for i in valid_idx]

        # per-group annualised metrics
        ann_ret = group_ret.mean(axis=0) * TRADING_DAYS_PER_YEAR
        ann_vol = group_ret.std(axis=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe  = ann_ret / (ann_vol + 1e-9)

        return BacktestResult(
            dates=dates, group_nav=cum_ret, group_daily_ret=group_ret,
            ic=ic, rank_ic=rank_ic, ic_ir=ic_ir,
            ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
        )

    # ---------- plotting (matches example figure) ---------- #

    def plot_backtest(
        self,
        result: BacktestResult,
        title: str = 'dev_fac | label4',
        save_path: Optional[str] = None,
        dpi: int = 150,
        figsize: Tuple[int, int] = (14, 6),
    ):
        """
        Plot 20-group cumulative net-value curves.

        Legend format per group (matches the provided example image):
          group1, -23.51%, 52.62%, -2.88
          ↑ name   ↑ ann_ret  ↑ ann_vol  ↑ sharpe
        """
        if result.group_nav.shape[0] == 0:
            logger.warning("No valid data to plot.")
            return

        n_days, n_groups = result.group_nav.shape
        dates = result.dates

        # x-axis tick sampling
        tick_step = max(1, n_days // 20)
        tick_idx = list(range(0, n_days, tick_step))
        tick_labels = [dates[i] for i in tick_idx] if dates else [str(i) for i in tick_idx]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for g in range(n_groups):
            nav = result.group_nav[:, g]
            ar = result.ann_ret[g] * 100
            av = result.ann_vol[g] * 100
            sh = result.sharpe[g]
            lbl = f"group{g+1}, {ar:.2f}%, {av:.2f}%, {sh:.2f}"
            ax.plot(range(n_days), nav, label=lbl, linewidth=0.9)

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7, ha='right')
        ax.set_xlabel('Date')
        ax.set_ylabel('Net Value')
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2, loc='upper left',
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
        ax.grid(True, linestyle='--', alpha=0.3)

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight')
            logger.info("Figure saved → %s", save_path)
        else:
            plt.show()
        plt.close(fig)

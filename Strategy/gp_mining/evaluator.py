from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from Strategy.data_io.saver import save_wide_table
from Strategy.gp_mining.config import GPMineConfig
from Strategy.gp_mining.data import GPDataBundle
from Strategy.gp_mining.expression import Node, build_function_specs

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TORCH_OPERATOR_CANDIDATES = (
    _PROJECT_ROOT / "torch_operators_user",
    _PROJECT_ROOT.parent / "torch_operators_user",
)
for _candidate in _TORCH_OPERATOR_CANDIDATES:
    if (_candidate / "gp_functions.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from gp_functions import func_map_dict  # noqa: E402


@dataclass
class EvaluationResult:
    formula: str
    formula_hash: str
    fitness: float
    accepted: bool
    direction: int
    metrics: dict[str, float] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)
    case_scores: tuple[float, ...] = ()
    error: Optional[str] = None


def formula_hash(formula: str) -> str:
    return hashlib.md5(formula.encode("utf-8")).hexdigest()


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom <= 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def _mean_daily_corr(
    factor: np.ndarray,
    label: np.ndarray,
    row_mask: np.ndarray,
    min_stocks: int,
    rank: bool = False,
) -> float:
    vals = []
    for i in np.flatnonzero(row_mask):
        f = factor[i]
        y = label[i]
        valid = np.isfinite(f) & np.isfinite(y)
        if int(valid.sum()) < min_stocks:
            continue
        fv = f[valid]
        yv = y[valid]
        if rank:
            fv = _rank_values(fv)
            yv = _rank_values(yv)
        corr = _corr_1d(fv, yv)
        if np.isfinite(corr):
            vals.append(corr)
    return float(np.mean(vals)) if vals else np.nan


def _daily_corr_series(
    factor: np.ndarray,
    label: np.ndarray,
    row_mask: np.ndarray,
    min_stocks: int,
    rank: bool = False,
) -> np.ndarray:
    vals = np.full(factor.shape[0], np.nan, dtype=np.float64)
    for i in np.flatnonzero(row_mask):
        f = factor[i]
        y = label[i]
        valid = np.isfinite(f) & np.isfinite(y)
        if int(valid.sum()) < min_stocks:
            continue
        fv = f[valid]
        yv = y[valid]
        if rank:
            fv = _rank_values(fv)
            yv = _rank_values(yv)
        vals[i] = _corr_1d(fv, yv)
    return vals


def _top_tail_metrics(
    factor: np.ndarray,
    label: np.ndarray,
    row_mask: np.ndarray,
    top_ks: tuple[int, ...],
    tail_ks: tuple[int, ...],
    min_stocks: int,
) -> dict[str, float]:
    top_values = {k: [] for k in top_ks}
    tail_values = {k: [] for k in tail_ks}
    for i in np.flatnonzero(row_mask):
        f = factor[i]
        y = label[i]
        valid = np.isfinite(f) & np.isfinite(y)
        if int(valid.sum()) < min_stocks:
            continue
        fv = f[valid]
        yv = y[valid]
        order_desc = np.argsort(-fv, kind="mergesort")
        order_asc = np.argsort(fv, kind="mergesort")
        for k in top_ks:
            take = min(k, len(order_desc))
            top_values[k].append(float(np.nanmean(yv[order_desc[:take]])))
        for k in tail_ks:
            take = min(k, len(order_asc))
            tail_values[k].append(float(np.nanmean(yv[order_asc[:take]])))

    metrics: dict[str, float] = {}
    for k, values in top_values.items():
        metrics[f"top{k}_mean"] = float(np.nanmean(values)) if values else np.nan
    for k, values in tail_values.items():
        metrics[f"tail{k}_mean"] = float(np.nanmean(values)) if values else np.nan
    spreads = []
    for k in sorted(set(top_ks).intersection(tail_ks)):
        top = metrics.get(f"top{k}_mean", np.nan)
        tail = metrics.get(f"tail{k}_mean", np.nan)
        if np.isfinite(top) and np.isfinite(tail):
            spreads.append(top - tail)
            metrics[f"spread{k}_mean"] = top - tail
    metrics["spread_mean"] = float(np.mean(spreads)) if spreads else np.nan
    return metrics


def _daily_top_tail_spread(
    factor: np.ndarray,
    label: np.ndarray,
    row_mask: np.ndarray,
    top_k: int,
    tail_k: int,
    min_stocks: int,
) -> np.ndarray:
    vals = np.full(factor.shape[0], np.nan, dtype=np.float64)
    for i in np.flatnonzero(row_mask):
        f = factor[i]
        y = label[i]
        valid = np.isfinite(f) & np.isfinite(y)
        if int(valid.sum()) < min_stocks:
            continue
        fv = f[valid]
        yv = y[valid]
        order_desc = np.argsort(-fv, kind="mergesort")
        order_asc = np.argsort(fv, kind="mergesort")
        top_take = min(top_k, len(order_desc))
        tail_take = min(tail_k, len(order_asc))
        vals[i] = float(np.nanmean(yv[order_desc[:top_take]]) - np.nanmean(yv[order_asc[:tail_take]]))
    return vals


def _mean_abs_corr_to_many(
    factor: np.ndarray,
    accepted: list[np.ndarray],
    row_mask: np.ndarray,
    min_stocks: int,
) -> float:
    if not accepted:
        return 0.0
    max_corr = 0.0
    for other in accepted:
        corr = _mean_daily_corr(factor, other, row_mask, min_stocks=min_stocks, rank=False)
        if np.isfinite(corr):
            max_corr = max(max_corr, abs(corr))
    return float(max_corr)


class FactorEvaluator:
    def __init__(self, config: GPMineConfig, data: GPDataBundle) -> None:
        self.config = config
        self.data = data
        self.function_specs = build_function_specs(func_map_dict, exclude=config.function_exclude)
        self.accepted_arrays: list[np.ndarray] = []
        self.accepted_tensors: list[torch.Tensor] = []
        self.cache: dict[str, EvaluationResult] = {}

    def evaluate_tensor(self, node: Node) -> torch.Tensor:
        if node.kind == "terminal":
            return self.data.terminal_tensors[str(node.value)]
        if node.kind == "const":
            return node.value
        spec = self.function_specs[str(node.value)]
        args = [self.evaluate_tensor(child) for child in node.children]
        out = spec.func(*args)
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Expression returned {type(out).__name__}, expected torch.Tensor")
        if tuple(out.shape) != tuple(self.data.label_tensor.shape):
            raise ValueError(f"Expression shape {tuple(out.shape)} != label shape {tuple(self.data.label_tensor.shape)}")
        return torch.where(torch.isfinite(out), out, torch.nan)

    def evaluate(self, node: Node) -> EvaluationResult:
        formula = str(node)
        h = formula_hash(formula)
        if h in self.cache:
            return self.cache[h]
        try:
            tensor = self.evaluate_tensor(node)
            arr = tensor.detach().float().cpu().numpy()
            label = self.data.label_tensor.detach().float().cpu().numpy()
            eval_mask = self.data.eval_mask.detach().cpu().numpy().astype(bool)
            oos_mask = self.data.oos_mask.detach().cpu().numpy().astype(bool)

            mean_ic = _mean_daily_corr(arr, label, eval_mask, self.config.min_stocks, rank=False)
            mean_rank_ic = _mean_daily_corr(arr, label, eval_mask, self.config.min_stocks, rank=True)
            direction = -1 if np.isfinite(mean_ic) and np.isfinite(mean_rank_ic) and mean_ic < 0 and mean_rank_ic < 0 else 1
            directed_arr = arr * direction
            daily_ic = _daily_corr_series(arr, label, eval_mask, self.config.min_stocks, rank=False) * direction
            daily_rank_ic = _daily_corr_series(arr, label, eval_mask, self.config.min_stocks, rank=True) * direction
            spread_k = self.config.top_ks[0] if self.config.top_ks else 50
            tail_k = self.config.tail_ks[0] if self.config.tail_ks else spread_k
            daily_spread = _daily_top_tail_spread(
                directed_arr, label, eval_mask, spread_k, tail_k, self.config.min_stocks
            )
            valid_daily_ic = daily_ic[np.isfinite(daily_ic)]
            valid_daily_rank_ic = daily_rank_ic[np.isfinite(daily_rank_ic)]

            metrics = {
                "mean_ic": mean_ic,
                "mean_rank_ic": mean_rank_ic,
                "abs_mean_ic": abs(mean_ic) if np.isfinite(mean_ic) else np.nan,
                "abs_mean_rank_ic": abs(mean_rank_ic) if np.isfinite(mean_rank_ic) else np.nan,
                "directed_mean_ic": float(np.nanmean(daily_ic)) if len(valid_daily_ic) else np.nan,
                "directed_mean_rank_ic": float(np.nanmean(daily_rank_ic)) if len(valid_daily_rank_ic) else np.nan,
                "ic_std": float(np.nanstd(valid_daily_ic)) if len(valid_daily_ic) else np.nan,
                "rank_ic_std": float(np.nanstd(valid_daily_rank_ic)) if len(valid_daily_rank_ic) else np.nan,
                "ic_ir": (
                    float(np.nanmean(valid_daily_ic) / np.nanstd(valid_daily_ic))
                    if len(valid_daily_ic) and np.nanstd(valid_daily_ic) > 0
                    else np.nan
                ),
                "rank_ic_ir": (
                    float(np.nanmean(valid_daily_rank_ic) / np.nanstd(valid_daily_rank_ic))
                    if len(valid_daily_rank_ic) and np.nanstd(valid_daily_rank_ic) > 0
                    else np.nan
                ),
                "ic_win_rate": float(np.mean(valid_daily_ic > 0)) if len(valid_daily_ic) else np.nan,
                "valid_eval_days": float(len(valid_daily_ic)),
                "oos_mean_ic": _mean_daily_corr(arr, label, oos_mask, self.config.min_stocks, rank=False),
                "oos_mean_rank_ic": _mean_daily_corr(arr, label, oos_mask, self.config.min_stocks, rank=True),
                "length": float(node.length),
                "depth": float(node.depth),
            }
            metrics.update(_top_tail_metrics(
                directed_arr,
                label,
                eval_mask,
                self.config.top_ks,
                self.config.tail_ks,
                self.config.min_stocks,
            ))
            metrics["max_corr_to_accepted"] = _mean_abs_corr_to_many(
                directed_arr,
                self.accepted_arrays,
                eval_mask,
                self.config.min_stocks,
            )

            spread = metrics.get("spread_mean", 0.0)
            spread_part = self.config.spread_weight * abs(spread) if np.isfinite(spread) else 0.0
            corr_penalty = self.config.corr_penalty * metrics["max_corr_to_accepted"]
            length_penalty = self.config.length_penalty * max(node.length - self.config.min_length, 0)
            fitness = (
                (metrics["abs_mean_ic"] if np.isfinite(metrics["abs_mean_ic"]) else -1.0)
                + (metrics["abs_mean_rank_ic"] if np.isfinite(metrics["abs_mean_rank_ic"]) else -1.0)
                + spread_part
                + self.config.ir_weight * (metrics["ic_ir"] if np.isfinite(metrics["ic_ir"]) else 0.0)
                + self.config.win_rate_weight * (metrics["ic_win_rate"] if np.isfinite(metrics["ic_win_rate"]) else 0.0)
                - corr_penalty
                - length_penalty
            )

            same_sign = np.isfinite(mean_ic) and np.isfinite(mean_rank_ic) and mean_ic * mean_rank_ic > 0
            accepted = (
                same_sign
                and abs(mean_ic) >= self.config.ic_threshold
                and abs(mean_rank_ic) >= self.config.rank_ic_threshold
                and metrics["max_corr_to_accepted"] < self.config.max_corr_threshold
            )
            objectives = {
                "ic": metrics["abs_mean_ic"] if np.isfinite(metrics["abs_mean_ic"]) else -1.0,
                "rank_ic": metrics["abs_mean_rank_ic"] if np.isfinite(metrics["abs_mean_rank_ic"]) else -1.0,
                "spread": metrics["spread_mean"] if np.isfinite(metrics["spread_mean"]) else -1.0,
                "stability": metrics["ic_ir"] if np.isfinite(metrics["ic_ir"]) else -1.0,
                "win_rate": metrics["ic_win_rate"] if np.isfinite(metrics["ic_win_rate"]) else 0.0,
                "decorrelation": -metrics["max_corr_to_accepted"],
                "parsimony": -float(node.length),
            }
            case_matrix = np.vstack([
                np.nan_to_num(daily_ic, nan=-1.0),
                np.nan_to_num(daily_rank_ic, nan=-1.0),
                np.nan_to_num(daily_spread, nan=-1.0),
            ])
            case_scores = tuple(np.nanmean(case_matrix, axis=0).tolist())

            result = EvaluationResult(
                formula=formula,
                formula_hash=h,
                fitness=float(fitness),
                accepted=bool(accepted),
                direction=direction,
                metrics=metrics,
                objectives=objectives,
                case_scores=case_scores,
            )
        except Exception as exc:
            result = EvaluationResult(
                formula=formula,
                formula_hash=h,
                fitness=float("-inf"),
                accepted=False,
                direction=1,
                objectives={
                    "ic": -1.0,
                    "rank_ic": -1.0,
                    "spread": -1.0,
                    "stability": -1.0,
                    "win_rate": 0.0,
                    "decorrelation": -1.0,
                    "parsimony": -float(node.length),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        self.cache[h] = result
        return result

    def remember_accepted(self, node: Node, result: EvaluationResult) -> None:
        tensor = self.evaluate_tensor(node) * result.direction
        arr = tensor.detach().float().cpu().numpy()
        self.accepted_arrays.append(arr)
        self.accepted_tensors.append(tensor.detach())

    def tensor_to_wide_df(self, node: Node, direction: int = 1) -> pd.DataFrame:
        tensor = self.evaluate_tensor(node) * direction
        arr = tensor.detach().float().cpu().numpy()
        return pd.DataFrame(arr, index=self.data.dates, columns=self.data.stocks)

    def save_factor(self, node: Node, result: EvaluationResult) -> Path:
        df = self.tensor_to_wide_df(node, direction=result.direction)
        name = f"gp_{self.config.label_tag}_{result.formula_hash[:12]}.fea"
        return save_wide_table(df, self.config.gp_factor_dir / name)


def result_to_row(result: EvaluationResult, gen: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gen": gen,
        "formula": result.formula,
        "formula_hash": result.formula_hash,
        "fitness": result.fitness,
        "accepted": result.accepted,
        "direction": result.direction,
        "error": result.error,
    }
    row.update(result.metrics)
    for name, value in result.objectives.items():
        row[f"obj_{name}"] = value
    return row

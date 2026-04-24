"""
Operator registry for RL factor mining.

Categorizes operators into:
  - Binary ops: combine two tensors (temp, feature)
  - Unary transforms: applied to a single feature before combining
  - TS transforms: time-series operators with window parameter
  - CS transforms: cross-sectional operators
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Callable, Tuple, Any

EPS = 1e-9


# ======================== Core Operators ======================== #

def op_add(x, y):
    return torch.add(x, y)

def op_sub(x, y):
    return torch.sub(x, y)

def op_mul(x, y):
    return torch.mul(x, y)

def op_div(x, y):
    return torch.where(torch.abs(y) > EPS, x / y, torch.full_like(x, torch.nan))

def op_max(x, y):
    return torch.maximum(x, y)

def op_min(x, y):
    return torch.minimum(x, y)


# ======================== Unary Operators ======================== #

def op_abs(x):
    return torch.abs(x)

def op_neg(x):
    return -x

def op_sqrt(x):
    return torch.sqrt(torch.abs(x))

def op_log(x):
    return torch.where(torch.abs(x) > EPS, torch.log(torch.abs(x)),
                       torch.full_like(x, torch.nan))

def op_sign(x):
    return torch.sign(x)

def op_inv(x):
    return torch.where(torch.abs(x) > EPS, 1.0 / x,
                       torch.full_like(x, torch.nan))

def op_sigmoid(x):
    return torch.sigmoid(x)

def op_identity(x):
    return x


# ======================== TS Operators ======================== #

def _unfold(x: torch.Tensor, d: int):
    """Create sliding window view along dim 0. Output: (T, n_stocks, d)"""
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return x.unfold(0, d, 1)


def ts_mean(x, d):
    w = _unfold(x, d)
    return torch.nanmean(w, dim=-1)


def ts_sum(x, d):
    w = _unfold(x, d)
    s = w.clone()
    s[torch.isnan(s)] = 0
    return s.sum(dim=-1)


def ts_stddev(x, d):
    w = _unfold(x, d)
    return torch.sqrt(torch.nanmean((w - torch.nanmean(w, dim=-1, keepdim=True)) ** 2, dim=-1) + EPS)


def ts_rank(x, d, pct=True):
    w = _unfold(x, d)
    last_val = w[..., -1:]
    ranks = (w < last_val).float().sum(dim=-1)
    valid_count = (~torch.isnan(w)).float().sum(dim=-1)
    if pct:
        ranks = ranks / valid_count.clamp(min=1)
    return ranks


def ts_delay(x, d):
    result = torch.full_like(x, torch.nan)
    if d < x.shape[0]:
        result[d:] = x[:-d]
    return result


def ts_delta(x, d):
    result = torch.full_like(x, torch.nan)
    if d < x.shape[0]:
        result[d:] = x[d:] - x[:-d]
    return result


def ts_min(x, d):
    w = _unfold(x, d)
    w_filled = w.clone()
    w_filled[torch.isnan(w_filled)] = float('inf')
    return w_filled.min(dim=-1).values


def ts_max(x, d):
    w = _unfold(x, d)
    w_filled = w.clone()
    w_filled[torch.isnan(w_filled)] = float('-inf')
    return w_filled.max(dim=-1).values


def ts_skew(x, d):
    w = _unfold(x, d)
    mu = torch.nanmean(w, dim=-1, keepdim=True)
    diff = w - mu
    n = (~torch.isnan(w)).float().sum(dim=-1, keepdim=True).clamp(min=3)
    m3 = torch.nanmean(diff ** 3, dim=-1)
    m2 = torch.nanmean(diff ** 2, dim=-1).clamp(min=EPS)
    return m3 / (m2 ** 1.5 + EPS)


def ts_kurt(x, d):
    w = _unfold(x, d)
    mu = torch.nanmean(w, dim=-1, keepdim=True)
    diff = w - mu
    m4 = torch.nanmean(diff ** 4, dim=-1)
    m2 = torch.nanmean(diff ** 2, dim=-1).clamp(min=EPS)
    return m4 / (m2 ** 2 + EPS) - 3.0


def ts_decay_linear(x, d):
    w = _unfold(x, d)
    weights = torch.arange(1, d + 1, dtype=x.dtype, device=x.device).float()
    weights = weights / weights.sum()
    w_filled = w.clone()
    w_filled[torch.isnan(w_filled)] = 0
    return (w_filled * weights).sum(dim=-1)


def ts_zscore(x, d):
    mu = ts_mean(x, d)
    sigma = ts_stddev(x, d)
    T = mu.shape[0]
    x_aligned = x[-T:]
    return (x_aligned - mu) / sigma.clamp(min=EPS)


def ts_corr(x, y, d):
    wx = _unfold(x, d)
    wy = _unfold(y, d)
    mx = torch.nanmean(wx, dim=-1, keepdim=True)
    my = torch.nanmean(wy, dim=-1, keepdim=True)
    dx = wx - mx
    dy = wy - my
    dx[torch.isnan(dx)] = 0
    dy[torch.isnan(dy)] = 0
    cov = (dx * dy).sum(dim=-1)
    sx = torch.sqrt((dx ** 2).sum(dim=-1).clamp(min=EPS))
    sy = torch.sqrt((dy ** 2).sum(dim=-1).clamp(min=EPS))
    return cov / (sx * sy + EPS)


# ======================== CS Operators ======================== #

def cs_rank(x, pct=True):
    """Cross-sectional rank along dim 1 (stocks)."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    mask = torch.isnan(x)
    x_filled = x.clone()
    x_filled[mask] = float('-inf')
    ranks = x_filled.argsort(dim=-1).argsort(dim=-1).float()
    valid_count = (~mask).float().sum(dim=-1, keepdim=True).clamp(min=1)
    if pct:
        ranks = ranks / valid_count
    ranks[mask] = float('nan')
    return ranks


def cs_demean(x):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x - torch.nanmean(x, dim=-1, keepdim=True)


def cs_normalize(x):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    mu = torch.nanmean(x, dim=-1, keepdim=True)
    std = torch.std(x, dim=-1, keepdim=True).clamp(min=EPS)
    return (x - mu) / std


# ======================== Operator Registry ======================== #

BINARY_OPS = [
    ('add', op_add),
    ('sub', op_sub),
    ('mul', op_mul),
    ('div', op_div),
    ('max', op_max),
    ('min', op_min),
]

UNARY_TRANSFORMS = [
    ('identity', op_identity),
    ('abs', op_abs),
    ('neg', op_neg),
    ('log', op_log),
    ('sign', op_sign),
    ('inv', op_inv),
    ('sqrt', op_sqrt),
    ('sigmoid', op_sigmoid),
    ('cs_rank', cs_rank),
    ('cs_demean', cs_demean),
    ('cs_normalize', cs_normalize),
]

TS_WINDOWS = [5, 10, 20, 40, 60]

TS_TRANSFORM_FUNCS = [
    ('ts_mean', ts_mean),
    ('ts_sum', ts_sum),
    ('ts_stddev', ts_stddev),
    ('ts_rank', ts_rank),
    ('ts_delay', ts_delay),
    ('ts_delta', ts_delta),
    ('ts_min', ts_min),
    ('ts_max', ts_max),
    ('ts_skew', ts_skew),
    ('ts_kurt', ts_kurt),
    ('ts_decay_linear', ts_decay_linear),
    ('ts_zscore', ts_zscore),
]


def _build_transform_list():
    """Build the complete list of transforms for the RL action space."""
    transforms = []
    for name, func in UNARY_TRANSFORMS:
        transforms.append((name, func, None))

    for ts_name, ts_func in TS_TRANSFORM_FUNCS:
        for w in TS_WINDOWS:
            transforms.append((f"{ts_name}_{w}", ts_func, w))

    return transforms


ALL_TRANSFORMS = _build_transform_list()


class OperatorRegistry:
    """Central registry providing operator metadata for the RL agent."""

    def __init__(self):
        self.binary_ops = BINARY_OPS
        self.transforms = ALL_TRANSFORMS

        self.n_binary_ops = len(self.binary_ops)
        self.n_transforms = len(self.transforms)

        self.binary_op_names = [name for name, _ in self.binary_ops]
        self.transform_names = [name for name, _, _ in self.transforms]

    def apply_binary_op(self, op_idx: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, func = self.binary_ops[op_idx]
        return func(x, y)

    def apply_transform(self, transform_idx: int, x: torch.Tensor) -> torch.Tensor:
        _, func, window = self.transforms[transform_idx]
        if window is not None:
            result = func(x, window)
            pad = x.shape[0] - result.shape[0]
            if pad > 0:
                padding = torch.full((pad, *result.shape[1:]),
                                     torch.nan, dtype=x.dtype, device=x.device)
                result = torch.cat([padding, result], dim=0)
            return result
        return func(x)

    def summary(self) -> str:
        lines = [f"OperatorRegistry: {self.n_binary_ops} binary ops, {self.n_transforms} transforms"]
        lines.append(f"  Binary ops: {self.binary_op_names}")
        lines.append(f"  Transforms ({self.n_transforms}): {self.transform_names[:5]}...")
        return '\n'.join(lines)

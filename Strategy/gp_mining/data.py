from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from Strategy.gp_mining.config import GPMineConfig
from Strategy.utils.helpers import ensure_tradedate_as_index


@dataclass
class GPDataBundle:
    dates: pd.DatetimeIndex
    stocks: pd.Index
    label_df: pd.DataFrame
    factor_dfs: dict[str, pd.DataFrame]
    terminal_tensors: dict[str, torch.Tensor]
    label_tensor: torch.Tensor
    eval_mask: torch.Tensor
    oos_mask: torch.Tensor
    device: torch.device
    dtype: torch.dtype


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def list_factor_names(factor_dir: Path) -> list[str]:
    return sorted(path.stem for path in Path(factor_dir).glob("*.fea"))


def load_wide_fea(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Wide table not found: {path}")
    return ensure_tradedate_as_index(pd.read_feather(path)).sort_index()


def load_gp_data(config: GPMineConfig, terminal_names: Optional[list[str]] = None) -> GPDataBundle:
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    label_df = load_wide_fea(config.label_path)
    names = terminal_names or config.terminal_names or list_factor_names(config.factor_dir)
    if not names:
        raise ValueError(f"No factor .fea files found in {config.factor_dir}")

    factor_dfs = {
        name: load_wide_fea(config.factor_dir / f"{name}.fea")
        for name in names
    }

    common_dates = label_df.index
    common_stocks = label_df.columns
    for fdf in factor_dfs.values():
        common_dates = common_dates.intersection(fdf.index)
        common_stocks = common_stocks.intersection(fdf.columns)
    common_dates = pd.DatetimeIndex(common_dates).sort_values()
    common_stocks = pd.Index(common_stocks).sort_values()
    if len(common_dates) == 0 or len(common_stocks) == 0:
        raise ValueError(
            f"No common dates/stocks after alignment: dates={len(common_dates)} stocks={len(common_stocks)}"
        )

    label_df = label_df.loc[common_dates, common_stocks]
    factor_dfs = {name: df.loc[common_dates, common_stocks] for name, df in factor_dfs.items()}

    terminal_tensors = {
        name: torch.as_tensor(df.to_numpy(dtype="float32", copy=True), device=device, dtype=dtype)
        for name, df in factor_dfs.items()
    }
    label_tensor = torch.as_tensor(label_df.to_numpy(dtype="float32", copy=True), device=device, dtype=dtype)

    dates = pd.DatetimeIndex(common_dates).normalize()
    eval_start = pd.Timestamp(config.eval_start).normalize()
    eval_end = pd.Timestamp(config.eval_end).normalize()
    oos_start = pd.Timestamp(config.oos_start).normalize()
    eval_mask = torch.as_tensor((dates >= eval_start) & (dates <= eval_end), device=device)
    oos_mask = torch.as_tensor(dates >= oos_start, device=device)

    return GPDataBundle(
        dates=common_dates,
        stocks=common_stocks,
        label_df=label_df,
        factor_dfs=factor_dfs,
        terminal_tensors=terminal_tensors,
        label_tensor=label_tensor,
        eval_mask=eval_mask,
        oos_mask=oos_mask,
        device=device,
        dtype=dtype,
    )

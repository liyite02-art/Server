"""
Label loader for separate label feather files.

Label file format (e.g. label2.fea):
  - Row index or first column = date string like "20210104" (or int 20210104)
  - Other columns = stock codes like "000001", "600519", …
  - Cell value = next-period return (or any label)

This loader aligns the label with MMapDataset's date/stock ordering,
producing (n_days, n_stocks) tensors that match the factor output exactly.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from bisect import bisect_left, bisect_right

logger = logging.getLogger(__name__)


class LabelLoader:
    """
    Load and serve daily label data from a standalone feather file.

    Usage:
        ll = LabelLoader("path/to/label2.fea", mmap_dates, mmap_stocks, device="cuda")
        label = ll.get("20210104", "20210331")  # → (n_days, n_stocks) tensor
    """

    def __init__(
        self,
        label_path: str,
        mmap_dates: List[str],
        mmap_stocks: List[str],
        device: str = 'cpu',
    ):
        self.device = device
        self.mmap_dates = mmap_dates
        self.mmap_stocks = mmap_stocks

        df = pd.read_feather(label_path)

        # --- identify date column ---
        date_col = self._find_date_column(df)
        if date_col is not None:
            df[date_col] = df[date_col].astype(str).str.strip()
            df = df.set_index(date_col)
        else:
            df.index = df.index.astype(str).str.strip()

        df.columns = df.columns.astype(str).str.strip()

        self._label_dates = sorted(df.index.tolist())
        self._label_stocks = list(df.columns)

        # Build full aligned matrix: (all_label_dates, n_mmap_stocks), float32
        stock_to_col: Dict[str, int] = {s: i for i, s in enumerate(self._label_stocks)}
        self._date_to_row: Dict[str, int] = {d: i for i, d in enumerate(self._label_dates)}

        n_label_dates = len(self._label_dates)
        n_mmap_stocks = len(mmap_stocks)

        mat = np.full((n_label_dates, n_mmap_stocks), np.nan, dtype=np.float32)
        raw = df.values.astype(np.float32)

        for ms_idx, ms_code in enumerate(mmap_stocks):
            if ms_code in stock_to_col:
                src_col = stock_to_col[ms_code]
                mat[:, ms_idx] = raw[:, src_col]

        self._full_matrix = mat  # (n_label_dates, n_mmap_stocks)

        n_overlap = len(set(self._label_dates) & set(mmap_dates))
        logger.info(
            "LabelLoader: %s | %d label dates, %d label stocks | "
            "%d dates overlap with mmap, %d/%d stocks matched",
            Path(label_path).name,
            n_label_dates, len(self._label_stocks),
            n_overlap,
            sum(1 for s in mmap_stocks if s in stock_to_col), n_mmap_stocks,
        )

    @staticmethod
    def _find_date_column(df: pd.DataFrame) -> Optional[str]:
        """Heuristic: find the column that looks like dates."""
        for col in df.columns:
            sample = df[col].dropna().head(5)
            if sample.empty:
                continue
            as_str = sample.astype(str)
            if all(len(s) == 8 and s.isdigit() for s in as_str):
                return col
        return None

    def get(self, date_start: str, date_end: str) -> torch.Tensor:
        """
        Return label tensor of shape (n_days, n_mmap_stocks) for dates
        that exist in BOTH the label file and the mmap dataset's date range.

        The returned tensor is aligned with mmap_dates ordering so that
        row i corresponds to mmap_dates[idx_start + i].
        """
        # mmap dates in the requested range
        idx_s = bisect_left(self.mmap_dates, date_start)
        idx_e = bisect_right(self.mmap_dates, date_end)
        target_dates = self.mmap_dates[idx_s:idx_e]

        n_days = len(target_dates)
        n_stocks = len(self.mmap_stocks)
        out = np.full((n_days, n_stocks), np.nan, dtype=np.float32)

        for i, d in enumerate(target_dates):
            if d in self._date_to_row:
                out[i] = self._full_matrix[self._date_to_row[d]]

        return torch.from_numpy(out).to(self.device)

    @property
    def available_dates(self) -> List[str]:
        return self._label_dates

"""
Efficient memory-mapped data loader for RL factor mining.

Provides fast random access to field data organized as:
  {field_name}.mmap -> shape (n_dates * n_minutes, n_stocks), float32
"""
import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from bisect import bisect_left, bisect_right

logger = logging.getLogger(__name__)


class MMapDataset:
    """
    Loads memory-mapped field data and provides efficient slicing.

    Usage:
        ds = MMapDataset("path/to/mmap_dir")
        tensor = ds.load_field("close", date_start="20210104", date_end="20210331")
    """

    def __init__(self, mmap_dir: str, device: str = 'cpu'):
        self.mmap_dir = mmap_dir
        self.device = device

        meta_path = os.path.join(mmap_dir, 'meta.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        self.dates: List[str] = self.meta['dates']
        self.stocks: List[str] = self.meta['stocks']
        self.fields: List[str] = self.meta['fields']
        self.n_minutes: int = self.meta['n_minutes']
        self.n_dates: int = self.meta['n_dates']
        self.n_stocks: int = self.meta['n_stocks']
        self.total_rows: int = self.n_dates * self.n_minutes

        self._mmap_cache: Dict[str, np.memmap] = {}

        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.stock_to_idx = {s: i for i, s in enumerate(self.stocks)}

        logger.info(f"MMapDataset: {self.n_dates} dates, {self.n_stocks} stocks, "
                     f"{len(self.fields)} fields, {self.n_minutes} min/day")

    def _get_mmap(self, field_name: str) -> np.memmap:
        if field_name not in self._mmap_cache:
            path = os.path.join(self.mmap_dir, f"{field_name}.mmap")
            if not os.path.exists(path):
                raise FileNotFoundError(f"MMap file not found: {path}")
            self._mmap_cache[field_name] = np.memmap(
                path, dtype=np.float32, mode='r',
                shape=(self.total_rows, self.n_stocks)
            )
        return self._mmap_cache[field_name]

    def _date_range_to_row_slice(self, date_start: str, date_end: str) -> Tuple[int, int]:
        idx_start = bisect_left(self.dates, date_start)
        idx_end = bisect_right(self.dates, date_end)
        row_start = idx_start * self.n_minutes
        row_end = idx_end * self.n_minutes
        return row_start, row_end

    def get_dates_in_range(self, date_start: str, date_end: str) -> List[str]:
        idx_start = bisect_left(self.dates, date_start)
        idx_end = bisect_right(self.dates, date_end)
        return self.dates[idx_start:idx_end]

    def load_field(self, field_name: str,
                   date_start: Optional[str] = None,
                   date_end: Optional[str] = None,
                   to_torch: bool = True) -> np.ndarray:
        """
        Load a field's data, optionally sliced by date range.

        Returns:
            Array of shape (n_selected_days * n_minutes, n_stocks)
        """
        mm = self._get_mmap(field_name)

        if date_start and date_end:
            r_start, r_end = self._date_range_to_row_slice(date_start, date_end)
            data = np.array(mm[r_start:r_end])
        else:
            data = np.array(mm[:])

        if to_torch:
            return torch.from_numpy(data).to(self.device)
        return data

    def load_fields(self, field_names: List[str],
                    date_start: Optional[str] = None,
                    date_end: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Load multiple fields into a dict of tensors."""
        return {f: self.load_field(f, date_start, date_end) for f in field_names}

    def load_daily_snapshot(self, field_name: str,
                            date_start: Optional[str] = None,
                            date_end: Optional[str] = None,
                            minute_idx: int = -1) -> torch.Tensor:
        """
        Load daily snapshot at a specific minute.

        Args:
            minute_idx: Which minute to take (-1 = last minute of day).

        Returns:
            Tensor of shape (n_days, n_stocks)
        """
        full = self.load_field(field_name, date_start, date_end, to_torch=False)
        n_rows = full.shape[0]
        n_days = n_rows // self.n_minutes

        reshaped = full.reshape(n_days, self.n_minutes, self.n_stocks)
        snapshot = reshaped[:, minute_idx, :]
        return torch.from_numpy(snapshot.copy()).to(self.device)

    def sample_date_windows(self, date_start: str, date_end: str,
                            window_days: int, n_samples: int = 1) -> List[Tuple[str, str]]:
        """Sample random date windows for RL training."""
        dates_in_range = self.get_dates_in_range(date_start, date_end)
        n_available = len(dates_in_range)
        if n_available < window_days:
            return [(dates_in_range[0], dates_in_range[-1])]

        windows = []
        for _ in range(n_samples):
            start_idx = np.random.randint(0, n_available - window_days)
            windows.append((dates_in_range[start_idx],
                            dates_in_range[start_idx + window_days - 1]))
        return windows

    def close(self):
        self._mmap_cache.clear()

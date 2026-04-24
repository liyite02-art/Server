"""
Convert .fea (feather) files to field-oriented .mmap (memory-mapped) files.

Layout:
  - Each field gets one mmap file: {field_name}.mmap
  - Shape: (n_dates * n_minutes, n_stocks) in float32, row-major
  - meta.json stores dates, stocks, fields, n_minutes
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


def scan_fea_directory(fea_dir: str):
    """Scan directory for .fea files and return sorted date list."""
    fea_files = sorted(Path(fea_dir).glob("*.fea"))
    dates = [f.stem for f in fea_files]
    logger.info(f"Found {len(dates)} fea files in {fea_dir}")
    return dates, fea_files


def build_universe(fea_files: list, sample_ratio: float = 1.0) -> tuple:
    """Build complete stock universe and feature list from fea files."""
    all_stocks = set()
    fields = None

    files_to_scan = fea_files
    if sample_ratio < 1.0:
        step = max(1, int(1.0 / sample_ratio))
        files_to_scan = fea_files[::step]

    for fea_path in tqdm(files_to_scan, desc="Scanning universe"):
        df = pd.read_feather(fea_path)
        all_stocks.update(df['code'].unique())
        if fields is None:
            fields = [c for c in df.columns if c not in ('code', 'second')]

    stocks = sorted(all_stocks)
    logger.info(f"Universe: {len(stocks)} stocks, {len(fields)} fields")
    return stocks, fields


def convert_fea_to_mmap(
    fea_dir: str,
    mmap_dir: str,
    n_minutes: int = 237,
    exclude_fields: Optional[List[str]] = None,
    target_fields: Optional[List[str]] = None,
):
    """
    Main conversion: read all .fea files and write per-field .mmap files.

    Args:
        fea_dir: Directory containing {date}.fea files.
        mmap_dir: Output directory for .mmap files.
        n_minutes: Number of intraday minutes (237 for A-shares).
        exclude_fields: Fields to skip (besides 'code' and 'second').
        target_fields: If set, only convert these fields.
    """
    os.makedirs(mmap_dir, exist_ok=True)
    exclude_fields = set(exclude_fields or [])
    exclude_fields.update({'code', 'second'})

    dates, fea_files = scan_fea_directory(fea_dir)
    if not dates:
        raise ValueError(f"No .fea files found in {fea_dir}")

    stocks, all_fields = build_universe(fea_files)

    if target_fields:
        fields = [f for f in all_fields if f in set(target_fields)]
    else:
        fields = [f for f in all_fields if f not in exclude_fields]

    n_dates = len(dates)
    n_stocks = len(stocks)
    total_rows = n_dates * n_minutes

    stock_to_idx = {s: i for i, s in enumerate(stocks)}

    logger.info(f"Creating mmap files: {n_dates} dates x {n_stocks} stocks x {n_minutes} minutes")
    logger.info(f"Fields to convert: {len(fields)}")

    mmap_handles = {}
    for field_name in fields:
        mmap_path = os.path.join(mmap_dir, f"{field_name}.mmap")
        mm = np.memmap(mmap_path, dtype=np.float32, mode='w+',
                       shape=(total_rows, n_stocks))
        mm[:] = np.nan
        mmap_handles[field_name] = mm

    for date_idx, (date_str, fea_path) in enumerate(
            tqdm(zip(dates, fea_files), total=n_dates, desc="Converting")):
        try:
            df = pd.read_feather(fea_path)
        except Exception as e:
            logger.warning(f"Failed to read {fea_path}: {e}")
            continue

        row_offset = date_idx * n_minutes

        grouped = df.groupby('code')
        for stock_code, group_df in grouped:
            if stock_code not in stock_to_idx:
                continue
            col_idx = stock_to_idx[stock_code]

            n_rows = min(len(group_df), n_minutes)

            for field_name in fields:
                if field_name not in group_df.columns:
                    continue
                values = group_df[field_name].values[:n_rows].astype(np.float32)
                mmap_handles[field_name][row_offset:row_offset + n_rows, col_idx] = values

    for mm in mmap_handles.values():
        mm.flush()

    meta = {
        'dates': dates,
        'stocks': stocks,
        'fields': fields,
        'n_minutes': n_minutes,
        'n_dates': n_dates,
        'n_stocks': n_stocks,
        'shape_per_field': [total_rows, n_stocks],
        'dtype': 'float32',
    }
    meta_path = os.path.join(mmap_dir, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Conversion complete. {len(fields)} mmap files saved to {mmap_dir}")
    return meta


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description="Convert .fea files to .mmap")
    parser.add_argument('--fea_dir', type=str, required=True)
    parser.add_argument('--mmap_dir', type=str, required=True)
    parser.add_argument('--n_minutes', type=int, default=237)
    args = parser.parse_args()

    convert_fea_to_mmap(args.fea_dir, args.mmap_dir, args.n_minutes)

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from prog_trade_reg.config import (
    EXPOSURE_PRE_END,
    EXPOSURE_PRE_START,
    POLICY_DATE_MAIN,
    TRADE_DAYS_PKL,
)
from prog_trade_reg.feather_io import read_feather_to_polars, UnreadableRawFeatherError
from prog_trade_reg.paths import build_manifest_path, exposure_pre_policy_path
from prog_trade_reg.raw_paths import trans_fea_path
from prog_trade_reg.trade_calendar import iter_dates_in_range


def _append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_daily_trans_exposure_features(trade_date: str) -> pl.DataFrame | None:
    """
    **One read** of ``trans_fea`` per calendar day; **one** ``group_by`` pass; multiple daily metrics.

    Currently computed (all from the same filtered trade table):

    - ``oib_day``: order-flow imbalance (buy−sell)/(buy+sell) for rows with ``bsFlag`` in ``B``/``S``.
    - ``n_trades_day``: count of such rows (trade intensity).
    - ``total_vol_day``: buy volume + sell volume (activity scale).

    Add more expressions in the same ``agg`` block to extend without re-reading the feather.

    Returns ``None`` if the file is missing.
    """
    path = trans_fea_path(trade_date)
    if not path.is_file():
        return None

    try:
        t = read_feather_to_polars(path)
    except UnreadableRawFeatherError:
        return None
    c = pl.col("code").cast(pl.Utf8)
    ex = (
        pl.when(c.str.ends_with(".SZ"))
        .then(pl.lit("SZ"))
        .when(c.str.ends_with(".SH"))
        .then(pl.lit("SH"))
        .otherwise(pl.lit(None))
    )
    t = t.with_columns(
        c.str.split(".").list.get(0).cast(pl.Int32, strict=False).alias("code"),
        ex.alias("exchange"),
        pl.col("tradeVolume").cast(pl.Float64).alias("vol"),
        pl.col("bsFlag").cast(pl.Utf8).str.strip_chars().alias("bs"),
    ).filter(pl.col("exchange").is_not_null())
    t = t.filter(pl.col("bs").is_in(["B", "S"]))
    t = t.with_columns(
        pl.when(pl.col("bs") == "B").then(pl.col("vol")).otherwise(0.0).alias("buy_vol"),
        pl.when(pl.col("bs") == "S").then(pl.col("vol")).otherwise(0.0).alias("sell_vol"),
    )
    g = t.group_by(["exchange", "code"]).agg(
        pl.col("buy_vol").sum(),
        pl.col("sell_vol").sum(),
        pl.len().alias("n_trades_day"),
    )
    g = g.filter((pl.col("buy_vol") + pl.col("sell_vol")) > 0).with_columns(
        (
            (pl.col("buy_vol") - pl.col("sell_vol"))
            / (pl.col("buy_vol") + pl.col("sell_vol"))
        ).alias("oib_day"),
        (pl.col("buy_vol") + pl.col("sell_vol")).alias("total_vol_day"),
        pl.lit(trade_date).alias("trade_date"),
    )
    return g.select(
        [
            "exchange",
            "code",
            "trade_date",
            "oib_day",
            "n_trades_day",
            "total_vol_day",
            "buy_vol",
            "sell_vol",
        ]
    )


def compute_daily_oib_from_trans(trade_date: str) -> pl.DataFrame | None:
    """Backward-compatible: same rows as before (OIB + buy/sell volume sums)."""
    df = compute_daily_trans_exposure_features(trade_date)
    if df is None:
        return None
    return df.select(["exchange", "code", "trade_date", "oib_day", "buy_vol", "sell_vol"])


def _winsor_z(
    df: pl.DataFrame,
    raw_mean_col: str,
    tag: str,
    lo: float,
    hi: float,
) -> pl.DataFrame:
    """Cross-sectional winsor on ``raw_mean_col``, then z-score → ``{tag}_mean``, ``E_{tag}_z``."""
    qlo = df.select(pl.col(raw_mean_col).quantile(lo)).item()
    qhi = df.select(pl.col(raw_mean_col).quantile(hi)).item()
    out = df.with_columns(
        pl.col(raw_mean_col).clip(lower_bound=qlo, upper_bound=qhi).alias(f"{tag}_mean")
    )
    mu = out.select(pl.col(f"{tag}_mean").mean()).item()
    sd = out.select(pl.col(f"{tag}_mean").std()).item()
    if sd is None or sd == 0:
        raise RuntimeError(f"Cannot z-score {tag}: zero std (column {raw_mean_col})")
    return out.with_columns(
        ((pl.col(f"{tag}_mean") - mu) / sd).alias(f"E_{tag}_z")
    )


def build_exposure_pre_policy(
    start: str | None = None,
    end: str | None = None,
    *,
    winsor_pct: tuple[float, float] = (0.01, 0.99),
    trade_days_pkl: Path | None = None,
) -> Path:
    """
    Time-invariant exposures: for each metric, **time-mean** over ``[start, end]``, then winsor + z.

    Reads each day's ``trans_fea`` **once**; in that pass computes OIB, trade count, and total volume
    (see :func:`compute_daily_trans_exposure_features`). Main regression column remains ``E_oib_z``;
    ``E_n_trades_z`` and ``E_total_vol_z`` are additional controls / robustness.

    Writes ``meta/exposure_pre_policy.parquet``.
    """
    start = start or EXPOSURE_PRE_START
    end = end or EXPOSURE_PRE_END
    if end >= POLICY_DATE_MAIN:
        raise ValueError(
            f"exposure end {end} must be strictly before policy date {POLICY_DATE_MAIN}"
        )

    pkl = trade_days_pkl or TRADE_DAYS_PKL
    days, _cal_mode = iter_dates_in_range(start, end, trade_days_pkl=pkl)
    parts: list[pl.DataFrame] = []
    missing = 0
    pbar = tqdm(days, desc="exposure (trans_fea)", unit="day")
    for d in pbar:
        pbar.set_postfix_str(d, refresh=False)
        part = compute_daily_trans_exposure_features(d)
        if part is None:
            missing += 1
            tp = trans_fea_path(d)
            if tp.is_file():
                tqdm.write(f"[exposure] {d} skip (unreadable or empty trans_fea): {tp}")
            continue
        if part.height > 0:
            parts.append(part)

    if not parts:
        raise RuntimeError("No trans_fea data found in range; check paths and dates.")

    long = pl.concat(parts, how="vertical")
    lo, hi = winsor_pct
    agg = long.group_by(["exchange", "code"]).agg(
        pl.col("oib_day").mean().alias("oib_mean_raw"),
        pl.col("n_trades_day").mean().alias("n_trades_mean_raw"),
        pl.col("total_vol_day").mean().alias("total_vol_mean_raw"),
        pl.col("trade_date").n_unique().alias("n_days"),
    )
    out = agg
    for raw_col, tag in (
        ("oib_mean_raw", "oib"),
        ("n_trades_mean_raw", "n_trades"),
        ("total_vol_mean_raw", "total_vol"),
    ):
        out = _winsor_z(out, raw_col, tag, lo, hi)

    out = out.select(
        [
            "exchange",
            "code",
            "n_days",
            "oib_mean_raw",
            "oib_mean",
            "E_oib_z",
            "n_trades_mean_raw",
            "n_trades_mean",
            "E_n_trades_z",
            "total_vol_mean_raw",
            "total_vol_mean",
            "E_total_vol_z",
        ]
    )

    outp = exposure_pre_policy_path()
    tmp = outp.with_suffix(".parquet.tmp")
    out.write_parquet(tmp)
    tmp.replace(outp)

    _append_manifest(
        build_manifest_path(),
        {
            "kind": "exposure_pre_policy",
            "start": start,
            "end": end,
            "policy_anchor": POLICY_DATE_MAIN,
            "calendar_mode": _cal_mode,
            "n_dates_in_range": len(days),
            "missing_trans_days": missing,
            "n_stocks": out.height,
            "out_path": str(outp),
            "columns": [
                "E_oib_z",
                "E_n_trades_z",
                "E_total_vol_z",
            ],
        },
    )
    return outp
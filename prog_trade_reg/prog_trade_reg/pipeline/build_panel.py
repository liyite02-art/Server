from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from prog_trade_reg.config import POLICY_DATE_MAIN
from prog_trade_reg.paths import (
    build_manifest_path,
    exposure_pre_policy_path,
    outcomes_dir,
    panel_dir,
)


def _append_manifest_panel(record: dict) -> None:
    path = build_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_panel_did_long(
    *,
    policy_date: str | None = None,
    write_csv: bool = True,
    write_parquet: bool = True,
    exposure_path: Path | None = None,
) -> dict[str, Path]:
    """
    Stack ``daily/outcomes/*.parquet``, left-join ``exposure_pre_policy.parquet``, add ``post`` and ``post_x_E*_z``.

    Writes under ``panel/``:

    - ``panel_did_long.parquet`` — Polars/pandas.
    - ``panel_did_long.csv`` — delimiter ``|`` (Stata: ``import delimited ..., delimiter("|")``).

    Main DID regressor column: ``post_x_E_oib_z`` (equals ``post * E_oib_z``). Other ``post_x_E*_z`` are optional controls.
    """
    if not write_csv and not write_parquet:
        raise ValueError("At least one of write_csv or write_parquet must be True")
    policy_date = policy_date or POLICY_DATE_MAIN
    exp_p = exposure_path or exposure_pre_policy_path()
    if not exp_p.is_file():
        raise FileNotFoundError(
            f"exposure parquet not found: {exp_p}. Run: python -m prog_trade_reg exposure"
        )

    odir = outcomes_dir()
    panel_dir().mkdir(parents=True, exist_ok=True)
    files = sorted(odir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No outcome parquet files under {odir}")

    t0 = time.perf_counter()
    lf = pl.concat([pl.scan_parquet(str(f)) for f in files], how="vertical")

    exp_df = pl.read_parquet(exp_p)
    z_cols = [c for c in exp_df.columns if c.startswith("E_") and c.endswith("_z")]
    exp = exp_df.lazy()

    lf = lf.with_columns(
        pl.col("code").cast(pl.Int64),
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("exchange").cast(pl.Utf8),
    )
    exp = exp.with_columns(
        pl.col("code").cast(pl.Int64),
        pl.col("exchange").cast(pl.Utf8),
    )

    out = lf.join(exp, on=["exchange", "code"], how="left")

    out = out.with_columns(
        (pl.col("trade_date") >= pl.lit(policy_date)).cast(pl.Int8).alias("post"),
        pl.lit(policy_date).alias("policy_date_main"),
    )

    for zc in z_cols:
        out = out.with_columns(
            (pl.col("post").cast(pl.Float64) * pl.col(zc)).alias(f"post_x_{zc}")
        )

    out = out.with_columns(
        pl.concat_str([pl.col("exchange"), pl.lit("_"), pl.col("code").cast(pl.Utf8)]).alias("panel_unit"),
        pl.when((pl.col("es_bps_den").is_not_null()) & (pl.col("es_bps_den") > 0))
        .then(pl.col("es_bps_num") / pl.col("es_bps_den"))
        .otherwise(None)
        .alias("vw_es_bps"),
    )

    df = out.collect(streaming=True)
    written: dict[str, Path] = {}
    pq_out = panel_dir() / "panel_did_long.parquet"
    csv_out = panel_dir() / "panel_did_long.csv"

    if write_parquet:
        tmp = pq_out.with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression="zstd")
        tmp.replace(pq_out)
        written["parquet"] = pq_out

    if write_csv:
        tmpc = csv_out.with_suffix(".csv.tmp")
        df.write_csv(tmpc, separator="|")
        tmpc.replace(csv_out)
        written["csv"] = csv_out

    elapsed = time.perf_counter() - t0
    _append_manifest_panel(
        {
            "kind": "panel_did_long",
            "policy_date": policy_date,
            "n_rows": df.height,
            "n_outcome_files": len(files),
            "exposure_path": str(exp_p),
            "parquet": str(written.get("parquet", "")),
            "csv": str(written.get("csv", "")),
            "z_interactions": [f"post_x_{zc}" for zc in z_cols],
            "elapsed_sec": round(elapsed, 3),
            "utc_iso": datetime.now(timezone.utc).isoformat(),
        }
    )
    return written

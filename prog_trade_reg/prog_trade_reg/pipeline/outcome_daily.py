from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl

from prog_trade_reg.config import TRADE_PRICE_SCALE
from prog_trade_reg.feather_io import read_feather_to_polars
from prog_trade_reg.paths import build_manifest_path, ensure_derived_dirs, outcome_daily_parquet_path
from prog_trade_reg.pipeline.lob_agg import aggregate_lob_by_code_time
from prog_trade_reg.raw_paths import lob_parquet_path, trans_fea_path


def _append_manifest_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prepare_trades(trans: pl.DataFrame, exchange: str) -> pl.DataFrame:
    ex = exchange.upper()
    suf = f".{ex}"
    code_u = pl.col("code").cast(pl.Utf8)
    t = trans.filter(code_u.str.ends_with(suf)).with_columns(
        code_u.str.split(".").list.get(0).cast(pl.Int32, strict=False).alias("code"),
        (pl.col("tradePrice").cast(pl.Float64) / TRADE_PRICE_SCALE).alias("price"),
        pl.col("tradeVolume").cast(pl.Float64).alias("vol"),
        pl.col("time").cast(pl.Int64),
    )
    return t.filter(pl.col("price") > 0, pl.col("vol") > 0).select(
        ["code", "time", "price", "vol"]
    )


def _effective_spread_stock_day(trades_prep: pl.DataFrame, lob_snap: pl.DataFrame) -> pl.DataFrame:
    """Join trades to LOB snapshots (as-of backward), aggregate VW effective spread."""
    lt = trades_prep.sort(["code", "time"])
    lb = lob_snap.sort(["code", "time"])
    j = lt.join_asof(
        lb,
        left_on="time",
        right_on="time",
        by_left="code",
        by_right="code",
        strategy="backward",
        check_sortedness=False,
    )
    j = j.filter(
        pl.col("bp1").is_not_nan(),
        pl.col("sp1").is_not_nan(),
        (pl.col("bp1") > 0) & (pl.col("sp1") > 0),
    )
    mid = (pl.col("bp1") + pl.col("sp1")) / 2.0
    j = j.with_columns(mid.alias("mid"))
    es_bps = 2.0 * (pl.col("price") - pl.col("mid")).abs() / pl.col("mid") * 10000.0
    j = j.with_columns(es_bps.alias("es_bps"))
    return j.group_by("code").agg(
        pl.len().alias("n_trades"),
        pl.col("vol").sum().alias("volume_sum"),
        (pl.col("price") * pl.col("vol")).sum().alias("dollar_vol"),
        (pl.col("es_bps") * pl.col("vol")).sum().alias("es_bps_num"),
        pl.col("vol").sum().alias("es_bps_den"),
    )


def _rv_from_mid(lob_snap: pl.DataFrame) -> pl.DataFrame:
    """Realized variance of log mids from consecutive valid LOB snapshots (within day)."""
    x = lob_snap.filter(
        pl.col("bp1").is_not_nan(),
        pl.col("sp1").is_not_nan(),
        (pl.col("bp1") > 0) & (pl.col("sp1") > 0),
    ).with_columns(((pl.col("bp1") + pl.col("sp1")) / 2.0).alias("mid"))
    x = x.sort(["code", "time"]).with_columns(
        pl.col("mid").log().diff().over("code").alias("lr")
    )
    x = x.filter(pl.col("lr").is_not_nan())
    return x.group_by("code").agg((pl.col("lr") ** 2).sum().alias("rv_mid"))


def _amihud_daily(lob_snap: pl.DataFrame, dollar_by_code: pl.DataFrame) -> pl.DataFrame:
    """|log(last_mid/first_mid)| / dollar volume (classic daily illiquidity flavor)."""
    x = lob_snap.filter(
        pl.col("bp1").is_not_nan(),
        pl.col("sp1").is_not_nan(),
        (pl.col("bp1") > 0) & (pl.col("sp1") > 0),
    ).with_columns(((pl.col("bp1") + pl.col("sp1")) / 2.0).alias("mid"))
    x = x.sort(["code", "time"])
    rng = x.group_by("code").agg(
        pl.col("mid").first().alias("m_open"),
        pl.col("mid").last().alias("m_close"),
    ).with_columns(
        (pl.col("m_close") / pl.col("m_open")).log().abs().alias("amihud_num")
    )
    j = rng.join(dollar_by_code, on="code", how="left")
    return j.with_columns(
        pl.when((pl.col("dollar_vol").is_not_null()) & (pl.col("dollar_vol") > 0))
        .then(pl.col("amihud_num") / pl.col("dollar_vol"))
        .otherwise(None)
        .alias("amihud")
    )


def _merge_es_rv(es: pl.DataFrame, rv: pl.DataFrame) -> pl.DataFrame:
    """Full join effective-spread aggregates with RV without duplicate ``code`` columns."""
    m = es.join(rv, on="code", how="full")
    if "code_right" in m.columns:
        m = m.with_columns(
            pl.coalesce(pl.col("code"), pl.col("code_right")).alias("code_k")
        ).drop(["code", "code_right"]).rename({"code_k": "code"})
    return m


def build_outcome_daily_for_date(
    trade_date: str,
    exchanges: Iterable[str] | None = None,
    *,
    write_aligned_trades: bool = False,
) -> None:
    """
    One pass: LOB -> (code,time) snapshot; trans_fea -> join_asof -> stock-day outcome metrics.

    Writes ``daily/outcomes/{trade_date}.parquet`` (SZ+SH rows, column ``exchange``).
    """
    if exchanges is None:
        exchanges = ("SZ", "SH")
    ex_list = [e.upper() for e in exchanges]

    trans_path = trans_fea_path(trade_date)
    if not trans_path.is_file():
        raise FileNotFoundError(f"trans_fea not found: {trans_path}")

    ensure_derived_dirs()
    t0 = time.perf_counter()

    trans_all = read_feather_to_polars(trans_path)

    parts: list[pl.DataFrame] = []
    for ex in ex_list:
        lob_p = lob_parquet_path(ex, trade_date)
        if not lob_p.is_file():
            raise FileNotFoundError(f"LOB file not found: {lob_p}")

        lob_snap = aggregate_lob_by_code_time(str(lob_p))
        trades_prep = _prepare_trades(trans_all, ex)
        es = _effective_spread_stock_day(trades_prep, lob_snap)
        rv = _rv_from_mid(lob_snap)
        dv = es.select(["code", "dollar_vol"])
        am = _amihud_daily(lob_snap, dv)

        out = (
            _merge_es_rv(es, rv)
            .join(am.select(["code", "amihud"]), on="code", how="left")
            .with_columns(
                pl.lit(trade_date).alias("trade_date"),
                pl.lit(ex).alias("exchange"),
            )
            .select(
                [
                    "code",
                    "trade_date",
                    "exchange",
                    "n_trades",
                    "volume_sum",
                    "dollar_vol",
                    "es_bps_num",
                    "es_bps_den",
                    "rv_mid",
                    "amihud",
                ]
            )
        )
        parts.append(out)

        if write_aligned_trades:
            _ = write_aligned_trades

    combined = pl.concat(parts, how="vertical")
    out_path = outcome_daily_parquet_path(trade_date)
    tmp = out_path.with_suffix(".parquet.tmp")
    combined.write_parquet(tmp)
    tmp.replace(out_path)

    elapsed = time.perf_counter() - t0
    _append_manifest_line(
        build_manifest_path(),
        {
            "trade_date": trade_date,
            "exchanges": ex_list,
            "status": "ok",
            "elapsed_sec": round(elapsed, 3),
            "trans_fea": str(trans_path),
            "outcome_daily_path": str(out_path),
            "utc_iso": datetime.now(timezone.utc).isoformat(),
        },
    )


def build_stock_day_for_date(*args, **kwargs) -> None:
    """Deprecated alias for :func:`build_outcome_daily_for_date`."""
    return build_outcome_daily_for_date(*args, **kwargs)


def default_exchanges() -> tuple[str, ...]:
    return ("SZ", "SH")

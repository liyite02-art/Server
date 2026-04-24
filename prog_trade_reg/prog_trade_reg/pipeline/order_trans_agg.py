from __future__ import annotations

"""
Optional: aggregate order_trans streams to (code_int, time) trade stats (type='T'),
matching the user's DuckDB pattern. Used for diagnostics or merged LOB+flow tables.
"""

import polars as pl


def aggregate_order_trans_trades_by_time(df: pl.DataFrame) -> pl.DataFrame:
    """
    Expect columns: code (categorical/str), type, volume, price, time.
    Produces integer ``code`` (first 6 digits) and vol/cnt/amt per (code, time).
    """
    s = df["code"].cast(pl.Utf8)
    code_int = s.str.split(".").list.get(0).cast(pl.Int64, strict=False)
    d = df.with_columns(code_int.alias("code_i")).filter(pl.col("type").cast(pl.Utf8) == "T")
    return d.group_by(["code_i", "time"]).agg(
        pl.col("volume").sum().cast(pl.Int64).alias("vol"),
        pl.len().alias("cnt"),
        (pl.col("price") * pl.col("volume")).sum().alias("amt"),
    ).rename({"code_i": "code"})

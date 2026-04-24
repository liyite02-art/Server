from __future__ import annotations

import polars as pl


def aggregate_lob_by_code_time(lob_path: str) -> pl.DataFrame:
    """
    Collapse multiple LOB rows per (code, time) to the row with max seq_num.

    Equivalent to DuckDB ``arg_max(col, seq_num)`` for each column after grouping
    by ``code, time`` (implemented as sort by seq_num then take last per group).
    """
    lf = pl.scan_parquet(lob_path)
    schema = lf.collect_schema()
    cols = [c for c in schema.names() if c not in ("code", "time")]
    if "seq_num" not in schema.names():
        raise ValueError("LOB schema must include seq_num for tie-breaking")
    agg_exprs = [pl.col(c).last() for c in cols]
    return (
        lf.sort(["code", "time", "seq_num"])
        .group_by(["code", "time"])
        .agg(agg_exprs)
        .sort(["code", "time"])
        .collect()
    )

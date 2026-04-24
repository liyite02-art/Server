"""
Prototype DID: two-way FE + continuous interaction, clustered SE (``linearmodels``).

Reads ``panel/panel_did_long.parquet`` or ``panel_did_long.csv`` under ``DERIVED_ROOT``.
"""

from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from prog_trade_reg.paths import panel_did_long_csv_path, panel_did_long_parquet_path

# 网格回归默认：日度产出列 + 三种事前暴露×post（与 exposure 流水线一致）
DEFAULT_GRID_OUTCOMES: tuple[str, ...] = (
    "vw_es_bps",
    "rv_mid",
    "amihud",
    "n_trades",
    "volume_sum",
    "dollar_vol",
)
DEFAULT_GRID_INTERACTIONS: tuple[str, ...] = (
    "post_x_E_oib_z",
    "post_x_E_n_trades_z",
    "post_x_E_total_vol_z",
)

HeadStrategy = Literal["first", "random", "ends"]


def _read_csv_tail_rows(path: Path, sep: str, n_tail: int) -> pd.DataFrame:
    """Last ``n_tail`` data lines plus header (for date-sorted panels: includes post period)."""
    if n_tail <= 0:
        return pd.DataFrame()
    proc = subprocess.run(
        ["tail", "-n", str(n_tail), str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    return pd.read_csv(StringIO(header + proc.stdout), sep=sep)


def _dedupe_panel_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("panel_unit", "trade_date") if c in df.columns]
    if keys:
        return df.drop_duplicates(subset=keys, keep="first", ignore_index=True)
    return df.drop_duplicates(ignore_index=True)


def load_panel_long(
    panel_path: Path | None = None,
    *,
    head: int | None = None,
    head_strategy: HeadStrategy = "ends",
) -> pd.DataFrame:
    """
    Load long panel. If ``head`` is set, subsample:

    - ``ends`` (default): first ``ceil(head/2)`` and last ``floor(head/2)`` rows — fast for CSV
      (no full read) when the file is roughly time-ordered; avoids ``post×E`` constant if pre
      and post both appear in the file.
    - ``first``: only the first ``head`` rows (``nrows`` on CSV); may be all pre-policy.
    - ``random``: uniform random ``head`` rows (reads entire file first).
    """
    pq = panel_did_long_parquet_path()
    csv = panel_did_long_csv_path()
    if panel_path is not None:
        p = Path(panel_path)
    elif pq.is_file():
        p = pq
    elif csv.is_file():
        p = csv
    else:
        raise FileNotFoundError(
            f"No panel file: expected {pq} or {csv}. Run: python -m prog_trade_reg panel-did"
        )
    is_parquet = p.suffix.lower() == ".parquet"

    if head is None or head <= 0:
        if is_parquet:
            return pd.read_parquet(p)
        return pd.read_csv(p, sep="|")

    if head_strategy == "first":
        if is_parquet:
            df = pd.read_parquet(p)
            return df.iloc[:head].copy()
        return pd.read_csv(p, sep="|", nrows=head)

    if head_strategy == "random":
        if is_parquet:
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, sep="|")
        n = min(head, len(df))
        return df.sample(n=n, random_state=0).reset_index(drop=True)

    # ends
    n_first = (head + 1) // 2
    n_tail = head // 2
    if is_parquet:
        df = pd.read_parquet(p)
        if len(df) <= head:
            return df.copy()
        top = df.iloc[:n_first]
        if n_tail <= 0:
            return top.copy()
        bot = df.iloc[-n_tail:]
        return _dedupe_panel_rows(pd.concat([top, bot], ignore_index=True))
    df_head = pd.read_csv(p, sep="|", nrows=n_first)
    if n_tail <= 0:
        return df_head.copy()
    df_tail = _read_csv_tail_rows(p, "|", n_tail)
    return _dedupe_panel_rows(pd.concat([df_head, df_tail], ignore_index=True))


def peek_panel_columns(panel_path: Path | None = None) -> list[str]:
    """Column names of ``panel_did_long`` without reading the full file (parquet: schema only)."""
    pq = panel_did_long_parquet_path()
    csv = panel_did_long_csv_path()
    if panel_path is not None:
        p = Path(panel_path)
    elif pq.is_file():
        p = pq
    elif csv.is_file():
        p = csv
    else:
        raise FileNotFoundError(f"No panel file: expected {pq} or {csv}")
    if p.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq_mod

        return list(pq_mod.read_schema(str(p)).names)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    return line.rstrip("\n").split("|")


def run_did_twfe(
    *,
    panel_path: Path | None = None,
    data: pd.DataFrame | None = None,
    y: str = "vw_es_bps",
    interaction: str = "post_x_E_oib_z",
    extra_controls: list[str] | None = None,
    cluster: Literal["time", "twoway", "none"] = "time",
    head: int | None = None,
    head_strategy: HeadStrategy = "ends",
) -> Any:
    """
    Estimate ``y ~ interaction + controls`` with entity & time FE.

    Parameters
    ----------
    y
        Outcome column (default volume-weighted effective spread, bps).
    interaction
        Main DID term (default ``post_x_E_oib_z``).
    extra_controls
        Additional regressors (e.g. ``post_x_E_n_trades_z``).
    cluster
        ``time`` = cluster on calendar date; ``twoway`` = entity+time (Driscoll-Kraay style in linearmodels);
        ``none`` = unadjusted (not recommended).
    head_strategy
        When ``head`` is set: ``ends`` (default) = bookend rows for time-sorted CSV; ``first`` = first N only;
        ``random`` = uniform sample (full read).
    data
        If provided, use this frame directly (``panel_path`` / ``head`` ignored). For batch grids to avoid
        reloading the panel.
    """
    if data is None:
        df = load_panel_long(panel_path, head=head, head_strategy=head_strategy)
    else:
        df = data.copy()
    if y not in df.columns:
        raise KeyError(f"Outcome {y!r} not in panel columns: {list(df.columns)[:20]}...")
    if interaction not in df.columns:
        raise KeyError(f"Regressor {interaction!r} not in panel columns")

    use = [y, interaction, "panel_unit", "trade_date"]
    controls = list(extra_controls or [])
    for c in controls:
        if c not in df.columns:
            raise KeyError(f"Control {c!r} not in panel columns")
    use.extend(controls)

    d = df[use].copy()
    d["td"] = pd.to_datetime(d["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    d = d.dropna(subset=["td", y, interaction])
    for c in controls:
        d = d.dropna(subset=[c])
    d = d.set_index(["panel_unit", "td"])

    xmat = d[[interaction] + controls].to_numpy(dtype=float)
    if not np.isfinite(xmat).all():
        raise ValueError("Regressors contain non-finite values after dropna.")
    if np.linalg.matrix_rank(xmat) < xmat.shape[1]:
        raise ValueError(
            f"Regressors are collinear or constant after dropna (rank {np.linalg.matrix_rank(xmat)} < "
            f"{xmat.shape[1]}). If you used --head first, try default --head-strategy ends or full sample."
        )

    rhs = [interaction] + controls
    formula = f"{y} ~ 1 + " + " + ".join(rhs) + " + EntityEffects + TimeEffects"

    mod = PanelOLS.from_formula(formula, data=d, check_rank=False)
    if cluster == "none":
        return mod.fit(low_memory=True)
    if cluster == "time":
        return mod.fit(
            cov_type="clustered",
            cluster_entity=False,
            cluster_time=True,
            low_memory=True,
        )
    if cluster == "twoway":
        return mod.fit(
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
            low_memory=True,
        )
    raise ValueError(cluster)


def interaction_coef_row(res: Any, interaction: str) -> dict[str, Any]:
    """One row of coefficient table for the main interaction (robust SE / t / p)."""
    names = list(res.params.index)
    key = interaction if interaction in names else None
    if key is None:
        # linearmodels 可能对公式项改名，取唯一非 Intercept 的斜率项
        slope = [n for n in names if n != "Intercept"]
        if len(slope) == 1:
            key = slope[0]
    if key is None:
        raise KeyError(f"No coefficient for {interaction!r}; available: {names}")
    out: dict[str, Any] = {
        "nobs": int(res.nobs),
        "coef": float(res.params[key]),
        "stderr": float(res.std_errors[key]),
        "t": float(res.tstats[key]),
        "pvalue": float(res.pvalues[key]),
    }
    return out


def run_did_grid(
    *,
    panel_path: Path | None = None,
    outcomes: list[str] | None = None,
    interactions: list[str] | None = None,
    cluster: Literal["time", "twoway", "none"] = "twoway",
    head: int | None = None,
    head_strategy: HeadStrategy = "ends",
    extra_controls: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load panel once, then run every (outcome × interaction) univariate TWFE spec.

    Missing columns or estimation failures become rows with ``error`` set.
    """
    ys = list(outcomes) if outcomes is not None else list(DEFAULT_GRID_OUTCOMES)
    xs = list(interactions) if interactions is not None else list(DEFAULT_GRID_INTERACTIONS)
    df = load_panel_long(panel_path, head=head, head_strategy=head_strategy)
    have = set(df.columns)
    rows: list[dict[str, Any]] = []
    for y in ys:
        for x in xs:
            row: dict[str, Any] = {"y": y, "interaction": x, "cluster": cluster}
            if y not in have:
                row["error"] = f"missing outcome column {y!r}"
                rows.append(row)
                continue
            if x not in have:
                row["error"] = f"missing interaction column {x!r}"
                rows.append(row)
                continue
            if extra_controls:
                miss = [c for c in extra_controls if c not in have]
                if miss:
                    row["error"] = f"missing control columns: {miss}"
                    rows.append(row)
                    continue
            try:
                res = run_did_twfe(
                    data=df,
                    y=y,
                    interaction=x,
                    extra_controls=extra_controls,
                    cluster=cluster,
                )
                row.update(interaction_coef_row(res, x))
            except Exception as exc:  # noqa: BLE001 — collect grid failures
                row["error"] = str(exc)
            rows.append(row)
    out = pd.DataFrame(rows)
    # stable column order
    front = ["y", "interaction", "cluster", "coef", "stderr", "t", "pvalue", "nobs", "error"]
    for c in front:
        if c not in out.columns:
            out[c] = np.nan
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def print_summary(res: Any) -> None:
    print(res.summary.as_text())

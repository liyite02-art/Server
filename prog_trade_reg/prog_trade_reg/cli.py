from __future__ import annotations

import argparse

from pathlib import Path

import pandas as pd

from prog_trade_reg.config import (
    DERIVED_ROOT,
    EXPOSURE_PRE_END,
    EXPOSURE_PRE_START,
    OUTCOMES_BATCH_END,
    OUTCOMES_BATCH_START,
    TRADE_DAYS_PKL,
)
from prog_trade_reg.paths import (
    exposure_pre_policy_path,
    outcome_daily_parquet_path,
    outcomes_dir,
    panel_dir,
)
from prog_trade_reg.did_regress import (
    DEFAULT_GRID_INTERACTIONS,
    DEFAULT_GRID_OUTCOMES,
    peek_panel_columns,
    print_summary,
    run_did_grid,
    run_did_twfe,
)
from prog_trade_reg.pipeline.build_panel import build_panel_did_long
from prog_trade_reg.pipeline.exposure import build_exposure_pre_policy
from prog_trade_reg.pipeline.outcome_batch import build_outcomes_range
from prog_trade_reg.pipeline.outcome_daily import build_outcome_daily_for_date, default_exchanges


def _trade_days_pkl_path(override: str | None) -> Path:
    return Path(override) if override else TRADE_DAYS_PKL


def _comma_list(s: str | None) -> list[str] | None:
    if s is None or not str(s).strip():
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _print_artifact_locations(*, note_exposure_deferred: bool = False) -> None:
    root = DERIVED_ROOT.resolve()
    print(f"[paths] 派生数据根目录: {root}", flush=True)
    print(f"[paths]   因变量目录: {outcomes_dir().resolve()}/", flush=True)
    print(f"[paths]   自变量文件: {exposure_pre_policy_path().resolve()}", flush=True)
    print(f"[paths]   DID 长面板目录: {panel_dir().resolve()}/", flush=True)
    if note_exposure_deferred:
        print(
            "[paths]   说明: exposure 阶段按日只读入内存，**全部交易日跑完后**才写入上述自变量 parquet。",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Programmatic trading regulation pipeline: daily outcomes (HF→daily) and pre-policy exposure."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_out = sub.add_parser(
        "outcomes",
        help="Build one Parquet per trade_date: ES/RV/Amihud under daily/outcomes/.",
    )
    p_out.add_argument(
        "trade_date",
        help="Trading calendar YYYYMMDD",
    )
    p_out.add_argument(
        "--exchange",
        action="append",
        choices=["SZ", "SH"],
        help="Restrict to one exchange; repeat or omit for both.",
    )
    p_out.add_argument(
        "--write-aligned-trades",
        action="store_true",
        help="If set, also write aligned_trades layer (not implemented yet).",
    )

    p_exp = sub.add_parser(
        "exposure",
        help="Build time-invariant continuous exposure E_oib_z from trans_fea (pre-policy window).",
    )
    p_exp.add_argument(
        "--start",
        default=None,
        help="YYYYMMDD inclusive; default from config.EXPOSURE_PRE_START",
    )
    p_exp.add_argument(
        "--end",
        default=None,
        help="YYYYMMDD inclusive; default from config.EXPOSURE_PRE_END (must be before main policy date)",
    )
    p_exp.add_argument(
        "--trade-days-pkl",
        default=None,
        help="Pickle path with key trade_days; default config.TRADE_DAYS_PKL",
    )

    p_range = sub.add_parser(
        "outcomes-range",
        help="Build daily/outcomes for each trading day in [start, end] (from trade_days pickle); skip missing raw.",
    )
    p_range.add_argument(
        "--start",
        default=None,
        help="YYYYMMDD inclusive; default config.OUTCOMES_BATCH_START",
    )
    p_range.add_argument(
        "--end",
        default=None,
        help="YYYYMMDD inclusive; default config.OUTCOMES_BATCH_END",
    )
    p_range.add_argument(
        "--exchange",
        action="append",
        choices=["SZ", "SH"],
        help="Restrict to one exchange; repeat or omit for both.",
    )
    p_range.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Do not skip missing/unreadable trans_fea/LOB; raise instead.",
    )
    p_range.add_argument(
        "--write-aligned-trades",
        action="store_true",
        help="If set, also write aligned_trades layer (not implemented yet).",
    )
    p_range.add_argument(
        "--trade-days-pkl",
        default=None,
        help="Pickle path with key trade_days; default config.TRADE_DAYS_PKL",
    )

    p_all = sub.add_parser(
        "build-all",
        help="Run exposure (pre-policy window) then outcomes-range for the panel sample.",
    )
    p_all.add_argument(
        "--skip-exposure",
        action="store_true",
        help="Only build daily outcomes.",
    )
    p_all.add_argument(
        "--skip-outcomes",
        action="store_true",
        help="Only build exposure_pre_policy.parquet.",
    )
    p_all.add_argument(
        "--outcomes-start",
        default=None,
        help="Overrides OUTCOMES_BATCH_START for the outcomes leg.",
    )
    p_all.add_argument(
        "--outcomes-end",
        default=None,
        help="Overrides OUTCOMES_BATCH_END for the outcomes leg.",
    )
    p_all.add_argument(
        "--exposure-start",
        default=None,
        help="Overrides EXPOSURE_PRE_START.",
    )
    p_all.add_argument(
        "--exposure-end",
        default=None,
        help="Overrides EXPOSURE_PRE_END.",
    )
    p_all.add_argument(
        "--exchange",
        action="append",
        choices=["SZ", "SH"],
        help="Restrict outcomes to one exchange; repeat or omit for both.",
    )
    p_all.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Passed to outcomes-range: stop on missing/unreadable raw files.",
    )
    p_all.add_argument(
        "--trade-days-pkl",
        default=None,
        help="Pickle path with key trade_days; default config.TRADE_DAYS_PKL",
    )

    p_panel = sub.add_parser(
        "panel-did",
        help="Merge daily/outcomes + exposure → panel/panel_did_long.(parquet|csv) for DID (post & post×E_z).",
    )
    p_panel.add_argument(
        "--policy-date",
        default=None,
        help="YYYYMMDD for post=1; default config.POLICY_DATE_MAIN (施行日)",
    )
    p_panel.add_argument(
        "--exposure",
        type=Path,
        default=None,
        help="Override path to exposure_pre_policy.parquet",
    )
    p_panel.add_argument(
        "--no-csv",
        action="store_true",
        help="Only write parquet",
    )
    p_panel.add_argument(
        "--no-parquet",
        action="store_true",
        help="Only write csv",
    )

    p_reg = sub.add_parser(
        "did-regress",
        help="Run two-way FE + post×E DID on panel_did_long (linearmodels); cluster SE by default.",
    )
    p_reg.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="panel_did_long.parquet or .csv; default under DERIVED_ROOT/panel/",
    )
    p_reg.add_argument(
        "--y",
        default="vw_es_bps",
        help="Outcome column (default vw_es_bps)",
    )
    p_reg.add_argument(
        "--x",
        dest="interaction",
        default="post_x_E_oib_z",
        help="Main interaction term (default post_x_E_oib_z)",
    )
    p_reg.add_argument(
        "--control",
        action="append",
        default=None,
        help="Extra regressors (repeatable), e.g. --control post_x_E_n_trades_z",
    )
    p_reg.add_argument(
        "--cluster",
        choices=["time", "twoway", "none"],
        default="time",
        help="SE clustering: time (default), twoway (entity+time), or none",
    )
    p_reg.add_argument(
        "--head",
        type=int,
        default=None,
        help="Subsample N rows for a quick run (default full sample). With CSV, default strategy is "
        "bookend (first+last rows) so post×E is not all zero.",
    )
    p_reg.add_argument(
        "--head-strategy",
        choices=["ends", "first", "random"],
        default="ends",
        help="How --head subsamples: ends=first half + last half rows (default, good for time-sorted CSV); "
        "first=only first N (may be all pre-policy); random=uniform sample (reads whole file).",
    )

    p_grid = sub.add_parser(
        "did-regress-grid",
        help="TWFE grid: each (outcome × post×E interaction) separately; default cluster twoway. Panel loaded once.",
    )
    p_grid.add_argument("--panel", type=Path, default=None, help="panel_did_long path; default DERIVED_ROOT/panel/")
    p_grid.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help=f"Comma-separated outcome columns. Default: {','.join(DEFAULT_GRID_OUTCOMES)}",
    )
    p_grid.add_argument(
        "--interactions",
        type=str,
        default=None,
        help=f"Comma-separated post×E columns. Default: {','.join(DEFAULT_GRID_INTERACTIONS)}",
    )
    p_grid.add_argument(
        "--cluster",
        choices=["time", "twoway", "none"],
        default="twoway",
        help="SE clustering (default twoway for robustness table)",
    )
    p_grid.add_argument("--head", type=int, default=None, help="Subsample rows (same semantics as did-regress)")
    p_grid.add_argument(
        "--head-strategy",
        choices=["ends", "first", "random"],
        default="ends",
        dest="head_strategy_grid",
    )
    p_grid.add_argument(
        "--control",
        action="append",
        default=None,
        help="Extra regressors in every spec (repeatable)",
    )
    p_grid.add_argument(
        "--list-columns",
        action="store_true",
        help="Print panel column names and exit (no regressions)",
    )

    args = p.parse_args()
    if args.cmd == "outcomes":
        _print_artifact_locations()
        print(
            f"[paths]   本日将写入: {outcome_daily_parquet_path(args.trade_date).resolve()}",
            flush=True,
        )
        ex = tuple(args.exchange) if args.exchange else default_exchanges()
        build_outcome_daily_for_date(
            args.trade_date,
            exchanges=ex,
            write_aligned_trades=args.write_aligned_trades,
        )
    elif args.cmd == "exposure":
        _print_artifact_locations(note_exposure_deferred=True)
        pkl = _trade_days_pkl_path(args.trade_days_pkl)
        build_exposure_pre_policy(
            start=args.start,
            end=args.end,
            trade_days_pkl=pkl,
        )
    elif args.cmd == "outcomes-range":
        _print_artifact_locations()
        ostart = args.start or OUTCOMES_BATCH_START
        oend = args.end or OUTCOMES_BATCH_END
        ex = tuple(args.exchange) if args.exchange else default_exchanges()
        pkl = _trade_days_pkl_path(args.trade_days_pkl)
        summary = build_outcomes_range(
            ostart,
            oend,
            exchanges=ex,
            skip_missing_raw=not args.fail_on_missing,
            write_aligned_trades=args.write_aligned_trades,
            trade_days_pkl=pkl,
        )
        print(
            "outcomes-range done:",
            summary,
            flush=True,
        )
    elif args.cmd == "build-all":
        _print_artifact_locations(note_exposure_deferred=True)
        pkl = _trade_days_pkl_path(args.trade_days_pkl)
        if not args.skip_exposure:
            build_exposure_pre_policy(
                start=args.exposure_start or EXPOSURE_PRE_START,
                end=args.exposure_end or EXPOSURE_PRE_END,
                trade_days_pkl=pkl,
            )
        if not args.skip_outcomes:
            ostart = args.outcomes_start or OUTCOMES_BATCH_START
            oend = args.outcomes_end or OUTCOMES_BATCH_END
            ex = tuple(args.exchange) if args.exchange else default_exchanges()
            summary = build_outcomes_range(
                ostart,
                oend,
                exchanges=ex,
                skip_missing_raw=not args.fail_on_missing,
                trade_days_pkl=pkl,
            )
            print("build-all outcomes leg:", summary, flush=True)
    elif args.cmd == "panel-did":
        _print_artifact_locations()
        written = build_panel_did_long(
            policy_date=args.policy_date,
            write_csv=not args.no_csv,
            write_parquet=not args.no_parquet,
            exposure_path=args.exposure,
        )
        print("[paths] panel-did written:", {k: str(v.resolve()) for k, v in written.items()}, flush=True)
    elif args.cmd == "did-regress":
        res = run_did_twfe(
            panel_path=args.panel,
            y=args.y,
            interaction=args.interaction,
            extra_controls=args.control,
            cluster=args.cluster,
            head=args.head,
            head_strategy=args.head_strategy,
        )
        print_summary(res)
    elif args.cmd == "did-regress-grid":
        if args.list_columns:
            cols = peek_panel_columns(args.panel)
            print("panel columns:", ", ".join(cols))
            yo = set(DEFAULT_GRID_OUTCOMES) & set(cols)
            xi = set(DEFAULT_GRID_INTERACTIONS) & set(cols)
            print("default outcomes present:", sorted(yo))
            print("default interactions present:", sorted(xi))
            return
        table = run_did_grid(
            panel_path=args.panel,
            outcomes=_comma_list(args.outcomes),
            interactions=_comma_list(args.interactions),
            cluster=args.cluster,
            head=args.head,
            head_strategy=args.head_strategy_grid,
            extra_controls=args.control,
        )
        with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_columns", None):
            print(table.to_string(index=False))
    else:
        raise RuntimeError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

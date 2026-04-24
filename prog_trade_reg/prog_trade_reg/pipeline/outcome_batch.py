from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from prog_trade_reg.config import TRADE_DAYS_PKL
from prog_trade_reg.feather_io import UnreadableRawFeatherError
from prog_trade_reg.pipeline.outcome_daily import build_outcome_daily_for_date, default_exchanges
from prog_trade_reg.trade_calendar import iter_dates_in_range


def build_outcomes_range(
    start: str,
    end: str,
    *,
    exchanges: Iterable[str] | None = None,
    skip_missing_raw: bool = True,
    write_aligned_trades: bool = False,
    trade_days_pkl: Path | None = None,
) -> dict[str, int]:
    """
    Build ``daily/outcomes/{YYYYMMDD}.parquet`` for each **trading day** in ``[start, end]``.

    Trading days come from ``trade_days_pkl`` (default ``config.TRADE_DAYS_PKL``).
    Raw inputs may still be absent, or ``trans_fea`` may be corrupt; those days are skipped when
    ``skip_missing_raw`` is True (including :class:`~prog_trade_reg.feather_io.UnreadableRawFeatherError`).

    Returns
    -------
    dict
        Counts ``ok``, ``skipped_missing``, ``skipped_unreadable``, ``calendar_mode``.
    """
    ex = tuple(exchanges) if exchanges is not None else default_exchanges()
    pkl = trade_days_pkl or TRADE_DAYS_PKL
    counts = {"ok": 0, "skipped_missing": 0, "skipped_unreadable": 0, "calendar_mode": ""}
    days, mode = iter_dates_in_range(start, end, trade_days_pkl=pkl)
    counts["calendar_mode"] = mode
    total = len(days)
    tqdm.write(f"[outcomes] calendar={mode} n_dates={total} [{start}..{end}]")
    pbar = tqdm(days, desc="daily/outcomes", unit="day")
    for d in pbar:
        pbar.set_postfix_str(d, refresh=False)
        try:
            build_outcome_daily_for_date(
                d,
                exchanges=ex,
                write_aligned_trades=write_aligned_trades,
            )
            counts["ok"] += 1
        except FileNotFoundError as err:
            if not skip_missing_raw:
                raise
            counts["skipped_missing"] += 1
            tqdm.write(f"[outcomes] {d} skip (missing raw): {err}")
        except UnreadableRawFeatherError as err:
            if not skip_missing_raw:
                raise
            counts["skipped_unreadable"] += 1
            tqdm.write(f"[outcomes] {d} skip (unreadable trans_fea): {err}")
        except Exception as err:
            print(f"[outcomes] {d} FAIL: {err}", file=sys.stderr, flush=True)
            raise
    return counts

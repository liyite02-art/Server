"""Read Arrow/Feather files into Polars; tolerate mixed .fea encodings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl


class UnreadableRawFeatherError(OSError):
    """File exists but is neither readable Feather (pandas) nor Arrow IPC (polars)."""


def read_feather_to_polars(path: Path | str) -> pl.DataFrame:
    """
    Load a ``.fea`` file as Polars.

    Tries, in order:

    1. ``pandas.read_feather`` → ``polars.from_pandas`` (matches many lz4 Feather builds).
    2. ``polars.read_ipc`` (some paths ship Arrow IPC under ``.fea``).

    If both fail, raises :class:`UnreadableRawFeatherError` (corrupt, truncated, or non-Arrow payload).
    """
    p = Path(path)
    err_feather: BaseException | None = None
    err_ipc: BaseException | None = None
    try:
        pdf = pd.read_feather(p)
        return pl.from_pandas(pdf)
    except Exception as e:
        err_feather = e
    try:
        return pl.read_ipc(str(p), memory_map=False)
    except Exception as e:
        err_ipc = e
    raise UnreadableRawFeatherError(
        f"{p}: not readable as Feather or IPC (feather={err_feather!r}; ipc={err_ipc!r})"
    ) from err_ipc

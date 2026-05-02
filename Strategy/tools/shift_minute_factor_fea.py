"""
Shift saved minute-factor feather files by one trading day in place.

This is intended for factors produced by ``Strategy.factor.min_factor`` with
``shift_output=False``.  The default pattern is ``*_min.fea`` so daily factors
that are already shifted are not touched.

Examples
--------
Dry run:
    python -m Strategy.tools.shift_minute_factor_fea

Apply in place:
    python -m Strategy.tools.shift_minute_factor_fea --apply

Apply with backups:
    python -m Strategy.tools.shift_minute_factor_fea --apply --backup-dir Strategy/outputs/factors_backup_pre_min_shift
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if os.environ.get(_thread_env) in ("", "0"):
        os.environ[_thread_env] = "1"

import pandas as pd

from Strategy import config
from Strategy.data_io.saver import save_wide_table
from Strategy.utils.helpers import ensure_tradedate_as_index


MANIFEST_NAME = ".minute_factor_shift_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"created_at": _utc_now(), "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _iter_factor_files(factor_dir: Path, pattern: str, names: Iterable[str] | None) -> list[Path]:
    if names:
        files = []
        for name in names:
            p = factor_dir / (name if name.endswith(".fea") else f"{name}.fea")
            files.append(p)
        return sorted(files)
    return sorted(factor_dir.glob(pattern))


def shift_one_file(
    path: Path,
    *,
    apply: bool,
    backup_dir: Path | None,
    force: bool,
    manifest: dict,
) -> tuple[str, str]:
    if not path.exists():
        return path.name, "missing"
    if path.name in manifest.get("files", {}) and not force:
        return path.name, "already_shifted_manifest_skip"

    raw = pd.read_feather(path)
    wide = ensure_tradedate_as_index(raw).sort_index()
    before_shape = wide.shape
    shifted = wide.shift(1)

    if not apply:
        return path.name, f"dry_run shape={before_shape}"

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)

    save_wide_table(shifted, path)
    manifest.setdefault("files", {})[path.name] = {
        "shifted_at": _utc_now(),
        "rows": int(before_shape[0]),
        "cols": int(before_shape[1]),
        "backup_dir": str(backup_dir) if backup_dir is not None else None,
    }
    return path.name, f"shifted shape={before_shape}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shift saved minute factor .fea files by one trading day.",
    )
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=config.FACTOR_OUTPUT_DIR,
        help="Directory containing factor .fea files. Default: Strategy config FACTOR_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--pattern",
        default="*_min.fea",
        help="Glob pattern under factor-dir. Default: *_min.fea.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional explicit factor names or .fea filenames. Overrides --pattern.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually overwrite files. Without this flag the script only prints what it would do.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional directory to copy original .fea files before overwriting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow shifting files already recorded in the manifest. Use carefully.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factor_dir = args.factor_dir
    files = _iter_factor_files(factor_dir, args.pattern, args.names)
    manifest_path = factor_dir / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)

    if not files:
        print(f"No files matched in {factor_dir} with pattern={args.pattern!r}")
        return

    print(f"factor_dir={factor_dir}")
    print(f"files={len(files)} apply={args.apply} backup_dir={args.backup_dir} force={args.force}")

    changed = 0
    for path in files:
        name, status = shift_one_file(
            path,
            apply=args.apply,
            backup_dir=args.backup_dir,
            force=args.force,
            manifest=manifest,
        )
        print(f"{name}: {status}")
        if status.startswith("shifted"):
            changed += 1

    if args.apply:
        _write_manifest(manifest_path, manifest)
        print(f"Done. shifted_files={changed}, manifest={manifest_path}")
    else:
        print("Dry run only. Re-run with --apply to overwrite files.")


if __name__ == "__main__":
    main()

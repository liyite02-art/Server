"""
将 outputs/factors 下已落盘的因子 .fea 宽表裁剪为仅沪深 (上交所 + 深交所)，去掉北交所等。

规则（与 data_io/loader 及 utils/helpers 一致）:
- 6 位数字列: 只保留 0/3/6 开头 (主板、创业板、科创板等 A 股常见代码)，剔除 4/8 等北证及 8 开头等
- 列名以 NE 开头 (北交所部分数据源) 的列整列删除
- 股票列写回时按 6 位代码数值从小到大排列 (000001, 000002, …)，整表
  `df[[TRADE_DATE] + 有序列]]` 选取，避免行与值错位

默认处理目录: config.FACTOR_OUTPUT_DIR (Strategy/outputs/factors/)。

使用:
    cd /root/autodl-tmp
    PYTHONPATH=/root/autodl-tmp python -m Strategy.tools.filter_factors_sse_szse

    # 只演练不写入
    PYTHONPATH=/root/autodl-tmp python -m Strategy.tools.filter_factors_sse_szse --dry-run
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from Strategy import config
from Strategy.data_io.saver import save_wide_table
from Strategy.utils.helpers import is_sh_or_sz_by_num, standardize_stock_column

logger = logging.getLogger(__name__)

_TRADE_DATE = "TRADE_DATE"
_NUM6 = re.compile(r"^(\d{1,6})$")


def _should_keep_column(name: str) -> bool:
    """
    是否保留为因子列。TRADE_DATE 在调用方单独处理。
    - NE* : 北交所相关前缀，删除
    - 6 位纯数字: 用 is_sh_or_sz_by_num (0/3/6 保留，4/8 等剔除)
    """
    s = str(name).strip()
    if s.upper().startswith("NE"):
        return False
    m = _NUM6.fullmatch(s)
    if m is not None:
        code6 = s.zfill(6)
        return is_sh_or_sz_by_num(code6)
    return False


def _code_int_for_sort(name) -> int:
    """6 位数字列名 → 整型，保证 1 < 2 < … < 10 < … < 000001 的数值序。"""
    s = str(name).strip().zfill(6)
    if not s.isdigit():
        raise ValueError(
            f"无法按股票代码排序的非数字列: {name!r}（应已在筛选中排除）"
        )
    return int(s)


def _ordered_stock_columns(stock_cols: List) -> List:
    return sorted(stock_cols, key=_code_int_for_sort)


def _filter_one_feather(
    path: Path,
    dry_run: bool,
) -> Tuple[int, int, List[str], bool]:
    """
    返回: (原股票列数, 新股票列数, 被删列名, 是否发生列重排/排序)
    """
    df = pd.read_feather(path)
    if _TRADE_DATE not in df.columns:
        raise ValueError(
            f"{path}: 缺少 {_TRADE_DATE} 列，非标准因子 feather"
        )
    all_cols = list(df.columns)
    stock_cols = [c for c in all_cols if c != _TRADE_DATE]
    to_drop = [c for c in stock_cols if not _should_keep_column(c)]
    to_drop_set = set(to_drop)
    kept = [c for c in stock_cols if c not in to_drop_set]
    kept_ordered = _ordered_stock_columns(kept)
    orig_kept_in_file_order = [c for c in stock_cols if c not in to_drop_set]
    need_sort = orig_kept_in_file_order != kept_ordered
    n_old, n_new = len(stock_cols), len(kept)
    if n_new == 0 and n_old > 0:
        logger.warning("裁剪后无股票列: %s (原 %d 列)", path.name, n_old)

    need_write = bool(to_drop) or need_sort
    if not need_write:
        return n_old, n_new, [], False

    if dry_run:
        return n_old, n_new, to_drop, need_sort

    # 同时完成删列与排序：只选目标列、按 code 升序，行索引不变、数据与列一一对应
    df = df[[_TRADE_DATE] + kept_ordered]
    out = df.set_index(_TRADE_DATE)
    out.columns = standardize_stock_column(out.columns)
    save_wide_table(out, path)
    return n_old, n_new, to_drop, need_sort


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="因子 .fea 仅保留沪深列，删除北证 (8/4 位、NE 等)"
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=config.FACTOR_OUTPUT_DIR,
        help="因子目录 (默认 outputs/factors)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将要删除的列，不写回文件",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )
    d: Path = args.input_dir
    if not d.is_dir():
        logger.error("目录不存在: %s", d)
        return 1

    files = sorted(
        f for f in d.glob("*.fea")
        if f.is_file() and not f.name.endswith(".part")
    )
    if not files:
        logger.warning("未找到 .fea: %s", d)
        return 0

    total_drop = 0
    n_sorted = 0
    for path in files:
        try:
            n_old, n_new, dropped, need_sort = _filter_one_feather(
                path, args.dry_run
            )
        except Exception as e:
            logger.exception("处理失败: %s — %s", path, e)
            return 1
        n_drop = n_old - n_new
        total_drop += n_drop
        if need_sort and n_new > 0:
            n_sorted += 1
        if n_drop or need_sort or args.verbose:
            msg = f"{path.name}: 股票列 {n_old} -> {n_new} (删 {n_drop} 列)"
            if need_sort and n_new > 0:
                msg += f"; 已按 code 升序(000001起)"
            if args.verbose and dropped and len(dropped) <= 20:
                msg += f" 例: {dropped[:20]!r}"
            elif args.verbose and dropped:
                msg += f" 删列数={len(dropped)}"
            logger.info("%s", msg)
    logger.info(
        "完成: %d 个文件, 累计删除列(次) %d, 需重排/排序的 file 数 %d %s",
        len(files),
        total_drop,
        n_sorted,
        "(dry-run)" if args.dry_run else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

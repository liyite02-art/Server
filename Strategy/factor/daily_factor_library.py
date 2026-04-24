"""
daily_factor_library.py

将 Strategy/factor/daily_factors_raw.py 的全部因子批量接入框架。

使用方式::

    from Strategy import config
    from Strategy.factor.daily_factor_library import DailyFactorLibraryAdapter
    adapter = DailyFactorLibraryAdapter()
    saved = adapter.compute_and_save_all(
        start_date=config.TRAIN_START, end_date=config.TRAIN_END,
    )
    print(f"共保存 {len(saved)} 个因子")

输出: 每个因子保存为 FACTOR_OUTPUT_DIR/{factor_name}.fea
      宽表格式 (index=TRADE_DATE, columns=6位股票代码, values=因子值)

字段依赖 (Daily_data/ 下的 pkl 文件):

    必选: CLOSE_PRICE, OPEN_PRICE, HIGHEST_PRICE, LOWEST_PRICE,
          DEAL_AMOUNT, CHG_PCT
    可选: TURNOVER_RATE, MARKET_VALUE
    推荐: VOLUME.pkl (由 generate_volume.py 生成; 不存在则以 DEAL_AMOUNT/CLOSE_PRICE 近似)

防未来数据:
    compute_daily_factors_panel 按 groupby-code 向量化计算, 每行使用该行及之前的数据.
    本 Adapter 在保存前统一 shift(1), 确保 T 日的因子值只用到 T-1 日及更早的数据.
"""
from __future__ import annotations

import datetime as _dt
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from Strategy import config
from Strategy.data_io.saver import save_wide_table
from Strategy.factor.daily_factors_raw import compute_daily_factors_panel

logger = logging.getLogger(__name__)

_EPS = 1e-9

# 可接受的日期类型: 与 config 中 TRAIN_START 等一致
DateLike = Union[str, int, pd.Timestamp, _dt.date, _dt.datetime]


def _to_timestamp(d: DateLike) -> pd.Timestamp:
    if isinstance(d, pd.Timestamp):
        return d.normalize()
    if isinstance(d, int):
        s = f"{d:08d}"
        return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    return pd.Timestamp(d).normalize()


class DailyFactorLibraryAdapter:
    """将 daily_factors.compute_daily_factors_panel 批量接入框架的适配器。

    主要步骤:
    1. 从 Daily_data/ 读取所需宽表字段
    2. 转换为长表 (date, code, open, high, ...) 格式
    3. 调用 compute_daily_factors_panel 向量化计算所有因子
    4. 每个因子列 shift(1) 后, 分别保存为宽表 .fea 文件
    """

    # Daily_data 文件名 -> panel 列名 (compute_daily_factors_panel 期望的列名)
    FIELD_MAP: Dict[str, str] = {
        "CLOSE_PRICE":   "close",
        "OPEN_PRICE":    "open",
        "HIGHEST_PRICE": "high",
        "LOWEST_PRICE":  "low",
        "DEAL_AMOUNT":   "amount",
        "CHG_PCT":       "chg_pct",
        "TURNOVER_RATE": "turnover_rate",
        "MARKET_VALUE":  "market_value",
        "VOLUME":        "volume",          # 推荐: 先生成 VOLUME.pkl
    }

    # 必选字段 (缺少任一则抛出异常)
    REQUIRED_KEYS = frozenset({
        "CLOSE_PRICE", "OPEN_PRICE", "HIGHEST_PRICE",
        "LOWEST_PRICE", "DEAL_AMOUNT", "CHG_PCT",
    })

    def __init__(
        self,
        daily_data_dir: Optional[Path] = None,
        factor_output_dir: Optional[Path] = None,
    ):
        self.data_dir = daily_data_dir or config.DAILY_DATA_DIR
        self.out_dir = factor_output_dir or config.FACTOR_OUTPUT_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 内部辅助 ──────────────────────────────────────────────────────────

    def _load_wide(self, key: str, as_of_date=None) -> Optional[pd.DataFrame]:
        path = self.data_dir / f"{key}.pkl"
        if not path.exists():
            return None
        try:
            df = pd.read_pickle(path)
            df.index = pd.DatetimeIndex(df.index)
            df.columns = pd.Index([str(c).zfill(6) for c in df.columns])
            if as_of_date is not None:
                cutoff = pd.Timestamp(str(as_of_date))
                df = df.loc[df.index <= cutoff]
            return df
        except Exception as e:
            logger.warning("读取 %s.pkl 失败: %s", key, e)
            return None

    @staticmethod
    def _wide_to_long(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """将宽表字典向量化转换为长表 (date, code, col1, col2, ...)。"""
        ref = next(iter(tables.values()))
        n_dates, n_stocks = ref.shape

        dates_rep = np.repeat(ref.index.values, n_stocks)
        codes_rep = np.tile(ref.columns.values, n_dates)

        long = pd.DataFrame({"date": dates_rep, "code": codes_rep})
        for col, wide in tables.items():
            aligned = wide.reindex(index=ref.index, columns=ref.columns)
            long[col] = aligned.values.flatten()
        return long

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def compute_and_save_all(
        self,
        as_of_date=None,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        skip_existing: bool = False,
        chunk_size: int = 80,
        lookback: int = 120,
    ) -> Dict[str, Path]:
        """执行完整因子计算流水线, 保存所有因子宽表。

        为避免将全部交易日 × 200 因子一次性加载到内存导致 OOM,
        本方法采用"分块计算 → 临时 parquet → 逐因子合并"三阶段策略:

          阶段①  按 chunk_size 个**输出**交易日切分, 每块带 lookback 预热
          阶段②  每块结果写入临时压缩 parquet, 立即释放内存
          阶段③  逐因子读取全部临时文件 (每次仅 3 列), 拼接后 shift(1) 保存

        `start_date` / `end_date` 可限定**输出**区间, 不计算全历史。
        首块会向前多取 `lookback` 个交易日用于指标预热; 不指定时默认全样本。

        Parameters
        ----------
        as_of_date
            从 Daily_data 读入时的截止日期 (防未来), 如 "2024-12-31" 或 20241231.
        start_date, end_date : 可选
            仅计算该区间内的**输出**交易日, 可传 int(YYYYMMDD) / 字符串 / datetime.date
            与 `Strategy.config` 中 TRAIN_START, TRAIN_END 配合使用.
        skip_existing
            True 时跳过已存在的 .fea 文件, 方便增量更新.
        chunk_size : int
            每块含多少个**输出**交易日, 默认 80。调小可让 tqdm 更细 (更多步),
            但块数、临时文件数、开销会增加; 调大可减步数.
        lookback : int
            每块向前多取的交易日 (滚动指标), 默认 120.

        Returns
        -------
        dict : {factor_name: saved_path}

        Notes
        -----
        正确性: 分块时向前补 ``lookback`` 个交易日, ``compute_daily_factors_panel`` 在块内
        子长表上按 code×date 与全表相同的 groupby/rolling; 再裁剪到本块「输出日」;
        最后各因子列 ``shift(1)`` 防前视, 与全量一次计算在相同日期上的数值一致 (边界块含预热).

        进度: ① 各 pkl 字段一行; ② 总量=输出侧交易日数, 每块结束按天 advance (块内向量化期间 postfix 显示日期段);
        ③ 每列因子一行, postfix 为当前列名. 更细的 ② 步可减小 ``chunk_size``.
        临时 parquet 在 ``<factor_output_dir>/_tmp_factor_chunks/`` , 结束自动清理.
        """
        # ── 阶段① 加载宽表 (可按 as_of 截断) ───────────────────────
        logger.info(
            "流水线: ①加载 Daily_data 字段 → ②分块向量化(含 lookback) → ③逐因子落盘 .fea"
        )
        logger.info("加载日频宽表数据 from %s ...", self.data_dir)
        tables: Dict[str, pd.DataFrame] = {}
        for key, col_name in tqdm(self.FIELD_MAP.items(), desc="① 宽表字段(pkl)", unit="field"):
            wide = self._load_wide(key, as_of_date)
            if wide is None:
                if key in self.REQUIRED_KEYS:
                    raise FileNotFoundError(
                        f"必选字段 {key}.pkl 缺失 (路径: {self.data_dir})"
                    )
                logger.info("  可选字段 %s 未找到, 跳过", key)
                continue
            tables[col_name] = wide
            logger.info("  已加载 %-20s shape=%s", key, wide.shape)

        if "volume" not in tables:
            if "amount" in tables and "close" in tables:
                logger.warning(
                    "VOLUME.pkl 不存在 — 以 DEAL_AMOUNT / CLOSE_PRICE 近似 volume。\n"
                    "  建议: python -m Strategy.data_io.generate_volume"
                )
                close_safe = tables["close"].replace(0, np.nan)
                tables["volume"] = tables["amount"] / close_safe
            else:
                logger.warning("volume 字段缺失且无法近似, 量价类因子将为 NaN")
                tables["volume"] = tables["close"] * np.nan

        logger.info("阶段① 完成: 已就绪字段 %s", list(tables.keys()))

        ref = next(iter(tables.values()))
        full_dates = ref.index.sort_values()

        if start_date is not None or end_date is not None:
            ts0 = _to_timestamp(start_date) if start_date is not None else full_dates[0]
            ts1 = _to_timestamp(end_date) if end_date is not None else full_dates[-1]
            out_mask = (full_dates >= ts0) & (full_dates <= ts1)
            out_dates = full_dates[out_mask]
            if len(out_dates) == 0:
                raise ValueError(
                    f"在 [{ts0}, {ts1}] 与数据交集上无可用交易日, 请检查 start/end 或 as_of"
                )
            idx0 = int(full_dates.get_indexer([pd.Timestamp(out_dates[0])])[0])
            idx1 = int(full_dates.get_indexer([pd.Timestamp(out_dates[-1])])[0])
            if idx0 < 0 or idx1 < 0 or idx0 > idx1:
                raise ValueError("无法对齐 start/end 到本地上交易日索引")
            i_load0 = max(0, idx0 - lookback)
            load_slice = full_dates[i_load0 : idx1 + 1]
            tables = {c: t.reindex(load_slice) for c, t in tables.items()}
            ref = next(iter(tables.values()))
            full_dates = ref.index.sort_values()
            logger.info(
                "限定输出区间 [%s, %s] 共 %d 个交易日, 为预热额外加载至索引 %d (早 %d 天)",
                str(out_dates[0])[:10], str(out_dates[-1])[:10], len(out_dates),
                i_load0, idx0 - i_load0,
            )
        else:
            out_dates = full_dates
            logger.info("未指定 start/end, 使用全部 %d 个交易日为输出", len(out_dates))

        n_out = len(out_dates)
        chunk_starts = list(range(0, n_out, chunk_size))
        n_chunks = len(chunk_starts)
        logger.info(
            "阶段②: 输出 %d 个交易日, 分 %d 块向量化 (chunk_size=%d, lookback=%d); "
            "每块在含 lookback 的窗口上算完全部因子, 再裁剪到本块输出日, 与全样本一次向量化在重叠区间一致",
            n_out, n_chunks, chunk_size, lookback,
        )

        # ── 阶段② 逐块计算 → 写入临时 parquet ────────────────────────
        tmp_dir = self.out_dir / "_tmp_factor_chunks"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        factor_cols_ref: Optional[List[str]] = None
        chunk_paths: List[Path] = []

        try:
            with tqdm(
                total=n_out,
                desc="② 分块向量化(输出日进度)",
                unit="day",
            ) as pbar:
                for ci, cs in enumerate(chunk_starts):
                    ce = min(cs + chunk_size, n_out)
                    d0 = out_dates[cs]
                    d1 = out_dates[ce - 1]
                    i0 = int(full_dates.get_indexer([pd.Timestamp(d0)])[0])
                    i1 = int(full_dates.get_indexer([pd.Timestamp(d1)])[0])
                    if i0 < 0 or i1 < 0:
                        raise RuntimeError("交易日索引失败")
                    i_data0 = max(0, i0 - lookback)
                    chunk_dates = full_dates[i_data0 : i1 + 1]

                    pbar.set_postfix_str(
                        f"块{ci+1}/{n_chunks} {str(d0)[:10]}~{str(d1)[:10]} 计算中",
                        refresh=True,
                    )

                    chunk_tables = {col: wide.reindex(chunk_dates) for col, wide in tables.items()}
                    long_df = self._wide_to_long(chunk_tables)
                    long_df = long_df.sort_values(["code", "date"]).reset_index(drop=True)

                    result_df, factor_cols = compute_daily_factors_panel(long_df)
                    del long_df
                    gc.collect()

                    if factor_cols_ref is None:
                        factor_cols_ref = factor_cols

                    real_date_min, real_date_max = d0, d1
                    mask = (result_df["date"] >= real_date_min) & (result_df["date"] <= real_date_max)
                    trimmed = result_df.loc[mask, ["date", "code"] + factor_cols].copy()
                    del result_df
                    gc.collect()

                    chunk_path = tmp_dir / f"chunk_{ci:04d}.parquet"
                    trimmed.to_parquet(chunk_path, index=False, compression="snappy")
                    chunk_paths.append(chunk_path)
                    del trimmed
                    gc.collect()

                    pbar.update(ce - cs)
                    pbar.set_postfix_str(
                        f"块{ci+1}/{n_chunks} 已累加至 {pbar.n}/{n_out} 天",
                        refresh=True,
                    )
                    days_here = ce - cs
                    logger.info(
                        "  块 %d/%d: 输出日 %s ~ %s (本块 %d 天, 累计 %d/%d 输出日)",
                        ci + 1, n_chunks, str(d0)[:10], str(d1)[:10], days_here, pbar.n, n_out,
                    )

            # 宽表不再需要, 释放内存
            del tables
            gc.collect()

            # ── 阶段③ 逐因子读取所有块 → 合并 → 保存 ──────────────────
            if factor_cols_ref is None:
                raise RuntimeError("未能处理任何数据块, 请检查输入数据")

            logger.info("阶段③: 共 %d 个因子列 → 合并各块并 shift(1) 写入 .fea", len(factor_cols_ref))
            saved: Dict[str, Path] = {}
            with tqdm(
                factor_cols_ref,
                desc="③ 落盘因子列",
                unit="col",
            ) as pbar3:
                for fname in pbar3:
                    pbar3.set_postfix_str(fname[:36] + ("…" if len(fname) > 36 else ""), refresh=True)
                    out_path = self.out_dir / f"{fname}.fea"
                    if skip_existing and out_path.exists():
                        saved[fname] = out_path
                        continue
                    try:
                        # 每次只读 date / code / 单因子列, 内存开销极小
                        wides: List[pd.DataFrame] = []
                        for cp in chunk_paths:
                            df_chunk = pd.read_parquet(cp, columns=["date", "code", fname])
                            wide = df_chunk.pivot_table(
                                index="date", columns="code", values=fname, aggfunc="first"
                            )
                            wide.index = pd.DatetimeIndex(wide.index)
                            wide.index.name = "TRADE_DATE"
                            wide.columns = pd.Index([str(c).zfill(6) for c in wide.columns])
                            wides.append(wide)

                        combined = pd.concat(wides).sort_index()
                        # ⚠️ shift(1): T 日存储的因子值仅用 T-1 日数据
                        combined = combined.shift(1)
                        path = save_wide_table(combined, out_path)
                        saved[fname] = path
                    except Exception as e:
                        logger.warning("保存因子 [%s] 失败: %s", fname, e)

        finally:
            # 无论成功或失败, 清理临时文件
            for cp in chunk_paths:
                try:
                    cp.unlink()
                except OSError:
                    pass
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

        logger.info(
            "DailyFactorLibrary 完成: %d / %d 个因子 -> %s",
            len(saved), len(factor_cols_ref or []), self.out_dir,
        )
        return saved

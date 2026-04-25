from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from Strategy import config
from Strategy.backtest.quick_backtest import (
    _load_daily_wide,
    build_tradeable_mask,
    topk_equal_weight_returns,
)
from Strategy.backtest.universe import load_ipo_dates, load_out_dates, load_st_status
from Strategy.label.label_generator import load_label, load_price
from Strategy.model.scorer import load_scores


LOG_PATH = "/root/autodl-tmp/.cursor/debug-da6e13.log"
RUN_ID = "quick-diagnostic-restore-variants"


def log(hypothesis_id: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": "da6e13",
        "runId": RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": "Strategy/tmp_quick_backtest_diagnostic.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    # endregion


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TRADE_DATE" in out.columns:
        out = out.set_index("TRADE_DATE")
    out.index = pd.DatetimeIndex(out.index)
    out.columns = pd.Index([str(c).zfill(6) for c in out.columns])
    return out


def load_raw_scores() -> pd.DataFrame:
    path = config.SCORE_OUTPUT_DIR / "SCORE_xgb_TWAP_1430_1457.fea"
    return standardize(pd.read_feather(path))


def make_raw_label(price: pd.DataFrame) -> pd.DataFrame:
    label = price.shift(-1) / price - 1
    label.index.name = "TRADE_DATE"
    return label


def perf(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {
            "n_days": 0,
            "simple_ann": np.nan,
            "compound_ann": np.nan,
            "mean_daily": np.nan,
            "total_ret": np.nan,
        }
    total_ret = (1 + ret).prod() - 1
    return {
        "n_days": int(len(ret)),
        "simple_ann": float(ret.mean() * 242),
        "compound_ann": float((1 + total_ret) ** (242 / max(len(ret), 1)) - 1),
        "mean_daily": float(ret.mean()),
        "total_ret": float(total_ret),
    }


def summarize_scenario(
    name: str,
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    tradeable_mask: pd.DataFrame,
    top_ks: tuple[int, ...] = (20, 50, 100),
) -> tuple[list[dict], pd.DataFrame]:
    ret_df, diag = topk_equal_weight_returns(
        score_df,
        label_df,
        ks=top_ks,
        start_date=config.VAL_START,
        tradeable_mask=tradeable_mask,
        return_diagnostics=True,
    )
    rows = []
    for k in top_ks:
        col = f"top{k}"
        p = perf(ret_df[col]) if col in ret_df else perf(pd.Series(dtype=float))
        rows.append(
            {
                "scenario": name,
                "topk": k,
                **p,
                "avg_selected": float(diag.get(f"{col}_selected", pd.Series(dtype=float)).mean()),
                "avg_label_missing": float(diag.get(f"{col}_label_missing", pd.Series(dtype=float)).mean()),
                "avg_tradeable": float(tradeable_mask.reindex(ret_df.index).sum(axis=1).mean())
                if len(ret_df)
                else np.nan,
            }
        )
    return rows, ret_df


def main() -> None:
    score_loaded = load_scores("xgb", "TWAP_1430_1457")
    score_raw = load_raw_scores()
    label_adj = standardize(load_label("TWAP_1430_1457"))
    price = standardize(load_price("TWAP_1430_1457"))
    label_raw = make_raw_label(price)

    limit_up = _load_daily_wide("LIMIT_UP_PRICE")
    ipo_dates = load_ipo_dates()
    out_dates = load_out_dates()
    st_status = load_st_status()
    current_mask, current_report = build_tradeable_mask(
        score_loaded,
        price_df=price,
        limit_up_df=limit_up,
        ipo_dates=ipo_dates,
        out_dates=out_dates,
        st_status=st_status,
    )
    current_no_limit_mask, current_no_limit_report = build_tradeable_mask(
        score_loaded,
        price_df=price,
        limit_up_df=limit_up,
        ipo_dates=ipo_dates,
        out_dates=out_dates,
        st_status=st_status,
        exclude_limit_up=False,
    )
    raw_score_current_mask, raw_score_current_report = build_tradeable_mask(
        score_raw,
        price_df=price,
        limit_up_df=limit_up,
        ipo_dates=ipo_dates,
        out_dates=out_dates,
        st_status=st_status,
    )
    no_hist_st_mask, no_hist_st_report = build_tradeable_mask(
        score_loaded,
        price_df=price,
        limit_up_df=limit_up,
        ipo_dates=ipo_dates,
        out_dates=out_dates,
        st_status=st_status,
        exclude_historical_st=False,
    )
    no_st_mask, no_st_report = build_tradeable_mask(
        score_loaded,
        price_df=price,
        limit_up_df=limit_up,
        ipo_dates=ipo_dates,
        out_dates=out_dates,
        st_status=st_status,
        exclude_st=False,
    )
    no_prefix_mask, no_prefix_report = build_tradeable_mask(
        score_loaded,
        price_df=price,
        limit_up_df=limit_up,
        ipo_dates=ipo_dates,
        out_dates=out_dates,
        st_status=st_status,
        excluded_prefixes=(),
    )
    loose_mask, loose_report = build_tradeable_mask(
        score_loaded,
        price_df=price,
        limit_up_df=limit_up,
        out_dates=None,
        st_status=None,
        min_listing_days=0,
        delist_buffer_days=0,
        exclude_st=False,
        excluded_prefixes=(),
    )

    common_idx = score_raw.index.intersection(label_adj.index)
    common_cols = score_raw.columns.intersection(label_adj.columns)
    legacy_label_nonnull_mask = (
        score_raw.reindex(index=common_idx, columns=common_cols).notna()
        & label_adj.reindex(index=common_idx, columns=common_cols).notna()
    )
    score_valid_mask = score_raw.reindex(index=common_idx, columns=common_cols).notna()
    common_idx_raw = score_raw.index.intersection(label_raw.index)
    common_cols_raw = score_raw.columns.intersection(label_raw.columns)
    legacy_raw_label_mask = (
        score_raw.reindex(index=common_idx_raw, columns=common_cols_raw).notna()
        & label_raw.reindex(index=common_idx_raw, columns=common_cols_raw).notna()
    )
    score_valid_raw_label_mask = score_raw.reindex(index=common_idx_raw, columns=common_cols_raw).notna()

    scenarios = [
        ("current_adj_label_loaded_score", score_loaded, label_adj, current_mask),
        ("current_no_limit_up_adj_label", score_loaded, label_adj, current_no_limit_mask),
        ("raw_score_current_filters_adj_label", score_raw, label_adj, raw_score_current_mask),
        ("no_historical_st_adj_label", score_loaded, label_adj, no_hist_st_mask),
        ("no_st_filter_adj_label", score_loaded, label_adj, no_st_mask),
        ("no_prefix_filter_adj_label", score_loaded, label_adj, no_prefix_mask),
        ("loose_tday_price_only_adj_label", score_loaded, label_adj, loose_mask),
        ("score_valid_only_adj_label", score_raw, label_adj, score_valid_mask),
        ("legacy_label_nonnull_adj_label", score_raw, label_adj, legacy_label_nonnull_mask),
        ("score_valid_only_raw_label", score_raw, label_raw, score_valid_raw_label_mask),
        ("legacy_label_nonnull_raw_label", score_raw, label_raw, legacy_raw_label_mask),
    ]

    all_rows = []
    ret_summary = {}
    for name, score, label, mask in scenarios:
        rows, ret_df = summarize_scenario(name, score, label, mask)
        all_rows.extend(rows)
        ret_summary[name] = {
            "ret_rows": int(len(ret_df)),
            "first_date": str(ret_df.index.min().date()) if len(ret_df) else None,
            "last_date": str(ret_df.index.max().date()) if len(ret_df) else None,
        }

    summary = pd.DataFrame(all_rows)
    out_path = config.BT_RESULT_DIR / "quick_backtest_diagnostic_variants.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    report_means = {
        "current": current_report.mean(numeric_only=True).to_dict(),
        "current_no_limit": current_no_limit_report.mean(numeric_only=True).to_dict(),
        "raw_score_current": raw_score_current_report.mean(numeric_only=True).to_dict(),
        "no_hist_st": no_hist_st_report.mean(numeric_only=True).to_dict(),
        "no_st": no_st_report.mean(numeric_only=True).to_dict(),
        "no_prefix": no_prefix_report.mean(numeric_only=True).to_dict(),
        "loose": loose_report.mean(numeric_only=True).to_dict(),
    }
    label_diff = (label_adj - label_raw.reindex_like(label_adj)).stack().replace([np.inf, -np.inf], np.nan).dropna()
    log(
        "H1,H2,H3,H4,H5",
        "quick backtest variant diagnostics completed",
        {
            "summary_path": str(out_path),
            "summary": summary.to_dict("records"),
            "report_means": report_means,
            "ret_summary": ret_summary,
            "score_loaded_nonnull": int(score_loaded.notna().sum().sum()),
            "score_raw_nonnull": int(score_raw.notna().sum().sum()),
            "score_price_mask_removed": int(score_raw.notna().sum().sum() - score_loaded.notna().sum().sum()),
            "label_adj_nonnull": int(label_adj.notna().sum().sum()),
            "label_raw_nonnull": int(label_raw.notna().sum().sum()),
            "label_adj_minus_raw_abs_p99": float(label_diff.abs().quantile(0.99)) if len(label_diff) else None,
            "label_adj_minus_raw_abs_max": float(label_diff.abs().max()) if len(label_diff) else None,
        },
    )
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

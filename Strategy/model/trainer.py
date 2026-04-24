"""
模型训练模块: 将因子宽表 + Label 宽表拼接为 Panel, 训练 XGBoost / MLP。

⚠️ 防未来数据:
- 训练集/验证集/测试集严格按时间划分 (见 config.py)
- 模型超参只能使用训练集和验证集调优, 严禁使用 OOS 区间
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from Strategy import config
from Strategy.utils.helpers import ensure_tradedate_as_index

logger = logging.getLogger(__name__)


def _stock_id_to_str6(x) -> str:
    """与宽表 6 位 code 列一致; 处理 int/float/字符串。"""
    if pd.isna(x):
        return ""
    try:
        v = int(float(x))
        return f"{v:06d}"
    except (ValueError, TypeError, OverflowError):
        s = str(x).strip()
        return s.zfill(6) if s.isdigit() else s


def _normalize_xgb_params(params: dict) -> dict:
    """
    XGBoost 2.x（CPU 版）不再接受 tree_method='gpu_hist'；GPU 时通常改为
    tree_method='hist' + device='cuda'（需安装 GPU 构建）。此处将旧参数自动回落为 hist。
    """
    p = dict(params)
    tm = p.get("tree_method")
    if tm in ("gpu_hist", "GPU_HIST"):
        logger.warning(
            "tree_method=%r 当前 xgboost 构建不支持，已改为 'hist'（CPU 直方图）。"
            "若已安装 GPU 版 XGBoost，可显式设置 tree_method='hist', device='cuda'。",
            tm,
        )
        p["tree_method"] = "hist"
    return p


def _align_panel_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一 merge 键类型。feather/宽表重载后，TRADE_DATE 常出现一侧 datetime64 一侧
    str/object，merge 会报错；这里统一为 datetime64[ns]。
    StockID 统一为 6 位字符串，避免 int 与 '000001' 或 1.0 与 '000001' 对不齐。
    """
    out = df.copy()
    if "TRADE_DATE" in out.columns:
        out["TRADE_DATE"] = pd.to_datetime(out["TRADE_DATE"], errors="coerce")
    if "StockID" in out.columns:
        out["StockID"] = out["StockID"].map(_stock_id_to_str6)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 数据准备: 宽表 -> 长表 Panel
# ═══════════════════════════════════════════════════════════════════════
def build_panel(
    factor_dict: Dict[str, pd.DataFrame],
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    将多个因子宽表 + Label 宽表对齐并展平为长表。

    高效路径: 先对公共 (日期 × 股票) 做一次 stack 得到 MultiIndex Series,
    再 concat 所有因子 + label, 最后 reset_index。避免 N 次大 merge 导致
    O(N × n_rows) 的性能问题。

    Returns
    -------
    pd.DataFrame
        columns: ['TRADE_DATE', 'StockID', factor1, factor2, ..., 'label']
        按 (TRADE_DATE, StockID) 对齐
    """
    label_df = ensure_tradedate_as_index(label_df)
    factor_dict = {k: ensure_tradedate_as_index(v) for k, v in factor_dict.items()}

    # ── 1. 计算公共交易日 & 股票 ───────────────────────────────────────
    common_dates = label_df.index
    for fdf in factor_dict.values():
        common_dates = common_dates.intersection(fdf.index)
    common_dates = common_dates.sort_values()

    common_stocks = label_df.columns
    for fdf in factor_dict.values():
        common_stocks = common_stocks.intersection(fdf.columns)
    common_stocks = common_stocks.sort_values()

    if len(common_dates) == 0 or len(common_stocks) == 0:
        raise ValueError(
            f"无公共交易日或股票: dates={len(common_dates)} stocks={len(common_stocks)}.\n"
            "可能原因: 因子行索引类型与 label 不一致 (已在 ensure_tradedate_as_index 中修复);\n"
            "或因子与 label 的股票列名不匹配 (检查是否 6 位数字字符串)。"
        )

    logger.info(
        "Panel 对齐: %d 交易日 × %d 只股票, %d 个因子",
        len(common_dates), len(common_stocks), len(factor_dict),
    )

    # ── 2. 一次性 stack: label ─────────────────────────────────────────
    # future_warning: stack(future_stack=True) in pandas 2.x
    label_sub = label_df.loc[common_dates, common_stocks]
    label_series = label_sub.stack(future_stack=True).rename("label")

    # ── 3. 所有因子 stack → concat (比 N 次 merge 快 10-100×) ─────────
    factor_series: List[pd.Series] = [label_series]
    for fname, fdf in factor_dict.items():
        s = fdf.loc[common_dates, common_stocks].stack(future_stack=True).rename(fname)
        factor_series.append(s)

    logger.info("正在 concat %d 个 Series ...", len(factor_series))
    panel_wide = pd.concat(factor_series, axis=1)

    # ── 4. 展平为长表 ──────────────────────────────────────────────────
    panel = panel_wide.reset_index()
    panel.columns = ["TRADE_DATE", "StockID"] + list(panel.columns[2:])

    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"], errors="coerce")
    panel["StockID"] = panel["StockID"].map(_stock_id_to_str6)

    logger.info("Panel 构建完成: shape=%s", panel.shape)
    return panel


def split_panel(
    panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按 config 中的时间划分拆分为训练集/验证集/OOS 测试集。

    ⚠️ OOS 测试集仅用于最终评估, 严禁用于任何参数决策!
    """
    if len(panel) == 0:
        raise ValueError(
            "split_panel: panel 为 0 行, 无 TRADE_DATE 可切分。请检查 build_panel 是否因 "
            "无公共交易日/股票而失败(见 train() 中 ValueError) 或因子未加载成功。"
        )
    dates = pd.to_datetime(panel["TRADE_DATE"], errors="coerce")
    if dates.isna().any():
        n_bad = int(dates.isna().sum())
        raise ValueError(f"panel 中 TRADE_DATE 有 {n_bad} 个无法解析为日期的行")
    if pd.api.types.is_datetime64tz_dtype(dates):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    dates = dates.dt.normalize()

    t0, t1 = (
        pd.Timestamp(config.TRAIN_START).normalize(),
        pd.Timestamp(config.TRAIN_END).normalize(),
    )
    v0, v1 = (
        pd.Timestamp(config.VAL_START).normalize(),
        pd.Timestamp(config.VAL_END).normalize(),
    )
    o0 = pd.Timestamp(config.OOS_START).normalize()

    dmin, dmax = dates.min(), dates.max()
    train_mask = (dates >= t0) & (dates <= t1)
    val_mask = (dates >= v0) & (dates <= v1)
    oos_mask = dates >= o0

    train = panel.loc[train_mask].copy()
    val = panel.loc[val_mask].copy()
    oos = panel.loc[oos_mask].copy()

    logger.info("panel 日期: [%s, %s]", dmin.date(), dmax.date())
    logger.info("Train: %d rows, Val: %d rows, OOS: %d rows", len(train), len(val), len(oos))
    if len(train) == 0:
        raise ValueError(
            f"训练集 0 行: panel 日期在 [{dmin.date()}, {dmax.date()}], "
            f"与 config 训练期 [{config.TRAIN_START}, {config.TRAIN_END}] 无交叠。\n"
            f"  若实际数据起始于 {dmin.date()} 之后，请把 TRAIN_END 调晚(或 TRAIN_START 调早)；"
            f"  或从因子/label 起算日覆盖训练所需区间(见 config 注释)。"
        )
    if len(val) == 0:
        raise ValueError(
            f"验证集 0 行: 请检查 VAL_START/VAL_END 与 panel 日期的交叠(当前 panel [{dmin.date()}, {dmax.date()}])"
        )
    return train, val, oos


# ═══════════════════════════════════════════════════════════════════════
# XGBoost 训练器
# ═══════════════════════════════════════════════════════════════════════
class XGBTrainer:
    """
    XGBoost 横截面训练器。

    Parameters
    ----------
    params : dict
        XGBoost 超参数
    num_boost_round : int
        迭代轮数
    early_stopping_rounds : int
        验证集早停轮数
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ):
        self.params = _normalize_xgb_params(
            params
            or {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 50,
                "tree_method": "hist",
                "verbosity": 0,
            }
        )
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names: List[str] = []

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """自动识别特征列 (排除 TRADE_DATE, StockID, label)"""
        exclude = {"TRADE_DATE", "StockID", "label"}
        return [c for c in df.columns if c not in exclude]

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "XGBTrainer":
        """
        训练 XGBoost 模型。

        Parameters
        ----------
        train_df : pd.DataFrame
            训练集 Panel (含 label 列)
        val_df : pd.DataFrame, optional
            验证集 Panel (用于早停)
        """
        import xgboost as xgb

        self.feature_names = self._get_feature_cols(train_df)
        logger.info("特征数=%d, 前若干列名(若有): %s", len(self.feature_names), self.feature_names[:8])
        if not self.feature_names:
            raise ValueError("无特征列: 请确认 panel 中除 TRADE_DATE/StockID/label 外有因子列")

        train_clean = train_df.dropna(subset=["label"])
        y_train = train_clean["label"].values.astype(np.float64)
        if y_train.size == 0:
            raise ValueError(
                "训练集在 dropna(subset=[label]) 后 0 行: 请检查 label 与 TRAIN 日期交叠"
            )
        if not np.isfinite(y_train).all():
            raise ValueError(
                "训练集 label 含 inf/NaN，请先检查 LABEL 与 merge 结果"
            )
        if float(np.std(y_train)) < 1e-12:
            raise ValueError(
                "训练集 label 为常数 (std<1e-12)；无法学习且会显示 train/val-rmse:0.00000。"
                "请检查 LABEL .fea 是否为真实 forward 收益、以及 build_panel 中因子列是否与 label 行对齐（特征全 NaN 时也可能得到虚假 RMSE=0）。"
            )

        X_train = (
            train_clean[self.feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )
        x_nan = float(np.isnan(X_train).mean())
        logger.info(
            "训练样本 n=%d; label: min=%.6g med=%.6g max=%.6g std=%.6g; 特征全表 NaN 占比=%.2f%%",
            len(y_train),
            float(np.nanmin(y_train)),
            float(np.nanmedian(y_train)),
            float(np.nanmax(y_train)),
            float(np.nanstd(y_train)),
            100.0 * x_nan,
        )
        if x_nan > 0.99:
            logger.warning(
                "特征全表 NaN 占比>99%%: 与因子/label 的日期或股票代码列 merge 可能未对齐, 将难以训练出有效树"
            )

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, "train")]
        if val_df is not None:
            val_clean = val_df.dropna(subset=["label"])
            y_val = val_clean["label"].values.astype(np.float64)
            if y_val.size == 0:
                raise ValueError("验证集在 dropna(subset=[label]) 后 0 行")
            X_val = (
                val_clean[self.feature_names]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64)
            )
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, "val"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if val_df is not None else None,
            verbose_eval=100,
        )

        logger.info(
            "训练完成. best_iteration=%s",
            getattr(self.model, "best_iteration", "N/A"),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """对 Panel 数据生成预测值"""
        import xgboost as xgb

        if self.model is None:
            raise RuntimeError("模型未训练, 请先调用 train()")
        X = df[self.feature_names].values
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmat)

    def save_model(self, path: Optional[Path] = None) -> Path:
        """保存模型"""
        path = path or (config.SCORE_OUTPUT_DIR / "xgb_model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)
        logger.info("模型已保存: %s", path)
        return path

    def load_model(self, path: Optional[Path] = None) -> "XGBTrainer":
        """加载模型"""
        path = path or (config.SCORE_OUTPUT_DIR / "xgb_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        logger.info("模型已加载: %s", path)
        return self

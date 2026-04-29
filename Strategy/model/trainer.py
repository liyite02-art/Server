"""
模型训练模块: 将因子宽表 + Label 宽表拼接为 Panel, 训练 XGBoost。

样本划分准则 (对应 strategy_rules.md):
  IS Train Set: config.IS_TRAIN_START ~ config.IS_TRAIN_END
    → 供 Rolling Val CV 内部使用 (train + val 均来自此段)
  IS Test Set:  config.IS_TEST_START ~ config.IS_TEST_END
    → 不参与梯度更新; 由 4-Fold Ensemble 推理评估泛化效果
  OOS:          config.OOS_START 之后
    → 严禁用于任何参数决策; 仅当 IS Test 达标后才可解锁

⚠️ 防未来数据:
  - 模型超参只能通过 IS Train CV 调优, 严禁在 IS Test / OOS 上调参
  - split_panel() 返回 (is_train, is_test, oos) 三元组
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


# ═══════════════════════════════════════════════════════════════════════
# 内部工具函数
# ═══════════════════════════════════════════════════════════════════════

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
    XGBoost 2.x（CPU 版）不再接受 tree_method='gpu_hist'。
    自动回落为 hist; GPU 版请显式设置 device='cuda'。
    """
    p = dict(params)
    tm = p.get("tree_method")
    if tm in ("gpu_hist", "GPU_HIST"):
        logger.warning(
            "tree_method=%r 当前构建不支持，已改为 'hist'。"
            "若已安装 GPU 版 XGBoost，可设置 tree_method='hist', device='cuda'。",
            tm,
        )
        p["tree_method"] = "hist"
    return p


def _winsorize_label_cross_section(
    panel: pd.DataFrame,
    sigma: float = 3.0,
) -> pd.DataFrame:
    """
    对 label 做截面 Winsorize: 每日截面内 clip 到 [μ - σ·sigma, μ + σ·sigma]。
    减少极端收益对 WPCC 梯度的干扰。
    """
    if sigma <= 0:
        return panel
    out = panel.copy()
    dates = pd.to_datetime(out["TRADE_DATE"])
    for dt_val in dates.unique():
        mask = dates == dt_val
        y = out.loc[mask, "label"]
        mu, s = y.mean(), y.std()
        if s > 1e-12:
            out.loc[mask, "label"] = y.clip(mu - sigma * s, mu + sigma * s)
    n_clipped = int((out["label"] != panel["label"]).sum())
    if n_clipped > 0:
        logger.info(
            "Label Winsorize (%.1fσ): %d 个样本被截断 (%.2f%%)",
            sigma, n_clipped, 100 * n_clipped / max(len(panel), 1),
        )
    return out


def _make_wpcc_objective(date_ids: np.ndarray):
    r"""
    为 XGBoost 生成 WPCC 自定义目标函数。

    数学:
      PCC_t = Cov(\hat{y}, y) / (σ_{\hat{y}} · σ_y)
      grad_j = -(1/(n·σ_\hat{y}·σ_y)) · [e_j - PCC · (σ_y/σ_\hat{y}) · d_j]
      hess_j ≈ 1/(n · σ_\hat{y} · σ_y)  (Fisher scoring)
    """
    unique_dates = np.unique(date_ids)
    date_slices = {d: np.where(date_ids == d)[0] for d in unique_dates}

    def wpcc_obj(predt, dtrain):
        labels = dtrain.get_label()
        grad = np.zeros_like(predt)
        hess = np.full_like(predt, 1e-6)

        for idx in date_slices.values():
            if len(idx) < 10:
                continue
            yh = predt[idx]
            y  = labels[idx]
            n  = len(idx)

            d = yh - yh.mean()
            e = y  - y.mean()
            s_yh = np.sqrt(np.mean(d ** 2)) + 1e-8
            s_y  = np.sqrt(np.mean(e ** 2)) + 1e-8
            pcc  = np.mean(d * e) / (s_yh * s_y)

            coeff = 1.0 / (n * s_yh * s_y)
            grad[idx] = -coeff * (e - pcc * (s_y / s_yh) * d)
            hess[idx] = coeff

        return grad, hess

    return wpcc_obj


def _make_wpcc_eval(date_ids: np.ndarray):
    """为 XGBoost 生成 WPCC 评估函数 (输出截面平均 IC)。"""
    unique_dates = np.unique(date_ids)
    date_slices = {d: np.where(date_ids == d)[0] for d in unique_dates}

    def wpcc_eval(predt, dtrain):
        labels = dtrain.get_label()
        corrs = []
        for idx in date_slices.values():
            if len(idx) < 10:
                continue
            yh = predt[idx]
            y  = labels[idx]
            d = yh - yh.mean()
            e = y  - y.mean()
            s_yh = np.sqrt(np.mean(d ** 2))
            s_y  = np.sqrt(np.mean(e ** 2))
            if s_yh < 1e-10 or s_y < 1e-10:
                continue
            corrs.append(np.mean(d * e) / (s_yh * s_y))
        return "wpcc", float(-np.mean(corrs)) if corrs else 0.0

    return wpcc_eval


def _make_wpcc_eval_train_val(
    dtrain, dval,
    train_date_ids: np.ndarray,
    val_date_ids: np.ndarray,
):
    """
    XGBoost 对 evals 里每个 DMatrix 调用同一个 custom_metric；
    必须按各自行的日期分组计算 WPCC。
    """
    ts = {d: np.where(train_date_ids == d)[0] for d in np.unique(train_date_ids)}
    vs = {d: np.where(val_date_ids == d)[0] for d in np.unique(val_date_ids)}

    def _mean_wpcc(predt, labels, slices: dict) -> float:
        corrs = []
        for idx in slices.values():
            if len(idx) < 10:
                continue
            yh = predt[idx]
            y = labels[idx]
            d = yh - yh.mean()
            e = y - y.mean()
            s_yh = np.sqrt(np.mean(d ** 2))
            s_y = np.sqrt(np.mean(e ** 2))
            if s_yh < 1e-10 or s_y < 1e-10:
                continue
            corrs.append(np.mean(d * e) / (s_yh * s_y))
        return float(-np.mean(corrs)) if corrs else 0.0

    def wpcc_eval(predt, dmat):
        labels = dmat.get_label()
        if dmat is dtrain:
            return "wpcc", _mean_wpcc(predt, labels, ts)
        if dval is not None and dmat is dval:
            return "wpcc", _mean_wpcc(predt, labels, vs)
        return "wpcc", 0.0

    return wpcc_eval


# ═══════════════════════════════════════════════════════════════════════
# 数据准备: 宽表 → 长表 Panel
# ═══════════════════════════════════════════════════════════════════════

def build_panel(
    factor_dict: Dict[str, pd.DataFrame],
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    将多个因子宽表 + Label 宽表对齐并展平为长表。

    高效路径: 先对公共 (日期 × 股票) 做一次 stack 得到 MultiIndex Series,
    再 concat 所有因子 + label, 最后 reset_index。

    Returns
    -------
    pd.DataFrame
        columns: ['TRADE_DATE', 'StockID', factor1, factor2, ..., 'label']
        按 (TRADE_DATE, StockID) 对齐
    """
    label_df = ensure_tradedate_as_index(label_df)
    factor_dict = {k: ensure_tradedate_as_index(v) for k, v in factor_dict.items()}

    # ── 1. 计算公共交易日 & 股票 ──────────────────────────────────────
    common_dates = label_df.index
    for fdf in factor_dict.values():
        common_dates = common_dates.intersection(fdf.index)
    common_dates = common_dates.sort_values()

    common_stocks = label_df.columns
    for fdf in factor_dict.values():
        common_stocks = common_stocks.intersection(fdf.columns)
    common_stocks = common_stocks.sort_values()

    if len(common_dates) == 0 or len(common_stocks) == 0:
        hints = []
        if len(label_df.columns) == 0:
            hints.append(
                "当前 Label 宽表 **无任何股票列**（常见于 LABEL_OPEN930_1000.fea 损坏或未正确生成："
                "需 `min_data` 分钟线在 09:30/10:00 有 open）。请删除 outputs/labels 下对应 LABEL 后重新运行 "
                "`generate_and_save_open930_1000_label()`。"
            )
        elif len(common_stocks) == 0 and len(factor_dict) > 0:
            f0 = next(iter(factor_dict.values()))
            hints.append(
                f"Label 有 {len(label_df.columns)} 只股票列，但与因子无交集；"
                f"请确认因子目录与 Label 为同一套股票代码口径（已自动 zfill(6)）。"
                f"示例 label 列={list(label_df.columns[:3])} 示例因子列={list(f0.columns[:3])}"
            )
        raise ValueError(
            f"无公共交易日或股票: dates={len(common_dates)} stocks={len(common_stocks)}.\n"
            + ("\n".join(hints) if hints else "")
            + "\n其他可能原因: 因子行索引与 label 日期类型不一致。"
        )

    logger.info(
        "Panel 对齐: %d 交易日 × %d 只股票, %d 个因子",
        len(common_dates), len(common_stocks), len(factor_dict),
    )

    # ── 2. 一次性 stack: label ────────────────────────────────────────
    label_sub = label_df.loc[common_dates, common_stocks]
    label_series = label_sub.stack(future_stack=True).rename("label")

    # ── 3. 所有因子 stack → concat ────────────────────────────────────
    factor_series: List[pd.Series] = [label_series]
    for fname, fdf in factor_dict.items():
        s = fdf.loc[common_dates, common_stocks].stack(future_stack=True).rename(fname)
        factor_series.append(s)

    logger.info("正在 concat %d 个 Series ...", len(factor_series))
    panel_wide = pd.concat(factor_series, axis=1)

    # ── 4. 展平为长表 ─────────────────────────────────────────────────
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
    按 config 中的时间划分拆分为 IS 训练集 / IS 测试集 / OOS 集。

    Returns
    -------
    is_train : pd.DataFrame
        IS 训练集 [IS_TRAIN_START, IS_TRAIN_END]，供 Rolling Val CV 使用
    is_test : pd.DataFrame
        IS 测试集 [IS_TEST_START, IS_TEST_END]，不参与梯度，仅评估泛化
    oos : pd.DataFrame
        OOS 集 [OOS_START, ∞)，严禁用于任何参数决策

    ⚠️ OOS 仅当 IS 测试集表现达标后才可解锁使用!
    """
    if len(panel) == 0:
        raise ValueError(
            "split_panel: panel 为 0 行，无 TRADE_DATE 可切分。"
            "请检查 build_panel 是否因无公共交易日/股票而失败。"
        )

    dates = pd.to_datetime(panel["TRADE_DATE"], errors="coerce")
    if dates.isna().any():
        n_bad = int(dates.isna().sum())
        raise ValueError(f"panel 中 TRADE_DATE 有 {n_bad} 个无法解析为日期的行")
    if pd.api.types.is_datetime64tz_dtype(dates):
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    dates = dates.dt.normalize()

    it0 = pd.Timestamp(config.IS_TRAIN_START).normalize()
    it1 = pd.Timestamp(config.IS_TRAIN_END).normalize()
    ist0 = pd.Timestamp(config.IS_TEST_START).normalize()
    ist1 = pd.Timestamp(config.IS_TEST_END).normalize()
    o0   = pd.Timestamp(config.OOS_START).normalize()

    is_train_mask = (dates >= it0)  & (dates <= it1)
    is_test_mask  = (dates >= ist0) & (dates <= ist1)
    oos_mask      = dates >= o0

    is_train = panel.loc[is_train_mask].copy()
    is_test  = panel.loc[is_test_mask].copy()
    oos      = panel.loc[oos_mask].copy()

    dmin, dmax = dates.min(), dates.max()
    logger.info("panel 日期: [%s, %s]", dmin.date(), dmax.date())
    logger.info(
        "IS Train: %d rows | IS Test: %d rows | OOS: %d rows",
        len(is_train), len(is_test), len(oos),
    )
    if len(is_train) == 0:
        raise ValueError(
            f"IS 训练集 0 行: panel 日期在 [{dmin.date()}, {dmax.date()}], "
            f"与 IS_TRAIN 期间 [{config.IS_TRAIN_START}, {config.IS_TRAIN_END}] 无交叠。"
        )
    return is_train, is_test, oos


# ═══════════════════════════════════════════════════════════════════════
# XGBoost 训练器
# ═══════════════════════════════════════════════════════════════════════

class XGBTrainer:
    """
    XGBoost 横截面训练器。

    设计原则 (对应 strategy_rules.md):
      - 截面模型 (Cross-sectional Model): 每日截面独立训练/推理
      - batch_size=1: 每次训练迭代的样本仅包含单日截面数据
      - 样本间在时间序列上无交叉依赖

    由 RollingTrainer 编排调用，不直接持有样本划分逻辑。

    Parameters
    ----------
    params : dict
        XGBoost 超参数
    num_boost_round : int
        最大迭代轮数
    early_stopping_rounds : int
        验证集早停轮数 (Val 仅用于早停，不影响参数决策)
    use_wpcc : bool
        True → 使用 WPCC 自定义目标函数直接优化截面 IC
        False → 回退到 MSE (reg:squarederror)
    label_winsorize_sigma : float
        截面 Label Winsorize 倍数, <=0 表示关闭
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        num_boost_round: int = 1500,
        early_stopping_rounds: int = 80,
        use_wpcc: bool = True,
        label_winsorize_sigma: float = config.LABEL_WINSORIZE_SIGMA,
    ):
        self.params = _normalize_xgb_params(
            params
            or {
                "max_depth": 4,
                "learning_rate": 0.02,
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "min_child_weight": 100,
                "max_leaves": 31,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "tree_method": "hist",
                "verbosity": 0,
            }
        )
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.use_wpcc = use_wpcc
        self.label_winsorize_sigma = label_winsorize_sigma
        self.model = None
        self.feature_names: List[str] = []

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """自动识别特征列 (排除 TRADE_DATE, StockID, label)。"""
        exclude = {"TRADE_DATE", "StockID", "label"}
        return [c for c in df.columns if c not in exclude]

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "XGBTrainer":
        """
        训练 XGBoost 模型。

        val_df 仅用于早停 (early stopping)，不参与超参决策。

        Parameters
        ----------
        train_df : pd.DataFrame
            训练集 Panel (含 label 列)
        val_df : pd.DataFrame, optional
            验证集 Panel (用于早停; 来自 Rolling Val CV 的 Val 窗口)
        """
        import xgboost as xgb

        self.feature_names = self._get_feature_cols(train_df)
        logger.info("特征数=%d", len(self.feature_names))
        if not self.feature_names:
            raise ValueError("无特征列")

        # ── Label Winsorize ──────────────────────────────────────────
        if self.label_winsorize_sigma > 0:
            train_df = _winsorize_label_cross_section(train_df, self.label_winsorize_sigma)
            if val_df is not None:
                val_df = _winsorize_label_cross_section(val_df, self.label_winsorize_sigma)

        train_clean = train_df.dropna(subset=["label"])
        y_train = train_clean["label"].values.astype(np.float64)
        if y_train.size == 0:
            raise ValueError("训练集 dropna(label) 后 0 行")
        if not np.isfinite(y_train).all():
            raise ValueError("训练集 label 含 inf/NaN")
        if float(np.std(y_train)) < 1e-12:
            raise ValueError("训练集 label 为常数")

        X_train = (
            train_clean[self.feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )
        logger.info(
            "训练样本 n=%d; label std=%.6g; 特征 NaN=%.2f%%",
            len(y_train), float(np.nanstd(y_train)),
            100.0 * float(np.isnan(X_train).mean()),
        )

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, "train")]
        dval = None
        val_date_ids = None
        if val_df is not None:
            val_clean = val_df.dropna(subset=["label"])
            y_val = val_clean["label"].values.astype(np.float64)
            if y_val.size == 0:
                logger.warning("验证集 dropna(label) 后 0 行，禁用早停")
            else:
                X_val = (
                    val_clean[self.feature_names]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=np.float64)
                )
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
                evals.append((dval, "val"))
                val_date_ids = pd.factorize(
                    pd.to_datetime(val_clean["TRADE_DATE"]).values
                )[0]

        # ── WPCC 或 MSE ───────────────────────────────────────────────
        obj_fn = None
        custom_metric = None
        params = dict(self.params)

        if self.use_wpcc:
            train_date_ids = pd.factorize(
                pd.to_datetime(train_clean["TRADE_DATE"]).values
            )[0]
            obj_fn = _make_wpcc_objective(train_date_ids)
            if val_date_ids is not None:
                custom_metric = _make_wpcc_eval_train_val(
                    dtrain, dval, train_date_ids, val_date_ids
                )
            else:
                custom_metric = _make_wpcc_eval(train_date_ids)
            params.pop("objective", None)
            params.pop("eval_metric", None)
            logger.info("使用 WPCC 自定义目标函数")
        else:
            params.setdefault("objective", "reg:squarederror")
            params.setdefault("eval_metric", "rmse")
            logger.info("使用 MSE 目标函数")

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            obj=obj_fn,
            custom_metric=custom_metric,
            early_stopping_rounds=self.early_stopping_rounds if dval is not None else None,
            verbose_eval=100,
        )

        logger.info(
            "训练完成. best_iteration=%s",
            getattr(self.model, "best_iteration", "N/A"),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """对 Panel 数据生成预测值。"""
        import xgboost as xgb

        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")
        X = df[self.feature_names].values
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmat)

    def save_model(self, path: Optional[Path] = None) -> Path:
        """保存模型。"""
        path = path or (config.SCORE_OUTPUT_DIR / "xgb_model.pkl")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)
        logger.info("模型已保存: %s", path)
        return path

    def load_model(self, path: Optional[Path] = None) -> "XGBTrainer":
        """加载模型。"""
        path = path or (config.SCORE_OUTPUT_DIR / "xgb_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        logger.info("模型已加载: %s", path)
        return self
"""
滚动式训练编排器 (Expanding Window)

核心逻辑:
  固定训练集起点, 每隔 val_months 月扩展一次训练集, 生成一个新 Fold。
  每个 Fold 独立训练一个模型, 仅用于预测其验证窗口内的日期。
  最终打分 = 拼接所有 Fold 的验证窗口预测 (严格无未来信息)。

Fold 示例 (val_months=3):
  Fold 1: Train=[2021-01, 2021-09]  Val=[2021-10, 2021-12]
  Fold 2: Train=[2021-01, 2021-12]  Val=[2022-01, 2022-03]
  Fold 3: Train=[2021-01, 2022-03]  Val=[2022-04, 2022-06]
  ...

使用:
  rt = RollingTrainer(XGBTrainer, model_kwargs={...})
  rt.train_all_folds(panel)
  score_wide = rt.score_all(panel)
  rt.save_model()
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from Strategy import config

logger = logging.getLogger(__name__)


@dataclass
class FoldInfo:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def generate_folds(
    train_start: str = "2021-01-01",
    data_end: Optional[str] = None,
    first_train_months: int = 9,
    val_months: int = config.ROLLING_VAL_MONTHS,
) -> List[FoldInfo]:
    """
    生成 Expanding Window Fold 列表。

    Parameters
    ----------
    train_start : str
        训练集固定起点
    data_end : str, optional
        数据最后日期, None 时自动检测
    first_train_months : int
        第一个 Fold 的初始训练集月数, 默认 9
    val_months : int
        每个 Fold 的验证窗口月数, 默认 3
    """
    ts = pd.Timestamp(train_start)
    folds = []
    fold_id = 1
    train_end = ts + relativedelta(months=first_train_months) - pd.Timedelta(days=1)

    while True:
        val_start = train_end + pd.Timedelta(days=1)
        val_end   = val_start + relativedelta(months=val_months) - pd.Timedelta(days=1)

        if data_end is not None and val_start > pd.Timestamp(data_end):
            break

        folds.append(FoldInfo(
            fold_id=fold_id,
            train_start=ts,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        ))

        train_end = val_end
        fold_id += 1

        if fold_id > 100:  # 安全上限
            break

    return folds


class RollingTrainer:
    """
    滚动式训练编排器, 支持任意模型类 (XGBTrainer / TransformerTrainer)。

    Parameters
    ----------
    model_class : type
        模型类, 需实现 train(train_df, val_df) / predict(df) / save_model / load_model
    model_kwargs : dict
        传给模型构造函数的参数
    train_start : str
        训练集固定起点
    first_train_months : int
        第一个 Fold 初始训练集月数
    val_months : int
        每个 Fold 验证窗口月数
    """

    def __init__(
        self,
        model_class: Type,
        model_kwargs: Optional[dict] = None,
        train_start: str = "2021-01-01",
        first_train_months: int = 9,
        val_months: int = config.ROLLING_VAL_MONTHS,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.train_start = train_start
        self.first_train_months = first_train_months
        self.val_months = val_months

        self.folds: List[FoldInfo] = []
        self.models: List[Any] = []         # 每个 Fold 一个模型
        self.fold_metrics: List[dict] = []  # 每个 Fold 的 IC 等指标
        self.feature_names: List[str] = []

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {"TRADE_DATE", "StockID", "label"}
        return [c for c in df.columns if c not in exclude]

    def train_all_folds(self, panel: pd.DataFrame) -> "RollingTrainer":
        """
        在 Panel 上训练所有 Fold。

        Parameters
        ----------
        panel : pd.DataFrame
            完整 Panel 长表 (含 TRADE_DATE, StockID, label, 因子列)
        """
        dates = pd.to_datetime(panel["TRADE_DATE"])
        data_end = dates.max().strftime("%Y-%m-%d")

        self.folds = generate_folds(
            train_start=self.train_start,
            data_end=data_end,
            first_train_months=self.first_train_months,
            val_months=self.val_months,
        )

        self.feature_names = self._get_feature_cols(panel)
        logger.info("滚动训练: %d 个 Fold, 特征数=%d", len(self.folds), len(self.feature_names))

        self.models = []
        self.fold_metrics = []

        for fold in self.folds:
            logger.info(
                "━━ Fold %d: Train=[%s, %s] Val=[%s, %s] ━━",
                fold.fold_id,
                fold.train_start.strftime("%Y-%m-%d"),
                fold.train_end.strftime("%Y-%m-%d"),
                fold.val_start.strftime("%Y-%m-%d"),
                fold.val_end.strftime("%Y-%m-%d"),
            )

            train_mask = (dates >= fold.train_start) & (dates <= fold.train_end)
            val_mask   = (dates >= fold.val_start)   & (dates <= fold.val_end)

            train_df = panel.loc[train_mask].copy()
            val_df   = panel.loc[val_mask].copy()

            if len(train_df) == 0:
                logger.warning("  Fold %d 训练集为空, 跳过", fold.fold_id)
                self.models.append(None)
                self.fold_metrics.append({"fold_id": fold.fold_id, "status": "skipped"})
                continue
            if len(val_df) == 0:
                logger.warning("  Fold %d 验证集为空, 跳过", fold.fold_id)
                self.models.append(None)
                self.fold_metrics.append({"fold_id": fold.fold_id, "status": "skipped"})
                continue

            logger.info("  Train: %d rows, Val: %d rows", len(train_df), len(val_df))

            # 训练
            model = self.model_class(**self.model_kwargs)
            model.train(train_df, val_df)
            self.models.append(model)

            # 计算 Fold 级 IC
            metrics = self._compute_fold_ic(model, val_df, fold)
            self.fold_metrics.append(metrics)

        logger.info("滚动训练完成: %d 个 Fold", len(self.folds))
        return self

    def _compute_fold_ic(self, model, val_df: pd.DataFrame, fold: FoldInfo) -> dict:
        """计算单个 Fold 在验证集上的 IC / Rank IC。"""
        from scipy import stats as sp_stats

        try:
            preds = model.predict(val_df)
        except Exception as e:
            logger.warning("  Fold %d predict 失败: %s", fold.fold_id, e)
            return {"fold_id": fold.fold_id, "status": "predict_failed"}

        val_df = val_df.copy()
        val_df["pred"] = preds
        val_dates = pd.to_datetime(val_df["TRADE_DATE"])

        ics, rank_ics = [], []
        for dt_val in val_dates.unique():
            sub = val_df.loc[val_dates == dt_val].dropna(subset=["label", "pred"])
            if len(sub) < 10:
                continue
            ic, _ = sp_stats.pearsonr(sub["pred"].values, sub["label"].values)
            ric, _ = sp_stats.spearmanr(sub["pred"].values, sub["label"].values)
            ics.append(ic)
            rank_ics.append(ric)

        mean_ic  = float(np.mean(ics))  if ics else float("nan")
        mean_ric = float(np.mean(rank_ics)) if rank_ics else float("nan")
        ic_std   = float(np.std(ics))   if ics else float("nan")
        icir     = mean_ic / ic_std if ic_std > 1e-8 else float("nan")

        logger.info(
            "  Fold %d Val IC: mean=%.4f  std=%.4f  ICIR=%.4f  RankIC=%.4f  days=%d",
            fold.fold_id, mean_ic, ic_std, icir, mean_ric, len(ics),
        )

        return {
            "fold_id": fold.fold_id,
            "status": "ok",
            "val_ic_mean": mean_ic,
            "val_ic_std": ic_std,
            "val_icir": icir,
            "val_rank_ic_mean": mean_ric,
            "n_val_days": len(ics),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        使用对应 Fold 的模型对每个日期进行预测。

        策略:
        - 日期在某个 Fold 的验证窗口内 → 用该 Fold 模型
        - 日期在第一个 Fold 训练期内 → 用第一个可用模型 (标注为样本内)
        - 日期超出最后一个 Fold → 用最后一个模型
        """
        dates = pd.to_datetime(df["TRADE_DATE"])
        unique_dates = sorted(dates.unique())

        # 建立 date -> fold_index 映射
        date_to_fold = {}
        for i, fold in enumerate(self.folds):
            for dt in unique_dates:
                ts = pd.Timestamp(dt)
                if fold.val_start <= ts <= fold.val_end:
                    date_to_fold[dt] = i

        # 未映射的日期: 训练期用 fold 0, 超出用最后一个
        last_valid_fold = len(self.models) - 1
        while last_valid_fold >= 0 and self.models[last_valid_fold] is None:
            last_valid_fold -= 1
        first_valid_fold = 0
        while first_valid_fold < len(self.models) and self.models[first_valid_fold] is None:
            first_valid_fold += 1

        for dt in unique_dates:
            if dt not in date_to_fold:
                ts = pd.Timestamp(dt)
                if len(self.folds) > 0 and ts < self.folds[0].val_start:
                    date_to_fold[dt] = first_valid_fold
                else:
                    date_to_fold[dt] = last_valid_fold

        # 按 fold 分批预测
        all_preds = np.full(len(df), np.nan, dtype=np.float64)

        fold_groups: Dict[int, List] = {}
        for dt in unique_dates:
            fi = date_to_fold.get(dt, last_valid_fold)
            fold_groups.setdefault(fi, []).append(dt)

        for fi, dt_list in fold_groups.items():
            model = self.models[fi]
            if model is None:
                logger.warning("Fold %d 模型为空, 跳过 %d 天", fi, len(dt_list))
                continue
            mask = dates.isin(dt_list)
            sub_df = df.loc[mask]
            if len(sub_df) == 0:
                continue
            preds = model.predict(sub_df)
            all_preds[mask.values] = preds

        return all_preds

    def score_all(self, panel: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        一键生成滚动模型打分宽表, 格式与 scorer.score_all() 完全一致。

        Parameters
        ----------
        panel : pd.DataFrame
            Panel 长表
        normalize : bool
            是否截面 Z-Score 标准化
        """
        preds = self.predict(panel)
        panel = panel.copy()
        panel["score"] = preds

        if normalize:
            panel["score"] = panel.groupby("TRADE_DATE")["score"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

        score_wide = panel.pivot(index="TRADE_DATE", columns="StockID", values="score")
        score_wide.index.name = "TRADE_DATE"
        return score_wide

    def fold_ic_report(self) -> pd.DataFrame:
        """返回每个 Fold 的 IC 汇总报告。"""
        return pd.DataFrame(self.fold_metrics)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        汇总所有 Fold 的 XGBoost feature importance。
        仅对 XGBTrainer 有效, 其他模型返回 None。
        """
        rows = []
        for i, model in enumerate(self.models):
            if model is None:
                continue
            xgb_model = getattr(model, "model", None)
            if xgb_model is None or not hasattr(xgb_model, "get_score"):
                continue
            scores = xgb_model.get_score(importance_type="gain")
            for feat, gain in scores.items():
                rows.append({"fold_id": i + 1, "feature": feat, "gain": gain})

        if not rows:
            return None

        df = pd.DataFrame(rows)
        summary = df.groupby("feature")["gain"].agg(["mean", "std", "count"]).reset_index()
        summary.columns = ["feature", "mean_gain", "std_gain", "n_folds"]
        summary = summary.sort_values("mean_gain", ascending=False)
        logger.info("Feature importance 汇总: %d 个特征, top5: %s",
                    len(summary), list(summary["feature"].head(5)))
        return summary

    def save_model(self, path: Optional[Path] = None) -> Path:
        """保存所有 Fold 模型 + 元数据。"""
        path = Path(path or (config.SCORE_OUTPUT_DIR / "rolling_model.pkl"))
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "folds": self.folds,
            "fold_metrics": self.fold_metrics,
            "feature_names": self.feature_names,
            "model_class_name": self.model_class.__name__,
            "model_kwargs": self.model_kwargs,
            "train_start": self.train_start,
            "first_train_months": self.first_train_months,
            "val_months": self.val_months,
        }

        # 各 Fold 模型单独保存
        model_states = []
        for i, model in enumerate(self.models):
            if model is None:
                model_states.append(None)
                continue
            # 通用方式: 序列化整个模型对象
            model_states.append(pickle.dumps(model))
        save_data["model_states"] = model_states

        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        logger.info("滚动模型已保存: %s (%d folds)", path, len(self.folds))
        return path

    def load_model(self, path: Optional[Path] = None) -> "RollingTrainer":
        """加载所有 Fold 模型。"""
        path = Path(path or (config.SCORE_OUTPUT_DIR / "rolling_model.pkl"))
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.folds = data["folds"]
        self.fold_metrics = data["fold_metrics"]
        self.feature_names = data["feature_names"]
        self.train_start = data["train_start"]
        self.first_train_months = data["first_train_months"]
        self.val_months = data["val_months"]

        self.models = []
        for state in data["model_states"]:
            if state is None:
                self.models.append(None)
            else:
                self.models.append(pickle.loads(state))

        logger.info("滚动模型已加载: %s (%d folds)", path, len(self.folds))
        return self

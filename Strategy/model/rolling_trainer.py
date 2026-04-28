"""
滚动验证训练编排器 (Rolling Val CV)

核心逻辑 (对应 strategy_rules.md §3-4):
  将 IS Train Set 按时间块 (季度) 切割为 K 个 Fold。
  每个 Fold: 取其中一个季度块作 Val (用于早停), 其余所有块拼接为 Train。
  由于 batch=1 且截面间无序列依赖, Train 的时间段可位于 Val 之后, 无数据泄露。

  Fold 示例 (val_months=3, IS Train=2021-01~2023-09):
    Fold 1:  Val=[2021-01, 2021-03]  Train=其余所有季度拼接
    Fold 2:  Val=[2021-04, 2021-06]  Train=其余所有季度拼接
    ...
    Fold 11: Val=[2023-07, 2023-09]  Train=其余所有季度拼接

IS Test Set 推理 (§4):
  选取 Val 窗口结束时间最近的 ENSEMBLE_N_FOLDS 个 Fold,
  各自独立推理后等权平均作为最终信号。

使用:
  rt = RollingTrainer(XGBTrainer, model_kwargs={...})
  rt.train_all_folds(is_train_panel)           # 在 IS Train 上训练所有 Fold
  score_wide = rt.predict_is_test(is_test_panel)   # 4-Fold Ensemble 推理
  rt.save_model()
"""
from __future__ import annotations

import io
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from Strategy import config

logger = logging.getLogger(__name__)

_ROLL_MDL_V3 = 3                         # 本版序列化版本号
_ROLL_KEY_TRANSFORMER = "transformer_torch"
_ROLL_KEY_PICKLE = "pickle"


# ═══════════════════════════════════════════════════════════════════════
# 序列化工具
# ═══════════════════════════════════════════════════════════════════════

def _pack_fold_model(model: Any, model_class: Type) -> Dict[str, Any]:
    """单折模型序列化: Transformer 存 state_dict, 其余 pickle。"""
    if model_class.__name__ == "TransformerTrainer" and getattr(model, "model", None) is not None:
        import torch
        buf = io.BytesIO()
        torch.save(
            {
                "state_dict": model.model.state_dict(),
                "feature_names": model.feature_names,
                "hparams": model._hparams,
            },
            buf,
        )
        return {"v": _ROLL_MDL_V3, "kind": _ROLL_KEY_TRANSFORMER, "b": buf.getvalue()}
    return {"v": _ROLL_MDL_V3, "kind": _ROLL_KEY_PICKLE, "b": pickle.dumps(model)}


def _unpack_fold_model(
    packed: Union[None, bytes, Dict[str, Any]],
    model_class: Type,
) -> Any:
    """与 _pack_fold_model 对称; 兼容旧版「整段 bytes = pickle」。"""
    if packed is None:
        return None
    if isinstance(packed, bytes):          # 旧版兼容
        return pickle.loads(packed)
    if not isinstance(packed, dict) or packed.get("v") not in (_ROLL_MDL_V3,):
        raise ValueError(f"无法识别的 model_states 格式: {packed.get('v')!r}")

    kind = packed["kind"]
    blob: bytes = packed["b"]

    if kind == _ROLL_KEY_TRANSFORMER:
        import torch
        from Strategy.model.transformer_trainer import CrossSectionalTransformerModel
        d = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        t = model_class()
        t.feature_names = d["feature_names"]
        t._hparams = d["hparams"]
        # 重建网络结构
        t.model = CrossSectionalTransformerModel(**{
            k: d["hparams"][k]
            for k in ("d_input", "d_model", "nhead", "num_layers", "d_ff", "dropout")
        })
        t.model.load_state_dict(d["state_dict"])
        t.model.eval()
        return t

    if kind == _ROLL_KEY_PICKLE:
        return pickle.loads(blob)

    raise ValueError(f"无法识别的 model kind: {kind!r}")


# ═══════════════════════════════════════════════════════════════════════
# Fold 数据结构
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FoldInfo:
    """单个 Fold 的元数据。"""
    fold_id: int
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    # train 由 IS Train Set 中除 val 窗口外的所有日期构成 (运行时按 panel 过滤)


# ═══════════════════════════════════════════════════════════════════════
# Fold 生成: 时间块切割 (Time-Block CV)
# ═══════════════════════════════════════════════════════════════════════

def generate_folds(
    is_train_start: str = None,
    is_train_end: str = None,
    val_months: int = None,
) -> List[FoldInfo]:
    """
    将 IS Train Set 按时间块切割为多个 Fold。

    每个 Fold 的 Val 窗口为一个时间块 (默认季度)；
    Train = IS Train Set 中除该时间块外的所有日期 (运行时拼接)。

    Parameters
    ----------
    is_train_start : str, optional
        IS 训练集起点，默认 config.IS_TRAIN_START
    is_train_end : str, optional
        IS 训练集终点，默认 config.IS_TRAIN_END
    val_months : int, optional
        每个 Val 块的月数，默认 config.ROLLING_VAL_MONTHS (3=季度)

    Returns
    -------
    List[FoldInfo]
        按 val_end 升序排列 (fold_id 从 1 开始)
    """
    ts_start = pd.Timestamp(is_train_start or config.IS_TRAIN_START)
    ts_end   = pd.Timestamp(is_train_end   or config.IS_TRAIN_END)
    vm = val_months or config.ROLLING_VAL_MONTHS

    folds: List[FoldInfo] = []
    fold_id = 1
    cur = ts_start

    while cur <= ts_end:
        val_start = cur
        val_end   = cur + relativedelta(months=vm) - pd.Timedelta(days=1)
        # 最后一块不超出 IS Train 边界
        val_end   = min(val_end, ts_end)

        folds.append(FoldInfo(
            fold_id=fold_id,
            val_start=val_start,
            val_end=val_end,
        ))

        cur = val_end + pd.Timedelta(days=1)
        fold_id += 1
        if fold_id > 200:
            break

    logger.info(
        "generate_folds: IS Train=[%s, %s], val_months=%d → %d Fold",
        ts_start.date(), ts_end.date(), vm, len(folds),
    )
    for f in folds:
        logger.debug(
            "  Fold %d: Val=[%s, %s]",
            f.fold_id, f.val_start.date(), f.val_end.date(),
        )
    return folds


# ═══════════════════════════════════════════════════════════════════════
# RollingTrainer
# ═══════════════════════════════════════════════════════════════════════

class RollingTrainer:
    """
    Rolling Val CV 训练编排器, 支持任意模型类 (XGBTrainer / TransformerTrainer)。

    训练逻辑 (strategy_rules.md §3):
      对 IS Train Set 内部做时间块切割 CV:
        - Val 窗口 = 一个季度块, 用于早停 (early stopping)
        - Train = IS Train Set 中其余所有日期拼接

    IS Test 推理逻辑 (strategy_rules.md §4):
      predict_is_test(is_test_panel):
        → 选取 Val 窗口结束时间最近的 ENSEMBLE_N_FOLDS 个 Fold
        → 各 Fold 模型独立推理, 等权平均

    Parameters
    ----------
    model_class : Type
        模型类, 需实现 train(train_df, val_df) / predict(df) / save_model / load_model
    model_kwargs : dict
        传给模型构造函数的参数
    is_train_start : str
        IS 训练集起点
    is_train_end : str
        IS 训练集终点
    val_months : int
        Val 块月数 (季度=3)
    n_ensemble : int
        IS Test 推理时选取的最近 Fold 数
    """

    def __init__(
        self,
        model_class: Type,
        model_kwargs: Optional[dict] = None,
        is_train_start: str = None,
        is_train_end: str = None,
        val_months: int = None,
        n_ensemble: int = None,
    ):
        self.model_class  = model_class
        self.model_kwargs = model_kwargs or {}
        self.is_train_start = is_train_start or str(config.IS_TRAIN_START)
        self.is_train_end   = is_train_end   or str(config.IS_TRAIN_END)
        self.val_months     = val_months or config.ROLLING_VAL_MONTHS
        self.n_ensemble     = n_ensemble or config.ENSEMBLE_N_FOLDS

        self.folds: List[FoldInfo] = []
        self.models: List[Any] = []
        self.fold_metrics: List[dict] = []
        self.feature_names: List[str] = []

    # ── 工具 ────────────────────────────────────────────────────────
    @staticmethod
    def _get_feature_cols(df: pd.DataFrame) -> List[str]:
        exclude = {"TRADE_DATE", "StockID", "label"}
        return [c for c in df.columns if c not in exclude]

    @staticmethod
    def _compute_ic(preds: np.ndarray, val_df: pd.DataFrame, fold_id: int) -> dict:
        """计算单个 Fold 在 Val 集上的 IC / Rank IC。"""
        from scipy import stats as sp_stats
        df = val_df.copy()
        df["pred"] = preds
        dates = pd.to_datetime(df["TRADE_DATE"])

        ics, rank_ics = [], []
        for dt_val in dates.unique():
            sub = df.loc[dates == dt_val].dropna(subset=["label", "pred"])
            if len(sub) < 10:
                continue
            ic,  _ = sp_stats.pearsonr(sub["pred"].values, sub["label"].values)
            ric, _ = sp_stats.spearmanr(sub["pred"].values, sub["label"].values)
            ics.append(ic)
            rank_ics.append(ric)

        mean_ic  = float(np.mean(ics))     if ics else float("nan")
        mean_ric = float(np.mean(rank_ics)) if rank_ics else float("nan")
        ic_std   = float(np.std(ics))      if ics else float("nan")
        icir     = mean_ic / ic_std if ic_std > 1e-8 else float("nan")

        logger.info(
            "  Fold %d Val  IC=%.4f  ICIR=%.4f  RankIC=%.4f  days=%d",
            fold_id, mean_ic, icir, mean_ric, len(ics),
        )
        return {
            "fold_id": fold_id,
            "status": "ok",
            "val_ic_mean": mean_ic,
            "val_ic_std": ic_std,
            "val_icir": icir,
            "val_rank_ic_mean": mean_ric,
            "n_val_days": len(ics),
        }

    # ── 训练 ────────────────────────────────────────────────────────
    def train_all_folds(self, is_train_panel: pd.DataFrame) -> "RollingTrainer":
        """
        在 IS Train Panel 上训练所有 Fold。

        对每个 Fold:
          - val_df  = is_train_panel 中属于 Val 窗口的行
          - train_df = is_train_panel 中属于其余所有时间块的行 (拼接)
          - 用 val_df 做早停, 保存最佳权重

        Parameters
        ----------
        is_train_panel : pd.DataFrame
            IS Train Set 的 Panel 长表 (含 TRADE_DATE / StockID / 因子列 / label)
        """
        self.folds = generate_folds(
            is_train_start=self.is_train_start,
            is_train_end=self.is_train_end,
            val_months=self.val_months,
        )
        self.feature_names = self._get_feature_cols(is_train_panel)
        dates = pd.to_datetime(is_train_panel["TRADE_DATE"])

        logger.info(
            "Rolling Val CV: %d Fold, 特征数=%d, IS Train=%s~%s",
            len(self.folds), len(self.feature_names),
            self.is_train_start, self.is_train_end,
        )

        self.models = []
        self.fold_metrics = []

        for fold in self.folds:
            logger.info(
                "━━ Fold %d/%d: Val=[%s, %s]  Train=IS Train-Val ━━",
                fold.fold_id, len(self.folds),
                fold.val_start.date(), fold.val_end.date(),
            )

            val_mask   = (dates >= fold.val_start) & (dates <= fold.val_end)
            train_mask = ~val_mask   # IS Train 中除 Val 窗口外的所有行

            val_df   = is_train_panel.loc[val_mask].copy()
            train_df = is_train_panel.loc[train_mask].copy()

            if len(train_df) == 0:
                logger.warning("  Fold %d Train 集为空，跳过", fold.fold_id)
                self.models.append(None)
                self.fold_metrics.append({"fold_id": fold.fold_id, "status": "skipped_no_train"})
                continue
            if len(val_df) == 0:
                logger.warning("  Fold %d Val 集为空，跳过", fold.fold_id)
                self.models.append(None)
                self.fold_metrics.append({"fold_id": fold.fold_id, "status": "skipped_no_val"})
                continue

            logger.info(
                "  Train: %d 行 (%d 日) | Val: %d 行 (%d 日)",
                len(train_df), dates[train_mask].nunique(),
                len(val_df),   dates[val_mask].nunique(),
            )

            # 训练
            model = self.model_class(**self.model_kwargs)
            model.train(train_df, val_df)
            self.models.append(model)

            # Val IC 指标
            try:
                preds = model.predict(val_df)
                metrics = self._compute_ic(preds, val_df, fold.fold_id)
            except Exception as exc:
                logger.warning("  Fold %d IC 计算失败: %s", fold.fold_id, exc)
                metrics = {"fold_id": fold.fold_id, "status": "ic_failed"}
            self.fold_metrics.append(metrics)

        logger.info("Rolling Val CV 完成: %d Fold 已训练", sum(m is not None for m in self.models))
        return self

    # ── IS Test 推理 (4-Fold Ensemble) ──────────────────────────────
    def _select_ensemble_folds(self) -> List[int]:
        """
        按 Val 窗口结束时间降序, 选取最近 n_ensemble 个有效 Fold 的索引。
        """
        valid_indices = [
            i for i, m in enumerate(self.models) if m is not None
        ]
        if len(valid_indices) == 0:
            raise RuntimeError("无可用 Fold 模型，请先调用 train_all_folds()")

        # 按 val_end 降序排列 (最近的在前)
        valid_indices_sorted = sorted(
            valid_indices,
            key=lambda i: self.folds[i].val_end,
            reverse=True,
        )
        selected = valid_indices_sorted[: self.n_ensemble]
        selected_fold_ids = [self.folds[i].fold_id for i in selected]
        selected_val_ends = [self.folds[i].val_end.date() for i in selected]

        logger.info(
            "IS Test Ensemble: 选取最近 %d 个 Fold → Fold IDs=%s  Val Ends=%s",
            len(selected), selected_fold_ids, selected_val_ends,
        )
        return selected

    def predict_is_test(
        self,
        is_test_panel: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        对 IS Test Set 执行 4-Fold Ensemble 推理。

        流程:
          1. 按 val_end 降序选取最近 ENSEMBLE_N_FOLDS 个 Fold
          2. 各 Fold 模型独立对 is_test_panel 推理
          3. 等权平均作为最终预测信号
          4. 可选截面 Z-Score 标准化

        Parameters
        ----------
        is_test_panel : pd.DataFrame
            IS Test Set 的 Panel 长表 (不含 label 也可; label 仅用于 IC 计算)
        normalize : bool
            是否对集成结果做截面 Z-Score 标准化

        Returns
        -------
        pd.DataFrame
            打分宽表 (index=TRADE_DATE, columns=StockID)
        """
        selected_indices = self._select_ensemble_folds()

        all_preds = []
        for i in selected_indices:
            model = self.models[i]
            preds = model.predict(is_test_panel)
            all_preds.append(preds)
            logger.info(
                "  Fold %d 推理完成: mean=%.4f std=%.4f",
                self.folds[i].fold_id, float(np.nanmean(preds)), float(np.nanstd(preds)),
            )

        # 等权平均
        ensemble_preds = np.nanmean(np.stack(all_preds, axis=0), axis=0)

        panel = is_test_panel.copy()
        panel["score"] = ensemble_preds

        if normalize:
            panel["score"] = panel.groupby("TRADE_DATE")["score"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

        score_wide = panel.pivot(index="TRADE_DATE", columns="StockID", values="score")
        score_wide.index.name = "TRADE_DATE"
        logger.info(
            "IS Test Ensemble 推理完成: %d dates × %d stocks",
            *score_wide.shape,
        )
        return score_wide

    # ── IS Train 截面打分 (每日用其所在 Fold 的模型) ────────────────
    def score_is_train(
        self,
        is_train_panel: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        对 IS Train 数据生成 CV 内模型的打分 (每日使用对应 Val Fold 的模型)。

        用途: 观察 IS Train 期间的模型行为, 不作为选模依据。

        Parameters
        ----------
        is_train_panel : pd.DataFrame
            IS Train Set 的 Panel 长表
        normalize : bool
            是否截面 Z-Score 标准化

        Returns
        -------
        pd.DataFrame
            打分宽表 (index=TRADE_DATE, columns=StockID)
        """
        dates = pd.to_datetime(is_train_panel["TRADE_DATE"])
        panel = is_train_panel.copy()
        panel["score"] = np.nan

        for i, fold in enumerate(self.folds):
            model = self.models[i] if i < len(self.models) else None
            if model is None:
                continue
            mask = (dates >= fold.val_start) & (dates <= fold.val_end)
            sub = is_train_panel.loc[mask]
            if len(sub) == 0:
                continue
            panel.loc[mask, "score"] = model.predict(sub)

        if normalize:
            panel["score"] = panel.groupby("TRADE_DATE")["score"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

        score_wide = panel.pivot(index="TRADE_DATE", columns="StockID", values="score")
        score_wide.index.name = "TRADE_DATE"
        return score_wide

    # ── IC 报告 ─────────────────────────────────────────────────────
    def fold_ic_report(self) -> pd.DataFrame:
        """返回每个 Fold 在 Val 集上的 IC 汇总报告。"""
        return pd.DataFrame(self.fold_metrics)

    def ensemble_fold_ids(self) -> List[int]:
        """返回当前会被用于 IS Test 集成的 Fold ID 列表。"""
        selected = self._select_ensemble_folds()
        return [self.folds[i].fold_id for i in selected]

    # ── Feature Importance ─────────────────────────────────────────
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        汇总所有 Fold 的 XGBoost feature importance (gain)。
        仅对 XGBTrainer 有效, 其他模型返回 None。
        """
        rows = []
        for i, model in enumerate(self.models):
            if model is None:
                continue
            xgb_model = getattr(model, "model", None)
            if xgb_model is None or not hasattr(xgb_model, "get_score"):
                continue
            for feat, gain in xgb_model.get_score(importance_type="gain").items():
                rows.append({"fold_id": i + 1, "feature": feat, "gain": gain})

        if not rows:
            return None

        df = pd.DataFrame(rows)
        summary = df.groupby("feature")["gain"].agg(["mean", "std", "count"]).reset_index()
        summary.columns = ["feature", "mean_gain", "std_gain", "n_folds"]
        summary = summary.sort_values("mean_gain", ascending=False)
        return summary

    # ── 持久化 ──────────────────────────────────────────────────────
    def save_model(self, path: Optional[Path] = None) -> Path:
        """保存所有 Fold 模型 + 元数据。"""
        path = Path(path or (config.SCORE_OUTPUT_DIR / "rolling_model.pkl"))
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data: Dict[str, Any] = {
            "version": _ROLL_MDL_V3,
            "folds": self.folds,
            "fold_metrics": self.fold_metrics,
            "feature_names": self.feature_names,
            "model_class_name": self.model_class.__name__,
            "model_kwargs": self.model_kwargs,
            "is_train_start": self.is_train_start,
            "is_train_end": self.is_train_end,
            "val_months": self.val_months,
            "n_ensemble": self.n_ensemble,
            "model_states": [
                _pack_fold_model(m, self.model_class) if m is not None else None
                for m in self.models
            ],
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        logger.info("滚动模型已保存: %s (%d folds)", path, len(self.folds))
        return path

    def load_model(self, path: Optional[Path] = None) -> "RollingTrainer":
        """加载所有 Fold 模型。"""
        path = Path(path or (config.SCORE_OUTPUT_DIR / "rolling_model.pkl"))
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.folds         = data["folds"]
        self.fold_metrics  = data["fold_metrics"]
        self.feature_names = data["feature_names"]
        self.is_train_start = data.get("is_train_start", self.is_train_start)
        self.is_train_end   = data.get("is_train_end",   self.is_train_end)
        self.val_months     = data.get("val_months",     self.val_months)
        self.n_ensemble     = data.get("n_ensemble",     self.n_ensemble)

        self.models = [
            _unpack_fold_model(state, self.model_class)
            for state in data["model_states"]
        ]
        logger.info("滚动模型已加载: %s (%d folds)", path, len(self.folds))
        return self
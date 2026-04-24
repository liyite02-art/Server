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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 数据准备: 宽表 -> 长表 Panel
# ═══════════════════════════════════════════════════════════════════════
def build_panel(
    factor_dict: Dict[str, pd.DataFrame],
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    将多个因子宽表 + Label 宽表对齐并展平为长表。

    Returns
    -------
    pd.DataFrame
        columns: ['TRADE_DATE', 'StockID', factor1, factor2, ..., 'label']
        按 (TRADE_DATE, StockID) 对齐
    """
    common_dates = label_df.index
    for name, fdf in factor_dict.items():
        common_dates = common_dates.intersection(fdf.index)
    common_dates = common_dates.sort_values()

    common_stocks = label_df.columns
    for name, fdf in factor_dict.items():
        common_stocks = common_stocks.intersection(fdf.columns)

    logger.info(
        "Panel 对齐: %d 交易日, %d 只股票, %d 因子",
        len(common_dates), len(common_stocks), len(factor_dict),
    )

    label_long = (
        label_df.loc[common_dates, common_stocks]
        .stack()
        .rename("label")
        .reset_index()
    )
    label_long.columns = ["TRADE_DATE", "StockID", "label"]

    panel = label_long.copy()
    for fname, fdf in factor_dict.items():
        f_long = (
            fdf.loc[common_dates, common_stocks]
            .stack()
            .rename(fname)
            .reset_index()
        )
        f_long.columns = ["TRADE_DATE", "StockID", fname]
        panel = panel.merge(f_long, on=["TRADE_DATE", "StockID"], how="left")

    return panel


def split_panel(
    panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按 config 中的时间划分拆分为训练集/验证集/OOS 测试集。

    ⚠️ OOS 测试集仅用于最终评估, 严禁用于任何参数决策!
    """
    dates = pd.to_datetime(panel["TRADE_DATE"])

    train_mask = (dates >= pd.Timestamp(config.TRAIN_START)) & (dates <= pd.Timestamp(config.TRAIN_END))
    val_mask = (dates >= pd.Timestamp(config.VAL_START)) & (dates <= pd.Timestamp(config.VAL_END))
    oos_mask = dates >= pd.Timestamp(config.OOS_START)

    train = panel.loc[train_mask].copy()
    val = panel.loc[val_mask].copy()
    oos = panel.loc[oos_mask].copy()

    logger.info("Train: %d rows, Val: %d rows, OOS: %d rows", len(train), len(val), len(oos))
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
        self.params = params or {
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
        logger.info("特征列: %s", self.feature_names)

        train_clean = train_df.dropna(subset=["label"])
        X_train = train_clean[self.feature_names].values
        y_train = train_clean["label"].values

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, "train")]
        if val_df is not None:
            val_clean = val_df.dropna(subset=["label"])
            X_val = val_clean[self.feature_names].values
            y_val = val_clean["label"].values
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

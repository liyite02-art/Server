"""
截面 Transformer 模型: 对每日截面中股票间做自注意力, 捕捉相对关系。

损失函数: WPCC (Weighted Pearson Correlation Coefficient)
    Loss = -Corr(y_pred, y_true)  —— 直接优化截面 IC

接口与 XGBTrainer 完全兼容:
    train(train_df, val_df) → self
    predict(df) → np.ndarray
    save_model(path) → Path
    load_model(path) → self
    feature_names: List[str]

使用示例::

    trainer = TransformerTrainer(
        d_model=64, nhead=4, num_layers=2, d_ff=128,
        epochs=50, lr=5e-4, batch_size=4,
    )
    trainer.train(train_df, val_df)
    trainer.save_model()

    # 与 scorer.py 无缝对接
    generate_scores(trainer, factor_dict, label_df, model_name="transformer")
"""
from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Strategy import config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# WPCC 损失函数
# ═══════════════════════════════════════════════════════════════════════

def wpcc_loss_fn(y_pred, y_true, mask, weights=None, eps=1e-8):
    """
    加权皮尔逊相关系数损失 (Weighted Pearson Correlation Coefficient).

    Loss = -WPCC(y_pred, y_true)，使模型直接优化截面 IC。

    Parameters
    ----------
    y_pred : Tensor, shape (batch, seq_len)
        模型预测值
    y_true : Tensor, shape (batch, seq_len)
        真实收益率
    mask : Tensor, shape (batch, seq_len)
        bool, True = 有效位置
    weights : Tensor, optional, shape (batch, seq_len)
        截面内样本权重, None = 等权
    eps : float
        防止除零

    Returns
    -------
    loss : Tensor, scalar
        -mean(WPCC across batch)
    """
    import torch

    # 每个日期独立计算 WPCC，取批次均值
    batch_size = y_pred.shape[0]
    corrs = []

    for i in range(batch_size):
        m = mask[i]                       # (seq_len,), bool
        n_valid = m.sum().item()
        if n_valid < 5:                   # 有效样本太少，跳过该截面
            continue

        pred_i = y_pred[i][m]             # (n_valid,)
        true_i = y_true[i][m]             # (n_valid,)

        if weights is not None:
            w = weights[i][m]
            w = w / (w.sum() + eps)       # 归一化权重
        else:
            w = torch.ones_like(pred_i) / n_valid

        # 加权均值
        mean_pred = (w * pred_i).sum()
        mean_true = (w * true_i).sum()

        # 加权协方差 & 方差
        dp = pred_i - mean_pred
        dt = true_i - mean_true
        cov    = (w * dp * dt).sum()
        var_p  = (w * dp * dp).sum()
        var_t  = (w * dt * dt).sum()

        denom = torch.sqrt(var_p * var_t + eps)
        corr  = cov / denom
        corrs.append(corr)

    if len(corrs) == 0:
        return y_pred.sum() * 0.0         # 无有效截面，返回 0 梯度

    return -torch.stack(corrs).mean()     # 最大化相关 = 最小化负相关


# ═══════════════════════════════════════════════════════════════════════
# 截面 Transformer 网络
# ═══════════════════════════════════════════════════════════════════════

class CrossSectionalTransformer:
    """
    截面 Transformer: 股票 = token, 因子 = token 特征。

    Architecture:
        Input (N_stocks × D_features)
          → Linear(D_features, d_model)
          → LayerNorm
          → L × TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
          → Linear(d_model, 1)
        Output (N_stocks × 1)

    注意: 不使用位置编码, 因为截面中股票没有固有顺序。
    """

    @staticmethod
    def build(d_input: int, d_model: int = 64, nhead: int = 4,
              num_layers: int = 2, d_ff: int = 128, dropout: float = 0.15):
        """构建模型, 延迟 import torch 以便本地无 GPU 也能导入本模块。"""
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(d_input, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,          # Pre-LN 更稳定
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers,
                )
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),
                )

            def forward(self, x, src_key_padding_mask=None):
                """
                Parameters
                ----------
                x : Tensor, (batch, seq_len, d_input)
                src_key_padding_mask : BoolTensor, (batch, seq_len)
                    True = padding 位置 (被忽略)

                Returns
                -------
                Tensor, (batch, seq_len)
                """
                h = self.input_proj(x)                # (B, N, d_model)
                h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
                out = self.head(h).squeeze(-1)        # (B, N)
                return out

        model = _Model()

        # 参数量统计
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "CrossSectionalTransformer: d_input=%d, d_model=%d, nhead=%d, "
            "num_layers=%d, d_ff=%d, dropout=%.2f, params=%.2fM",
            d_input, d_model, nhead, num_layers, d_ff, dropout,
            n_params / 1e6,
        )
        return model


# ═══════════════════════════════════════════════════════════════════════
# 数据整理: Panel 长表 → 按日期分组的截面 batch
# ═══════════════════════════════════════════════════════════════════════

def _panel_to_daily_groups(
    panel: pd.DataFrame,
    feature_names: List[str],
) -> List[Tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]]:
    """
    将 Panel 长表按日期分组, 返回 [(date, X, y, valid_mask), ...]
    X: (n_stocks, n_features), y: (n_stocks,), valid_mask: (n_stocks,) bool
    """
    groups = []
    for date, grp in panel.groupby("TRADE_DATE"):
        X = grp[feature_names].values.astype(np.float32)
        y = grp["label"].values.astype(np.float32) if "label" in grp.columns else np.zeros(len(grp), dtype=np.float32)
        # valid = 特征和 label 均非 NaN
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        # 用 0 填充 NaN（padding 位置会被 mask 掉）
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0)
        groups.append((pd.Timestamp(date), X, y, valid))
    return groups


def _collate_daily_batch(batch_groups, device):
    """
    将多个日期的截面数据拼成一个 padded batch。

    Returns
    -------
    X : Tensor, (batch, max_stocks, n_features)
    y : Tensor, (batch, max_stocks)
    padding_mask : Tensor, (batch, max_stocks), True = padding
    valid_mask : Tensor, (batch, max_stocks), True = 有效(非pad & 非NaN)
    """
    import torch

    max_n = max(g[1].shape[0] for g in batch_groups)
    d = batch_groups[0][1].shape[1]
    B = len(batch_groups)

    X = np.zeros((B, max_n, d), dtype=np.float32)
    y = np.zeros((B, max_n), dtype=np.float32)
    padding = np.ones((B, max_n), dtype=bool)     # True = padding
    valid   = np.zeros((B, max_n), dtype=bool)

    for i, (_, Xi, yi, vi) in enumerate(batch_groups):
        n = Xi.shape[0]
        X[i, :n] = Xi
        y[i, :n] = yi
        padding[i, :n] = False
        valid[i, :n] = vi

    return (
        torch.tensor(X,       device=device),
        torch.tensor(y,       device=device),
        torch.tensor(padding, device=device),
        torch.tensor(valid,   device=device),
    )


# ═══════════════════════════════════════════════════════════════════════
# TransformerTrainer: 与 XGBTrainer 兼容的训练器
# ═══════════════════════════════════════════════════════════════════════

class TransformerTrainer:
    """
    截面 Transformer 训练器。

    接口与 XGBTrainer 完全一致, 可无缝接入 scorer.py / backtest 流水线。

    Parameters
    ----------
    d_model : int
        Transformer 隐藏维度, 默认 64
    nhead : int
        注意力头数, 默认 4
    num_layers : int
        编码器层数, 默认 2
    d_ff : int
        前馈网络维度, 默认 128
    dropout : float
        Dropout 比率, 默认 0.15
    epochs : int
        最大训练轮数, 默认 50
    lr : float
        初始学习率, 默认 5e-4
    weight_decay : float
        L2 正则化系数, 默认 0.01
    batch_size : int
        每批交易日数 (每日 ~5000 stocks), 默认 4
    early_stopping_patience : int
        验证集无改善的容忍轮数, 默认 8
    device : str
        训练设备, 默认 "cuda" (自动回退到 "cpu")
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.15,
        epochs: int = 50,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        early_stopping_patience: int = 8,
        device: str = "cuda",
        use_amp: bool = True,
        label_winsorize_sigma: float = 3.0,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device_str = device
        self.use_amp = use_amp
        self.label_winsorize_sigma = label_winsorize_sigma

        self.model = None
        self.feature_names: List[str] = []
        self._hparams: dict = {}   # 用于保存/加载时恢复

    def _get_device(self):
        import torch
        if self.device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device_str == "cuda":
            logger.warning("CUDA 不可用, 回退到 CPU")
        return torch.device("cpu")

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {"TRADE_DATE", "StockID", "label"}
        return [c for c in df.columns if c not in exclude]

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "TransformerTrainer":
        """
        训练截面 Transformer 模型。

        Parameters
        ----------
        train_df : pd.DataFrame
            训练集 Panel (含 label 列)
        val_df : pd.DataFrame, optional
            验证集 Panel (用于早停)
        """
        import torch

        device = self._get_device()
        logger.info("训练设备: %s, AMP: %s", device, self.use_amp)

        self.feature_names = self._get_feature_cols(train_df)
        d_input = len(self.feature_names)
        if d_input == 0:
            raise ValueError("无特征列")

        # Label Winsorize
        if self.label_winsorize_sigma > 0 and "label" in train_df.columns:
            from Strategy.model.trainer import _winsorize_label_cross_section
            train_df = _winsorize_label_cross_section(train_df, self.label_winsorize_sigma)
            if val_df is not None:
                val_df = _winsorize_label_cross_section(val_df, self.label_winsorize_sigma)

        self._hparams = dict(
            d_input=d_input, d_model=self.d_model, nhead=self.nhead,
            num_layers=self.num_layers, d_ff=self.d_ff, dropout=self.dropout,
        )

        self.model = CrossSectionalTransformer.build(
            d_input=d_input, d_model=self.d_model, nhead=self.nhead,
            num_layers=self.num_layers, d_ff=self.d_ff, dropout=self.dropout,
        ).to(device)

        train_groups = _panel_to_daily_groups(train_df, self.feature_names)
        val_groups   = _panel_to_daily_groups(val_df, self.feature_names) if val_df is not None else None

        logger.info("训练集: %d 日, 验证集: %s 日, 特征: %d",
                    len(train_groups), len(val_groups) if val_groups else "无", d_input)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.lr * 0.01)
        amp_enabled = self.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            np.random.shuffle(train_groups)
            train_losses = []

            for batch_start in range(0, len(train_groups), self.batch_size):
                batch = train_groups[batch_start : batch_start + self.batch_size]
                X, y, pad_mask, valid_mask = _collate_daily_batch(batch, device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred = self.model(X, src_key_padding_mask=pad_mask)
                    loss = wpcc_loss_fn(pred, y, valid_mask)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train = np.mean(train_losses) if train_losses else float("nan")

            avg_val = float("nan")
            if val_groups:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_start in range(0, len(val_groups), self.batch_size):
                        batch = val_groups[batch_start : batch_start + self.batch_size]
                        X, y, pad_mask, valid_mask = _collate_daily_batch(batch, device)
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            pred = self.model(X, src_key_padding_mask=pad_mask)
                            loss = wpcc_loss_fn(pred, y, valid_mask)
                        val_losses.append(loss.item())
                avg_val = np.mean(val_losses) if val_losses else float("nan")

            if epoch % 5 == 0 or epoch <= 3:
                logger.info("  Epoch %d/%d  train=%.5f  val=%.5f  lr=%.2e",
                            epoch, self.epochs, avg_train, avg_val,
                            optimizer.param_groups[0]["lr"])

            if val_groups and not np.isnan(avg_val):
                if avg_val < best_val_loss - 1e-6:
                    best_val_loss = avg_val
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info("  Early stop epoch %d (best_val=%.5f)", epoch, best_val_loss)
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("已恢复最佳模型 (val=%.5f)", best_val_loss)

        self.model.eval()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        对 Panel 数据生成预测值。

        与 XGBTrainer.predict 接口一致:
        输入: 含 feature_names 列的 Panel 长表
        输出: 一维 numpy 数组, 长度 = len(df)
        """
        import torch

        if self.model is None:
            raise RuntimeError("模型未训练, 请先调用 train() 或 load_model()")

        device = self._get_device()
        self.model = self.model.to(device)
        self.model.eval()

        groups = _panel_to_daily_groups(df, self.feature_names)

        # 逐日预测, 拼回原始顺序
        all_preds = []
        with torch.no_grad():
            for _, Xi, _, _ in groups:
                X_t = torch.tensor(Xi, device=device).unsqueeze(0)  # (1, N, D)
                pad = torch.zeros(1, Xi.shape[0], dtype=torch.bool, device=device)
                pred = self.model(X_t, src_key_padding_mask=pad)    # (1, N)
                all_preds.append(pred.squeeze(0).cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def save_model(self, path: Optional[Path] = None) -> Path:
        """保存模型参数 + 超参 + feature_names"""
        import torch

        path = path or (config.SCORE_OUTPUT_DIR / "transformer_model.pkl")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "state_dict": self.model.state_dict() if self.model else None,
            "feature_names": self.feature_names,
            "hparams": self._hparams,
        }
        torch.save(save_dict, path)
        logger.info("Transformer 模型已保存: %s", path)
        return path

    def load_model(self, path: Optional[Path] = None) -> "TransformerTrainer":
        """加载模型参数"""
        import torch

        path = path or (config.SCORE_OUTPUT_DIR / "transformer_model.pkl")
        save_dict = torch.load(path, map_location="cpu", weights_only=False)

        self.feature_names = save_dict["feature_names"]
        self._hparams = save_dict["hparams"]

        self.model = CrossSectionalTransformer.build(**self._hparams)
        self.model.load_state_dict(save_dict["state_dict"])
        self.model.eval()

        logger.info("Transformer 模型已加载: %s (features=%d)", path, len(self.feature_names))
        return self

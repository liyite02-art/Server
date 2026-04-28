"""
截面 Transformer 训练器。

网络设计:
  股票 = token, 因子 = token 特征。
  通过自注意力捕捉截面内股票间的相对关系。

损失函数: WPCC (Weighted Pearson Correlation Coefficient)
  Loss = -Corr(y_pred, y_true) → 直接优化截面 IC

接口与 XGBTrainer 完全兼容:
  train(train_df, val_df) → self
  predict(df)             → np.ndarray
  save_model(path)        → Path
  load_model(path)        → self
  feature_names: List[str]

由 RollingTrainer 编排调用, 不直接持有样本划分逻辑。
val_df 仅用于早停 (early stopping)。
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Strategy import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# WPCC 损失函数
# ═══════════════════════════════════════════════════════════════════════

def wpcc_loss_fn(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    加权皮尔逊相关系数损失。

    Loss = -WPCC(y_pred, y_true)，使模型直接优化截面 IC。
    在 float32 中计算 (AMP 下避免 NaN)。

    Parameters
    ----------
    y_pred : Tensor (batch, seq_len)
    y_true : Tensor (batch, seq_len)
    mask   : Tensor (batch, seq_len), bool, True = 有效位置
    weights: Tensor (batch, seq_len), optional, 截面样本权重
    eps    : float, 防除零
    """
    y_pred = y_pred.float()
    y_true = y_true.float()
    if weights is not None:
        weights = weights.float()

    corrs = []
    for i in range(y_pred.shape[0]):
        m = mask[i]
        n_valid = m.sum().item()
        if n_valid < 5:
            continue

        pred_i = y_pred[i][m]
        true_i = y_true[i][m]

        if weights is not None:
            w = weights[i][m]
            w = w / (w.sum() + eps)
        else:
            w = torch.ones_like(pred_i) / n_valid

        mean_pred = (w * pred_i).sum()
        mean_true = (w * true_i).sum()

        dp = pred_i - mean_pred
        dt = true_i - mean_true
        cov   = (w * dp * dt).sum()
        var_p = (w * dp * dp).sum()
        var_t = (w * dt * dt).sum()

        denom = torch.sqrt(var_p * var_t + eps)
        corr  = cov / denom
        if torch.isfinite(corr).item():
            corrs.append(corr)

    if len(corrs) == 0:
        return y_pred.nansum() * 0.0

    return -torch.stack(corrs).mean()


# ═══════════════════════════════════════════════════════════════════════
# 网络结构
# ═══════════════════════════════════════════════════════════════════════

class CrossSectionalTransformerModel(nn.Module):
    """
    截面 Transformer 网络。

    必须为模块级类 (非嵌套), 以保证 pickle / state_dict 可序列化。
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        return self.head(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 数据整理: Panel 长表 → 按日期分组的截面列表
# ═══════════════════════════════════════════════════════════════════════

def _panel_to_daily_groups(
    panel: pd.DataFrame,
    feature_names: List[str],
) -> List[Tuple]:
    """
    将 Panel 长表按日期分组。

    Returns
    -------
    List[(date, X, y, valid_mask)]
      X: (n_stocks, n_features), y: (n_stocks,), valid_mask: (n_stocks,) bool
    """
    groups = []
    for date, grp in panel.groupby("TRADE_DATE"):
        X = grp[feature_names].values.astype(np.float32)
        y = (
            grp["label"].values.astype(np.float32)
            if "label" in grp.columns
            else np.zeros(len(grp), dtype=np.float32)
        )
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0)
        groups.append((pd.Timestamp(date), X, y, valid))
    return groups


def _collate_batch(batch_groups, device):
    """
    将多个日期的截面数据拼成 padded batch。

    Returns: X, y, padding_mask, valid_mask
    """
    max_n = max(g[1].shape[0] for g in batch_groups)
    d = batch_groups[0][1].shape[1]
    B = len(batch_groups)

    X       = np.zeros((B, max_n, d), dtype=np.float32)
    y       = np.zeros((B, max_n),    dtype=np.float32)
    padding = np.ones( (B, max_n),    dtype=bool)
    valid   = np.zeros((B, max_n),    dtype=bool)

    for i, (_, Xi, yi, vi) in enumerate(batch_groups):
        n = Xi.shape[0]
        X[i, :n]       = Xi
        y[i, :n]       = yi
        padding[i, :n] = False
        valid[i, :n]   = vi

    return (
        torch.tensor(X,       device=device),
        torch.tensor(y,       device=device),
        torch.tensor(padding, device=device),
        torch.tensor(valid,   device=device),
    )


# ═══════════════════════════════════════════════════════════════════════
# TransformerTrainer
# ═══════════════════════════════════════════════════════════════════════

class TransformerTrainer:
    """
    截面 Transformer 训练器。

    接口与 XGBTrainer 完全一致，可无缝接入 RollingTrainer / scorer.py。

    注意: batch_size 指每次梯度更新使用的「天数」(非股票数)。
    每日截面内所有股票作为一个 token 序列输入网络，符合 batch=1 的设计原则。

    Parameters
    ----------
    d_model : int
        Transformer 隐藏维度
    nhead : int
        注意力头数
    num_layers : int
        编码器层数
    d_ff : int
        前馈网络维度
    dropout : float
        Dropout 比率
    epochs : int
        最大训练轮数
    lr : float
        初始学习率
    weight_decay : float
        L2 正则化系数
    batch_size : int
        每批天数 (每天 ~5000 stocks 作为一个截面)
    early_stopping_patience : int
        验证集无改善的容忍轮数 (Val 仅用于早停)
    device : str
        训练设备, 默认 "cuda" (自动回退 cpu)
    use_amp : bool
        是否启用 CUDA 混合精度
    label_winsorize_sigma : float
        截面 Label Winsorize 倍数, <=0 关闭
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
        use_amp: bool = False,
        label_winsorize_sigma: float = 3.0,
    ):
        self.d_model    = d_model
        self.nhead      = nhead
        self.num_layers = num_layers
        self.d_ff       = d_ff
        self.dropout    = dropout
        self.epochs     = epochs
        self.lr         = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device_str = device
        self.use_amp    = use_amp
        self.label_winsorize_sigma = label_winsorize_sigma

        self.model: Optional[CrossSectionalTransformerModel] = None
        self.feature_names: List[str] = []
        self._hparams: dict = {}

    def _get_device(self) -> torch.device:
        if self.device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device_str == "cuda":
            logger.warning("CUDA 不可用，回退到 CPU")
        return torch.device("cpu")

    @staticmethod
    def _get_feature_cols(df: pd.DataFrame) -> List[str]:
        exclude = {"TRADE_DATE", "StockID", "label"}
        return [c for c in df.columns if c not in exclude]

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "TransformerTrainer":
        """
        训练截面 Transformer。

        val_df 仅用于早停，不参与超参决策。
        由 RollingTrainer 编排，train_df = IS Train 中除 Val 窗口外的所有日期拼接。

        Parameters
        ----------
        train_df : pd.DataFrame
            训练集 Panel (含 label 列)
        val_df : pd.DataFrame, optional
            验证集 Panel (用于早停; 来自 Rolling Val CV 的 Val 窗口)
        """
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
        self.model = CrossSectionalTransformerModel(**self._hparams).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "CrossSectionalTransformer: d_input=%d d_model=%d nhead=%d layers=%d params=%.2fM",
            d_input, self.d_model, self.nhead, self.num_layers, n_params / 1e6,
        )

        train_groups = _panel_to_daily_groups(train_df, self.feature_names)
        val_groups   = _panel_to_daily_groups(val_df, self.feature_names) if val_df is not None else None

        logger.info(
            "训练集: %d 日 | 验证集: %s 日 | 特征: %d",
            len(train_groups), len(val_groups) if val_groups else "无", d_input,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        amp_enabled = self.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_val_loss = float("inf")
        best_state    = None
        patience_ctr  = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            np.random.shuffle(train_groups)
            train_losses = []
            did_step = False

            for start in range(0, len(train_groups), self.batch_size):
                batch = train_groups[start: start + self.batch_size]
                X, y, pad_mask, valid_mask = _collate_batch(batch, device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred = self.model(X, src_key_padding_mask=pad_mask)
                loss = wpcc_loss_fn(pred, y, valid_mask)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    did_step = True
                train_losses.append(loss.item())

            if did_step:
                scheduler.step()

            avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
            avg_val   = float("nan")

            if val_groups:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for start in range(0, len(val_groups), self.batch_size):
                        batch = val_groups[start: start + self.batch_size]
                        X, y, pad_mask, valid_mask = _collate_batch(batch, device)
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            pred = self.model(X, src_key_padding_mask=pad_mask)
                        val_losses.append(wpcc_loss_fn(pred, y, valid_mask).item())
                avg_val = float(np.mean(val_losses)) if val_losses else float("nan")

            if epoch % 5 == 0 or epoch <= 3:
                logger.info(
                    "  Epoch %d/%d  train=%.5f  val=%.5f  lr=%.2e",
                    epoch, self.epochs, avg_train, avg_val,
                    optimizer.param_groups[0]["lr"],
                )

            # 早停 (仅针对 val_df 提供的情况)
            if val_groups and not np.isnan(avg_val):
                if avg_val < best_val_loss - 1e-6:
                    best_val_loss = avg_val
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.early_stopping_patience:
                        logger.info("  Early stop @ epoch %d (best_val=%.5f)", epoch, best_val_loss)
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("已恢复最佳模型 (val_loss=%.5f)", best_val_loss)

        self.model.eval()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        对 Panel 数据生成预测值。

        输入: 含 feature_names 列的 Panel 长表
        输出: 一维 numpy 数组, 长度 = len(df)
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train() 或 load_model()")

        device = self._get_device()
        self.model = self.model.to(device)
        self.model.eval()

        groups = _panel_to_daily_groups(df, self.feature_names)
        all_preds = []
        with torch.no_grad():
            for _, Xi, _, _ in groups:
                X_t = torch.tensor(Xi, device=device).unsqueeze(0)  # (1, N, D)
                pad = torch.zeros(1, Xi.shape[0], dtype=torch.bool, device=device)
                pred = self.model(X_t, src_key_padding_mask=pad)    # (1, N)
                all_preds.append(pred.squeeze(0).cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def save_model(self, path: Optional[Path] = None) -> Path:
        """保存模型参数 + 超参 + feature_names。"""
        path = Path(path or (config.SCORE_OUTPUT_DIR / "transformer_model.pt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model else None,
                "feature_names": self.feature_names,
                "hparams": self._hparams,
            },
            path,
        )
        logger.info("Transformer 模型已保存: %s", path)
        return path

    def load_model(self, path: Optional[Path] = None) -> "TransformerTrainer":
        """加载模型参数。"""
        path = Path(path or (config.SCORE_OUTPUT_DIR / "transformer_model.pt"))
        save_dict = torch.load(path, map_location="cpu", weights_only=False)

        self.feature_names = save_dict["feature_names"]
        self._hparams      = save_dict["hparams"]
        self.model = CrossSectionalTransformerModel(**self._hparams)
        self.model.load_state_dict(save_dict["state_dict"])
        self.model.eval()

        logger.info("Transformer 模型已加载: %s (features=%d)", path, len(self.feature_names))
        return self
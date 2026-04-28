"""
截面 MLP 训练器 (深度前馈网络)。

逐股独立映射: 因子向量 -> 预测值, 与 Transformer 相同使用按日截面的 WPCC 损失,
训练循环与 ``TransformerTrainer`` 一致, 可接入 ``RollingTrainer`` / ``scorer.py``。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Strategy import config
from Strategy.model.transformer_trainer import (
    _collate_batch,
    _panel_to_daily_groups,
    wpcc_loss_fn,
)

logger = logging.getLogger(__name__)


class CrossSectionalMLPModel(nn.Module):
    """
    对截面内每只股票做同一套 MLP: (d_input,) -> 1。

    输入布局与 ``CrossSectionalTransformerModel`` 相同: (batch, n_stocks, d_input)。
    必须为模块级类, 以便 ``torch`` 序列化与 ``RollingTrainer`` 恢复。
    """

    def __init__(
        self,
        d_input: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.15,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = d_input
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(d, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # padding 位置仍前向; 损失由 valid_mask 屏蔽 (与 Transformer 行为一致)
        return self.net(x).squeeze(-1)


class MLPTrainer:
    """
    截面 MLP 训练器; 接口与 ``XGBTrainer`` / ``TransformerTrainer`` 一致。

    Parameters
    ----------
    hidden_dims : tuple of int
        隐藏层宽度序列
    dropout, epochs, lr, weight_decay, batch_size, early_stopping_patience, device, use_amp
        同 ``TransformerTrainer``；默认 ``batch_size=config.NN_TRAINER_BATCH_SIZE`` (strategy_rules.md)。
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.15,
        epochs: int = 50,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        batch_size: int = config.NN_TRAINER_BATCH_SIZE,
        early_stopping_patience: int = 8,
        device: str = "cuda",
        use_amp: bool = False,
        label_winsorize_sigma: float = config.LABEL_WINSORIZE_SIGMA,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device_str = device
        self.use_amp = use_amp
        self.label_winsorize_sigma = label_winsorize_sigma

        self.model: Optional[CrossSectionalMLPModel] = None
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
    ) -> "MLPTrainer":
        device = self._get_device()
        logger.info("MLP 训练设备: %s, AMP: %s", device, self.use_amp)

        self.feature_names = self._get_feature_cols(train_df)
        d_input = len(self.feature_names)
        if d_input == 0:
            raise ValueError("无特征列")

        if self.label_winsorize_sigma > 0 and "label" in train_df.columns:
            from Strategy.model.trainer import _winsorize_label_cross_section

            train_df = _winsorize_label_cross_section(train_df, self.label_winsorize_sigma)
            if val_df is not None:
                val_df = _winsorize_label_cross_section(val_df, self.label_winsorize_sigma)

        self._hparams = dict(
            d_input=d_input,
            hidden_dims=tuple(self.hidden_dims),
            dropout=self.dropout,
        )
        self.model = CrossSectionalMLPModel(**self._hparams).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "CrossSectionalMLP: d_input=%d hidden=%s params=%.2fM",
            d_input,
            self.hidden_dims,
            n_params / 1e6,
        )

        train_groups = _panel_to_daily_groups(train_df, self.feature_names)
        val_groups = (
            _panel_to_daily_groups(val_df, self.feature_names) if val_df is not None else None
        )

        logger.info(
            "训练集: %d 日 | 验证集: %s 日 | 特征: %d",
            len(train_groups),
            len(val_groups) if val_groups else "无",
            d_input,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        amp_enabled = self.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_val_loss = float("inf")
        best_state: Optional[dict] = None
        patience_ctr = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            np.random.shuffle(train_groups)
            train_losses: List[float] = []
            did_step = False

            for start in range(0, len(train_groups), self.batch_size):
                batch = train_groups[start : start + self.batch_size]
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
            avg_val = float("nan")

            if val_groups:
                self.model.eval()
                val_losses: List[float] = []
                with torch.no_grad():
                    for start in range(0, len(val_groups), self.batch_size):
                        batch = val_groups[start : start + self.batch_size]
                        X, y, pad_mask, valid_mask = _collate_batch(batch, device)
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            pred = self.model(X, src_key_padding_mask=pad_mask)
                        val_losses.append(wpcc_loss_fn(pred, y, valid_mask).item())
                avg_val = float(np.mean(val_losses)) if val_losses else float("nan")

            if epoch % 5 == 0 or epoch <= 3:
                logger.info(
                    "  Epoch %d/%d  train=%.5f  val=%.5f  lr=%.2e",
                    epoch,
                    self.epochs,
                    avg_train,
                    avg_val,
                    optimizer.param_groups[0]["lr"],
                )

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
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train() 或 load_model()")

        device = self._get_device()
        self.model = self.model.to(device)
        self.model.eval()

        groups = _panel_to_daily_groups(df, self.feature_names)
        all_preds: List[np.ndarray] = []
        with torch.no_grad():
            for _, Xi, _, _ in groups:
                X_t = torch.tensor(Xi, device=device).unsqueeze(0)
                pad = torch.zeros(1, Xi.shape[0], dtype=torch.bool, device=device)
                pred = self.model(X_t, src_key_padding_mask=pad)
                all_preds.append(pred.squeeze(0).cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def save_model(self, path: Optional[Path] = None) -> Path:
        path = Path(path or (config.SCORE_OUTPUT_DIR / "mlp_model.pt"))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model else None,
                "feature_names": self.feature_names,
                "hparams": self._hparams,
            },
            path,
        )
        logger.info("MLP 模型已保存: %s", path)
        return path

    def load_model(self, path: Optional[Path] = None) -> "MLPTrainer":
        path = Path(path or (config.SCORE_OUTPUT_DIR / "mlp_model.pt"))
        save_dict = torch.load(path, map_location="cpu", weights_only=False)

        self.feature_names = save_dict["feature_names"]
        self._hparams = save_dict["hparams"]
        self.model = CrossSectionalMLPModel(**self._hparams)
        self.model.load_state_dict(save_dict["state_dict"])
        self.model.eval()

        logger.info("MLP 模型已加载: %s (features=%d)", path, len(self.feature_names))
        return self

"""
Global configuration for RL Factor Mining system.
"""
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class DataConfig:
    fea_dir: str = ""
    mmap_dir: str = ""
    label_file: str = ""              # path to label .fea file (row=date, col=stock)
    label_shift: int = 1              # factor[t] -> label[t+label_shift]
    n_minutes: int = 237
    train_start: str = "20210104"
    train_end: str = "20211231"
    val_start: str = "20220104"
    val_end: str = "20220630"
    test_start: str = "20220701"
    test_end: str = "20221230"
    exclude_fields: List[str] = field(default_factory=lambda: ["code", "second"])


@dataclass
class EnvConfig:
    max_steps: int = 6
    n_groups: int = 20
    reward_type: str = "composite"
    ic_weight: float = 1.0
    icir_weight: float = 0.5
    ret_weight: float = 0.5
    mono_weight: float = 0.3
    turnover_penalty: float = 0.1
    improvement_weight: float = 0.5
    sample_n_days: int = 60


@dataclass
class AgentConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    hidden_dim: int = 256
    n_transformer_layers: int = 2
    n_heads: int = 4


@dataclass
class TrainConfig:
    n_episodes: int = 50000
    n_envs: int = 16
    rollout_length: int = 128
    log_interval: int = 50
    save_interval: int = 2000
    eval_interval: int = 500
    top_k_save: int = 50
    output_dir: str = "outputs"
    device: str = "cuda"
    seed: int = 42


@dataclass
class BacktestConfig:
    n_groups: int = 20
    commission: float = 0.001
    figure_dpi: int = 150
    figure_size: List[int] = field(default_factory=lambda: [14, 6])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        cfg = cls()
        for section_name in ['data', 'env', 'agent', 'train', 'backtest']:
            if section_name in d:
                section_cls = {'data': DataConfig, 'env': EnvConfig,
                               'agent': AgentConfig, 'train': TrainConfig,
                               'backtest': BacktestConfig}[section_name]
                setattr(cfg, section_name, section_cls(**d[section_name]))
        return cfg

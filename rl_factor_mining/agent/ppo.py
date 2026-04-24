"""
PPO (Proximal Policy Optimization) agent with Transformer-based policy
for formulaic alpha factor mining.

Architecture:
  - Embedding layers for operators, features, and transforms
  - Transformer encoder for expression history
  - Factored action heads for (binary_op, feature, transform)
  - Value head for advantage estimation

References:
  - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
  - "Generating Synergistic Formulaic Alphas" (AAAI 2024)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    obs: Dict[str, np.ndarray]
    action: Tuple[int, int, int]
    reward: float
    done: bool
    log_prob: float
    value: float


class ExpressionEncoder(nn.Module):
    """Encode expression building history with a Transformer."""

    def __init__(self, n_binary_ops: int, n_features: int, n_transforms: int,
                 max_steps: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps

        self.op_embed = nn.Embedding(n_binary_ops + 1, d_model // 4, padding_idx=0)
        self.feat_embed = nn.Embedding(n_features + 1, d_model // 4, padding_idx=0)
        self.trans_embed = nn.Embedding(n_transforms + 1, d_model // 4, padding_idx=0)
        self.step_embed = nn.Embedding(max_steps + 1, d_model // 4)

        self.input_proj = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.stat_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: dict with keys: step, prev_ops, prev_features, prev_transforms,
                 current_ic, current_rank_ic

        Returns:
            Encoded representation (batch, d_model)
        """
        batch_size = obs['step'].shape[0]
        device = obs['step'].device

        ops = obs['prev_ops'] + 1
        feats = obs['prev_features'] + 1
        trans = obs['prev_transforms'] + 1

        step_indices = torch.arange(self.max_steps, device=device).unsqueeze(0).expand(batch_size, -1)

        op_emb = self.op_embed(ops)
        feat_emb = self.feat_embed(feats)
        trans_emb = self.trans_embed(trans)
        step_emb = self.step_embed(step_indices)

        token_emb = torch.cat([op_emb, feat_emb, trans_emb, step_emb], dim=-1)
        token_emb = self.input_proj(token_emb)

        current_step = obs['step'].squeeze(-1)
        mask = step_indices >= current_step.unsqueeze(-1)

        encoded = self.transformer(token_emb, src_key_padding_mask=mask)

        pooled = encoded.mean(dim=1)

        stats = torch.cat([obs['current_ic'], obs['current_rank_ic']], dim=-1)
        stat_emb = self.stat_proj(stats)

        return pooled + stat_emb


class PolicyNetwork(nn.Module):
    """
    PPO policy network with factored action space.

    Actions are decomposed into three components:
      1. Binary operator selection
      2. Feature selection
      3. Transform selection
    """

    def __init__(self, n_binary_ops: int, n_features: int, n_transforms: int,
                 max_steps: int, hidden_dim: int = 256, n_heads: int = 4,
                 n_layers: int = 2):
        super().__init__()
        self.n_binary_ops = n_binary_ops
        self.n_features = n_features
        self.n_transforms = n_transforms

        self.encoder = ExpressionEncoder(
            n_binary_ops, n_features, n_transforms,
            max_steps, hidden_dim, n_heads, n_layers
        )

        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_binary_ops),
        )

        self.feat_head = nn.Sequential(
            nn.Linear(hidden_dim + n_binary_ops, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features),
        )

        self.trans_head = nn.Sequential(
            nn.Linear(hidden_dim + n_binary_ops + n_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_transforms),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: Dict[str, torch.Tensor]):
        """
        Returns:
            op_dist, feat_dist, trans_dist: Categorical distributions
            value: state value estimate
        """
        h = self.encoder(obs)

        op_logits = self.op_head(h)
        op_dist = Categorical(logits=op_logits)

        op_onehot = F.one_hot(op_dist.probs.argmax(dim=-1),
                              self.n_binary_ops).float().detach()
        feat_input = torch.cat([h, op_onehot], dim=-1)
        feat_logits = self.feat_head(feat_input)
        feat_dist = Categorical(logits=feat_logits)

        feat_onehot = F.one_hot(feat_dist.probs.argmax(dim=-1),
                                self.n_features).float().detach()
        trans_input = torch.cat([h, op_onehot, feat_onehot], dim=-1)
        trans_logits = self.trans_head(trans_input)
        trans_dist = Categorical(logits=trans_logits)

        value = self.value_head(h).squeeze(-1)

        return op_dist, feat_dist, trans_dist, value

    def get_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """Sample or select action."""
        op_dist, feat_dist, trans_dist, value = self.forward(obs)

        if deterministic:
            op_action = op_dist.probs.argmax(dim=-1)
            feat_action = feat_dist.probs.argmax(dim=-1)
            trans_action = trans_dist.probs.argmax(dim=-1)
        else:
            op_action = op_dist.sample()
            feat_action = feat_dist.sample()
            trans_action = trans_dist.sample()

        log_prob = (op_dist.log_prob(op_action) +
                    feat_dist.log_prob(feat_action) +
                    trans_dist.log_prob(trans_action))

        entropy = (op_dist.entropy() + feat_dist.entropy() + trans_dist.entropy())

        return (op_action, feat_action, trans_action), log_prob, value, entropy

    def evaluate_action(self, obs: Dict[str, torch.Tensor],
                        action: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Evaluate log_prob and entropy for given actions (for PPO update)."""
        op_dist, feat_dist, trans_dist, value = self.forward(obs)

        op_action, feat_action, trans_action = action
        log_prob = (op_dist.log_prob(op_action) +
                    feat_dist.log_prob(feat_action) +
                    trans_dist.log_prob(trans_action))
        entropy = (op_dist.entropy() + feat_dist.entropy() + trans_dist.entropy())

        return log_prob, entropy, value


class RolloutBuffer:
    """Stores transitions for PPO updates."""

    def __init__(self):
        self.observations: List[Dict[str, np.ndarray]] = []
        self.actions: List[Tuple[int, int, int]] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def add(self, t: Transition):
        self.observations.append(t.obs)
        self.actions.append(t.action)
        self.rewards.append(t.reward)
        self.dones.append(t.done)
        self.log_probs.append(t.log_prob)
        self.values.append(t.value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, gamma: float, gae_lambda: float, last_value: float = 0.0):
        """Compute Generalized Advantage Estimation."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        return advantages, returns

    def to_batch(self, device: str = 'cpu'):
        """Convert buffer to batched tensors."""
        obs_batch = {}
        keys = self.observations[0].keys()
        for k in keys:
            obs_batch[k] = torch.tensor(
                np.stack([o[k] for o in self.observations]), device=device
            )

        actions = (
            torch.tensor([a[0] for a in self.actions], device=device, dtype=torch.long),
            torch.tensor([a[1] for a in self.actions], device=device, dtype=torch.long),
            torch.tensor([a[2] for a in self.actions], device=device, dtype=torch.long),
        )
        log_probs = torch.tensor(self.log_probs, device=device, dtype=torch.float32)

        return obs_batch, actions, log_probs


class PPOAgent:
    """PPO Agent for factor mining."""

    def __init__(
        self,
        n_binary_ops: int,
        n_features: int,
        n_transforms: int,
        max_steps: int,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        device: str = 'cpu',
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.policy = PolicyNetwork(
            n_binary_ops, n_features, n_transforms,
            max_steps, hidden_dim, n_heads, n_layers
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(self, obs: Dict[str, np.ndarray], deterministic: bool = False):
        """Select action given observation."""
        obs_tensor = {
            k: torch.tensor(v, device=self.device).unsqueeze(0)
            if v.ndim < 2 else torch.tensor(v, device=self.device).unsqueeze(0)
            for k, v in obs.items()
        }

        with torch.no_grad():
            (op_a, feat_a, trans_a), log_prob, value, _ = self.policy.get_action(
                obs_tensor, deterministic=deterministic
            )

        action = (op_a.item(), feat_a.item(), trans_a.item())
        return action, log_prob.item(), value.item()

    def store_transition(self, obs, action, reward, done, log_prob, value):
        self.buffer.add(Transition(obs, action, reward, done, log_prob, value))

    def update(self) -> Dict[str, float]:
        """Run PPO update on collected rollout data."""
        if len(self.buffer) == 0:
            return {}

        advantages, returns = self.buffer.compute_gae(self.gamma, self.gae_lambda)

        adv_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        ret_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        obs_batch, actions_batch, old_log_probs = self.buffer.to_batch(self.device)

        n = len(self.buffer)
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                idx = indices[start:end]

                mb_obs = {k: v[idx] for k, v in obs_batch.items()}
                mb_actions = (actions_batch[0][idx], actions_batch[1][idx], actions_batch[2][idx])
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = adv_tensor[idx]
                mb_returns = ret_tensor[idx]

                new_log_probs, entropy, values = self.policy.evaluate_action(mb_obs, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                vf_loss = F.mse_loss(values, mb_returns)

                entropy_loss = -entropy.mean()

                loss = pg_loss + self.value_coef * vf_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()

        return {
            'pg_loss': total_pg_loss / max(n_updates, 1),
            'vf_loss': total_vf_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

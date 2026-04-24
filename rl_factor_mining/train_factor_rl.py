"""
Training entry for RL factor mining.

1) Convert fea -> mmap (run once offline, or auto-detect)
2) Load LabelLoader from separate label.fea file
3) Build MMapDataset, OperatorRegistry, FactorMiningEnv
4) Train PPOAgent to search for high-IC expressions
5) Periodically log / save best expressions and run quick backtests
"""
import os
import logging
from typing import List, Tuple

import numpy as np

from config import Config
from preprocess.fea_to_mmap import convert_fea_to_mmap
from data.mmap_dataset import MMapDataset
from data.label_loader import LabelLoader
from operators.registry import OperatorRegistry
from env.factor_env import FactorMiningEnv
from agent.ppo import PPOAgent
from backtest.backtester import FactorBacktester


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_mmap(cfg: Config):
    if not os.path.exists(os.path.join(cfg.data.mmap_dir, "meta.json")):
        logger.info("MMap not found, converting fea -> mmap ...")
        convert_fea_to_mmap(
            cfg.data.fea_dir,
            cfg.data.mmap_dir,
            n_minutes=cfg.data.n_minutes,
            exclude_fields=cfg.data.exclude_fields,
        )
    else:
        logger.info("Existing mmap found, skip conversion.")


def build_env_and_agent(cfg: Config):
    dataset = MMapDataset(cfg.data.mmap_dir, device=cfg.train.device)
    op_registry = OperatorRegistry()

    # --- label from separate feather file ---
    label_loader = LabelLoader(
        label_path=cfg.data.label_file,
        mmap_dates=dataset.dates,
        mmap_stocks=dataset.stocks,
        device=cfg.train.device,
    )

    feature_names: List[str] = [
        f for f in dataset.fields if f not in set(cfg.data.exclude_fields)
    ]

    env = FactorMiningEnv(
        dataset=dataset,
        feature_names=feature_names,
        label_loader=label_loader,
        op_registry=op_registry,
        max_steps=cfg.env.max_steps,
        n_groups=cfg.env.n_groups,
        reward_type=cfg.env.reward_type,
        ic_weight=cfg.env.ic_weight,
        icir_weight=cfg.env.icir_weight,
        ret_weight=cfg.env.ret_weight,
        mono_weight=cfg.env.mono_weight,
        turnover_penalty=cfg.env.turnover_penalty,
        improvement_weight=cfg.env.improvement_weight,
        sample_n_days=cfg.env.sample_n_days,
        date_start=cfg.data.train_start,
        date_end=cfg.data.train_end,
        label_shift=cfg.data.label_shift,
        device=cfg.train.device,
    )

    agent = PPOAgent(
        n_binary_ops=env.action_dims[0],
        n_features=env.action_dims[1],
        n_transforms=env.action_dims[2],
        max_steps=cfg.env.max_steps,
        hidden_dim=cfg.agent.hidden_dim,
        n_heads=cfg.agent.n_heads,
        n_layers=cfg.agent.n_transformer_layers,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        clip_eps=cfg.agent.clip_eps,
        entropy_coef=cfg.agent.entropy_coef,
        value_coef=cfg.agent.value_coef,
        max_grad_norm=cfg.agent.max_grad_norm,
        ppo_epochs=cfg.agent.ppo_epochs,
        mini_batch_size=cfg.agent.mini_batch_size,
        device=cfg.train.device,
    )

    return env, agent, dataset, label_loader, op_registry, feature_names


def train(cfg: Config):
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    cfg_path = os.path.join(cfg.train.output_dir, "config.yaml")
    cfg.save(cfg_path)

    ensure_mmap(cfg)

    env, agent, dataset, label_loader, op_registry, feature_names = build_env_and_agent(cfg)

    backtester = FactorBacktester(
        dataset=dataset,
        label_loader=label_loader,
        op_registry=op_registry,
        feature_names=feature_names,
        n_groups=cfg.backtest.n_groups,
        label_shift=cfg.data.label_shift,
        device=cfg.train.device,
    )

    best_expressions: List[Tuple[float, float, str]] = []

    for episode in range(1, cfg.train.n_episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, done, log_prob, value)
            obs = next_obs
            ep_reward += reward

        stats = agent.update()

        if episode % cfg.train.log_interval == 0:
            logger.info(
                "Ep %d | EpReward %.4f | pg_loss %.4f | vf_loss %.4f | entropy %.4f",
                episode, ep_reward,
                stats.get("pg_loss", 0.0),
                stats.get("vf_loss", 0.0),
                stats.get("entropy", 0.0),
            )

        if "metrics" in info and "expression" in info:
            m = info["metrics"]
            ic_val = abs(m.get("rank_ic", 0.0))
            ret_val = m.get("long_short_ret", 0.0)
            expr_str = info["expression"]
            best_expressions.append((ic_val, ret_val, expr_str))
            best_expressions = sorted(best_expressions, key=lambda x: -x[0])[: cfg.train.top_k_save]

            if ic_val > 0.02:
                logger.info(
                    "IC: %.4f, Ret: %.4f, Expression: %s",
                    m.get("ic", 0.0), ret_val, expr_str,
                )

        if episode % cfg.train.save_interval == 0:
            ckpt_path = os.path.join(cfg.train.output_dir, f"ppo_ep{episode}.pt")
            agent.save(ckpt_path)
            logger.info("Checkpoint saved to %s", ckpt_path)

        if episode % cfg.train.eval_interval == 0 and best_expressions:
            top_ic, top_ret, top_expr = best_expressions[0]
            logger.info("Eval best | IC %.4f, Ret %.4f | %s", top_ic, top_ret, top_expr)
            try:
                res = backtester.backtest_expression(
                    top_expr, cfg.data.val_start, cfg.data.val_end
                )
                fig_path = os.path.join(cfg.train.output_dir, f"dev_fac_ep{episode}.png")
                backtester.plot_backtest(
                    res, title="dev_fac | label",
                    save_path=fig_path,
                    dpi=cfg.backtest.figure_dpi,
                    figsize=tuple(cfg.backtest.figure_size),
                )
            except Exception as e:
                logger.warning("Eval backtest failed: %s", e)

    expr_path = os.path.join(cfg.train.output_dir, "best_expressions.txt")
    with open(expr_path, "w", encoding="utf-8") as f:
        for ic_val, ret_val, expr in best_expressions:
            f.write(f"IC: {ic_val:.4f}, Ret: {ret_val:.4f}, Expression: {expr}\n")
    logger.info("Best expressions saved to %s", expr_path)


if __name__ == "__main__":
    cfg = Config()
    # ---- 你需要修改以下路径 ----
    # cfg.data.fea_dir    = r"D:\data\fea_files"
    # cfg.data.mmap_dir   = r"D:\data\mmap_fields"
    # cfg.data.label_file = r"D:\data\label2.fea"
    # cfg.data.label_shift = 1  # factor[t] -> label[t+1]
    train(cfg)

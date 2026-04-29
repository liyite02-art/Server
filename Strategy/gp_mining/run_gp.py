from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from Strategy.gp_mining.config import GPMineConfig
from Strategy.gp_mining.data import load_gp_data
from Strategy.gp_mining.engine_v2 import GPEngineV2, GPRunResult
from Strategy.gp_mining.evaluator import FactorEvaluator
from Strategy.gp_mining.expression import ExpressionFactory


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        if columns:
            ordered = [col for col in columns if col in df.columns]
            rest = [col for col in df.columns if col not in ordered]
            df = df[ordered + rest]
    else:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def _prepare_output_dir(config: GPMineConfig) -> None:
    if config.experiment_dir.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"Experiment output already exists: {config.experiment_dir}. "
                "Use overwrite=True or --overwrite to replace it."
            )
        shutil.rmtree(config.experiment_dir)
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    if config.save_factors:
        config.gp_factor_dir.mkdir(parents=True, exist_ok=True)


def mine_gp_factors(
    label_tag: str = "OPEN930_1000",
    experiment: str = "gp_dev",
    generations: int = 32,
    population: int = 64,
    save_factors: bool = False,
    terminal_names: Optional[list[str]] = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> GPRunResult:
    """Mine GP factors from existing Strategy wide-table factor .fea files."""
    config = GPMineConfig(
        label_tag=label_tag,
        experiment=experiment,
        generations=generations,
        population=population,
        save_factors=save_factors,
        terminal_names=terminal_names,
        overwrite=overwrite,
        **kwargs,
    )
    _prepare_output_dir(config)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    data = load_gp_data(config, terminal_names=terminal_names)
    evaluator = FactorEvaluator(config, data)
    factory = ExpressionFactory(
        function_specs=evaluator.function_specs,
        terminal_names=list(data.terminal_tensors.keys()),
        delay_list=config.delay_list,
        quan_list=config.quan_list,
        exp_list=config.exp_list,
        rng=random.Random(config.random_seed),
    )
    engine = GPEngineV2(config, factory, evaluator)
    result = engine.run()

    saved_paths: dict[str, str] = {}
    if config.save_factors:
        for node, eval_result in result.accepted_items:
            path = evaluator.save_factor(node, eval_result)
            saved_paths[eval_result.formula_hash] = str(path)
        for row in result.accepted_rows:
            row["factor_path"] = saved_paths.get(row["formula_hash"])

    accepted_columns = [
        "gen", "phase", "formula", "formula_hash", "fitness", "accepted", "direction",
        "mean_ic", "mean_rank_ic", "abs_mean_ic", "abs_mean_rank_ic",
        "directed_mean_ic", "directed_mean_rank_ic", "ic_ir", "rank_ic_ir", "ic_win_rate",
        "top50_mean", "tail50_mean", "top100_mean", "tail100_mean",
        "spread_mean", "max_corr_to_accepted", "length", "depth",
        "age", "pareto_rank", "crowding_distance",
        "obj_ic", "obj_rank_ic", "obj_spread", "obj_stability",
        "obj_win_rate", "obj_decorrelation", "obj_parsimony",
        "oos_mean_ic", "oos_mean_rank_ic", "factor_path", "error",
    ]
    _write_csv(result.leaderboard_rows, config.experiment_dir / "leaderboard.csv")
    _write_csv(result.logbook_rows, config.experiment_dir / "logbook.csv")
    _write_csv(result.accepted_rows, config.experiment_dir / "accepted_formulas.csv", columns=accepted_columns)
    with open(config.experiment_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_json_dict(), f, ensure_ascii=False, indent=2)

    return result


def _parse_terminal_names(text: Optional[str]) -> Optional[list[str]]:
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strategy-native GP factor mining")
    parser.add_argument("--label-tag", default="OPEN930_1000")
    parser.add_argument("--experiment", default="gp_dev")
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-factors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--terminal-names",
        default=None,
        help="Comma-separated factor names from Strategy/outputs/factors without .fea suffix.",
    )
    parser.add_argument("--ic-threshold", type=float, default=0.015)
    parser.add_argument("--rank-ic-threshold", type=float, default=0.015)
    parser.add_argument("--max-corr-threshold", type=float, default=0.8)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--parent-selection", choices=["epsilon_lexicase", "tournament"], default="epsilon_lexicase")
    parser.add_argument("--epsilon-lexicase-epsilon", type=float, default=0.002)
    parser.add_argument("--lexicase-case-sample-ratio", type=float, default=0.35)
    parser.add_argument("--p-reproduction", type=float, default=0.05)
    parser.add_argument("--nsga-complexity-weight", type=float, default=1.0)
    parser.add_argument("--nsga-corr-weight", type=float, default=1.0)
    parser.add_argument("--nsga-age-weight", type=float, default=0.25)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = mine_gp_factors(
        label_tag=args.label_tag,
        experiment=args.experiment,
        generations=args.generations,
        population=args.population,
        save_factors=args.save_factors,
        terminal_names=_parse_terminal_names(args.terminal_names),
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        device=args.device,
        ic_threshold=args.ic_threshold,
        rank_ic_threshold=args.rank_ic_threshold,
        max_corr_threshold=args.max_corr_threshold,
        max_depth=args.max_depth,
        max_length=args.max_length,
        parent_selection=args.parent_selection,
        epsilon_lexicase_epsilon=args.epsilon_lexicase_epsilon,
        lexicase_case_sample_ratio=args.lexicase_case_sample_ratio,
        p_reproduction=args.p_reproduction,
        nsga_complexity_weight=args.nsga_complexity_weight,
        nsga_corr_weight=args.nsga_corr_weight,
        nsga_age_weight=args.nsga_age_weight,
    )
    print(
        f"GP mining finished: evaluated={len(result.leaderboard_rows)} "
        f"accepted={len(result.accepted_rows)}"
    )


if __name__ == "__main__":
    main()

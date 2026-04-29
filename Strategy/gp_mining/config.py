from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from Strategy import config as strategy_config


@dataclass
class GPMineConfig:
    """Configuration for Strategy-native GP factor mining."""

    label_tag: str = "OPEN930_1000"
    experiment: str = "gp_dev"
    random_seed: int = 100

    generations: int = 32
    population: int = 64
    tournament_size: int = 4
    elite_size: int = 8
    hall_of_fame_size: int = 32
    parent_selection: str = "epsilon_lexicase"
    survivor_selection: str = "nsga2"
    epsilon_lexicase_epsilon: float = 0.002
    lexicase_case_sample_ratio: float = 0.35
    nsga_complexity_weight: float = 1.0
    nsga_corr_weight: float = 1.0
    nsga_age_weight: float = 0.25

    init_min_depth: int = 2
    init_max_depth: int = 4
    min_length: int = 2
    max_depth: int = 16
    max_length: int = 64
    mutation_max_depth: int = 3
    p_crossover: float = 0.8
    p_mutation: float = 0.2
    p_reproduction: float = 0.05

    ic_threshold: float = 0.015
    rank_ic_threshold: float = 0.015
    max_corr_threshold: float = 0.8
    min_stocks: int = 10
    top_ks: tuple[int, ...] = (50, 100)
    tail_ks: tuple[int, ...] = (50, 100)

    length_penalty: float = 0.001
    corr_penalty: float = 0.2
    spread_weight: float = 0.5
    ir_weight: float = 0.25
    win_rate_weight: float = 0.05

    delay_list: tuple[int, ...] = (1, 3, 5, 15)
    quan_list: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95)
    exp_list: tuple[float, ...] = (-3, -2, -1, -0.5, 0.5, 2, 3)

    device: str = "auto"
    dtype: str = "float32"
    save_factors: bool = False
    overwrite: bool = False
    terminal_names: Optional[list[str]] = None

    factor_dir: Path = strategy_config.FACTOR_OUTPUT_DIR
    label_dir: Path = strategy_config.LABEL_OUTPUT_DIR
    output_root: Path = strategy_config.OUTPUT_DIR / "gp_mining"
    gp_factor_root: Path = strategy_config.OUTPUT_DIR / "gp_factors"

    eval_start: Any = strategy_config.IS_TRAIN_START
    eval_end: Any = strategy_config.IS_TEST_END
    oos_start: Any = strategy_config.OOS_START

    function_exclude: set[str] = field(default_factory=lambda: {
        # These operators are expensive or domain-specific enough to keep out
        # of the default v1 search space. Users can opt in later.
        "cs_marketcorr",
    })

    @property
    def experiment_dir(self) -> Path:
        return self.output_root / self.experiment

    @property
    def gp_factor_dir(self) -> Path:
        return self.gp_factor_root / self.experiment

    @property
    def label_path(self) -> Path:
        return self.label_dir / f"LABEL_{self.label_tag}.fea"

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("factor_dir", "label_dir", "output_root", "gp_factor_root"):
            data[key] = str(data[key])
        data["eval_start"] = str(data["eval_start"])
        data["eval_end"] = str(data["eval_end"])
        data["oos_start"] = str(data["oos_start"])
        data["function_exclude"] = sorted(data["function_exclude"])
        return data

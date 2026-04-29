from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from Strategy.gp_mining.config import GPMineConfig


OBJECTIVE_NAMES = (
    "ic",
    "rank_ic",
    "spread",
    "stability",
    "win_rate",
    "decorrelation",
    "parsimony",
    "age",
)


def objective_value(individual: Any, name: str, config: GPMineConfig) -> float:
    if name == "age":
        return -float(individual.age) * config.nsga_age_weight
    value = individual.result.objectives.get(name, float("-inf"))
    if name == "parsimony":
        value *= config.nsga_complexity_weight
    elif name == "decorrelation":
        value *= config.nsga_corr_weight
    if value is None or not np.isfinite(value):
        return float("-inf")
    return float(value)


def dominates(left: Any, right: Any, config: GPMineConfig) -> bool:
    left_vals = [objective_value(left, name, config) for name in OBJECTIVE_NAMES]
    right_vals = [objective_value(right, name, config) for name in OBJECTIVE_NAMES]
    return all(a >= b for a, b in zip(left_vals, right_vals)) and any(
        a > b for a, b in zip(left_vals, right_vals)
    )


def fast_nondominated_sort(individuals: list[Any], config: GPMineConfig) -> list[list[Any]]:
    dominates_map: dict[int, list[int]] = {}
    dominated_count: dict[int, int] = {}
    fronts: list[list[int]] = [[]]

    for i, left in enumerate(individuals):
        dominates_map[i] = []
        dominated_count[i] = 0
        for j, right in enumerate(individuals):
            if i == j:
                continue
            if dominates(left, right, config):
                dominates_map[i].append(j)
            elif dominates(right, left, config):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front: list[int] = []
        for i in fronts[front_idx]:
            for j in dominates_map[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        front_idx += 1
        if next_front:
            fronts.append(next_front)

    return [[individuals[i] for i in front] for front in fronts if front]


def assign_crowding_distance(front: list[Any], config: GPMineConfig) -> None:
    if not front:
        return
    for individual in front:
        individual.crowding_distance = 0.0
    if len(front) <= 2:
        for individual in front:
            individual.crowding_distance = float("inf")
        return

    for name in OBJECTIVE_NAMES:
        front.sort(key=lambda ind: objective_value(ind, name, config))
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        lo = objective_value(front[0], name, config)
        hi = objective_value(front[-1], name, config)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi == lo:
            continue
        for i in range(1, len(front) - 1):
            prev_val = objective_value(front[i - 1], name, config)
            next_val = objective_value(front[i + 1], name, config)
            if math.isfinite(front[i].crowding_distance):
                front[i].crowding_distance += (next_val - prev_val) / (hi - lo)


def nsga2_select(individuals: list[Any], k: int, config: GPMineConfig) -> list[Any]:
    selected: list[Any] = []
    fronts = fast_nondominated_sort(individuals, config)
    for rank, front in enumerate(fronts):
        for individual in front:
            individual.pareto_rank = rank
        assign_crowding_distance(front, config)
        if len(selected) + len(front) <= k:
            selected.extend(front)
            continue
        front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
        selected.extend(front[: k - len(selected)])
        break
    return selected


def tournament_select(individuals: list[Any], rng: random.Random, tournament_size: int) -> Any:
    k = min(tournament_size, len(individuals))
    sample = rng.sample(individuals, k)
    sample.sort(
        key=lambda ind: (
            ind.pareto_rank,
            -ind.crowding_distance,
            -ind.result.fitness,
            ind.age,
        )
    )
    return sample[0]


def epsilon_lexicase_select(
    individuals: list[Any],
    rng: random.Random,
    epsilon: float,
    case_sample_ratio: float,
    fallback_tournament_size: int,
) -> Any:
    with_cases = [ind for ind in individuals if ind.result.case_scores]
    if not with_cases:
        return tournament_select(individuals, rng, fallback_tournament_size)

    n_cases = len(with_cases[0].result.case_scores)
    if n_cases == 0:
        return tournament_select(individuals, rng, fallback_tournament_size)
    sample_n = max(1, min(n_cases, int(n_cases * case_sample_ratio)))
    cases = rng.sample(range(n_cases), sample_n)
    rng.shuffle(cases)

    candidates = with_cases
    for case_idx in cases:
        scores = [ind.result.case_scores[case_idx] for ind in candidates]
        finite_scores = [score for score in scores if np.isfinite(score)]
        if not finite_scores:
            continue
        best = max(finite_scores)
        candidates = [
            ind for ind in candidates
            if np.isfinite(ind.result.case_scores[case_idx])
            and ind.result.case_scores[case_idx] >= best - epsilon
        ]
        if len(candidates) <= 1:
            break

    if not candidates:
        return tournament_select(individuals, rng, fallback_tournament_size)
    return rng.choice(candidates)

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Optional

import numpy as np

from Strategy.gp_mining.config import GPMineConfig
from Strategy.gp_mining.evaluator import EvaluationResult, FactorEvaluator, result_to_row
from Strategy.gp_mining.expression import ExpressionFactory, Node
from Strategy.gp_mining.selection import (
    epsilon_lexicase_select,
    nsga2_select,
    tournament_select,
)


@dataclass
class Individual:
    node: Node
    birth_gen: int
    result: Optional[EvaluationResult] = None
    age: int = 0
    pareto_rank: int = 10**9
    crowding_distance: float = 0.0


@dataclass
class GPRunResult:
    leaderboard_rows: list[dict[str, Any]]
    logbook_rows: list[dict[str, Any]]
    accepted_rows: list[dict[str, Any]]
    accepted_items: list[tuple[Node, EvaluationResult]]
    population: list[Node]


class GPEngineV2:
    """DEAP-style engine with lexicase parents and NSGA-II survivors."""

    def __init__(
        self,
        config: GPMineConfig,
        factory: ExpressionFactory,
        evaluator: FactorEvaluator,
    ) -> None:
        if config.survivor_selection != "nsga2":
            raise ValueError(f"Unsupported survivor_selection: {config.survivor_selection}")
        self.config = config
        self.factory = factory
        self.evaluator = evaluator
        self.rng = random.Random(config.random_seed)
        self.seen_accepted: set[str] = set()

    def _valid_tree(self, node: Node) -> bool:
        return (
            node.depth <= self.config.max_depth
            and self.config.min_length <= node.length <= self.config.max_length
        )

    def _new_random_tree(self) -> Node:
        for _ in range(100):
            node = self.factory.random_tree(
                max_depth=self.rng.randint(self.config.init_min_depth, self.config.init_max_depth),
                min_depth=self.config.init_min_depth,
            )
            if self._valid_tree(node):
                return node
        return self.factory.random_tree(max_depth=self.config.init_max_depth)

    def _initial_population(self) -> list[Individual]:
        pop: list[Individual] = []
        seen: set[str] = set()
        attempts = 0
        while len(pop) < self.config.population and attempts < self.config.population * 50:
            node = self._new_random_tree()
            key = str(node)
            attempts += 1
            if key in seen or not self._valid_tree(node):
                continue
            seen.add(key)
            pop.append(Individual(node=node, birth_gen=0))
        if not pop:
            raise RuntimeError("Failed to generate initial GP population")
        return pop

    def _evaluate(self, individuals: list[Individual], gen: int) -> None:
        for individual in individuals:
            individual.age = max(0, gen - individual.birth_gen)
            individual.result = self.evaluator.evaluate(individual.node)
        nsga2_select(individuals, len(individuals), self.config)

    def _parent(self, population: list[Individual]) -> Individual:
        if self.config.parent_selection == "epsilon_lexicase":
            return epsilon_lexicase_select(
                population,
                self.rng,
                self.config.epsilon_lexicase_epsilon,
                self.config.lexicase_case_sample_ratio,
                self.config.tournament_size,
            )
        if self.config.parent_selection == "tournament":
            return tournament_select(population, self.rng, self.config.tournament_size)
        raise ValueError(f"Unsupported parent_selection: {self.config.parent_selection}")

    def _make_child(self, population: list[Individual], birth_gen: int) -> Individual:
        parent1 = self._parent(population)
        child = parent1.node

        if self.rng.random() < self.config.p_reproduction:
            return Individual(node=child, birth_gen=birth_gen)

        if self.rng.random() < self.config.p_crossover and len(population) > 1:
            parent2 = self._parent(population)
            child, _ = self.factory.crossover(parent1.node, parent2.node)

        if self.rng.random() < self.config.p_mutation:
            child = self.factory.mutate_subtree(child, self.config.mutation_max_depth)

        if not self._valid_tree(child):
            child = self._new_random_tree()
        return Individual(node=child, birth_gen=birth_gen)

    def _offspring(self, population: list[Individual], birth_gen: int) -> list[Individual]:
        children: list[Individual] = []
        seen = {str(ind.node) for ind in population}
        attempts = 0
        while len(children) < self.config.population and attempts < self.config.population * 100:
            attempts += 1
            child = self._make_child(population, birth_gen)
            key = str(child.node)
            if key in seen:
                continue
            seen.add(key)
            children.append(child)
        while len(children) < self.config.population:
            children.append(Individual(node=self._new_random_tree(), birth_gen=birth_gen))
        return children

    def _record_rows(
        self,
        individuals: list[Individual],
        gen: int,
        phase: str,
        leaderboard_rows: list[dict[str, Any]],
        accepted_rows: list[dict[str, Any]],
        accepted_items: list[tuple[Node, EvaluationResult]],
    ) -> None:
        for individual in individuals:
            if individual.result is None:
                continue
            row = result_to_row(individual.result, gen)
            row["phase"] = phase
            row["age"] = individual.age
            row["pareto_rank"] = individual.pareto_rank
            row["crowding_distance"] = individual.crowding_distance
            leaderboard_rows.append(row)
            if individual.result.accepted and individual.result.formula_hash not in self.seen_accepted:
                self.seen_accepted.add(individual.result.formula_hash)
                self.evaluator.remember_accepted(individual.node, individual.result)
                accepted_rows.append(row.copy())
                accepted_items.append((individual.node, individual.result))

    def _logbook_row(self, population: list[Individual], gen: int, offspring_eval: int) -> dict[str, Any]:
        valid = [ind for ind in population if ind.result is not None]
        finite = [ind.result.fitness for ind in valid if ind.result and np.isfinite(ind.result.fitness)]
        best = max(valid, key=lambda ind: ind.result.fitness if ind.result else float("-inf"))
        front0 = sum(1 for ind in valid if ind.pareto_rank == 0)
        accepted = sum(1 for ind in valid if ind.result and ind.result.accepted)
        return {
            "gen": gen,
            "eval": len(valid) + offspring_eval,
            "survivors": len(population),
            "front0": front0,
            "accepted_in_population": accepted,
            "avg_length": float(np.mean([ind.node.length for ind in valid])) if valid else np.nan,
            "avg_age": float(np.mean([ind.age for ind in valid])) if valid else np.nan,
            "avg_fitness": float(np.mean(finite)) if finite else np.nan,
            "best_length": best.node.length,
            "best_fitness": best.result.fitness if best.result else np.nan,
            "best_rank": best.pareto_rank,
            "best_formula": str(best.node),
        }

    def run(self) -> GPRunResult:
        population = self._initial_population()
        leaderboard_rows: list[dict[str, Any]] = []
        accepted_rows: list[dict[str, Any]] = []
        accepted_items: list[tuple[Node, EvaluationResult]] = []
        logbook_rows: list[dict[str, Any]] = []

        self._evaluate(population, gen=0)
        for gen in range(self.config.generations + 1):
            self._record_rows(population, gen, "population", leaderboard_rows, accepted_rows, accepted_items)

            if gen >= self.config.generations:
                logbook_rows.append(self._logbook_row(population, gen, offspring_eval=0))
                break

            children = self._offspring(population, birth_gen=gen + 1)
            self._evaluate(children, gen=gen + 1)
            self._record_rows(children, gen + 1, "offspring", leaderboard_rows, accepted_rows, accepted_items)

            combined = population + children
            for individual in combined:
                individual.age = max(0, gen + 1 - individual.birth_gen)
            population = list(nsga2_select(combined, self.config.population, self.config))
            logbook_rows.append(self._logbook_row(population, gen, offspring_eval=len(children)))

        return GPRunResult(
            leaderboard_rows=leaderboard_rows,
            logbook_rows=logbook_rows,
            accepted_rows=accepted_rows,
            accepted_items=accepted_items,
            population=[individual.node for individual in population],
        )

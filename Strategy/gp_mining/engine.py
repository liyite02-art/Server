from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from Strategy.gp_mining.config import GPMineConfig
from Strategy.gp_mining.evaluator import EvaluationResult, FactorEvaluator, result_to_row
from Strategy.gp_mining.expression import ExpressionFactory, Node


@dataclass
class GPRunResult:
    leaderboard_rows: list[dict[str, Any]]
    logbook_rows: list[dict[str, Any]]
    accepted_rows: list[dict[str, Any]]
    accepted_items: list[tuple[Node, EvaluationResult]]
    population: list[Node]


class GPEngine:
    def __init__(
        self,
        config: GPMineConfig,
        factory: ExpressionFactory,
        evaluator: FactorEvaluator,
    ) -> None:
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

    def _initial_population(self) -> list[Node]:
        pop: list[Node] = []
        seen: set[str] = set()
        attempts = 0
        while len(pop) < self.config.population and attempts < self.config.population * 50:
            node = self._new_random_tree()
            key = str(node)
            attempts += 1
            if key in seen or not self._valid_tree(node):
                continue
            seen.add(key)
            pop.append(node)
        if not pop:
            raise RuntimeError("Failed to generate initial GP population")
        return pop

    def _tournament(self, scored: list[tuple[Node, EvaluationResult]]) -> Node:
        k = min(self.config.tournament_size, len(scored))
        sample = self.rng.sample(scored, k)
        sample.sort(key=lambda x: x[1].fitness, reverse=True)
        return sample[0][0]

    def _next_population(self, scored: list[tuple[Node, EvaluationResult]]) -> list[Node]:
        scored = sorted(scored, key=lambda x: x[1].fitness, reverse=True)
        elite_n = min(max(self.config.elite_size, 1), len(scored), self.config.population)
        next_pop = [node for node, _ in scored[:elite_n]]
        seen = {str(node) for node in next_pop}

        while len(next_pop) < self.config.population:
            parent1 = self._tournament(scored)
            child1 = parent1
            if self.rng.random() < self.config.p_crossover and len(scored) > 1:
                parent2 = self._tournament(scored)
                child1, _ = self.factory.crossover(parent1, parent2)
            if self.rng.random() < self.config.p_mutation:
                child1 = self.factory.mutate_subtree(child1, self.config.mutation_max_depth)
            if not self._valid_tree(child1):
                child1 = self._new_random_tree()
            key = str(child1)
            if key in seen:
                child1 = self._new_random_tree()
                key = str(child1)
            seen.add(key)
            next_pop.append(child1)
        return next_pop

    def run(self) -> GPRunResult:
        population = self._initial_population()
        leaderboard_rows: list[dict[str, Any]] = []
        accepted_rows: list[dict[str, Any]] = []
        accepted_items: list[tuple[Node, EvaluationResult]] = []
        logbook_rows: list[dict[str, Any]] = []

        for gen in range(self.config.generations + 1):
            scored = [(node, self.evaluator.evaluate(node)) for node in population]
            scored.sort(key=lambda x: x[1].fitness, reverse=True)

            finite_fitness = [res.fitness for _, res in scored if np.isfinite(res.fitness)]
            lengths = [node.length for node, _ in scored]
            best_node, best_res = scored[0]
            logbook_rows.append({
                "gen": gen,
                "eval": len(scored),
                "avg_length": float(np.mean(lengths)) if lengths else np.nan,
                "avg_fitness": float(np.mean(finite_fitness)) if finite_fitness else np.nan,
                "best_length": best_node.length,
                "best_fitness": best_res.fitness,
                "best_formula": str(best_node),
            })

            for node, res in scored:
                row = result_to_row(res, gen)
                leaderboard_rows.append(row)
                if res.accepted and res.formula_hash not in self.seen_accepted:
                    self.seen_accepted.add(res.formula_hash)
                    self.evaluator.remember_accepted(node, res)
                    accepted_rows.append(row)
                    accepted_items.append((node, res))

            if gen >= self.config.generations:
                break
            population = self._next_population(scored)

        return GPRunResult(
            leaderboard_rows=leaderboard_rows,
            logbook_rows=logbook_rows,
            accepted_rows=accepted_rows,
            accepted_items=accepted_items,
            population=population,
        )

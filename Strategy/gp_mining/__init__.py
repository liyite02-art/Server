"""
Strategy-native genetic programming factor mining.

This package mines daily wide-table factors without depending on
FactorFramework, mmap, deap, or the original compiled GP modules.
"""

from .config import GPMineConfig

__all__ = ["GPMineConfig", "mine_gp_factors"]


def __getattr__(name: str):
    if name == "mine_gp_factors":
        from .run_gp import mine_gp_factors

        return mine_gp_factors
    raise AttributeError(name)

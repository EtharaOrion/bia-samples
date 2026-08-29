"""Locate the earliest evaluation point whose seed-mean clears the target.

Ported from a prior voided attempt at this project. The prior verifier searched a
grid of optimizer STEPS; this one searches a grid of TRAINING TOKENS, because the
graded quantity here is tokens consumed and batch size is deliberately free. With
batch size free, two submissions reaching the same loss at the same step number
have not done equal work, so a step grid would measure the wrong thing.

Pure functions over measurements the verifier produced. Nothing here reads an
agent-authored number, and nothing here touches a GPU, so every branch is
unit-testable without hardware.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Crossing:
    training_tokens: int | None
    mean_loss: float | None
    n_seeds: int
    margin_achieved: float | None
    reason: str


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def common_grid(per_seed: dict[int, dict[int, float]]) -> list[int]:
    """Token counts measured on every seed, with a finite loss on every seed.

    The intersection is the structural fix for the cadence defect. A submission
    cannot gain by being evaluated more often, because a point present on one
    seed and absent on another never enters the search.
    """
    if not per_seed:
        return []
    grids = []
    for losses in per_seed.values():
        grids.append({t for t, v in losses.items() if _finite(v)})
    return sorted(set.intersection(*grids)) if grids else []


def seed_means(per_seed: dict[int, dict[int, float]], grid: list[int]) -> dict[int, float]:
    n = len(per_seed)
    return {t: sum(losses[t] for losses in per_seed.values()) / n for t in grid}


def clears_bar(mean_loss: float, n_seeds: int, target_loss: float, margin: float) -> bool:
    """Substrate significance rule: (target - mean) * sqrt(n) >= margin."""
    return (target_loss - mean_loss) * math.sqrt(n_seeds) >= margin


def find_crossing(
    per_seed: dict[int, dict[int, float]],
    target_loss: float,
    margin: float,
    min_seeds: int,
) -> Crossing:
    """Earliest token count on the common grid whose seed-mean clears the bar.

    Fails closed: an absent measurement is never treated as a pass.
    """
    n = len(per_seed)
    if n == 0:
        return Crossing(None, None, 0, None, "no-seeds-measured")
    if n < min_seeds:
        return Crossing(None, None, n, None, "insufficient-seeds")

    grid = common_grid(per_seed)
    if not grid:
        return Crossing(None, None, n, None, "no-common-evaluation-grid")

    means = seed_means(per_seed, grid)
    for tokens in grid:
        if clears_bar(means[tokens], n, target_loss, margin):
            achieved = (target_loss - means[tokens]) * math.sqrt(n)
            return Crossing(tokens, means[tokens], n, achieved, "crossing-found")

    best = min(means.values())
    shortfall = (target_loss - best) * math.sqrt(n)
    return Crossing(None, best, n, shortfall, "target-never-cleared")

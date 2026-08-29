"""Locate the earliest optimizer step whose seed-mean clears the target.

Pure functions over measurements the verifier produced. Nothing here reads an
agent-authored number, nothing here touches a GPU, and every branch is testable
without hardware.

The crossing step is the graded quantity of this task. It is found here, from
losses computed by the verifier, and it is the only input to the reward
magnitude. The previous verifier computed the reward from `claimed_steps`, a
field read verbatim out of the submission's own claim file, so an agent that
declared the target step count collected full marks for declaring it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Crossing:
    step: int | None
    mean_loss: float | None
    n_seeds: int
    margin_achieved: float | None
    reason: str


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def common_grid(per_seed: dict[int, dict[int, float]]) -> list[int]:
    """Steps measured with a finite loss on every seed.

    The intersection is what stops a submission gaining by being evaluated at a
    point some other seed never reached.
    """
    if not per_seed:
        return []
    grids = [{s for s, v in losses.items() if _finite(v)} for losses in per_seed.values()]
    return sorted(set.intersection(*grids)) if grids else []


def identical_cadence(per_seed: dict[int, tuple[int, ...]]) -> bool:
    """Every seed checkpointed at exactly the same steps.

    This is the uniform-stopping rule made mechanical. Upstream rule 5 permits
    choosing a stopping point from validation losses only when the choice is
    applied identically across all trials; choosing per run against that run's
    own loss is forbidden. A submission that stops per run produces a different
    milestone set on different seeds, and that difference is visible here. The
    previous verifier asserted this checker as a literal True.
    """
    if not per_seed:
        return False
    sets = list(per_seed.values())
    return all(set(s) == set(sets[0]) for s in sets) and bool(sets[0])


def seed_means(per_seed: dict[int, dict[int, float]], grid: list[int]) -> dict[int, float]:
    n = len(per_seed)
    return {s: sum(losses[s] for losses in per_seed.values()) / n for s in grid}


def clears_bar(mean_loss: float, n_seeds: int, target_loss: float, margin: float) -> bool:
    """Substrate significance rule: (target - mean) * sqrt(n) >= margin."""
    return (target_loss - mean_loss) * math.sqrt(n_seeds) >= margin


def seed_interface_live(per_seed: dict[int, dict[int, float]]) -> bool:
    """Two seeds must move the measured loss.

    Measured from checkpoints the verifier evaluated, never from a substring of
    the submission's source. The previous verifier satisfied this by searching
    the source text for the literal BIA_SEED, which a comment supplies.
    """
    if len(per_seed) < 2:
        return False
    curves = list(per_seed.values())
    shared = set(curves[0])
    for c in curves[1:]:
        shared &= set(c)
    if not shared:
        return False
    return any(len({round(c[s], 12) for c in curves}) > 1 for s in shared)


def find_crossing(per_seed: dict[int, dict[int, float]], target_loss: float,
                  margin: float, min_seeds: int) -> Crossing:
    n = len(per_seed)
    if n == 0:
        return Crossing(None, None, 0, None, "no-seeds-measured")
    if n < min_seeds:
        return Crossing(None, None, n, None, "insufficient-seeds")
    grid = common_grid(per_seed)
    if not grid:
        return Crossing(None, None, n, None, "no-common-evaluation-grid")
    means = seed_means(per_seed, grid)
    for step in grid:
        if clears_bar(means[step], n, target_loss, margin):
            return Crossing(step, means[step], n,
                            (target_loss - means[step]) * math.sqrt(n), "crossing-found")
    best = min(means.values())
    return Crossing(None, best, n, (target_loss - best) * math.sqrt(n),
                    "target-never-cleared")

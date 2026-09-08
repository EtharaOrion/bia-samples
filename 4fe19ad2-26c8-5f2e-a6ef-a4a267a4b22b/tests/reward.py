"""Anchored reward, and the only place a reward file is written.

The reward is one float on the closed interval [0.0, 1.0], higher is better, and
it is never binary: between the baseline and the target it is the anchored ratio
and it moves continuously with the consolidated step count.

    raw   = (baseline_metric - agent_metric) / (baseline_metric - target_metric)
    score = min(max(raw, 0.0), 1.0)

Reaching the target scores exactly 1.0 and beating it also scores 1.0. The
target is a bar, not a point on a gradient.

Every zero carries a machine-readable reason. There is no path through this
module that writes a bare number.
"""
from __future__ import annotations

import json
import os
import pathlib

REWARD_DIR = pathlib.Path(os.environ.get("BIA_REWARD_DIR", "/logs/verifier"))
REWARD_JSON = "reward.json"
REWARD_FLOAT = "reward.txt"
SCORE_DOC = "score.json"

# The bound runtime paths, written literally so an outside reader can grep them:
#   /logs/verifier/reward.json   the reward document
#   /logs/verifier/reward.txt    the single float the runtime reads
#   /logs/verifier/score.json    the score document
BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_REWARD_DOCUMENT = "/logs/verifier/reward.json"
BOUND_SCORE_DOCUMENT = "/logs/verifier/score.json"


def anchored(agent_metric, baseline: int, target: int) -> float:
    if agent_metric is None:
        return 0.0
    span = float(baseline) - float(target)
    if span <= 0.0:
        return 0.0
    raw = (float(baseline) - float(agent_metric)) / span
    return min(max(raw, 0.0), 1.0)


def write(reward: float, reason: str, metric: dict, checkers: list | None = None) -> pathlib.Path:
    """Write the reward document, the single float and the score document.

    The float file is last, because it is the artifact the runtime reads and a
    partially written score document beside a missing float is a clearer failure
    than a float beside a missing document.
    """
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    value = min(max(float(reward), 0.0), 1.0)
    document = {"reward": value, "reason": reason, "metric": metric}
    (REWARD_DIR / REWARD_JSON).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    (REWARD_DIR / SCORE_DOC).write_text(
        json.dumps({"reward": value, "reason": reason, "metric": metric,
                    "checkers": checkers or []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (REWARD_DIR / REWARD_FLOAT).write_text("{:.6f}\n".format(value), encoding="utf-8")
    return REWARD_DIR / REWARD_FLOAT

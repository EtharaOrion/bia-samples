"""FROZEN. The standard mixed precision recipe that supplies the baseline anchor.

This file is red-lined and hashed. It is byte-for-byte the policy that ships as the starter at
`submission/precision_policy.py`, and it exists as a separate frozen file for one reason: the
baseline anchor must be a fixed quantity, so the recipe that produces it cannot live in the
file the agent is told to edit. Before this split the baseline was whatever the agent had just
written, which made the denominator of the score agent-controlled.

Like every other policy, this one is hosted by `policy_worker.py` in its own process. The
driver runs the baseline through the identical channel it runs the submission through, so the
two anchors and the graded phase differ in the policy and in nothing else.
"""

from __future__ import annotations

from typing import Any, Dict, List


class StandardMixedPrecisionPolicy:
    """fp32 master weights, one global dynamic loss scale, chain accumulation."""

    master_dtype = "float32"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.n_chunks = int(config["micro_batches"])
        self.scale_exp = 12
        self.clean_run = 0
        self.growth_interval = 200

    def loss_scale(self, step: int) -> int:
        return self.scale_exp

    def plan(self, step: int, n_chunks: int, stats: List[Dict[str, float]]) -> Dict[str, Any]:
        return {
            "order": list(range(n_chunks)),
            "shape": "chain",
            "compensated": False,
            "leaf_exp": [0] * n_chunks,
            "align": "max",
            "node_exp": 0,
        }

    def on_overflow(self, step: int, info: Dict[str, Any]) -> str:
        self.scale_exp = max(-32, self.scale_exp - 1)
        self.clean_run = 0
        return "skip"


def build_policy(config: Dict[str, Any]) -> StandardMixedPrecisionPolicy:
    return StandardMixedPrecisionPolicy(config)

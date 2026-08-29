"""STARTER SUBMISSION. This is the file you edit. Everything else is frozen.

What ships here is the standard mixed precision recipe: fp32 master weights, a single global
dynamic loss scale that backs off when an overflow is reported, and naive left to right
accumulation of the micro batch gradients. It is a working, runnable policy and it is the
baseline your score is measured against. It is not a good policy under BIA-E4M3N-v1.

The contract you must satisfy is below. build_policy is the only entry point the harness
calls, and it must return an object exposing master_dtype, loss_scale, plan and on_overflow.
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
    """Return the precision policy the graded run will use.

    config carries the resolved operating point: micro_batches, steps, seq_len, d_model,
    n_layer, vocab_size, seed and mode. It never carries gradients or losses.
    """
    return StandardMixedPrecisionPolicy(config)

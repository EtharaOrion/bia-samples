"""PRIVATE REFERENCE POLICY for bia slot S07. Never ships inside environment/.

The whole argument is a bound on the stored magnitude at every node of the reduction, which
is a statement about accumulation order rather than about a configuration.

Let every leaf be pre scaled so its stored magnitude is at most B = 32, two binades under the
240.0 ceiling of BIA-E4M3N-v1. Combine with align "min" and node_exp -1, so a node's exponent
is min(ea, eb) - 1. Both operands are then shifted by a strictly negative amount, so neither
rescale can grow a word, and each lands at or below B/2. Their sum is at or below B. By
induction every node in the tree is at or below B, and the compensation words add at most a
factor (1 + 2**-4) per level, so a depth 4 tree stays under 41. No value in the reduction can
reach 248.0, so the overflow counter cannot fire. That is a proof rather than a tuning.

Shape is the load bearing choice. Under a chain the same node_exp of -1 would shrink the
running sum by 2**-15 and flush the whole gradient to zero, and a chain with node_exp 0 lets
the partial sum grow by the number of micro batches, which is what overflows. Only the
pairwise shape makes a per level halving affordable, so the shape and the node exponent have
to be chosen together.

Ordering by ascending chunk magnitude is the second half. With a machine epsilon of 2**-4 a
running sum swamps any term more than about sixteen times smaller than itself, so combining
like sized terms first is what keeps the small contributions from vanishing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

TARGET_STORED_ABSMAX = 32.0
LEAF_EXP_LIMIT = 64


class BlockExponentTreePolicy:
    master_dtype = "float32"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.n_chunks = int(config["micro_batches"])
        self.overflow_seen = 0

    def loss_scale(self, step: int) -> int:
        return 0

    def _leaf_exponent(self, absmax: float) -> int:
        if not (absmax > 0.0) or not math.isfinite(absmax):
            return 0
        e = int(math.floor(math.log2(TARGET_STORED_ABSMAX / absmax)))
        return max(-LEAF_EXP_LIMIT, min(LEAF_EXP_LIMIT, e))

    def plan(self, step: int, n_chunks: int, stats: List[Dict[str, float]]) -> Dict[str, Any]:
        leaf_exp = [0] * n_chunks
        for s in stats:
            leaf_exp[int(s["index"])] = self._leaf_exponent(float(s["absmax"]))
        order = sorted(range(n_chunks), key=lambda i: float(stats[i]["absmax"]))
        return {
            "order": order,
            "shape": "pairwise",
            "compensated": True,
            "leaf_exp": leaf_exp,
            "align": "min",
            "node_exp": -1,
        }

    def on_overflow(self, step: int, info: Dict[str, Any]) -> str:
        self.overflow_seen += 1
        return "continue"


def build_policy(config: Dict[str, Any]) -> BlockExponentTreePolicy:
    return BlockExponentTreePolicy(config)

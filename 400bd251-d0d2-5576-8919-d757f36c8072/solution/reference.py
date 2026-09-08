#!/usr/bin/env python3
"""The OER-28 oracle: emit the execution plan the authoring sweep arrived at.

WHAT THIS IS AND IS NOT
    This file writes five enumerated choices. It does not time anything and it does
    not know either anchor's latency. The plan it writes is the OUTPUT of a sweep run
    during authoring over this exact harness, recorded in solution/grounding.md.

    It is NOT the verifier's private reference plan. The two choose different fused
    attention backends, so the reference arm of the gate is never the reference plan
    being timed against itself.

WHAT THE SWEEP FOUND, AND WHAT IT DISPROVED
    The step spends its time in two places, and only one of them is worth attacking.

    The attention backend is the lever. The textbook MATH backend materialises the
    full score matrix and measured 107.2ms per step; the two fused backends measured
    91.7ms (flash) and 92.8ms (memory-efficient), a speedup of 1.17x and 1.16x. That
    is essentially the entire span of this task.

    Chunking the cross-entropy is NOT a lever, and the sweep expected it to be. The
    unchunked loss head materialises a 4096-by-50304 float32 logit tensor and squares
    it, which looks like exactly the thing to break into pieces. Measured, every
    chunking made the step SLOWER -- 8 chunks cost 99.2ms against 91.7ms unchunked,
    16 chunks 104.6ms, and more chunks cost monotonically more. Chunking trades a few
    large matmuls for many small ones and pays for it in launches. It saves memory; it
    does not save time. `loss_chunks` is a real axis with a real answer, and the answer
    is 1.

    Reduced precision in the auxiliary computations is not a lever either. Carrying the
    rotary and the qk normalisation in bfloat16 measured slower, not faster. Carrying
    the logit softcap and the cross-entropy in bfloat16 measured slower AND is refused
    by the equivalence gate, which is the only place in this space where the two
    failure modes coincide.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

PLAN = {
    "schema": "oer-kernel-plan/v1",
    "notes": (
        "The memory-efficient fused attention backend, an unchunked cross-entropy and "
        "float32 everywhere else. The sweep behind this plan measured the attention "
        "backend to be the only choice here that is worth making: both fused backends "
        "beat the textbook one by about 1.16x and land within 1.2ms of each other, "
        "while every other axis in the schema is a distractor that costs time. Chunking "
        "the loss head looks like the obvious optimisation and is measurably slower at "
        "every chunk count offered, because it trades a few large matmuls for many small "
        "ones. bfloat16 in the rotary and the qk norm is slower. bfloat16 in the logit "
        "softcap is slower and is refused by the equivalence gate on the gradient."
    ),
    "attention": "efficient",
    "loss_chunks": 1,
    "logit_dtype": "fp32",
    "rotary_dtype": "fp32",
    "qk_norm_dtype": "fp32",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(PLAN, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

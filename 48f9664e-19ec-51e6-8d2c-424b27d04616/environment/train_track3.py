#!/usr/bin/env python3
"""Frozen track-3 training loop. The agent does not edit this file.

What is frozen here and why each piece is frozen:

- The dataset shards and the held-out evaluation split are opened read-only and
  their identities are recorded into the harness-owned telemetry. Training on
  the held-out split is a red line.
- The batch size and the architecture are read out of frozen_contract.json and
  are never taken from the update rule.
- Exactly ONE forward-backward pass runs per optimizer step. The loop counts its
  own passes and aborts if a rule tries to induce a second one.

What is NOT here, deliberately:

- No crossing is computed by this file. This loop records a raw evaluation
  series and a weight ledger and hands them to the verifier. The verifier
  schedules its own evaluation points and reads the crossing off its own
  numbers. Anything this loop prints is for the agent's own use and is not on
  the graded path.
- No smoothing is applied to anything this loop records. A rule that wants a
  smoothed readout for its own schedule computes it inside itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "frozen_contract.json").read_text(encoding="utf-8"))

# Frozen axes. Read, never chosen here and never taken from the update rule.
BATCH_SIZE = 512
ARCHITECTURE = "nanogpt-track3-frozen"
PASSES_PER_STEP = CONTRACT["passes_per_step"]
TARGET_LOSS = CONTRACT["target_validation_loss"]

# Where the harness writes what it OWNS. The verifier reads these; the agent may
# read them too, and nothing the agent writes into them reaches the reward.
TELEMETRY_DIR = Path(os.environ.get("OER05_TELEMETRY_DIR", "/logs/harness"))


class FrozenPassBudget(RuntimeError):
    """Raised when a step attempts more than one forward-backward pass."""


class HarnessLedger:
    """The harness's own record of what happened, keyed by step.

    The weight digest recorded per step is the custody proof: the verifier
    evaluates the weights this ledger names at the graded step, so a checkpoint
    the submission selected and handed over is never the thing evaluated.
    """

    def __init__(self) -> None:
        self.weights: list = []
        self.deltas: list = []

    def record_step(self, step: int, weights_digest: str, delta_signature: list) -> None:
        self.weights.append(
            {"step": step, "weights_digest": weights_digest, "custody": "harness-owned-digest-bound"}
        )
        self.deltas.append({"step": step, "delta_signature": delta_signature})

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"schema": "oer05.harness_ledger/v1", "weights": self.weights, "deltas": self.deltas},
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )


def load_update_rule(shape, hyper):
    """Import the agent's rule from the working tree. Only the harness does this.

    The GRADING process never performs this import. It happens here, inside the
    agent's own training container, which is a different process on a different
    surface from tests/grade.py.
    """
    import update_rule

    return update_rule.build_update_rule(shape, hyper)


def train(seed: int, max_steps: int, shape, hyper) -> dict:
    """One seeded run. Returns the raw series and the harness ledger.

    This function never decides whether the run crossed. It records.
    """
    rule = load_update_rule(shape, hyper)
    ledger = HarnessLedger()
    series = []
    state = {}
    params = [0.0] * shape[0]
    passes = 0
    for step in range(1, max_steps + 1):
        grads, passes_used = forward_backward(params, seed, step)
        passes += passes_used
        if passes_used != PASSES_PER_STEP:
            raise FrozenPassBudget(
                "step " + str(step) + " used " + str(passes_used) + " passes; the frozen rule is one"
            )
        before = list(params)
        params, state = rule.step(params, grads, state)
        ledger.record_step(step, weights_digest_of(params), delta_signature(before, params))
        series.append({"step": step, "train_loss_raw": train_loss_of(params, seed, step)})
    return {
        "seed": seed,
        "steps_run": max_steps,
        "batch_size": BATCH_SIZE,
        "architecture": ARCHITECTURE,
        "passes_total": passes,
        "series_raw": series,
        "ledger": ledger,
    }


def forward_backward(params, seed: int, step: int):
    """One forward and one backward pass. Returns gradients and the pass count."""
    grads = [((seed * 7 + step * 13 + index * 3) % 11 - 5) / 10.0 for index in range(len(params))]
    return grads, 1


def train_loss_of(params, seed: int, step: int) -> float:
    """The TRAINING loss. Never graded. The verifier evaluates the held-out split."""
    magnitude = sum(value * value for value in params)
    return round(4.0 / (1.0 + step / 400.0) + magnitude * 1e-6, 6)


def weights_digest_of(params) -> str:
    import hashlib

    payload = json.dumps([round(value, 9) for value in params], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def delta_signature(before, after) -> list:
    return [round(b - a, 9) for a, b in zip(before, after)]


def main() -> int:
    shape = (8, 8)
    hyper = {}
    result = train(seed=0, max_steps=64, shape=shape, hyper=hyper)
    result["ledger"].dump(TELEMETRY_DIR / "ledger_seed0.json")
    print(
        json.dumps(
            {
                "note": "this output is for the agent's own use and is not on the graded path",
                "steps_run": result["steps_run"],
                "passes_total": result["passes_total"],
                "target_loss": TARGET_LOSS,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

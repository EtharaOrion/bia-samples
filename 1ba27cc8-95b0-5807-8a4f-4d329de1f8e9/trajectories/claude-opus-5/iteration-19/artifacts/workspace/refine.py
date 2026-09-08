#!/usr/bin/env python3
"""refine.py - emit one recipe for this attempt. Stdlib only, self-contained."""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

SEC_PER_STEP = 0.170
SAFETY = 0.90
TARGET_STEPS = 1380

RECIPE = {
    "optimizer": "ortho_momentum",
    "lr": 0.021,
    "beta1": 0.92,
    "beta2": 0.95,
    "eps": 1e-10,
    "weight_decay": 0.0,
    "schedule": "wsd",
    "warmup_frac": 0.015,
    "decay_frac": 0.78,
    "final_lr_frac": 0.0,
    "grad_clip": 1.0,
    "ns_steps": 6,
    "embed_lr_mult": 32.0,
    "head_lr_mult": 5.0,
    "init_scheme": "scaled_normal",
    "init_scale": 1.0,
}


def budget_seconds(state) -> float:
    for container in (state, state.get("shape", {}) or {}, state.get("anchors", {}) or {}):
        if isinstance(container, dict):
            for key, mult in (("budget_hours", 3600.0), ("budget_seconds", 1.0)):
                v = container.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v) * mult
    return 300.0


def main() -> int:
    state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
    digest = state["summary_digest"]
    shape = state.get("shape", {}) or {}
    ceiling = int(shape.get("max_steps", 2000) or 2000)

    affordable = int(budget_seconds(state) * SAFETY / SEC_PER_STEP)
    steps = max(600, min(ceiling, affordable, TARGET_STEPS))

    recipe = dict(RECIPE)
    recipe["max_steps"] = steps
    (HERE / "proposal.json").write_text(
        json.dumps(
            {
                "recipe": recipe,
                "summary_digest": digest,
                "inherits_from": [],
                "report": {
                    "crossing_step": None,
                    "claimed_val_loss": None,
                    "readout": "raw; verifier measures the crossing",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

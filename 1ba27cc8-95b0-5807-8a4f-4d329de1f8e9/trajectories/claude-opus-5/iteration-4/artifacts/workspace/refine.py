#!/usr/bin/env python3
"""refine.py - emit one recipe for this attempt. Stdlib only, self-contained."""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# Iteration 3 crossed at 1370 out of 2000 steps while the wsd factor was still
# 0.63 of peak, so the crossing was limited by the learning rate still being
# high, not by the tokens seen. Compressing the same schedule into a shorter
# horizon anneals the rate sooner at every step and pulls the crossing forward,
# with enough tail left for the sustain window.
RECIPE = {
    "optimizer": "ortho_momentum",
    "lr": 0.022,
    "beta1": 0.95,
    "beta2": 0.95,
    "eps": 1e-10,
    "weight_decay": 0.0,
    "schedule": "wsd",
    "warmup_frac": 0.02,
    "decay_frac": 0.7,
    "final_lr_frac": 0.0,
    "grad_clip": 1.0,
    "ns_steps": 5,
    "embed_lr_mult": 10.0,
    "head_lr_mult": 10.0,
    "init_scheme": "scaled_normal",
    "init_scale": 1.0,
    "max_steps": 1350,
}


def main() -> int:
    state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
    digest = state["summary_digest"]
    shape = state.get("shape", {}) or {}
    ceiling = int(shape.get("max_steps", 2000) or 2000)
    recipe = dict(RECIPE)
    recipe["max_steps"] = min(int(recipe["max_steps"]), ceiling)
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

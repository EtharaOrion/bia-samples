#!/usr/bin/env python3
"""refine.py - emit one recipe for this attempt. Stdlib only, self-contained."""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

RECIPE = {
    "optimizer": "ortho_momentum",
    "lr": 0.021,
    "beta1": 0.95,
    "beta2": 0.95,
    "eps": 1e-10,
    "weight_decay": 0.0,
    "schedule": "wsd",
    "warmup_frac": 0.02,
    # iteration 3 crossed at 1370 with decay_frac 0.5 (decay begins ~step 1000);
    # iteration 4 showed the run is token-limited, so keep the full 2000-step
    # horizon and instead start the anneal earlier so the LR at every step in
    # the 1100-1400 region is lower, which lowers the anytime loss there.
    "decay_frac": 0.8,
    "final_lr_frac": 0.0,
    "grad_clip": 1.0,
    "ns_steps": 5,
    # orthogonalized updates on the 50304x384 embedding/head carry ~1/sqrt(50304)
    # per entry against ~1/sqrt(384) in the hidden blocks, so those groups need a
    # larger multiplier to move at a comparable rate.
    "embed_lr_mult": 10.0,
    "head_lr_mult": 10.0,
    "init_scheme": "scaled_normal",
    "init_scale": 1.0,
}


def main() -> int:
    state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
    digest = state["summary_digest"]
    shape = state.get("shape", {}) or {}
    ceiling = int(shape.get("max_steps", 2000) or 2000)
    recipe = dict(RECIPE)
    recipe["max_steps"] = ceiling
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

#!/usr/bin/env python3
"""refine.py — emit one recipe for this attempt.

Self-contained, stdlib only. Reads state.json from its own working directory,
writes proposal.json beside it.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
    history = state.get("attempts", [])
    digest = state["summary_digest"]
    shape = state.get("shape", {})
    ceiling = shape.get("max_steps", 2000)

    # Carry any within-run history: if a prior attempt in this run crossed,
    # reuse its recipe and nudge; otherwise use the campaign-informed default.
    best = None
    for row in history:
        cs = row.get("crossing_step")
        if cs is not None and (best is None or cs < best[0]):
            best = (cs, row.get("recipe", {}))

    recipe = {
        "optimizer": "adamw",
        "lr": 0.004,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-10,
        "weight_decay": 0.05,
        "schedule": "wsd",
        "warmup_frac": 0.02,
        "decay_frac": 0.4,
        "final_lr_frac": 0.0,
        "grad_clip": 1.0,
        "embed_lr_mult": 1.0,
        "head_lr_mult": 1.0,
        "init_scheme": "scaled_normal",
        "init_scale": 1.0,
        "max_steps": int(ceiling),
    }
    if best is not None:
        inherited = dict(best[1])
        inherited.update({k: v for k, v in recipe.items() if k not in inherited})
        recipe = inherited
        recipe["max_steps"] = int(ceiling)

    (HERE / "proposal.json").write_text(
        json.dumps(
            {
                "recipe": recipe,
                "summary_digest": digest,
                "inherits_from": [row["index"] for row in history],
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

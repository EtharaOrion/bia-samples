#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
    recipe = {
        "optimizer": "adamw",
        "lr": 0.0022,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.1,
        "schedule": "wsd",
        "warmup_frac": 0.02,
        "decay_frac": 0.4,
        "final_lr_frac": 0.05,
        "grad_clip": 1.0,
        "ns_steps": 5,
        "embed_lr_mult": 1.0,
        "head_lr_mult": 1.0,
        "init_scheme": "scaled_normal",
        "init_scale": 1.0,
        "max_steps": state["shape"]["max_steps"],
    }
    proposal = {
        "recipe": recipe,
        "summary_digest": state["summary_digest"],
        "inherits_from": [],
        "report": {"crossing_step": None, "claimed_val_loss": None, "readout": "raw"},
    }
    (HERE / "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

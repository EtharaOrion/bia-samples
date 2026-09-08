#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

BOOTSTRAP = [
    {"optimizer": "momentum", "lr": 0.02, "beta1": 0.9, "schedule": "constant",
     "grad_clip": 1.0, "init_scheme": "normal", "init_scale": 1.0},
    {"optimizer": "ortho_momentum", "lr": 0.02, "beta1": 0.95, "ns_steps": 5,
     "schedule": "linear_decay", "warmup_frac": 0.02, "grad_clip": 1.0,
     "init_scheme": "scaled_normal", "init_scale": 1.0},
    {"optimizer": "adamw", "lr": 0.006, "beta1": 0.9, "beta2": 0.95, "weight_decay": 0.01,
     "schedule": "cosine", "warmup_frac": 0.05, "grad_clip": 1.0,
     "init_scheme": "scaled_normal", "init_scale": 1.0},
]

LADDER = [
    {"embed_lr_mult": 2.0},
    {"lr": "x1.6"},
    {"head_lr_mult": 0.5},
    {"warmup_frac": 0.08},
    {"lr": "x1.25"},
    {"beta2": 0.98},
    {"weight_decay": 0.02},
    {"schedule": "wsd", "decay_frac": 0.3},
    {"lr": "x0.85"},
    {"grad_clip": 0.5},
    {"final_lr_frac": 0.05},
    {"init_scale": 0.9},
    {"beta1": 0.92},
    {"lr": "x1.1"},
    {"embed_lr_mult": 3.0},
    {"lr": "x0.95"},
]

CONVERGED = {
    "optimizer": "ortho_momentum", "lr": 0.024, "beta1": 0.92, "beta2": 0.95,
    "eps": 1e-10, "weight_decay": 0.0, "schedule": "wsd", "warmup_frac": 0.015,
    "decay_frac": 0.80, "final_lr_frac": 0.0, "grad_clip": 1.0, "ns_steps": 6,
    "embed_lr_mult": 32.0, "head_lr_mult": 5.0, "init_scheme": "scaled_normal",
    "init_scale": 1.0, "max_steps": 1320,
}

def single_attempt(state: dict) -> bool:
    try:
        return int(state.get("bindings", {}).get("max_attempts", 0)) == 1
    except (TypeError, ValueError):
        return False

def apply_patch(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, str) and value.startswith("x"):
            out[key] = round(float(out.get(key, 1.0)) * float(value[1:]), 8)
        else:
            out[key] = value
    return out

def best_of(history: list):
    crossed = [row for row in history if row.get("sustained_crossing_step") is not None]
    if not crossed:
        return None
    return min(crossed, key=lambda row: (int(row["sustained_crossing_step"]), int(row["index"])))

STEP_PROBE = (400, 800, 1600, 3200)

def step_budget(state: dict, history: list) -> int:
    ceiling = int(state["shape"]["max_steps"])
    stride = int(state["shape"]["eval_stride"])
    window = int(state["shape"]["sustain_window"])
    sustain = (window + 1) * stride
    if single_attempt(state):
        return max(sustain, min(ceiling, int(CONVERGED["max_steps"])))
    base = best_of(history)
    if base is not None:
        return max(sustain, min(ceiling, int(base["sustained_crossing_step"]) + sustain))
    rung = STEP_PROBE[min(len(history) // len(BOOTSTRAP), len(STEP_PROBE) - 1)]
    return max(sustain, min(ceiling, rung))

def propose(state: dict) -> dict:
    history = state["attempts"]
    max_steps = step_budget(state, history)
    if single_attempt(state):
        recipe = dict(CONVERGED)
        recipe["max_steps"] = max_steps
        return {"recipe": recipe, "inherits_from": []}
    if len(history) < len(BOOTSTRAP):
        recipe = dict(BOOTSTRAP[len(history)])
        recipe["max_steps"] = max_steps
        return {"recipe": recipe, "inherits_from": []}
    base = best_of(history)
    if base is None:
        recipe = dict(BOOTSTRAP[len(history) % len(BOOTSTRAP)])
        recipe["max_steps"] = max_steps
        recipe["lr"] = round(float(recipe["lr"]) * (1.0 + 0.25 * (len(history) % 4)), 8)
        return {"recipe": recipe, "inherits_from": []}
    rung = LADDER[(len(history) - len(BOOTSTRAP)) % len(LADDER)]
    recipe = apply_patch(base["recipe"], rung)
    recipe["max_steps"] = max_steps
    return {"recipe": recipe, "inherits_from": [int(base["index"])]}

def main() -> int:
    state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
    chosen = propose(state)
    (HERE / "proposal.json").write_text(
        json.dumps(
            {
                "recipe": chosen["recipe"],
                "summary_digest": state["summary_digest"],
                "inherits_from": chosen["inherits_from"],
                "report": {"crossing_step": None, "claimed_val_loss": None, "readout": "raw"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

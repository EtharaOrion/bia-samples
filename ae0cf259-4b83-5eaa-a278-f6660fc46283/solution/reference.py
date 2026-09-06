#!/usr/bin/env python3
"""The reference solution: a refinement policy, not a configuration.

It is submission-shaped on purpose. The harness copies it to `refine.py` and
invokes it once per attempt, in a fresh directory, with `state.json` beside it.
Everything it knows about the session it reads out of that file.

The policy is a hill climb with a bootstrap, and it is the reason a single
fortunate attempt cannot reproduce it:

- The first three attempts probe three DIFFERENT optimizer families, so the
  session has real evidence rather than one point.
- After that every proposal is built ON TOP of the best attempt so far, which
  the state file identifies by its reward. That base is what `inherits_from`
  names, and the components it keeps are what makes the inheritance real rather
  than claimed.
- The patch applied to that base walks a fixed ladder, so a direction that stops
  paying is left behind by construction: the ladder moves on and the base does
  not move backwards.

It reports no crossing. Claiming a crossing it measured on the development split
would be claiming a number the verifier did not measure, and the DIVERGENCE
checker exists precisely so that such a claim costs something. Reporting nothing
is the honest position for a policy that cannot see the graded split.
"""
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

# Each rung is a patch onto the running best. The ladder is ordered so that the
# structural moves come first and the fine ones last, which is what makes the
# late attempts consolidate a level rather than rediscover it.
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


def apply_patch(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, str) and value.startswith("x"):
            out[key] = round(float(out.get(key, 1.0)) * float(value[1:]), 8)
        else:
            out[key] = value
    return out


def best_of(history: list):
    """The attempt the session should build on: the lowest sustained crossing so far."""
    crossed = [row for row in history if row.get("sustained_crossing_step") is not None]
    if not crossed:
        return None
    return min(crossed, key=lambda row: (int(row["sustained_crossing_step"]), int(row["index"])))


def propose(state: dict) -> dict:
    history = state["attempts"]
    max_steps = int(state["shape"]["max_steps"])
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

#!/usr/bin/env python3
"""Starting point for refine.py. Rename it, edit it, submit it.

The harness invokes this file once per attempt, in a fresh directory, with
state.json beside it. state.json carries every prior approach and the reward it
earned. Write proposal.json beside it and exit.

Nothing here reaches the network, a clock or a random source, and nothing here
needs to: the whole input is state.json.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
    history = state["attempts"]           # every prior attempt, its recipe, its crossing, its reward
    digest = state["summary_digest"]      # echo this back or the attempt is graded stale

    # A first pass that ignores `history` will score zero for
    # graded-outcome-not-carried however good any single recipe in it is.
    recipe = {
        "optimizer": "momentum",
        "lr": 0.005,
        "beta1": 0.9,
        "schedule": "constant",
        "grad_clip": 1.0,
        "max_steps": state["shape"]["max_steps"],
    }

    (HERE / "proposal.json").write_text(
        json.dumps(
            {
                "recipe": recipe,
                "summary_digest": digest,
                "inherits_from": [row["index"] for row in history[-2:]],
                "report": {"crossing_step": None, "claimed_val_loss": None, "readout": "raw"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

def main() -> int:
    state = json.loads((HERE / "state.json").read_text(encoding="utf-8"))
    history = state["attempts"]
    digest = state["summary_digest"]

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

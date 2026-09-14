#!/usr/bin/env python3
"""The frozen trainer. The HARNESS runs this; a submission never does.

Every axis this script reads comes from environment/frozen_recipe.json and none of
them is a submission input. The one thing that varies between runs is the token feed
the pipeline produced, which is exactly the free surface of this slot.

This file is agent-visible so the agent can read what is frozen. It is not a file the
agent edits: the harness digests the frozen axes at open and at close, and a run whose
frozen axes moved is graded as having mutated a frozen axis rather than as an
improvement.

Reading a validation loss out of this script's stdout is not the graded path. The
harness evaluates the model itself, unsmoothed, on the frozen held-out split, at the
bound evaluation point and at every sustain point the harness schedules.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RECIPE = HERE / "frozen_recipe.json"


def load_recipe() -> dict:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    recipe = load_recipe()
    axes = recipe["frozen_axes"]
    if len(argv) < 2:
        print("usage: train_frozen.py <feed.txt>", file=sys.stderr)
        print("frozen axes: " + json.dumps(axes, sort_keys=True), file=sys.stderr)
        return 2
    feed = pathlib.Path(argv[1])
    if not feed.is_file():
        print("feed absent: " + str(feed), file=sys.stderr)
        return 2
    # The accelerator-bound training body is supplied by the harness image at run
    # time. This module refuses to invent one, because a trainer that silently
    # degrades to a stub would report a loss nobody trained for.
    print(
        json.dumps(
            {
                "trainer": "frozen",
                "feed": feed.name,
                "max_steps": axes["max_steps"],
                "note": "the harness supplies the accelerator-bound body and owns the weights",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

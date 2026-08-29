#!/usr/bin/env python3
"""Frozen training entry point for OER-09.

Everything this script does is frozen by environment/frozen_config.yaml: the token budget,
the model, the optimizer and the evaluation split. The only thing a submission changes is the
pool handed to `--pool`.

What this script does NOT do, on purpose:

  * it does not decide the graded loss. The verifier recomputes the graded validation loss
    itself on the frozen held-out split, from harness-owned weights, at the bound evaluation
    point, unsmoothed. Any loss this script prints is for the operator's eyes.
  * it does not choose which checkpoint is evaluated. The weights ledger is harness-owned.
  * it does not decide when to stop. The evaluation schedule is the verifier's; halting early
    is graded as not having established the loss.

As it feeds, the harness records its own observation of the pool, the feed ledger and the
held-out leak audit under /logs/harness/. Those records are the graded state. This script
writes none of them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "frozen_config.yaml"


def load_config() -> dict:
    import yaml

    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_pool(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="train under the frozen budget on a curated pool")
    parser.add_argument("--pool", required=True)
    args = parser.parse_args(argv)

    config = load_config()
    budget = int(config["budget"]["token_budget_tokens"])
    rows = load_pool(Path(args.pool))
    if not rows:
        print("refusing to train on an empty pool", file=sys.stderr)
        return 2

    available = sum(int(row.get("tokens", 0)) for row in rows)
    print("pool documents: " + str(len(rows)))
    print("pool tokens available: " + str(available))
    print("frozen token budget: " + str(budget))
    print("model: " + str(config["model"]["arch"]) + ", layers=" + str(config["model"]["layers"]))
    print("optimizer: " + str(config["optimizer"]["name"]))
    print("evaluation split: " + str(config["evaluation"]["split_id"]))
    print(
        "verifier evaluation points: bound="
        + str(config["evaluation"]["bound_point"])
        + " sustain="
        + str(config["evaluation"]["sustain_points"])
    )
    print("")
    print("the graded validation loss is recomputed by the verifier and is not printed here")
    return 0


if __name__ == "__main__":
    sys.exit(main())

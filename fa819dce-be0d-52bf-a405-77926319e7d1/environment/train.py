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
held-out leak audit under /logs/harness/, through environment/observe.py. Those records are
ADVISORY: /logs/harness sits inside this container, so tests/harness.py recomputes every one
of them inside the verifier and overwrites them before a checker reads one. This script writes
none of them itself.

What this script does leave beside the curated pool is `train_receipt.json`. That receipt is a
CLAIM. The verifier captures it, and it can establish only that a training run happened and
which pool digest it ran against. Every graded number is recomputed by the verifier from the
curated pool itself, so a receipt can cause a rejection and can never raise a score.
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


def _import_observe():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import observe

    return observe


def _publish_feed_observation(fed_rows, tokens_fed, budget, config, register_rows) -> None:
    try:
        observe = _import_observe()
    except ImportError:
        return
    held = [row for row in register_rows if row.get("holdout") is True]
    held_ids = {row.get("doc_id") for row in held}
    fed_ids = {row.get("doc_id") for row in fed_rows}
    observe.observe_feed(
        fed_rows,
        tokens_fed,
        budget,
        config.get("register") or {},
        config.get("evaluation") or {},
        len(held),
        len(held_ids & fed_ids),
    )


def _write_receipt(pool_path: Path, rows: list, tokens_fed: int, budget: int) -> Path:
    observe = _import_observe()
    receipt = {
        "status": "trained",
        "pool_path": str(pool_path),
        "pool_documents": len(rows),
        "pool_digest": observe.pool_digest(rows),
        "tokens_fed": int(tokens_fed),
        "frozen_budget_tokens": int(budget),
    }
    target = pool_path.parent / "train_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="train under the frozen budget on a curated pool")
    parser.add_argument("--pool", required=True)
    args = parser.parse_args(argv)

    config = load_config()
    budget = int(config["budget"]["token_budget_tokens"])
    pool_path = Path(args.pool)
    rows = load_pool(pool_path)
    if not rows:
        print("refusing to train on an empty pool", file=sys.stderr)
        return 2

    available = sum(int(row.get("tokens", 0)) for row in rows)
    if available <= 0:
        print("refusing to train on a pool carrying no tokens", file=sys.stderr)
        return 2

    # The budget is fed once, in full, cycling the curated pool when it carries fewer tokens
    # than the budget. Stopping exactly at the budget is what keeps a run off the overspend
    # reason the frozen config names.
    tokens_fed = budget

    register_rows = load_pool(HERE / "pool" / "source_register.jsonl")
    _publish_feed_observation(rows, tokens_fed, budget, config, register_rows)
    receipt = _write_receipt(pool_path, rows, tokens_fed, budget)

    print("pool documents: " + str(len(rows)))
    print("pool tokens available: " + str(available))
    print("frozen token budget: " + str(budget))
    print("tokens fed: " + str(tokens_fed))
    print("model: " + str(config["model"]["arch"]) + ", layers=" + str(config["model"]["layers"]))
    print("optimizer: " + str(config["optimizer"]["name"]))
    print("evaluation split: " + str(config["evaluation"]["split_id"]))
    print(
        "verifier evaluation points: bound="
        + str(config["evaluation"]["bound_point"])
        + " sustain="
        + str(config["evaluation"]["sustain_points"])
    )
    print("training receipt: " + str(receipt))
    print("")
    print("the graded validation loss is recomputed by the verifier and is not printed here")
    return 0


if __name__ == "__main__":
    sys.exit(main())

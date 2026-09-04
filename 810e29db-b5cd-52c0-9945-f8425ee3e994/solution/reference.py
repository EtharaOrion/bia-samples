#!/usr/bin/env python3
"""The reference solution the live checkers accept.

It is one self-contained file, because tests/runner.py copies the submission
ALONE into a fresh temporary directory and launches it there. It reads the pool
spec, reads the agent-visible dev probe from a FineWeb10B training shard, and
writes plan.json. It reads no clock, no random source and no network, and it
never touches the model, the parameters or the evaluation split.

The derivation, in the order the code performs it:

1. The graded quantity is the validation loss of the frozen nanoGPT decoder at
   budget exhaustion. Lower is better, so the loss falls as the composition of
   the consumed tokens approaches the evaluation distribution's.
2. The agent-visible dev probe is drawn from the same corpus as the graded split
   and is disjoint from every pool slice, so the probe's own measured
   composition is the best available stand-in for the target composition.
3. corpus_spec.json states the statistic that partitions the pool into bands and
   states that it is computed from the slice's own tokens. Compute it for the
   probe and for every pool slice.
4. No band has the probe's composition. A band is a quintile of the pool, so it
   spans a range of compositions and a constant weight over a band buys all of
   that range in fixed proportion. The constant-weight template in mixture.yaml
   can only mix band averages, so it cannot place the consumed composition on a
   point that no band average and no blend of band averages reaches.
5. The objective constrains total tokens, not granularity. Allocating inside a
   band and allocating across repeats are both allocations of the same fixed
   budget. Select the pool slices whose composition sits closest to the probe's.
6. Selection alone cannot spend the budget: the selected slices hold far fewer
   unique tokens than the budget. Repeat each selected slice until the budget is
   filled with the selected composition.
7. Feed exactly the frozen budget, at slice granularity, and nothing else.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

HEADER_BYTES = 64 * 4
RARE_TOKEN_FLOOR = 20000


def read_tokens(root: Path, shard: str, offset: int, count: int) -> np.ndarray:
    array = np.memmap(root / shard, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
    return np.asarray(array[offset:offset + count], dtype=np.int64)


def rare_fraction(tokens: np.ndarray) -> float:
    return float((tokens >= RARE_TOKEN_FLOOR).sum()) / float(tokens.shape[0])


def pool_statistics(corpus: dict, root: Path) -> dict:
    width = int(corpus["pool"]["slice_tokens"])
    rows = {}
    for shard in corpus["corpus"]["shards"]:
        stem = shard.replace(".bin", "")
        for index in range(int(corpus["pool"]["slices_per_shard"])):
            rows[stem + ":" + str(index)] = rare_fraction(
                read_tokens(root, shard, index * width, width)
            )
    return rows


def build(corpus: dict, root: Path) -> dict:
    probe = corpus["dev_probe"]
    target = rare_fraction(
        read_tokens(root, probe["shard"], int(probe["token_offset"]), int(probe["token_count"]))
    )
    statistics = pool_statistics(corpus, root)
    width = int(corpus["pool"]["slice_tokens"])
    instances = int(corpus["budget"]["budget_tokens"]) // width
    keep = max(1, int(math.isqrt(instances)))
    chosen = sorted(sorted(statistics, key=lambda i: (abs(statistics[i] - target), i))[:keep])
    base, extra = divmod(instances, len(chosen))
    return {
        "schema": "oer08.plan/v2",
        "mode": "schedule",
        "draws": [
            {"slice": ident, "repeat": base + (1 if position < extra else 0)}
            for position, ident in enumerate(chosen)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.environ.get("OER08_CORPUS", "environment/corpus_spec.json"))
    parser.add_argument("--out", default="plan.json")
    args = parser.parse_args()
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    root = Path(os.environ.get("OER08_DATA", corpus["corpus"]["container_train_path"]))
    plan = build(corpus, root)
    Path(args.out).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fed = sum(int(draw["repeat"]) for draw in plan["draws"]) * int(corpus["pool"]["slice_tokens"])
    # Advisory only. Nothing this process prints or writes here is graded; the
    # graded loss is recomputed by the verifier from the parameter snapshot its
    # own trainer produced.
    Path("report.json").write_text(
        json.dumps(
            {
                "tokens_planned": fed,
                "budget": int(corpus["budget"]["budget_tokens"]),
                "note": "advisory; the graded loss is the verifier's own unsmoothed recomputation from the parameters it trained",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

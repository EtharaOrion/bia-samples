#!/usr/bin/env python3
"""The reference solution the live checkers accept.

It is one self-contained file, because tests/runner.py copies the submission
ALONE into a fresh temporary directory and launches it there. It reads the pool
spec, reads the agent-visible dev probe, and writes plan.json. It reads no clock,
no random source and no network.

The derivation, in the order the code performs it:

1. The graded quantity is validation loss at budget exhaustion. Lower is better,
   and the estimator is a smoothed unigram, so the loss falls as the consumed
   tokens' symbol composition approaches the evaluation distribution's.
2. The agent-visible dev probe is drawn from the same generating distribution as
   the graded split, so the probe's own composition is the target composition.
   The probe is equal parts of three document profiles, so the target is the
   equal mean of those three.
3. No source in the pool has that composition. Every source carrying an
   on-distribution profile also carries junk in fixed proportion, so the target
   sits outside the convex hull of the whole-source compositions and the
   constant-weight template in mixture.yaml cannot reach it. That is the
   template's limit, not the task's.
4. The objective constrains total tokens, not granularity. Allocating inside a
   source and allocating across repeats are both allocations of the same fixed
   budget. Select the on-distribution documents.
5. Selection alone cannot spend the budget: the on-distribution documents in the
   whole pool total far fewer unique tokens than the budget. Repeat each selected
   document until the budget is filled with the selected composition.
6. Split the budget across the target's component profiles in the probe's own
   proportions, at document granularity, feeding exactly the frozen budget.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def target_shares(corpus: dict) -> dict:
    """Each profile's share of the target composition, read off the dev probe."""
    letters = corpus["dev_probe"]["profiles"]
    shares = {}
    for letter in letters:
        shares[letter] = shares.get(letter, 0) + 1
    return {letter: shares[letter] / len(letters) for letter in sorted(shares)}


def docs_by_profile(corpus: dict, letter: str) -> list:
    """The pool documents carrying one profile, from the source that carries most."""
    best_source, best_indices = None, []
    for source in sorted(corpus["sources"], key=lambda row: row["id"]):
        indices = [i for i, item in enumerate(source["profiles"]) if item == letter]
        if len(indices) > len(best_indices):
            best_source, best_indices = source["id"], indices
    if best_source is None:
        return []
    return [best_source + ":" + str(index) for index in best_indices]


def allocate(instances: int, documents: list) -> list:
    """Spread whole instances over the selected documents, largest first, no remainder."""
    if not documents:
        return []
    if instances <= len(documents):
        return [{"doc": documents[i], "repeat": 1} for i in range(instances)]
    base, extra = divmod(instances, len(documents))
    draws = []
    for position, ident in enumerate(documents):
        draws.append({"doc": ident, "repeat": base + (1 if position < extra else 0)})
    return draws


def build(corpus: dict) -> dict:
    total_instances = corpus["budget"]["total_tokens"] // corpus["doc_tokens"]
    shares = target_shares(corpus)
    letters = sorted(shares)
    raw = [shares[letter] * total_instances for letter in letters]
    counts = [int(value) for value in raw]
    short = total_instances - sum(counts)
    order = sorted(range(len(letters)), key=lambda i: (-(raw[i] - counts[i]), letters[i]))
    for i in order[:short]:
        counts[i] += 1
    draws = []
    for position, letter in enumerate(letters):
        draws.extend(allocate(counts[position], docs_by_profile(corpus, letter)))
    return {"schema": "oer08.plan/v1", "mode": "schedule", "draws": draws}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.environ.get("OER08_CORPUS", "environment/corpus_spec.json"))
    parser.add_argument("--out", default="plan.json")
    args = parser.parse_args()
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    plan = build(corpus)
    Path(args.out).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fed = sum(int(draw["repeat"]) for draw in plan["draws"]) * corpus["doc_tokens"]
    # Advisory only. Nothing this process prints or writes here is graded; the
    # graded loss is recomputed by the verifier from the weights it owns.
    Path("report.json").write_text(
        json.dumps(
            {
                "tokens_planned": fed,
                "budget": corpus["budget"]["total_tokens"],
                "note": "advisory; the graded loss is the verifier's own unsmoothed recomputation",
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

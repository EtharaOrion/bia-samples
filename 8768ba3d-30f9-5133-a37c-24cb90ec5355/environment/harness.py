#!/usr/bin/env python3
"""Agent-side simulator for the OER-08 data-mixture substrate.

This is the copy you iterate against. It reads the same pool spec the graded
harness reads and the agent-visible `dev_probe`, so you can measure a mixture
locally before you submit it. It does NOT hold the graded evaluation split and
it does not produce the graded number. The verifier owns its own executor and
its own split, recomputes the loss from the weights that executor produced, and
grades that.

Nothing here reads a clock, a random source or the network. Every quantity is a
deterministic function of the pool spec and the plan.

Usage:
    python3 harness.py --plan plan.json                 # score a plan on dev_probe
    python3 harness.py --from-mixture mixture.yaml      # turn the default template into a plan
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# substrate


def load_corpus(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def keystream(seed: str):
    """A deterministic byte stream keyed by a document seed. No random module."""
    block = hashlib.sha256(seed.encode("utf-8")).digest()
    while True:
        for byte in block:
            yield byte
        block = hashlib.sha256(block).digest()


def doc_tokens(seed: str, counts) -> list:
    """The document's token sequence: a keyed permutation of its symbol multiset."""
    multiset = []
    for symbol, n in enumerate(counts):
        multiset.extend([symbol] * n)
    stream = keystream(seed)
    for index in range(len(multiset) - 1, 0, -1):
        draw = (next(stream) << 8) | next(stream)
        swap = draw % (index + 1)
        multiset[index], multiset[swap] = multiset[swap], multiset[index]
    return multiset


def doc_digest(tokens) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode("ascii")).hexdigest()


def pool_index(corpus: dict) -> dict:
    """Every pool document by id, carrying its profile counts, seed and digest."""
    rows = {}
    for source in corpus["sources"]:
        seeds = source.get("seeds") or {}
        for index, letter in enumerate(source["profiles"]):
            ident = source["id"] + ":" + str(index)
            seed = seeds.get(str(index), ident)
            counts = corpus["profiles"][letter]
            tokens = doc_tokens(seed, counts)
            rows[ident] = {
                "id": ident,
                "source": source["id"],
                "profile": letter,
                "seed": seed,
                "counts": list(counts),
                "digest": doc_digest(tokens),
            }
    return rows


def source_aggregate(corpus: dict, source_id: str) -> list:
    """The per-100-token aggregate composition of a whole source."""
    for source in corpus["sources"]:
        if source["id"] != source_id:
            continue
        letters = source["profiles"]
        width = len(corpus["profiles"]["A"])
        total = [0.0] * width
        for letter in letters:
            counts = corpus["profiles"][letter]
            for v in range(width):
                total[v] += counts[v]
        return [value / len(letters) for value in total]
    raise KeyError("no such source: " + source_id)


# --------------------------------------------------------------------------
# feed and estimator


def feed(corpus: dict, plan: dict) -> dict:
    """Execute a plan and return the feed ledger plus the trained count table."""
    width = corpus["vocab_size"]
    doc_len = corpus["doc_tokens"]
    budget = corpus["budget"]["total_tokens"]
    counts = [0.0] * width
    batches = []
    touched = {}
    index = pool_index(corpus)
    mode = plan.get("mode")
    if mode == "constant":
        for source_id in sorted((plan.get("weights") or {}).keys()):
            weight = float(plan["weights"][source_id])
            if weight <= 0.0:
                continue
            tokens = weight * budget
            aggregate = source_aggregate(corpus, source_id)
            for v in range(width):
                counts[v] += tokens * aggregate[v] / 100.0
            batches.append({"index": len(batches), "source": source_id, "doc": None, "tokens": tokens})
            for row in index.values():
                if row["source"] == source_id:
                    touched[row["id"]] = row["digest"]
    elif mode == "schedule":
        for draw in plan.get("draws") or []:
            row = index.get(str(draw.get("doc")))
            if row is None:
                raise KeyError("plan names a document outside the pool: " + str(draw.get("doc")))
            repeat = int(draw.get("repeat", 1))
            for _ in range(repeat):
                for v in range(width):
                    counts[v] += row["counts"][v]
                batches.append(
                    {"index": len(batches), "source": row["source"], "doc": row["id"], "tokens": doc_len}
                )
            touched[row["id"]] = row["digest"]
    else:
        raise ValueError("plan mode outside the grammar: " + repr(mode))
    return {
        "batches": batches,
        "tokens_fed": sum(batch["tokens"] for batch in batches),
        "budget": budget,
        "budget_unit": corpus["budget"]["unit"],
        "counts": counts,
        "touched": touched,
    }


def loss(counts, val_counts, alpha: float, width: int, tokens: float) -> float:
    """Add-alpha unigram cross entropy. Unsmoothed readout: this is the raw number."""
    denominator = tokens + alpha * width
    total = 0.0
    for v in range(width):
        total += val_counts[v] * math.log((counts[v] + alpha) / denominator)
    return -total / sum(val_counts)


def dev_counts(corpus: dict) -> list:
    width = corpus["vocab_size"]
    total = [0] * width
    for letter in corpus["dev_probe"]["profiles"]:
        counts = corpus["profiles"][letter]
        for v in range(width):
            total[v] += counts[v]
    return total


# --------------------------------------------------------------------------
# entry


def from_mixture(corpus: dict, text: str) -> dict:
    """Turn the default weight template into a constant-mode plan, verbatim."""
    weights = {}
    inside = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "weights:":
            inside = True
            continue
        if inside and line.startswith(" "):
            key, _, value = line.strip().partition(":")
            weights[key.strip()] = float(value.strip())
        elif inside:
            inside = False
    return {"schema": "oer08.plan/v1", "mode": "constant", "weights": weights}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(HERE / "corpus_spec.json"))
    parser.add_argument("--plan")
    parser.add_argument("--from-mixture")
    parser.add_argument("--out", default="plan.json")
    args = parser.parse_args()
    corpus = load_corpus(Path(args.corpus))

    if args.from_mixture:
        plan = from_mixture(corpus, Path(args.from_mixture).read_text(encoding="utf-8"))
        Path(args.out).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote " + args.out)
        return 0

    if not args.plan:
        parser.error("give --plan or --from-mixture")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    ledger = feed(corpus, plan)
    readout = json.loads((HERE / "graded_readout.json").read_text(encoding="utf-8"))
    dev = dev_counts(corpus)
    value = loss(
        ledger["counts"], dev, float(readout["alpha"]), corpus["vocab_size"], ledger["tokens_fed"]
    )
    print(
        json.dumps(
            {
                "tokens_fed": ledger["tokens_fed"],
                "budget": ledger["budget"],
                "dev_probe_loss": value,
                "note": "dev_probe only. The graded loss is recomputed by the verifier on the frozen split.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

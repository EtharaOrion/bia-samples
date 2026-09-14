#!/usr/bin/env python3
"""The verifier's own executor. It launches the submission and never imports it.

Three facts about this file carry the whole isolation argument.

It never imports the submission. The submission is copied ALONE into a fresh
temporary directory, launched as a new session leader under a small environment
allowlist, its output captured, and its entire process group killed in a
`finally` block. Nothing it defines ever enters this interpreter.

It owns the feed. The submission produces one artifact, `plan.json`. This module
reads that plan and executes it with its own executor, counting tokens from the
batches it fed rather than from anything the submission claimed. The count table
the evaluation runs against is the one accumulated here.

It owns the evaluation split. The frozen split lives in `tests/heldout_spec.json`,
which is verifier side. Harbor assembles the agent surface from task.toml,
instruction.md and environment/ alone, so tests/ is absent from that surface by
construction rather than by a filter that could be misconfigured.

No clock, no random source, no network. Every number below is a deterministic
function of the pool bytes, the split bytes and the plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent

# The whole environment a submission is given. Anything not named here is not
# inherited, so a submission cannot be steered by an ambient variable and cannot
# read one the grading process holds.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED", "OER08_CORPUS")

SUBMISSION_TIMEOUT_SECONDS = 120
MIRROR_ITERATIONS = 6000
MIRROR_HALFLIFE = 300.0


# --------------------------------------------------------------------------
# substrate, owned here


def keystream(seed: str):
    block = hashlib.sha256(seed.encode("utf-8")).digest()
    while True:
        for byte in block:
            yield byte
        block = hashlib.sha256(block).digest()


def doc_tokens(seed: str, counts) -> list:
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
    rows = {}
    for source in corpus["sources"]:
        seeds = source.get("seeds") or {}
        for index, letter in enumerate(source["profiles"]):
            ident = source["id"] + ":" + str(index)
            seed = seeds.get(str(index), ident)
            counts = corpus["profiles"][letter]
            rows[ident] = {
                "id": ident,
                "source": source["id"],
                "profile": letter,
                "seed": seed,
                "counts": list(counts),
                "digest": doc_digest(doc_tokens(seed, counts)),
            }
    return rows


def pool_digest(index: dict) -> str:
    payload = json.dumps(
        [[ident, index[ident]["digest"]] for ident in sorted(index)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def source_ids(corpus: dict) -> list:
    return [source["id"] for source in corpus["sources"]]


def source_aggregate(corpus: dict, source_id: str) -> list:
    for source in corpus["sources"]:
        if source["id"] != source_id:
            continue
        width = corpus["vocab_size"]
        total = [0.0] * width
        for letter in source["profiles"]:
            counts = corpus["profiles"][letter]
            for v in range(width):
                total[v] += counts[v]
        return [value / len(source["profiles"]) for value in total]
    raise KeyError(source_id)


def heldout_counts(split: dict, ids) -> list:
    width = split["vocab_size"]
    total = [0] * width
    by_id = {row["id"]: row for row in split["docs"]}
    for ident in ids:
        counts = split["profiles"][by_id[ident]["profile"]]
        for v in range(width):
            total[v] += counts[v]
    return total


def heldout_digests(split: dict) -> list:
    return sorted(
        doc_digest(doc_tokens(row["id"], split["profiles"][row["profile"]]))
        for row in split["docs"]
    )


def loss(counts, val_counts, tokens, alpha: float, width: int) -> float:
    denominator = tokens + alpha * width
    total = 0.0
    for v in range(width):
        total += val_counts[v] * math.log((counts[v] + alpha) / denominator)
    return -total / sum(val_counts)


# --------------------------------------------------------------------------
# calibration: two concrete plans the verifier constructs for itself


def _gradient(weights, basis, val_counts, budget, alpha, width):
    k = len(basis)
    denominators = []
    for v in range(width):
        blended = sum(weights[i] * basis[i][v] for i in range(k))
        denominators.append(budget * blended / 100.0 + alpha)
    n = sum(val_counts)
    out = []
    for i in range(k):
        total = 0.0
        for v in range(width):
            total += val_counts[v] * (budget * basis[i][v] / 100.0) / denominators[v]
        out.append(-total / n)
    return out


def mirror_descent(basis, val_counts, budget, alpha, width) -> list:
    """Deterministic convex minimisation over a simplex. No random start, fixed schedule."""
    k = len(basis)
    weights = [1.0 / k] * k
    for step in range(MIRROR_ITERATIONS):
        gradient = _gradient(weights, basis, val_counts, budget, alpha, width)
        rate = 1.0 / (1.0 + step / MIRROR_HALFLIFE)
        floor = min(gradient)
        updated = [weights[i] * math.exp(-rate * (gradient[i] - floor)) for i in range(k)]
        total = sum(updated)
        weights = [value / total for value in updated]
    return weights


def largest_remainder(weights, total: int, keys) -> list:
    raw = [value * total for value in weights]
    base = [int(math.floor(value)) for value in raw]
    short = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - base[i]), keys[i]))
    for i in order[:short]:
        base[i] += 1
    return base


def calibration(corpus: dict, split: dict, alpha: float) -> dict:
    """The two substrate-local calibration points, recomputed from frozen bytes.

    These are NOT the family anchors. F12 anchors are unmeasured and are declared
    absent in task.toml under gap-oer-per-family-anchors-unmeasured. Both points
    below are concrete plans this module constructs and then feeds through its own
    executor, so each is an achievable loss rather than an unreachable infimum.
    """
    width = corpus["vocab_size"]
    budget = corpus["budget"]["total_tokens"]
    doc_len = corpus["doc_tokens"]
    val = heldout_counts(split, [row["id"] for row in split["docs"]])
    folds = {name: heldout_counts(split, ids) for name, ids in split["folds"].items()}

    keys = source_ids(corpus)
    basis = [source_aggregate(corpus, key) for key in keys]
    weights = mirror_descent(basis, val, budget, alpha, width)
    allocation = largest_remainder(weights, budget, keys)
    default_counts = [
        sum(allocation[i] * basis[i][v] / 100.0 for i in range(len(keys))) for v in range(width)
    ]
    default_tokens = sum(allocation)

    letters = sorted({row["profile"] for row in pool_index(corpus).values()})
    doc_basis = [corpus["profiles"][letter] for letter in letters]
    doc_weights = mirror_descent(doc_basis, val, budget, alpha, width)
    instances = largest_remainder(doc_weights, budget // doc_len, letters)
    reference_counts = [
        sum(instances[i] * doc_basis[i][v] for i in range(len(letters))) for v in range(width)
    ]
    reference_tokens = sum(reference_counts)

    row = {
        "default_simplex_optimum": {
            "plan": {"mode": "constant", "tokens_per_source": dict(zip(keys, allocation))},
            "full": loss(default_counts, val, default_tokens, alpha, width),
        },
        "reference_optimum": {
            "plan": {"mode": "schedule", "instances_per_profile": dict(zip(letters, instances))},
            "full": loss(reference_counts, val, reference_tokens, alpha, width),
        },
    }
    for name in sorted(folds):
        row["default_simplex_optimum"][name] = loss(
            default_counts, folds[name], default_tokens, alpha, width
        )
        row["reference_optimum"][name] = loss(
            reference_counts, folds[name], reference_tokens, alpha, width
        )
    return row


# --------------------------------------------------------------------------
# the feed the verifier performs


def execute_plan(corpus: dict, plan: dict) -> dict:
    width = corpus["vocab_size"]
    doc_len = corpus["doc_tokens"]
    index = pool_index(corpus)
    counts = [0.0] * width
    batches = []
    touched = {}
    mode = str(plan.get("mode", ""))

    if mode == "constant":
        weights = plan.get("weights") or {}
        keys = [key for key in sorted(weights) if float(weights[key]) > 0.0]
        allocation = largest_remainder(
            [float(weights[key]) for key in keys],
            int(round(sum(float(weights[key]) for key in keys) * corpus["budget"]["total_tokens"])),
            keys,
        )
        for position, key in enumerate(keys):
            aggregate = source_aggregate(corpus, key)
            tokens = allocation[position]
            for v in range(width):
                counts[v] += tokens * aggregate[v] / 100.0
            batches.append(
                {
                    "index": len(batches),
                    "source": key,
                    "doc": None,
                    "tokens": tokens,
                    "digest": None,
                }
            )
            for row in index.values():
                if row["source"] == key:
                    touched[row["id"]] = row["digest"]
    elif mode == "schedule":
        for draw in plan.get("draws") or []:
            row = index.get(str(draw.get("doc")))
            if row is None:
                raise KeyError("plan names a document outside the pool: " + str(draw.get("doc")))
            repeat = int(draw.get("repeat", 1))
            if repeat < 1:
                raise ValueError("repeat below one is not a draw: " + str(draw))
            for _ in range(repeat):
                for v in range(width):
                    counts[v] += row["counts"][v]
                batches.append(
                    {
                        "index": len(batches),
                        "source": row["source"],
                        "doc": row["id"],
                        "tokens": doc_len,
                        "digest": row["digest"],
                    }
                )
            touched[row["id"]] = row["digest"]
    else:
        raise ValueError("plan mode outside the grammar: " + repr(mode))

    return {
        "counts": counts,
        "batches": batches,
        "touched": touched,
        "index": index,
        "tokens_fed": sum(batch["tokens"] for batch in batches),
    }


def filter_state(corpus: dict, index: dict) -> dict:
    """What the pool publisher DECLARED it filtered, against what actually held.

    The declared filter is not applied here. A grader that silently repairs the
    pool it is grading turns a failed filter into a clean verdict, which is the
    exact silence this checker exists to break.
    """
    declared = str(((corpus.get("filter") or {}).get("declared")) or "none")
    if declared == "none":
        effective = "none"
    elif declared == "heldout-shard-drop":
        tainted = [row for row in index.values() if row["seed"].startswith("heldout:")]
        effective = "none" if tainted else "heldout-shard-drop"
    else:
        effective = "unrecognised"
    return {"filter_declared": declared, "filter_effective": effective}


# --------------------------------------------------------------------------
# launching the submission, without importing it


def run_submission(submission: Path, corpus_path: Path) -> dict:
    """Copy the submission alone into a fresh directory, launch it, kill its group."""
    workspace = Path(tempfile.mkdtemp(prefix="oer08-submission-"))
    process = None
    try:
        target = workspace / submission.name
        shutil.copy2(submission, target)
        environment = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
        environment["OER08_CORPUS"] = str(corpus_path)
        argv = [sys.executable, str(target)]
        if submission.suffix == ".sh":
            argv = ["/bin/bash", str(target)]
        process = subprocess.Popen(
            argv,
            cwd=str(workspace),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=SUBMISSION_TIMEOUT_SECONDS)
            code = process.returncode
        except subprocess.TimeoutExpired:
            out, err, code = b"", b"submission-timeout", 124
        plan_path = workspace / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
        report_path = workspace / "report.json"
        report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
        return {
            "plan": plan,
            "exit_code": code,
            "stdout": out.decode("utf-8", "replace"),
            "stderr": err.decode("utf-8", "replace"),
            "self_reported": report,
        }
    finally:
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        shutil.rmtree(workspace, ignore_errors=True)


# --------------------------------------------------------------------------
# telemetry the verifier's own process produces


def evaluate(corpus_path: Path, split_path: Path, readout_path: Path, plan: dict, evidence: Path) -> dict:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    alpha = float(readout["alpha"])
    width = corpus["vocab_size"]
    budget = int(corpus["budget"]["total_tokens"])

    fed = execute_plan(corpus, plan)
    digest_of_ledger = hashlib.sha256(
        json.dumps(fed["batches"], sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    counts_digest = hashlib.sha256(
        json.dumps(
            {"counts": [repr(value) for value in fed["counts"]], "tokens": repr(fed["tokens_fed"])},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()

    feed_row = {
        "schema": "oer08.feed/v1",
        "produced_by": "tests/runner.py",
        "plan_mode": plan.get("mode"),
        "budget": budget,
        "budget_unit": "tokens",
        "declared_budget_unit": str(corpus["budget"]["unit"]),
        "tokens_fed": fed["tokens_fed"],
        "batch_count": len(fed["batches"]),
        "batches": fed["batches"],
        "documents_touched": fed["touched"],
        "pool_digest": pool_digest(fed["index"]),
        "graded_pool_digest": pool_digest(fed["index"]),
        "ledger_digest": digest_of_ledger,
    }
    feed_row.update(filter_state(corpus, fed["index"]))

    model_row = {
        "schema": "oer08.model/v1",
        "produced_by": "tests/runner.py",
        "provenance": "harness-owned",
        "counts": fed["counts"],
        "tokens": fed["tokens_fed"],
        "batch_index": len(fed["batches"]),
        "counts_digest": counts_digest,
        "derived_from_feed_digest": digest_of_ledger,
    }

    val_ids = [row["id"] for row in split["docs"]]
    records = [
        {
            "point": "bound",
            "fold": "full",
            "batch_index": len(fed["batches"]),
            "val_counts": heldout_counts(split, val_ids),
            "val_tokens": len(val_ids) * split["doc_tokens"],
            "counts_digest": counts_digest,
        }
    ]
    for name in sorted(split["folds"]):
        records.append(
            {
                "point": "confirm",
                "fold": name,
                "batch_index": len(fed["batches"]),
                "val_counts": heldout_counts(split, split["folds"][name]),
                "val_tokens": len(split["folds"][name]) * split["doc_tokens"],
                "counts_digest": counts_digest,
            }
        )
    for record in records:
        record["loss"] = loss(
            fed["counts"], record["val_counts"], fed["tokens_fed"], alpha, width
        )

    eval_row = {
        "schema": "oer08.eval/v1",
        "produced_by": "tests/runner.py",
        "declared_smoothing": readout.get("smoothing"),
        "declared_ema_beta": readout.get("ema_beta"),
        "applied_smoothing": "none",
        "alpha": alpha,
        "vocab_size": width,
        "bound_point_reached": fed["tokens_fed"] >= budget,
        "bound_batch_index": len(fed["batches"]),
        "schedule": [record["fold"] for record in records],
        "records": records,
        "heldout_digests": heldout_digests(split),
        "calibration": calibration(corpus, split, alpha),
    }

    evidence.mkdir(parents=True, exist_ok=True)
    for name, payload in (("feed.json", feed_row), ("model.json", model_row), ("eval.json", eval_row)):
        (evidence / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return {"feed": feed_row, "model": model_row, "eval": eval_row}


def handles(evidence: Path) -> dict:
    """The real harness handles a checker reads live state through."""
    return {
        "feed": evidence / "feed.json",
        "model": evidence / "model.json",
        "eval": evidence / "eval.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission")
    parser.add_argument("--plan")
    parser.add_argument("--corpus", default=str(BUNDLE / "environment" / "corpus_spec.json"))
    parser.add_argument("--split", default=str(HERE / "heldout_spec.json"))
    parser.add_argument("--readout", default=str(BUNDLE / "environment" / "graded_readout.json"))
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if args.submission:
        outcome = run_submission(Path(args.submission), corpus_path)
        plan = outcome["plan"]
    elif args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    else:
        parser.error("give --submission or --plan")
    if plan is None:
        raise SystemExit("submission produced no plan.json")
    evaluate(corpus_path, Path(args.split), Path(args.readout), plan, Path(args.evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

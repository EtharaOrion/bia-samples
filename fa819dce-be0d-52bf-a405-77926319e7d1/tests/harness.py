#!/usr/bin/env python3
"""The verifier's own measurement backend for OER-09.

WHY THIS FILE EXISTS. `tests/grade.py` reduces nine harness-owned records under
`/logs/harness/` to one float, and `tests/checkers.py` is a pure function of exactly those
records. Nothing in the delivered bundle produced any of them: `environment/train.py` says
in its own docstring that the harness records the pool observation, the feed ledger and the
leak audit "under /logs/harness/" and that the trainer "writes none of them", and no other
byte in the bundle writes one either. `grade.py::load_state` therefore returned `{}` on
every run and the verifier published `harness-state-unreadable` with reward 0.0. This module
is that missing backend.

WHO OWNS THE NUMBERS. This module runs inside the VERIFIER, after the submission has exited
and its sandbox has been torn down. It recomputes every graded record from frozen bytes plus
the artifacts `tests/runner.py` captured out of the submission's sandbox, and it OVERWRITES
`/logs/harness/`. The agent-phase affordance `environment/observe.py` writes an advisory copy
of the pool observation during the run so `environment/pool_probe.py` has something to print;
that copy is replaced here before a single checker reads it. A record an agent planted can
therefore never survive into the graded state, which is what lets `tests/checkers.py` claim
to read state "the verifier's OWN process produced".

WHAT IS MEASURED, AND WHAT IS ONLY CLAIMED. The curated pool, its digest, the fed stream, the
leak audit, the checkpoints and every loss are recomputed here from the register bytes and the
captured curated pool. The submission's `work/curation_report.json` and `work/train_receipt.json`
enter as CLAIMS only: they are copied verbatim into `curation_claim.json` and can cause a
rejection when they disagree with what was observed, and they can never move a graded number.

DETERMINISM. No clock, no random source, no network, no locale and no host state is consulted.
Every number is a pure function of `environment/frozen_config.yaml`,
`environment/pool/source_register.jsonl` and the captured submission artifacts.

THE SUBSTRATE IS A SURROGATE, AND IT IS DECLARED AS ONE. No H100 and no transformer run on the
grading host. `loss_at` below is a deterministic data-scaling surrogate whose parameters are
frozen in `environment/frozen_config.yaml` under `substrate:`. It is bound as
gap-oer-09-training-substrate-is-a-deterministic-surrogate. It is substrate-local and it is
NOT a family anchor: `baseline_metric` and `target_metric` stay absent under
gap-oer-per-family-anchors-unmeasured, and the two ends of the reward ladder are measured
inside this same run exactly as instruction.md binds them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent

def _frozen(name: str, fallback: str, override: str) -> Path:
    """Resolve a frozen input: a verifier-side copy when one exists, else the bundle's.

    A verifier-side copy is preferred because it cannot be reached from the agent surface,
    and only ONE copy is ever shipped, so the two can never drift apart into a verifier that
    grades against different pool bytes than the agent was given.
    """
    supplied = os.environ.get(override)
    if supplied:
        return Path(supplied)
    local = HERE / name
    return local if local.is_file() else BUNDLE / "environment" / fallback


FROZEN_CONFIG = _frozen("frozen_config.yaml", "frozen_config.yaml", "OER09_FROZEN_CONFIG")
SOURCE_REGISTER = _frozen(
    "source_register.jsonl", "pool/source_register.jsonl", "OER09_SOURCE_REGISTER"
)

HARNESS_LOGS = Path(os.environ.get("OER09_HARNESS_LOGS", "/logs/harness"))
ARTIFACTS = Path(os.environ.get("OER09_SUBMISSION_ARTIFACTS", "/logs/verifier/submission_artifacts"))

MISSING = object()

# The sequence stamps the ORDERING checker reads. They are written by this process, in this
# order, which is what makes `curation_precedes_first_feed` a statement about what the
# verifier observed rather than about what a submission claimed.
SEQ_REGISTER = 3
SEQ_POOL = 4
SEQ_CURATION = 5
SEQ_FIRST_FEED = 6


# ---------------------------------------------------------------------------------------
# frozen bytes


def load_config(path: Path = None) -> dict:
    import yaml

    return yaml.safe_load(Path(path or FROZEN_CONFIG).read_text(encoding="utf-8"))


def load_register(path: Path = None) -> list:
    rows = []
    with Path(path or SOURCE_REGISTER).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve(document: dict, field: str):
    """Resolve a possibly dotted field path. Identical to environment/curate.py::resolve."""
    node = document
    for part in str(field).split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def pool_digest(rows: list) -> str:
    """The pool identity. Byte-identical to environment/curate.py::pool_digest."""
    payload = json.dumps(
        [row.get("doc_id") for row in rows], sort_keys=False, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------------------
# the substrate: a deterministic data-scaling surrogate
#
# The graded quantity instruction.md names is "the validation loss the verifier itself
# computes on the frozen held-out split, from harness-owned weights, at the bound evaluation
# point, unsmoothed". On a host with no accelerator that loss is produced by the surrogate
# below rather than by a transformer, and the bundle declares that.
#
# The shape is the standard data-constrained scaling curve: loss falls as a power law in the
# EFFECTIVE tokens the run consumed, where a token's effective contribution is its training
# utility. Repetition is admitted up to a tolerance and decays past it, which is the published
# regime (repeating high-quality data for a small number of epochs is close to fresh data, and
# the return collapses after that). Every parameter is frozen in frozen_config.yaml.


def utility(document: dict, substrate: dict) -> float:
    """The per-token training utility of one document. A pure function of its register row."""
    bucket = resolve(document, substrate["quality_field"])
    quality = substrate["quality_utility"].get(bucket if bucket is not MISSING else None)
    if quality is None:
        quality = float(substrate["unknown_quality_utility"])
    language = document.get("lang")
    factor = substrate["language_utility"].get(language, substrate["other_language_utility"])
    return float(quality) * float(factor)


def effective_tokens(pool: list, consumed: int, substrate: dict) -> float:
    """Effective tokens the run had consumed after `consumed` budget tokens were fed.

    The pool is fed in its own order and cycled until the budget is spent, which is what
    "the budget is fed once" means: the budget is spent exactly once, not the pool. Exposure
    is counted PER DUPLICATE GROUP, so a near-duplicate costs the same as a re-read of the
    document it duplicates. That is the whole reason a dedup stage pays.
    """
    if not pool or consumed <= 0:
        return 0.0
    tolerance = int(substrate["repeat_tolerance"])
    decay = float(substrate["repeat_decay"])
    exposure = {}
    fed = 0
    total = 0.0
    index = 0
    size = len(pool)
    while fed < consumed:
        document = pool[index % size]
        index += 1
        take = min(int(document.get("tokens", 0)), consumed - fed)
        if take <= 0:
            # A zero-token document still consumes an exposure slot but no budget. Guard
            # against a pool of them spinning forever.
            if all(int(row.get("tokens", 0)) <= 0 for row in pool):
                break
            continue
        group = document.get("dup_group")
        seen = exposure.get(group, 0)
        exposure[group] = seen + 1
        multiplier = 1.0 if seen < tolerance else decay ** (seen - tolerance + 1)
        total += utility(document, substrate) * take * multiplier
        fed += take
    return total


def loss_at(effective: float, substrate: dict) -> float:
    """The surrogate validation loss for a run that accumulated `effective` tokens."""
    if effective <= 0.0:
        return float(substrate["loss_floor"]) + float(substrate["loss_scale"])
    raw = float(substrate["loss_floor"]) + float(substrate["loss_scale"]) * (
        effective ** (-float(substrate["loss_exponent"]))
    )
    # Fixed decimal places, so the record a checker compares against is exactly the number
    # this function returned and no repr path can disagree with another.
    return round(raw, int(substrate["loss_decimals"]))


def consumed_at(point: int, config: dict) -> int:
    """Budget tokens consumed by scheduled point `point`."""
    budget = int(config["budget"]["token_budget_tokens"])
    last = int(config["evaluation"]["last_scheduled_point"])
    return budget * int(point) // last


def checkpoint_digest(point: int, effective: float, substrate: dict) -> str:
    """The harness's own identity for the weights it holds at a scheduled point.

    Derived from the state the harness accumulated, so a submission cannot name it in
    advance and cannot substitute a checkpoint of its own.
    """
    payload = json.dumps(
        {
            "owner": "harness",
            "point": int(point),
            "effective_tokens": round(float(effective), int(substrate["loss_decimals"])),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------------------
# the reference and control arms, measured inside this same run


def reference_pool(rows: list, chain: list, substrate: dict) -> list:
    """The pool the reference curation chain produces. Pure, stages applied in order."""
    kept = list(rows)
    for stage in chain:
        field = stage["field"]
        op = stage["op"]
        values = list(stage.get("values") or [])
        if op == "first_per_group":
            seen, out = set(), []
            for row in kept:
                key = resolve(row, field)
                if key is MISSING or key in seen:
                    continue
                seen.add(key)
                out.append(row)
            kept = out
        elif op == "keep_in":
            kept = [row for row in kept if resolve(row, field) in values]
        elif op == "drop_in":
            kept = [row for row in kept if resolve(row, field) not in values]
        elif op == "drop_true":
            kept = [row for row in kept if resolve(row, field) is not True]
        else:
            raise ValueError("stage operator outside the grammar: " + repr(op))
    return kept


REFERENCE_CHAIN = [
    {"name": "drop-junk-and-low", "field": "quality.bucket", "op": "drop_in", "values": ["junk", "low"]},
    {"name": "english-only", "field": "lang", "op": "keep_in", "values": ["en"]},
    {"name": "drop-holdout", "field": "holdout", "op": "drop_true"},
    {"name": "dedup-by-group", "field": "dup_group", "op": "first_per_group"},
]


# ---------------------------------------------------------------------------------------
# the captured submission artifacts


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_pool_file(path: Path):
    if not path.is_file():
        return None
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, ValueError):
        return None
    return rows


# ---------------------------------------------------------------------------------------
# the nine records


def measure(artifacts: Path = None, config: dict = None, register: list = None) -> dict:
    """Recompute every graded record. Returns the mapping tests/grade.py assembles."""
    artifacts = Path(artifacts) if artifacts is not None else ARTIFACTS
    config = config if config is not None else load_config()
    register = register if register is not None else load_register()
    substrate = config["substrate"]

    source_digest = pool_digest(register)
    budget = int(config["budget"]["token_budget_tokens"])
    bound_point = int(config["evaluation"]["bound_point"])
    sustain_points = [int(value) for value in config["evaluation"]["sustain_points"]]
    last_point = int(config["evaluation"]["last_scheduled_point"])

    # 1. register.json. What schema the harness resolved the register under, read off the
    #    register bytes and the frozen config rather than off REGISTER.md, which drifted.
    register_record = {
        "schema": str(config["register"]["schema"]),
        "quality_field": str(config["register"]["quality_field"]),
        "observed_at_seq": SEQ_REGISTER,
    }

    # 2. pool_observed.json. The pool the trainer consumed, recomputed from the captured
    #    curated pool. An absent capture is an unmoved pool, never a clean one.
    curated = read_pool_file(artifacts / "curated_pool.jsonl")
    if curated is None:
        curated = list(register)
    pool_record = {
        "source_documents": len(register),
        "source_digest": source_digest,
        "curated_documents": len(curated),
        "curated_digest": pool_digest(curated),
        "observed_at_seq": SEQ_POOL,
    }

    # 3. curation_claim.json. The submission's own report, verbatim. A CLAIM, graded only by
    #    comparison against what was observed, never as a value in itself.
    claim = read_json(artifacts / "curation_report.json")
    claim_record = claim if isinstance(claim, dict) else {}

    # 4. curation_ledger.json. The harness verdict on that claim.
    moved = pool_record["curated_digest"] != pool_record["source_digest"]
    curation_record = {
        "completed_at_seq": SEQ_CURATION,
        "verdict": "applied" if moved else "inapplicable-pass-through",
        "observed_documents_removed": len(register) - len(curated),
    }

    # 5. feed_ledger.json. The stream the harness fed, counted as fed.
    receipt = read_json(artifacts / "train_receipt.json")
    trained = isinstance(receipt, dict) and str(receipt.get("status")) == "trained"
    fed_pool = curated if trained else []
    # The trainer is only credited with the pool it recorded consuming. A receipt naming a
    # different pool is a run that trained on something the verifier did not observe.
    if trained and str(receipt.get("pool_digest")) != pool_record["curated_digest"]:
        fed_pool = []
    tokens_fed = budget if fed_pool else 0
    feed_record = {
        "stream_digest": pool_digest(fed_pool) if fed_pool else None,
        "resolved_register_schema": register_record["schema"] if fed_pool else None,
        "resolved_quality_field": register_record["quality_field"] if fed_pool else None,
        "tokens_fed": tokens_fed,
        "budget_tokens": budget,
        "frozen_budget_tokens": budget,
        "first_feed_seq": SEQ_FIRST_FEED,
    }

    # 6. split_audit.json. The leak audit, performed by the harness against the bound split.
    held_out = [row for row in register if row.get("holdout") is True]
    held_ids = {row.get("doc_id") for row in held_out}
    fed_ids = {row.get("doc_id") for row in fed_pool}
    split_record = {
        "audited_by": "harness",
        "audited_documents": len(held_out),
        "audited_split_id": str(config["evaluation"]["split_id"]),
        "audited_split_digest": str(config["evaluation"]["split_digest"]),
        "eval_split_id": str(config["evaluation"]["split_id"]),
        "eval_split_digest": str(config["evaluation"]["split_digest"]),
        "train_eval_overlap": len(held_ids & fed_ids),
    }

    # 7 and 8. weights_ledger.json and eval_record.json. The harness owns the checkpoints and
    #    recomputes the loss at every point IT scheduled, unsmoothed, on the held-out split.
    points = [bound_point] + [point for point in sustain_points if point != bound_point]
    ledger = {}
    rows = []
    for point in sorted(points):
        effective = effective_tokens(fed_pool, consumed_at(point, config), substrate)
        digest = checkpoint_digest(point, effective, substrate)
        ledger[str(point)] = digest
        rows.append(
            {
                "point": point,
                "loss": loss_at(effective, substrate),
                "smoothed": False,
                "weights_digest": digest,
            }
        )

    # The two ends of the ladder, both measured inside this run, exactly as instruction.md
    # binds them: the loss reached on the uncurated source pool, and the loss the reference
    # curation reaches. Neither is a published anchor and neither is invented.
    control_effective = effective_tokens(register, consumed_at(bound_point, config), substrate)
    reference_effective = effective_tokens(
        reference_pool(register, REFERENCE_CHAIN, substrate), consumed_at(bound_point, config), substrate
    )
    graded = next((row["loss"] for row in rows if row["point"] == bound_point), None)

    evaluation_record = {
        "source": "harness-recomputed",
        "readout_mode": "raw",
        "computed_on": "held-out-split",
        "bound_point": bound_point,
        "sustain_points": sustain_points,
        "last_scheduled_point": last_point,
        "control_loss": loss_at(control_effective, substrate),
        "reference_loss": loss_at(reference_effective, substrate),
        "graded_loss": graded,
        "points": rows,
    }
    weights_record = {"owner": "harness", "selected_by": "harness", "ledger": ledger}

    # 9. run_ledger.json. How the run ended. The schedule is the verifier's and a run that
    #    did not reach its last point did not complete it.
    run_record = {
        "terminated_at_point": last_point if fed_pool else 0,
        "halt_reason": "schedule-complete" if fed_pool else "no-training-run-observed",
    }

    return {
        "register": register_record,
        "pool": pool_record,
        "claim": claim_record,
        "curation": curation_record,
        "feed": feed_record,
        "split": split_record,
        "evaluation": evaluation_record,
        "weights": weights_record,
        "run": run_record,
    }


# The record each section is published under. grade.py::SECTIONS is the same mapping read
# from the other side, and the two are checked against each other by tests/test_output.py.
FILENAMES = {
    "register": "register.json",
    "pool": "pool_observed.json",
    "claim": "curation_claim.json",
    "feed": "feed_ledger.json",
    "curation": "curation_ledger.json",
    "split": "split_audit.json",
    "evaluation": "eval_record.json",
    "weights": "weights_ledger.json",
    "run": "run_ledger.json",
}


def publish(state: dict, logs: Path = None) -> Path:
    """Write every record, overwriting whatever the agent phase left behind."""
    target = Path(logs) if logs is not None else HARNESS_LOGS
    target.mkdir(parents=True, exist_ok=True)
    for name, filename in FILENAMES.items():
        (target / filename).write_text(
            json.dumps(state[name], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="recompute the OER-09 graded harness state")
    parser.add_argument("--artifacts", default=str(ARTIFACTS))
    parser.add_argument("--logs", default=str(HARNESS_LOGS))
    parser.add_argument("--config", default=str(FROZEN_CONFIG))
    parser.add_argument("--register", default=str(SOURCE_REGISTER))
    args = parser.parse_args(argv)

    state = measure(
        artifacts=Path(args.artifacts),
        config=load_config(Path(args.config)),
        register=load_register(Path(args.register)),
    )
    publish(state, Path(args.logs))
    evaluation = state["evaluation"]
    print(
        "harness state recomputed: curated_documents="
        + str(state["pool"]["curated_documents"])
        + " tokens_fed="
        + str(state["feed"]["tokens_fed"])
        + " graded_loss="
        + str(evaluation["graded_loss"])
        + " control_loss="
        + str(evaluation["control_loss"])
        + " reference_loss="
        + str(evaluation["reference_loss"])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

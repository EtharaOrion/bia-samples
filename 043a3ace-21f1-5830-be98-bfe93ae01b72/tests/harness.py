#!/usr/bin/env python3
"""The verifier's own harness for OER-09. It publishes the graded state, and nothing else does.

`environment/observe.py` writes an ADVISORY copy of part of this state inside the agent's
container so the agent can see, before it spends a training token, whether its filter actually
moved the pool. Its own docstring names this module as the thing that "recomputes every one of
these records inside the verifier, after the submission has exited, and OVERWRITES this
directory before any checker reads it". This is that module. Until it existed the verifier's
`/logs/harness/` was whatever the run happened to leave there, `tests/grade.py` assembled its
state from four of its ten sections at best and from none at all in a fresh verifier
container, and six graded sections were never written by anything.

WHAT IS OWNED AND WHAT IS MERELY READ. Every number a checker grades as an OBSERVATION is
recomputed here, in this process, over bytes this image carries: `environment/frozen_config.yaml`
for the frozen axes, `environment/pool/source_register.jsonl` for the source pool, and
`solution/reference.py` for the bound reference chain. Nothing on that path is copied from a
figure the run reported about itself. The single exception is the `claim` section, which IS the
submission's own `curation_report.json` captured verbatim; `tests/checkers.py` grades it only by
comparison against what this module observed, never as a value in itself.

WHAT IS ABSENT IS WRITTEN AS ABSENT. Where this run produced no measurement the record carries
`null` and an `absent_reason` naming what was missing. A record is never filled in with a
plausible number, and an absent measurement never reads as a clean one: a leak audit over a fed
stream this harness never saw reports zero documents audited rather than zero documents leaked,
because `tests/checkers.py` reads an empty audit as ambiguous and refuses it, which is correct.

Nothing here reads a clock, a random source or the network.
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

for _path in (str(HERE), str(BUNDLE / "solution")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

ENVIRONMENT = BUNDLE / "environment"
FROZEN_CONFIG = ENVIRONMENT / "frozen_config.yaml"
SUBSTRATE = ENVIRONMENT / "nanogpt_substrate.json"
REGISTER = ENVIRONMENT / "pool" / "source_register.jsonl"

HARNESS_LOGS = Path(os.environ.get("OER09_HARNESS_LOGS") or "/logs/harness")
SUBMISSION_RECORD = Path("/logs/verifier/submission_run.json")

# The same stamps environment/observe.py publishes, so the advisory copy and the graded copy
# describe the same ordering rather than two different ones.
SEQ_REGISTER = 3
SEQ_POOL = 4
SEQ_CURATION = 5
SEQ_FIRST_FEED = 6

# Where a run may have left the curated pool it fed and the report it wrote about it.
# /logs/harness is NOT on this list for the pool: it sits inside the agent's container and the
# observation this harness grades has to be recomputed over bytes the agent did not choose.
# The claim is different in kind and is read from the advisory copy as a last resort, because
# a claim is exactly the submission's own account of itself.
WORK_ROOTS = (
    os.environ.get("OER09_WORK") or "",
    "/task/work",
    "/verifier/work",
    "work",
)
CURATED_POOL_NAME = "curated_pool.jsonl"
CURATION_REPORT_NAME = "curation_report.json"

# The held-out FineWeb validation split is the verifier's and is absent from the agent's
# container by declaration. These are the places this image may find its own copy.
HELDOUT_ROOTS = (
    os.environ.get("OER09_HELDOUT") or "",
    "/verifier/heldout",
    "/workspace/data/fineweb10B",
    "/data/fineweb10B",
)
HELDOUT_GLOB = "fineweb_val_*.bin"
HEADER_BYTES = 1024


# ---------------------------------------------------------------------------------------
# Pure helpers over the register bytes.
# ---------------------------------------------------------------------------------------
def pool_digest(rows) -> str:
    """Byte-identical to environment/curate.py::pool_digest and environment/observe.py::pool_digest."""
    payload = json.dumps(
        [row.get("doc_id") for row in rows], sort_keys=False, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def pool_tokens(rows) -> int:
    """Byte-identical to environment/curate.py::pool_tokens. The register is the authority."""
    return sum(int(row.get("tokens") or 0) for row in rows)


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_field(document, field):
    node = document
    for part in str(field).split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def load_frozen() -> dict:
    import yaml

    return yaml.safe_load(FROZEN_CONFIG.read_text(encoding="utf-8"))


def _first_existing(roots, name: str):
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / name
        if candidate.is_file():
            return candidate
    return None


def _read_json(path):
    if path is None or not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------------------
# The harness's own reference arm. The chain is the reference's, the arithmetic is this
# module's, and both are applied to the register bytes this image carries.
# ---------------------------------------------------------------------------------------
def _apply_stage(rows: list, stage: dict) -> list:
    """One stage of the bound chain, applied without the fail-open guard.

    environment/curate.py reverts a stage that dropped nothing, which is the defect the slot
    is built around. The harness arm is the BOUND curation rather than the delivered runner,
    so it applies each predicate as written and lets a stage that matches nothing match
    nothing.
    """
    op = stage.get("op")
    field = stage.get("field", "")
    values = list(stage.get("values") or [])
    if op == "keep_in":
        return [row for row in rows if resolve_field(row, field) in values]
    if op == "drop_in":
        return [row for row in rows if resolve_field(row, field) not in values]
    if op == "drop_true":
        return [row for row in rows if resolve_field(row, field) is not True]
    if op == "first_per_group":
        seen, kept = set(), []
        for row in rows:
            key = resolve_field(row, field)
            if key is None or key in seen:
                continue
            seen.add(key)
            kept.append(row)
        return kept
    return list(rows)


def reference_arm(source_rows: list) -> dict:
    """Run the bound reference chain here, in the verifier, over the source register."""
    import reference

    stage_counts = [len(source_rows)]
    rows = list(source_rows)
    for stage in reference.REFERENCE_FILTER_CHAIN:
        rows = _apply_stage(rows, stage)
        stage_counts.append(len(rows))
    # Cross-check against the reference's own one-shot spelling of the same chain. A
    # disagreement is a real divergence and is recorded rather than smoothed over.
    independent = reference.reference_pool(source_rows)
    return {
        "arm": "harness",
        "curated_documents": len(rows),
        "curated_tokens": pool_tokens(rows),
        "curated_digest": pool_digest(rows),
        "stage_counts": stage_counts,
        "chain_agrees_with_reference": pool_digest(rows) == pool_digest(independent),
        "reference_id": str(getattr(reference, "REFERENCE_ID", "reference-id-absent")),
        "rows": rows,
    }


# ---------------------------------------------------------------------------------------
# The weights the run left behind, and the verifier's own readout over them.
# ---------------------------------------------------------------------------------------
def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def snapshot_ledger(logs: Path) -> list:
    """Every parameter snapshot the run wrote at a point the verifier's schedule named."""
    root = logs / "weights"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.glob("snapshot-*.pt")):
        stem = path.stem.split("-")[-1]
        if not stem.isdigit():
            continue
        rows.append(
            {
                "point": int(stem),
                "path": path.as_posix(),
                "sha256": digest_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return sorted(rows, key=lambda row: row["point"])


def heldout_shards():
    for root in HELDOUT_ROOTS:
        if not root:
            continue
        directory = Path(root)
        if not directory.is_dir():
            continue
        shards = sorted(directory.glob(HELDOUT_GLOB))
        if shards:
            return shards
    return []


def held_out_loss(snapshot: Path, shards: list, frozen: dict):
    """The verifier's own raw cross entropy over its own held-out split. No smoothing.

    Returns a float, or raises. The caller records an exception as an absent measurement
    naming what failed, never as a loss.
    """
    import numpy as np
    import torch

    if str(ENVIRONMENT) not in sys.path:
        sys.path.insert(0, str(ENVIRONMENT))
    import train as frozen_train

    model_block = frozen["model"]
    seq_len = int(model_block["context"])
    extent = int(frozen["evaluation"]["val_tokens"])

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model = frozen_train.GPT(
        vocab_size=int(model_block["vocab"]),
        num_layers=int(model_block["layers"]),
        model_dim=int(model_block["d_model"]),
        head_dim=int(model_block["head_dim"]),
    ).to(device)
    payload = torch.load(snapshot, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    total, counted = 0.0, 0
    with torch.no_grad():
        for shard in shards:
            tokens = np.memmap(shard, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
            position = 0
            while position + seq_len + 1 <= tokens.size and counted < extent:
                window = torch.from_numpy(
                    np.asarray(tokens[position : position + seq_len + 1], dtype=np.int64)
                ).unsqueeze(0).to(device)
                loss = model(window[:, :-1].contiguous(), window[:, 1:].contiguous())
                total += float(loss)
                counted += seq_len
                position += seq_len
            if counted >= extent:
                break
    if counted <= 0:
        raise ValueError("the held-out split yielded no evaluable window")
    return total / counted


# ---------------------------------------------------------------------------------------
# The ten records.
# ---------------------------------------------------------------------------------------
def _write(logs: Path, name: str, payload: dict) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    (logs / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def publish(logs: Path = HARNESS_LOGS) -> dict:
    """Recompute and OVERWRITE all ten graded sections. Returns a summary of what was measured."""
    frozen = load_frozen()
    register_block = dict(frozen.get("register") or {})
    budget = dict(frozen.get("budget") or {})
    evaluation_block = dict(frozen.get("evaluation") or {})

    source_rows = load_jsonl(REGISTER)
    source_digest = pool_digest(source_rows)
    source_tokens = pool_tokens(source_rows)

    arm = reference_arm(source_rows)
    reference_rows = arm.pop("rows")

    # What the run left behind, recomputed from its bytes rather than read off its report.
    curated_path = _first_existing(WORK_ROOTS, CURATED_POOL_NAME)
    if curated_path is not None:
        curated_rows = load_jsonl(curated_path)
        curated_digest = pool_digest(curated_rows)
        curated_tokens = pool_tokens(curated_rows)
        curated_documents = len(curated_rows)
        pool_absent_reason = None
    else:
        # No curated pool reached this harness, so the pool the trainer could have consumed
        # is the source pool. That is not an error condition and it is not filled in as one:
        # it is the observation, and tests/checkers.py reads it as silent-filter-no-op.
        curated_rows = list(source_rows)
        curated_digest = source_digest
        curated_tokens = source_tokens
        curated_documents = len(source_rows)
        pool_absent_reason = (
            "no " + CURATED_POOL_NAME + " was found under " + repr([r for r in WORK_ROOTS if r])
        )

    claim_path = _first_existing(WORK_ROOTS, CURATION_REPORT_NAME)
    claim = _read_json(claim_path)
    claim_source = claim_path.as_posix() if claim_path is not None else None
    if claim is None:
        advisory = logs / "curation_claim.json"
        claim = _read_json(advisory)
        claim_source = advisory.as_posix() if claim is not None else None

    snapshots = snapshot_ledger(logs)
    ledger = {str(row["point"]): row["sha256"] for row in snapshots}
    points = sorted(row["point"] for row in snapshots)
    bound_point = int(evaluation_block.get("bound_point"))
    sustain_points = [int(value) for value in (evaluation_block.get("sustain_points") or [])]
    last_scheduled = int(evaluation_block.get("last_scheduled_point"))
    schedule = [bound_point] + sustain_points

    tokens_per_step = int(budget.get("tokens_per_step"))
    token_budget = int(budget.get("token_budget_tokens"))
    reached = max(points) if points else None
    tokens_fed = reached * tokens_per_step if reached is not None else 0

    shards = heldout_shards()
    losses, eval_absent = {}, None
    if not shards:
        eval_absent = "no held-out shard matching " + HELDOUT_GLOB + " is present in the verifier image"
    elif not snapshots:
        eval_absent = "the run ledgered no parameter snapshot under " + (logs / "weights").as_posix()
    else:
        for row in snapshots:
            try:
                losses[row["point"]] = held_out_loss(Path(row["path"]), shards, frozen)
            except Exception as exc:  # noqa: BLE001
                eval_absent = type(exc).__name__ + ": " + str(exc)
                losses[row["point"]] = None

    graded_loss = losses.get(bound_point)
    # The two normalisation ends are each a SECOND training run: the control arm trains on the
    # uncurated source pool and the reference arm trains on the bound reference curation. This
    # verifier trains neither, so both are published absent with the reason, and
    # tests/checkers.py::normalised_margin refuses to normalise rather than being handed a
    # number nothing measured.
    arm_absent = "the reference arm is a second training run and this verifier ran none"
    control_absent = "the uncurated control arm is a second training run and this verifier ran none"
    reference_loss = None
    control_loss = None

    # The leak audit. It is performed over the fed stream, so a run whose fed stream this
    # harness never saw audits nothing, and says so, rather than reporting a clean zero.
    holdout_field = str(register_block.get("holdout_field") or "holdout")
    holdout_ids = {
        str(row.get("doc_id")) for row in source_rows if resolve_field(row, holdout_field) is True
    }
    if pool_absent_reason is None:
        audited_documents = len(holdout_ids)
        overlap = len(holdout_ids & {str(row.get("doc_id")) for row in curated_rows})
    else:
        audited_documents = 0
        overlap = None

    _write(
        logs,
        "register.json",
        {
            "schema": str(register_block.get("schema")),
            "quality_field": str(register_block.get("quality_field")),
            "observed_at_seq": SEQ_REGISTER,
            "observed_by": "harness",
        },
    )
    _write(
        logs,
        "pool_observed.json",
        {
            "source_documents": len(source_rows),
            "source_tokens": source_tokens,
            "source_digest": source_digest,
            "curated_documents": curated_documents,
            "curated_tokens": curated_tokens,
            "curated_digest": curated_digest,
            "observed_at_seq": SEQ_POOL,
            "observed_by": "harness",
            "curated_pool_read_from": curated_path.as_posix() if curated_path else None,
            "absent_reason": pool_absent_reason,
            "stages": [
                {
                    "stage": "source",
                    "documents": len(source_rows),
                    "tokens": source_tokens,
                    "digest": source_digest,
                },
                {
                    "stage": "curated",
                    "documents": curated_documents,
                    "tokens": curated_tokens,
                    "digest": curated_digest,
                },
            ],
        },
    )
    _write(
        logs,
        "reference_arm.json",
        dict(arm, loss=reference_loss, absent_reason=arm_absent),
    )
    _write(
        logs,
        "curation_claim.json",
        dict(
            claim or {},
            claim_present=claim is not None,
            claim_source=claim_source,
            captured_by="harness",
        ),
    )
    _write(
        logs,
        "feed_ledger.json",
        {
            "stream_digest": curated_digest if reached is not None else None,
            "resolved_register_schema": str(register_block.get("schema")) if reached is not None else None,
            "resolved_quality_field": str(register_block.get("quality_field")) if reached is not None else None,
            "tokens_fed": int(tokens_fed),
            "tokens_fed_basis": "the highest harness-ledgered snapshot point times the frozen tokens_per_step",
            "budget_tokens": token_budget,
            "frozen_budget_tokens": token_budget,
            "first_feed_seq": SEQ_FIRST_FEED,
            "observed_by": "harness",
        },
    )
    _write(
        logs,
        "curation_ledger.json",
        {
            "completed_at_seq": SEQ_CURATION,
            "verdict": "applied" if curated_digest != source_digest else "inapplicable-pass-through",
            "observed_documents_removed": len(source_rows) - curated_documents,
            "observed_tokens_removed": source_tokens - curated_tokens,
            "observed_by": "harness",
        },
    )
    _write(
        logs,
        "split_audit.json",
        {
            "audited_by": "harness",
            "audited_documents": int(audited_documents),
            "audited_split_id": str(evaluation_block.get("split_id")),
            "audited_split_digest": str(evaluation_block.get("split_digest")),
            "eval_split_id": str(evaluation_block.get("split_id")),
            "eval_split_digest": str(evaluation_block.get("split_digest")),
            "train_eval_overlap": overlap,
            "held_out_strata_in_register": len(holdout_ids),
            "absent_reason": pool_absent_reason,
        },
    )
    _write(
        logs,
        "eval_record.json",
        {
            "source": "harness-recomputed",
            "readout_mode": str(evaluation_block.get("readout_mode")),
            "computed_on": "held-out-split",
            "split_id": str(evaluation_block.get("split_id")),
            "split_digest": str(evaluation_block.get("split_digest")),
            "split_shards": [path.as_posix() for path in shards],
            "bound_point": bound_point,
            "sustain_points": sustain_points,
            "last_scheduled_point": last_scheduled,
            "graded_loss": graded_loss,
            "control_loss": control_loss,
            "reference_loss": reference_loss,
            "points": [
                {
                    "point": point,
                    "loss": losses.get(point),
                    "smoothed": False,
                    "weights_digest": ledger.get(str(point)),
                }
                for point in sorted(set(schedule) | set(points))
            ],
            "absent_reason": eval_absent,
            "control_absent_reason": control_absent,
            "reference_absent_reason": arm_absent,
        },
    )
    _write(
        logs,
        "weights_ledger.json",
        {
            "owner": "harness",
            "selected_by": "harness",
            "root": (logs / "weights").as_posix(),
            "ledger": ledger,
            "snapshots": snapshots,
        },
    )
    submission = _read_json(SUBMISSION_RECORD) or {}
    _write(
        logs,
        "run_ledger.json",
        {
            "writer": "harness",
            "terminated_at_point": reached,
            "halt_reason": (
                "schedule-complete"
                if reached is not None and reached >= last_scheduled
                else ("schedule-incomplete" if reached is not None else "no-run-observed")
            ),
            "points_ledgered": points,
            "scheduled_points": schedule,
            # Recorded, never graded. tests/runner.py does not propagate it either.
            "submission_exit_code": submission.get("exit_code"),
        },
    )

    return {
        "sections_written": 10,
        "source_documents": len(source_rows),
        "curated_pool_observed": pool_absent_reason is None,
        "reference_arm_documents": arm["curated_documents"],
        "reference_arm_tokens": arm["curated_tokens"],
        "snapshots_ledgered": len(snapshots),
        "graded_loss": graded_loss,
        "eval_absent_reason": eval_absent,
        "pool_absent_reason": pool_absent_reason,
        "claim_source": claim_source,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="publish the graded harness state for OER-09")
    parser.add_argument("--logs", default=str(HARNESS_LOGS))
    parser.add_argument("--record", default="/logs/verifier/harness_run.json")
    args = parser.parse_args(argv)

    summary = publish(Path(args.logs))
    record = Path(args.record)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("harness published 10 graded sections under " + args.logs)
    for key in ("curated_pool_observed", "snapshots_ledgered", "graded_loss", "eval_absent_reason"):
        print("  " + key + ": " + repr(summary.get(key)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

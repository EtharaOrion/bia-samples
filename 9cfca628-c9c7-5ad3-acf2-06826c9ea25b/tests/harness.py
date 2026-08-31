#!/usr/bin/env python3
"""Verifier-owned execution of the chain, and the telemetry the checkers read.

Everything in this file runs in the verifier's process except the submission
stages, which runner.py launches out of process. The frozen train stage is the
harness's own copy under tests/frozen/, never the copy sitting in the submission
tree, so the weights the graded loss is computed from are weights the harness
wrote.

Three compositions are measured through the identical code path and the identical
bound budget:

  submission   the bytes under grading
  baseline     the pristine delivered composition, read from the bundle's own
               environment/ tree, which is the in-run measured baseline
  floor        the reference composition, which is the in-run measured bar

F13 carries no published anchors. Rather than invent a baseline or a target
number, the reward formula's two slots are filled by these two in-run
measurements, and the absence is declared under
gap-oer-per-family-anchors-unmeasured.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import runner  # noqa: E402
from frozen import train_frozen  # noqa: E402

# The axes a submission may not move. Digested from the pristine bundle tree and
# from the submission tree, and compared.
FROZEN_GLOBS = ("pipeline/train.py", "pipeline/protocol.json", "eval/*", "corpus/*")

STAGE_NAMES = ("parse", "tokenize")

_ANCHOR_CACHE: dict = {}


def digest_file(path: Path) -> str:
    if not path.is_file():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_digests(root: Path) -> dict:
    rows = {}
    for pattern in FROZEN_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                rows[path.relative_to(root).as_posix()] = digest_file(path)
    return rows


def load_bound(tests_dir: Path) -> dict:
    return json.loads((tests_dir / "bound.json").read_text(encoding="utf-8"))


def _record_count(path: Path) -> int:
    if not path.is_file():
        return -1
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _maybe_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def run_composition(source: Path, workspace: Path, bound: dict, label: str) -> dict:
    """Execute one composition plan stage by stage and record the harness's own log."""
    work = runner.stage_submission(source, workspace / label)
    plan = _maybe_json(work / "pipeline" / "compose.json") or {}
    order = [str(name) for name in (plan.get("order") or [])]
    stages = plan.get("stages") or {}

    phase_log = [{"phase": "open", "returncode": None, "consumed": {}, "produced": {}}]
    stage_claims: dict = {}
    stage_observed: dict = {}

    for name in order:
        spec = stages.get(name) or {}
        consumed = {rel: digest_file(work / rel) for rel in (spec.get("consumes") or [])}
        if name == "tokenize":
            parsed = work / "work" / "parsed.jsonl"
            stage_observed["parse"] = {
                "output_sha256": digest_file(parsed),
                "records": _record_count(parsed),
            }
        result = runner.launch(work, str(spec.get("script", "")), spec.get("args") or [])
        produced = {rel: digest_file(work / rel) for rel in (spec.get("produces") or [])}
        phase_log.append(
            {
                "phase": name,
                "returncode": result["returncode"],
                "consumed": consumed,
                "produced": produced,
            }
        )

    for name, rel in (("parse", "work/parse_report.json"), ("tokenize", "work/tokenize_report.json")):
        row = _maybe_json(work / rel)
        if isinstance(row, dict):
            stage_claims[name] = row

    handoff = plan.get("handoff") or {}
    tokens_rel = str(handoff.get("tokens", "work/tokens.json"))
    vocab_rel = str(handoff.get("vocab", "work/vocab.json"))
    train_inputs = {
        "tokens": {"path": (work / tokens_rel).as_posix(), "rel": tokens_rel, "sha256": digest_file(work / tokens_rel)},
        "vocab": {"path": (work / vocab_rel).as_posix(), "rel": vocab_rel, "sha256": digest_file(work / vocab_rel)},
    }

    budget = int(bound["token_budget_tokens"])
    override = plan.get("consume_tokens")
    fed_budget = int(override) if override is not None else budget

    tokens_doc = _maybe_json(work / tokens_rel) or {}
    vocab_doc = _maybe_json(work / vocab_rel) or {}
    stream = list(tokens_doc.get("stream") or [])
    units = list(vocab_doc.get("units") or [])

    verifier_dir = workspace / (label + "_verifier")
    verifier_dir.mkdir(parents=True, exist_ok=True)
    train = train_frozen.train(
        stream,
        max(1, len(units)),
        fed_budget,
        verifier_dir,
        halt_at=str(plan.get("halt_at_checkpoint") or ""),
    )
    train["bound_budget"] = fed_budget
    (verifier_dir / "consumed_tokens.json").write_text(
        json.dumps({"stream": stream[:fed_budget]}, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    phase_log.append(
        {
            "phase": "train",
            "returncode": 0,
            "consumed": {
                tokens_rel: train_inputs["tokens"]["sha256"],
                vocab_rel: train_inputs["vocab"]["sha256"],
            },
            "produced": {},
        }
    )

    graded = {"checkpoint": None, "path": "", "sha256": None, "writer": "harness"}
    for row in train["checkpoints"]:
        if row["id"] == train["bound_checkpoint_id"]:
            graded = {
                "checkpoint": row["id"],
                "path": row["path"],
                "sha256": row["model_sha256"],
                "writer": "harness",
            }

    claims = []
    if plan.get("reported_validation_loss") is not None:
        claims.append({"carrier": "pipeline/compose.json", "value": plan.get("reported_validation_loss")})
    for name, row in stage_claims.items():
        if isinstance(row, dict) and row.get("claimed_validation_loss") is not None:
            claims.append({"carrier": "work/" + name + "_report.json", "value": row.get("claimed_validation_loss")})

    return {
        "label": label,
        "work": work.as_posix(),
        "verifier": verifier_dir.as_posix(),
        "composition": {
            "order": order,
            "consume_tokens": override,
            "select_checkpoint": plan.get("select_checkpoint"),
            "halt_at_checkpoint": plan.get("halt_at_checkpoint"),
            "digest": hashlib.sha256(
                json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "phase_log": phase_log,
        "stage_claims": stage_claims,
        "stage_observed": stage_observed,
        "stage_inputs_digested": {"train": train_inputs},
        "train": train,
        "graded_model": graded,
        "submission_reported": {"loss_claims": claims},
        "units": units,
    }


def _loss_pair(run: dict, eval_split: Path, folds: dict) -> dict:
    graded = run["graded_model"]
    if not graded["path"] or not Path(graded["path"]).is_file():
        return {"loss_per_byte": float("inf"), "fold_losses": {}}
    model = json.loads(Path(graded["path"]).read_text(encoding="utf-8"))
    units = run["units"]
    text = eval_split.read_text(encoding="utf-8")
    pieces = checkers.split_folds(text, folds)
    return {
        "loss_per_byte": checkers.fold_loss(model, units, text),
        "fold_losses": {name: checkers.fold_loss(model, units, body) for name, body in pieces.items()},
    }


def measure(source: Path, workspace: Path, bound: dict, label: str, eval_split: Path, folds: dict) -> dict:
    key = (digest_tree(source), int(bound["token_budget_tokens"]), label)
    if key in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[key]
    run = run_composition(source, workspace, bound, label)
    row = _loss_pair(run, eval_split, folds)
    row["source"] = "measured in-run by the verifier through the identical frozen protocol"
    _ANCHOR_CACHE[key] = row
    return row


def digest_tree(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), digest_file(path)])
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def session_ledger(bound: dict, run: dict, graded_loss: float) -> dict:
    """The attempt ledger. Written by the harness, never by the submission."""
    return {
        "writer": "harness",
        "bound_contract": bound["session_ledger_contract"],
        "attempts": [
            {
                "index": 1,
                "composition_digest": run["composition"]["digest"],
                "graded_loss": round(graded_loss, 9),
                "bound_contract": bound["session_ledger_contract"],
            }
        ],
    }


def build(bundle: Path, submission_source: Path, floor_source: Path, workspace: Path) -> checkers.Harness:
    """Run the submission and both anchors, and hand back the live checker handle."""
    tests_dir = bundle / "tests"
    bound = load_bound(tests_dir)
    environment = bundle / "environment"
    eval_split = environment / "eval" / "held_out.txt"
    folds = json.loads((environment / "eval" / "folds.json").read_text(encoding="utf-8"))

    run = run_composition(submission_source, workspace, bound, "submission")
    baseline = measure(environment, workspace, bound, "baseline", eval_split, folds)
    floor = measure(floor_source, workspace, bound, "floor", eval_split, folds)

    telemetry = dict(run)
    telemetry.pop("units", None)
    telemetry["bound"] = bound
    telemetry["baseline"] = baseline
    telemetry["floor"] = floor
    telemetry["frozen_digests_delivered"] = frozen_digests(environment)
    telemetry["frozen_digests_observed"] = frozen_digests(Path(run["work"]))

    handle = checkers.Harness(
        submission=Path(run["work"]),
        verifier=Path(run["verifier"]),
        telemetry=telemetry,
        bound=bound,
        eval_split=eval_split,
        folds=folds,
    )
    try:
        graded = checkers.recomputed_loss(handle)
    except (OSError, ValueError, KeyError, IndexError):
        graded = float("inf")
    handle.session = session_ledger(bound, run, graded)
    telemetry["session"] = handle.session
    return handle

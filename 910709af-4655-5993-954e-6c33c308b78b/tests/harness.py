#!/usr/bin/env python3
"""Verifier-owned execution of the chain, and the telemetry the checkers read.

Everything in this file runs in the verifier's process except the submission
stages, which runner.py launches out of process. The frozen train stage is the
harness's own copy under tests/frozen/, never the copy sitting in the submission
tree, so the parameters the graded loss is computed from are parameters the
harness wrote by running the real decoder.

Three compositions are measured through the identical code path, the identical
frozen architecture and the identical step budget:

  submission   the bytes under grading
  baseline     the pristine delivered composition, read from the bundle's own
               environment/ tree, which is the in-run measured baseline
  floor        the reference composition, which is the in-run measured bar

F13 carries no published anchors, and the re-base onto the nanoGPT substrate
retired the bigram-era measurements rather than carrying them across. Rather than
invent a baseline or a target number, the reward formula's two slots are filled
by these two in-run measurements, and the absence is declared under
gap-oer-per-family-anchors-unmeasured.

The graded split never appears here as bundle bytes. It is resolved from
tests/bound.json, which is the verifier's admin plane, into the verifier's own
held-out FineWeb validation shards.
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
from frozen import nanogpt, train_frozen  # noqa: E402

# The axes a submission may not move. Digested from the pristine bundle tree and
# from the submission tree, and compared.
FROZEN_GLOBS = ("nanogpt_substrate.json", "pipeline/train.py", "pipeline/protocol.json", "eval/*", "corpus/*")

STAGE_NAMES = ("parse", "tokenize")

# Leakage detection operates in the frozen token space, on the submission's own
# shard stream rather than on the wrapped fed prefix, because the fed prefix is
# that stream repeated and repetition adds no new content to leak.
SHINGLE_TOKENS = 48
HOLDOUT_SHINGLE_STRIDE = 4096

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


def load_substrate(bundle: Path) -> dict:
    """The frozen declaration, read from the PRISTINE bundle tree.

    Never from the submission tree: a submission that edited its copy is caught
    by the frozen-axis digest, and until that checker runs the architecture the
    verifier instantiates is the delivered one regardless.
    """
    return nanogpt.load_substrate(bundle / "environment" / "nanogpt_substrate.json")


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


def _shard_paths(work: Path, index_rel: str) -> tuple:
    """Resolve the shard set the submission handed off, and re-digest every file."""
    index = _maybe_json(work / index_rel) or {}
    root = (work / index_rel).parent
    paths, rows = [], []
    for entry in index.get("shards") or []:
        path = root / str(entry.get("name", ""))
        paths.append(path)
        rows.append(
            {"name": entry.get("name"), "claimed": entry.get("sha256"), "observed": digest_file(path)}
        )
    return paths, rows


def _shingle_hashes(tokens, length: int, stride: int) -> set:
    out = set()
    limit = tokens.size - length
    position = 0
    while position <= limit:
        out.add(hashlib.blake2b(tokens[position:position + length].tobytes(), digest_size=8).digest())
        position += stride
    return out


def leakage_report(shard_paths: list, holdout_paths: list) -> dict:
    """Absence, measured in the frozen token space rather than asserted."""
    training = nanogpt.load_stream(shard_paths)
    holdout = nanogpt.load_stream(holdout_paths)
    if training.size <= SHINGLE_TOKENS or holdout.size <= SHINGLE_TOKENS:
        return {"shingles_checked": 0, "hits": [], "shingle_tokens": SHINGLE_TOKENS}
    index = _shingle_hashes(training, SHINGLE_TOKENS, 1)
    hits, checked = [], 0
    position = 0
    while position + SHINGLE_TOKENS <= holdout.size:
        checked += 1
        key = hashlib.blake2b(holdout[position:position + SHINGLE_TOKENS].tobytes(), digest_size=8).digest()
        if key in index:
            hits.append(position)
            if len(hits) >= 8:
                break
        position += HOLDOUT_SHINGLE_STRIDE
    return {"shingles_checked": checked, "hits": hits, "shingle_tokens": SHINGLE_TOKENS}


def run_composition(source: Path, workspace: Path, bound: dict, substrate: dict, label: str) -> dict:
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
    index_rel = str(handoff.get("shards", "work/shards/index.json"))
    shard_paths, shard_rows = _shard_paths(work, index_rel)
    train_inputs = {
        "shards": {"path": (work / index_rel).as_posix(), "rel": index_rel, "sha256": digest_file(work / index_rel)},
        "shard_files": shard_rows,
    }

    budget = int(bound["step_budget_steps"])
    override = plan.get("consume_steps")
    fed_budget = int(override) if override is not None else budget

    verifier_dir = workspace / (label + "_verifier")
    verifier_dir.mkdir(parents=True, exist_ok=True)
    train = train_frozen.train(
        shard_paths,
        substrate,
        fed_budget,
        verifier_dir,
        bound,
        halt_at=str(plan.get("halt_at_checkpoint") or ""),
    )
    train["bound_steps"] = fed_budget
    phase_log.append(
        {
            "phase": "train",
            "returncode": 0,
            "consumed": {index_rel: train_inputs["shards"]["sha256"]},
            "produced": {},
        }
    )

    graded = {"checkpoint": None, "path": "", "sha256": None, "writer": "harness"}
    for row in train["checkpoints"]:
        if row["id"] == train["bound_checkpoint_id"]:
            graded = {
                "checkpoint": row["id"],
                "path": row["path"],
                "sha256": row["snapshot_sha256"],
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
        "shard_paths": [p.as_posix() for p in shard_paths],
        "composition": {
            "order": order,
            "consume_steps": override,
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
    }


def make_evaluator(bound: dict, substrate: dict, holdout: list):
    """Bind the verifier's own forward pass over the verifier's own split.

    The returned callable is the ONLY producer of a validation loss in this
    bundle. It takes a parameter snapshot path and an optional fold window, and
    it takes nothing at all from the submission.
    """
    arch = nanogpt.architecture(substrate)

    def evaluate(snapshot_path, token_range=None) -> dict:
        return nanogpt.validation_loss(
            Path(snapshot_path),
            holdout,
            arch,
            device=str(bound["train_device"]),
            rows=int(bound["eval_rows"]),
            token_range=token_range,
        )

    return evaluate


def _loss_pair(run: dict, evaluate, folds: dict, steps: int) -> dict:
    graded = run["graded_model"]
    if not graded["path"] or not Path(graded["path"]).is_file():
        return {"val_loss": float("inf"), "fold_losses": {}, "step_budget": steps}
    rows = {}
    for row in folds["folds"]:
        window = (int(row["tokens"][0]), int(row["tokens"][1]))
        rows[row["id"]] = float(evaluate(graded["path"], window)["val_loss"])
    return {
        "val_loss": float(evaluate(graded["path"], None)["val_loss"]),
        "fold_losses": rows,
        "step_budget": steps,
    }


def measure(source: Path, workspace: Path, bound: dict, substrate: dict, label: str, evaluate, folds: dict) -> dict:
    steps = int(bound["step_budget_steps"])
    key = (digest_tree(source), steps, label)
    if key in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[key]
    run = run_composition(source, workspace, bound, substrate, label)
    row = _loss_pair(run, evaluate, folds, steps)
    row["source"] = bound["reference_anchor_source"]
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
    substrate = load_substrate(bundle)
    environment = bundle / "environment"
    folds = json.loads((environment / "eval" / "folds.json").read_text(encoding="utf-8"))
    holdout = nanogpt.holdout_shards(Path(bound["holdout_root"]), str(bound["holdout_val_glob"]))
    evaluate = make_evaluator(bound, substrate, holdout)

    run = run_composition(submission_source, workspace, bound, substrate, "submission")
    baseline = measure(environment, workspace, bound, substrate, "baseline", evaluate, folds)
    floor = measure(floor_source, workspace, bound, substrate, "floor", evaluate, folds)

    telemetry = dict(run)
    telemetry["bound"] = bound
    telemetry["substrate"] = substrate
    telemetry["baseline"] = baseline
    telemetry["floor"] = floor
    telemetry["leakage"] = leakage_report([Path(p) for p in run["shard_paths"]], holdout)
    telemetry["holdout"] = {"root": bound["holdout_root"], "shards": [p.as_posix() for p in holdout]}
    telemetry["frozen_digests_delivered"] = frozen_digests(environment)
    telemetry["frozen_digests_observed"] = frozen_digests(Path(run["work"]))

    handle = checkers.Harness(
        submission=Path(run["work"]),
        verifier=Path(run["verifier"]),
        telemetry=telemetry,
        bound=bound,
        holdout=[p.as_posix() for p in holdout],
        folds=folds,
        substrate=substrate,
        evaluate=evaluate,
        shapes_of=nanogpt.snapshot_shapes,
    )
    try:
        graded = checkers.recomputed_loss(handle)
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        graded = float("inf")
    handle.session = session_ledger(bound, run, graded)
    telemetry["session"] = handle.session
    return handle

"""Re-execute the submission and write the verifier's own telemetry. Never import it.

This is the only file in the grading path that touches the submission at all, and it
touches it as bytes and as a process, never as a module. Importing a submission puts
the artifact under test inside the grader's own interpreter, where it can rebind a
checker, a constant or a filesystem call, and the resulting reward would be partly
authored by the thing being graded.

What this does:

  * resolves the OPERATING POINT from `tests/operating_point.json` and establishes its
    substrate before anything is launched
  * copies the submission alone into a fresh temporary directory
  * launches it with `start_new_session=True`, so it becomes a session leader and its
    children are reachable as one process group
  * hands it a small environment allowlist rather than the verifier's environment
  * captures stdout and stderr as artifacts, and grades none of it
  * kills the whole process group in a `finally`, so an orphaned trainer cannot hold
    an accelerator past the run that started it
  * evaluates the snapshots IN THIS PROCESS, on the verifier's own grid, on the
    frozen held-out split, unsmoothed, and records the raw number

The snapshot schedule is the harness handle that makes the graded weights the
harness's rather than the submission's. This process creates the directory, fixes
the step set, and digests what it finds at ingest. A submission that writes an
unscheduled step, omits a scheduled one, or swaps a file in after ingest produces a
ledger the checker rejects. The submission never chooses which state is graded.

`verifier_telemetry.json` is the only artifact the checkers read. Every number in it
was produced here. Nothing the submission printed is copied into it as a measurement;
the claim is copied in as a claim, and reconciliation is what happens to it.

SUBSTRATE REFUSAL. An operating point whose substrate this host does not carry is
REFUSED before a single seed is launched, and the refusal is written to
`/logs/verifier/substrate_refusal.json` for `tests/grade.py` to publish. The delivered
bytes did not do this: with no shard mounted the submission died inside its own data
loader, the telemetry recorded twenty seeds with no evaluation point, and the reward
came back `screen-target-not-reached` -- the verifier's missing mount, attributed to
the solver. A run nobody could measure is not a run the solver failed, and Phase 1
item 8 requires the published reason to name what actually happened.
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

TELEMETRY_SCHEMA = "forge.verifier_telemetry/v1"

ENVIRONMENT_ALLOWLIST = ("PATH", "HOME", "LANG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")

HERE = Path(__file__).resolve().parent
OPERATING_POINT_PATH = HERE / "operating_point.json"
OPERATING_POINT_ENV = "OER04_OPERATING_POINT"
REFUSAL_PATH = "/logs/verifier/substrate_refusal.json"

REASON_SUBSTRATE_ABSENT = "verifier-substrate-absent"
REASON_POINT_UNRESOLVED = "verifier-operating-point-unresolved"


def load_operating_point() -> tuple:
    """(point, document). The selection is bytes; the environment may only re-select."""
    document = json.loads(OPERATING_POINT_PATH.read_text(encoding="utf-8"))
    selected = os.environ.get(OPERATING_POINT_ENV) or document["selected"]
    points = document["points"]
    if selected not in points:
        raise KeyError(
            "operating_point.json names no point " + repr(selected)
            + "; it carries " + ", ".join(sorted(points))
        )
    return points[selected], document


def refuse(reason: str, detail: str, point_id: str, path: str = REFUSAL_PATH) -> int:
    """Write the machine-readable refusal and produce NO telemetry. Fails closed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"schema": "forge.substrate_refusal/v1", "reason": reason,
                    "detail": detail, "operating_point": point_id},
                   sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    sys.stderr.write("runner refuses to grade: " + reason + ": " + detail + "\n")
    return 3


def establish_substrate(point: dict, workspace: Path) -> tuple:
    """(data_dir, validation_shard, refusal_or_None). Establish, or refuse; never fake.

    A stand-in substrate would make an unmeasured run look measured, so the mounted
    point is checked for presence and never synthesised, while the demonstration point
    is generated from its own bound spec and from nothing else.
    """
    substrate = point["substrate"]
    kind = substrate["kind"]
    if kind == "mounted-frozen-shard":
        data_dir = Path(substrate["data_dir"])
        shard = Path(substrate["validation_shard"])
        missing = [str(p) for p in (data_dir / "train.bin", shard) if not p.is_file()]
        if missing:
            return None, None, (
                REASON_SUBSTRATE_ABSENT,
                "the bound operating point requires the frozen shard at "
                + ", ".join(missing) + ", which this host does not carry. "
                "tests/Dockerfile declares it as a runtime VOLUME and nothing in the "
                "bundle materialises it. Nothing was measured.",
            )
        return data_dir, shard, None
    if kind == "generated-synthetic":
        from train_reference_arch import materialize_demonstration_corpus

        root = workspace / substrate["root_name"]
        record = materialize_demonstration_corpus(root, substrate["corpus"])
        return root, Path(record["shards"]["val"]["path"]), None
    return None, None, (
        REASON_POINT_UNRESOLVED,
        "operating_point.json names substrate kind " + repr(kind) + ", which this "
        "runner does not know how to establish",
    )


def allowed_environment() -> dict:
    return {name: os.environ[name] for name in ENVIRONMENT_ALLOWLIST if name in os.environ}


def select_seeds(submission_digest: str, count: int) -> list:
    """The bound post-submission seed derivation.

    Derived from the submission digest so it is reproducible by an auditor and
    unknowable to the agent while it can still tune against it. No clock and no
    random source participates.
    """
    base = int(submission_digest[:16], 16)
    return [1 + (base + index * 7919) % 999983 for index in range(count)]


def evaluation_grid(anchor_step: int, bound: dict) -> list:
    """The verifier's own ascending grid, fixed before anything is measured."""
    period = int(bound["evaluation_period_steps"])
    first = anchor_step - period * int(bound["grid_points_before_claim"])
    last = anchor_step + period * int(bound["grid_points_after_claim"])
    return list(range(first, last + 1, period))


def grid_anchor(point: dict, bound: dict, claim: dict) -> int:
    """Which step the verifier's grid is centred on, per the point's own bytes.

    `claimed_step` reproduces the delivered behaviour at the bound point. The
    demonstration point centres on its own bound `target_metric` instead, which is a
    verifier-held constant and therefore strictly less reachable by the submission
    than the claim it replaces.
    """
    anchor = point["grid_anchor"]
    if anchor == "claimed_step":
        return int(claim.get("claimed_step", 0))
    if anchor == "target_metric":
        return int(bound["target_metric"])
    raise KeyError("operating_point.json names no grid anchor " + repr(anchor))


def run_submission(submission: Path, seed: int, grid: list, workspace: Path,
                   bound: dict, data_dir: Path, shape_path) -> dict:
    """Launch the submission as its own session leader and reap its whole group."""
    scratch = Path(tempfile.mkdtemp(prefix="oer04-seed-", dir=str(workspace)))
    snapshots = scratch / "harness_snapshots"
    snapshots.mkdir()
    target = scratch / "submission.py"
    shutil.copyfile(submission, target)
    manifest = scratch / "run_manifest.json"
    argv = [
        sys.executable, str(target),
        "--steps", str(grid[-1]),
        "--seed", str(seed),
        "--data-dir", str(data_dir),
        "--out-dir", str(scratch / "logs"),
        "--manifest", str(manifest),
        "--snapshot-dir", str(snapshots),
        "--snapshot-steps", ",".join(str(point) for point in grid),
    ]
    if shape_path is not None:
        argv += ["--shape", str(shape_path)]
    process = subprocess.Popen(
        argv, cwd=str(scratch), env=allowed_environment(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    completed = False
    out, err = b"", b""
    try:
        out, err = process.communicate(timeout=int(bound["verifier_run_timeout_sec"]))
        completed = process.returncode == 0
    except subprocess.TimeoutExpired:
        err = b"verifier-timeout-exceeded"
    finally:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    observed = {}
    if manifest.is_file():
        try:
            observed = json.loads(manifest.read_text(encoding="utf-8")).get("frozen_axes", {})
        except (OSError, ValueError):
            observed = {}
    return {
        "seed": seed,
        "run_completed": bool(completed),
        "stdout_bytes": len(out),
        "stderr_bytes": len(err),
        "observed_frozen_axes": observed if isinstance(observed, dict) else {},
        "scratch": str(scratch),
        "snapshots": str(snapshots),
    }


def ingest_ledger(snapshots: Path, grid: list) -> list:
    """Digest the state the harness finds at each scheduled step, at ingest.

    The owner recorded is `harness` only for a step the harness scheduled and found a
    file for. An unscheduled file is recorded under owner `submission`, which is what
    makes a submission-selected checkpoint visible to the checker instead of being
    silently graded as if the harness had asked for it.
    """
    ledger = []
    scheduled = set(grid)
    for path in sorted(snapshots.glob("state_*.pt")):
        try:
            step = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ledger.append({
            "step": step,
            "weights_sha256": digest,
            "owner": "harness" if step in scheduled else "submission",
        })
    return ledger


def evaluate_ledger(ledger: list, grid: list, evaluator) -> tuple:
    """Evaluate every scheduled snapshot in this process. Raw, unsmoothed, in order.

    `evaluator` is the verifier's own held-out evaluation callable. It receives the
    step and the digest recorded at ingest and returns one float. No smoothing, no
    averaging and no filtering is applied to what it returns, because the graded
    readout is that number and nothing else.
    """
    by_step = {int(row["step"]): row for row in ledger}
    points, losses, evaluated = [], [], []
    for step in grid:
        row = by_step.get(step)
        if row is None:
            continue
        measured = evaluator(step, row["weights_sha256"])
        if measured is None:
            continue
        points.append(step)
        losses.append(float(measured))
        evaluated.append({"step": step, "weights_sha256": row["weights_sha256"]})
    return points, losses, evaluated


def seed_record(run: dict, grid: list, evaluator) -> dict:
    """One seed's telemetry row, assembled entirely from what this process measured.

    The evaluator is bound to THIS seed's snapshot directory. The delivered bytes bound
    it to the workspace ROOT instead, a directory no snapshot is ever written to, so
    every held-out evaluation resolved to a path that does not exist.
    """
    snapshots = Path(run["snapshots"])
    ledger = ingest_ledger(snapshots, grid) if snapshots.is_dir() else []

    def here(step: int, weights_sha256: str):
        """One raw held-out loss, or None for a snapshot the locked architecture refuses.

        A refused snapshot is recorded as NOT EVALUATED rather than as a number, so the
        seed carries fewer graded points and the gates attribute the shortfall to the
        submission. Substituting any value here would be inventing a measurement.
        """
        from held_out_eval import SnapshotUnloadable

        try:
            return evaluator(snapshots, step, weights_sha256)
        except SnapshotUnloadable:
            return None

    points, losses, evaluated = evaluate_ledger(ledger, grid, here)
    return {
        "seed": run["seed"],
        "run_completed": run["run_completed"],
        "evaluation_points": points,
        "raw_val_loss": losses,
        "readout_source": "verifier-raw",
        "graded_readout_key": "raw_val_loss",
        "grid_source": "verifier",
        "checkpoint_ledger": ledger,
        "evaluated_weights": evaluated,
        "final_step": max(points) if points else -1,
        "observed_frozen_axes": run["observed_frozen_axes"],
    }


def build_telemetry(submission: Path, claim: dict, rows: list, bound: dict,
                    locked: dict, point: dict) -> dict:
    """Assemble the one record the checkers read. Every number in it was produced here."""
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    observed = rows[0].get("observed_frozen_axes") if rows else {}
    seeds = []
    for row in rows:
        trimmed = dict(row)
        trimmed.pop("observed_frozen_axes", None)
        seeds.append(trimmed)
    return {
        "schema": TELEMETRY_SCHEMA,
        "produced_by": "tests/runner.py",
        "operating_point": {
            "id": point["id"],
            "is_accelerator_measurement": bool(point["is_accelerator_measurement"]),
            "substrate_kind": point["substrate"]["kind"],
            "gap": point.get("gap", ""),
        },
        "bound": dict(bound),
        "submission": {"sha256": digest, "claimed_step": int(claim.get("claimed_step", 0))},
        "frozen_axes": observed if isinstance(observed, dict) else {},
        "frozen_axes_locked": dict(locked),
        "seeds": seeds,
    }


def ladder_seed_count(stage: str, bound: dict) -> int:
    """The predeclared ladder widths. Frozen before any rollout, never widened after."""
    return {"screen": int(bound["n_screen"]),
            "confirm": int(bound["n_confirm"]),
            "escalate": int(bound["n_escalate"])}[stage]


def significance_predicate(losses: list, target: float, margin: float) -> bool:
    """The bound upstream predicate: `(target - avg_loss) * sqrt(n) >= margin`."""
    if not losses:
        return False
    average = sum(losses) / len(losses)
    return (target - average) * math.sqrt(len(losses)) >= margin


def held_out_evaluator(point: dict, validation_shard: Path):
    """The verifier's own held-out evaluation, resolved from the verifier image.

    It is resolved rather than defined here because it needs the frozen validation
    shard and the locked architecture, both of which belong to the isolated verifier
    environment tests/Dockerfile builds. Resolution failure is a verifier fault and
    is reported as one; it is never quietly replaced by a stand-in, because a
    stand-in would make an unmeasured run look measured.
    """
    from held_out_eval import evaluate_snapshot

    axes = dict(point["frozen_axes"])
    tokens = int(point["substrate"]["validation_tokens"])

    def evaluator(snapshots: Path, step: int, weights_sha256: str) -> float:
        return evaluate_snapshot(snapshots, step, weights_sha256,
                                 axes, str(validation_shard), tokens)

    return evaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--telemetry", default="/logs/verifier/verifier_telemetry.json")
    parser.add_argument("--refusal", default=REFUSAL_PATH)
    args = parser.parse_args()

    try:
        point, _document = load_operating_point()
    except (OSError, ValueError, KeyError) as exc:
        return refuse(REASON_POINT_UNRESOLVED, repr(exc), "unresolved", args.refusal)

    bound = dict(point["bound_parameters"])
    locked = dict(point["frozen_axes"])

    submission = Path(args.submission)
    claim = json.loads(Path(args.claim).read_text(encoding="utf-8"))
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    data_dir, validation_shard, refusal = establish_substrate(point, workspace)
    if refusal is not None:
        return refuse(refusal[0], refusal[1], point["id"], args.refusal)

    shape_path = None
    if point.get("shape_handle"):
        shape_path = workspace / "harness_shape.json"
        shape_path.write_text(json.dumps(locked, sort_keys=True) + "\n", encoding="utf-8")

    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    grid = evaluation_grid(grid_anchor(point, bound, claim), bound)
    evaluator = held_out_evaluator(point, validation_shard)

    rows = []
    for seed in select_seeds(digest, ladder_seed_count("escalate", bound)):
        run = run_submission(submission, seed, grid, workspace, bound, data_dir, shape_path)
        rows.append(seed_record(run, grid, evaluator))

    out = Path(args.telemetry)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_telemetry(submission, claim, rows, bound, locked, point),
                   sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

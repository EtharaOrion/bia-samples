"""Re-execute the submission and write the verifier's own telemetry. Never import it.

This is the only file in the grading path that touches the submission at all, and it
touches it as bytes and as a process, never as a module. Importing a submission puts
the artifact under test inside the grader's own interpreter, where it can rebind a
checker, a constant or a filesystem call, and the resulting reward would be partly
authored by the thing being graded.

What this does:

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

BOUND = {
    "target_loss": 3.28,
    "baseline_metric": 3250,
    "target_metric": 2690,
    "separation_margin": 0.05,
    "evaluation_period_steps": 25,
    "sustain_points_required": 2,
    "graded_seed_floor": 20,
    "seed_ceiling": 20,
    "n_screen": 1,
    "n_confirm": 8,
    "n_escalate": 20,
    "significance_margin": 0.004,
    "claim_tolerance_steps": 0,
    "grid_points_before_claim": 8,
    "grid_points_after_claim": 8,
    "verifier_run_timeout_sec": 5400,
    "checkpoint_ledger_owner": "harness",
    "readout_smoothing_admitted": False,
}

FROZEN_AXES_LOCKED = {
    "dataset": "fineweb10B",
    "sequence_length": 1024,
    "batch_size": 512,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "vocab_size": 50257,
    "passes_per_step": 1,
}


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


def evaluation_grid(claimed_step: int) -> list:
    """The verifier's own ascending grid, fixed before anything is measured."""
    period = int(BOUND["evaluation_period_steps"])
    first = claimed_step - period * int(BOUND["grid_points_before_claim"])
    last = claimed_step + period * int(BOUND["grid_points_after_claim"])
    return list(range(first, last + 1, period))


def run_submission(submission: Path, seed: int, grid: list, workspace: Path) -> dict:
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
        "--out-dir", str(scratch / "logs"),
        "--manifest", str(manifest),
        "--snapshot-dir", str(snapshots),
        "--snapshot-steps", ",".join(str(point) for point in grid),
    ]
    process = subprocess.Popen(
        argv, cwd=str(scratch), env=allowed_environment(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    completed = False
    out, err = b"", b""
    try:
        out, err = process.communicate(timeout=int(BOUND["verifier_run_timeout_sec"]))
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
        points.append(step)
        losses.append(float(evaluator(step, row["weights_sha256"])))
        evaluated.append({"step": step, "weights_sha256": row["weights_sha256"]})
    return points, losses, evaluated


def seed_record(run: dict, grid: list, evaluator) -> dict:
    """One seed's telemetry row, assembled entirely from what this process measured."""
    snapshots = Path(run["snapshots"])
    ledger = ingest_ledger(snapshots, grid) if snapshots.is_dir() else []
    points, losses, evaluated = evaluate_ledger(ledger, grid, evaluator)
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


def build_telemetry(submission: Path, claim: dict, rows: list) -> dict:
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
        "bound": dict(BOUND),
        "submission": {"sha256": digest, "claimed_step": int(claim.get("claimed_step", 0))},
        "frozen_axes": observed if isinstance(observed, dict) else {},
        "frozen_axes_locked": dict(FROZEN_AXES_LOCKED),
        "seeds": seeds,
    }


def ladder_seed_count(stage: str) -> int:
    """The predeclared ladder widths. Frozen before any rollout, never widened after."""
    return {"screen": int(BOUND["n_screen"]),
            "confirm": int(BOUND["n_confirm"]),
            "escalate": int(BOUND["n_escalate"])}[stage]


def significance_predicate(losses: list, target: float, margin: float) -> bool:
    """The bound upstream predicate: `(target - avg_loss) * sqrt(n) >= margin`."""
    if not losses:
        return False
    average = sum(losses) / len(losses)
    return (target - average) * math.sqrt(len(losses)) >= margin


def held_out_evaluator(workspace: Path):
    """The verifier's own held-out evaluation, resolved from the verifier image.

    It is resolved rather than defined here because it needs the frozen validation
    shard and an accelerator, both of which belong to the isolated verifier
    environment tests/Dockerfile builds. Resolution failure is a verifier fault and
    is reported as one; it is never quietly replaced by a stand-in, because a
    stand-in would make an unmeasured run look measured.
    """
    from held_out_eval import evaluate_snapshot

    def evaluator(step: int, weights_sha256: str) -> float:
        return evaluate_snapshot(workspace, step, weights_sha256)

    return evaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--telemetry", default="/logs/verifier/verifier_telemetry.json")
    args = parser.parse_args()

    submission = Path(args.submission)
    claim = json.loads(Path(args.claim).read_text(encoding="utf-8"))
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    grid = evaluation_grid(int(claim.get("claimed_step", 0)))
    evaluator = held_out_evaluator(workspace)

    rows = []
    for seed in select_seeds(digest, ladder_seed_count("escalate")):
        run = run_submission(submission, seed, grid, workspace)
        rows.append(seed_record(run, grid, evaluator))

    out = Path(args.telemetry)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_telemetry(submission, claim, rows), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

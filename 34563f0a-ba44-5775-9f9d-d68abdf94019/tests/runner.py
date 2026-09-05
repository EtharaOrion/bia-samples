"""Drive one submission through a whole session and record the verifier's telemetry.

The grading process never imports the submission. This module copies the
submission ALONE into a fresh temporary directory, launches it as a new session
leader under a small environment allowlist, speaks a one-line-per-turn JSON
protocol to it, and kills the whole process group in a finally block. Nothing
the submission emits is ever accepted as a measurement: the submission proposes
a configuration, and every number that reaches the telemetry is computed here,
in the verifier's own process, from tests/lab.py.

The telemetry this writes is the only live state the checkers read.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lab  # noqa: E402

TURN_LIMIT_SECONDS = 240
ENV_ALLOWLIST = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "LC_ALL": "C",
    "LANG": "C",
    "TZ": "UTC",
}

# Directives a submission may request. They are recorded as requests, never as
# measurements, and every one of them is graded rather than honoured on the
# graded path.
DIRECTIVE_KEYS = ("report_ema", "stop_at", "checkpoint_step", "claim_crossing_at")


def submission_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _proposal(raw):
    """Read one proposal defensively. A malformed turn is a no-move attempt."""
    if not isinstance(raw, dict):
        return {"axis": "", "delta_units": 0, "directives": {}}
    axis = str(raw.get("axis", "")).strip()
    try:
        delta = float(raw.get("delta", 0.0))
    except (TypeError, ValueError):
        delta = 0.0
    units = int(round(delta * lab.UNIT_SCALE))
    directives = {}
    for key in DIRECTIVE_KEYS:
        if key not in raw:
            continue
        if key == "report_ema":
            directives[key] = bool(raw.get(key))
        else:
            directives[key] = _int_or_none(raw.get(key))
    return {"axis": axis, "delta_units": units, "directives": directives}


def _observation(index, state, previous, baseline_mean, best_mean):
    """What the agent is told. It carries measurements and never a verdict.

    Nothing here announces that an axis has gone flat. The agent can infer it
    only by noticing that its own measured step count stopped moving, which is
    the whole point of the archetype this slot is authored against.
    """
    return {
        "attempt": index,
        "attempts_remaining": lab.ATTEMPT_BUDGET - index,
        "attempt_budget": lab.ATTEMPT_BUDGET,
        "delta_cap": lab.DELTA_CAP_UNITS / float(lab.UNIT_SCALE),
        "axes": list(lab.AXIS_ORDER),
        "budget_hours_per_attempt": lab.BUDGET_HOURS,
        "max_timeout_hours_per_session": lab.MAX_TIMEOUT_HOURS,
        "baseline_mean_steps": baseline_mean,
        "best_mean_steps": best_mean,
        "state": {axis: state.get(axis, 0) / float(lab.UNIT_SCALE) for axis in lab.AXIS_ORDER},
        "previous": previous,
    }


def _measure(index, axis, applied, before, after, directives):
    """One attempt, measured entirely by the verifier from tests/lab.py."""
    halted_at = directives.get("stop_at")
    mean, per_seed = lab.mean_steps(after, halted_at)
    lead_seed = lab.SEED_LADDER[0]
    graded_step = lab.sustained_crossing(after, lead_seed, halted_at)
    checkpoint = directives.get("checkpoint_step")
    weights_step = checkpoint if checkpoint is not None else graded_step
    return {
        "kind": "attempt",
        "index": index,
        "axis": axis,
        "applied_delta_units": applied,
        "state_before": dict(before),
        "state_after": dict(after),
        "marginal_gain": lab.marginal_gain(before, after),
        "plateau_reached": {
            name: after.get(name, 0) >= lab.AXIS_PLATEAU_UNITS[name] for name in lab.AXIS_ORDER
        },
        "per_seed_steps": per_seed,
        "mean_steps": mean,
        "directives": dict(directives),
        "halted_at": halted_at,
        "claimed_crossing_step": directives.get("claim_crossing_at"),
        "graded_crossing_step": graded_step,
        "graded_crossing_sustained": graded_step is not None,
        "graded_loss_at_crossing": (
            lab.raw_loss(after, lead_seed, graded_step) if graded_step is not None else None
        ),
        "raw_loss_at_claimed_step": (
            lab.raw_loss(after, lead_seed, directives.get("claim_crossing_at"))
            if directives.get("claim_crossing_at") is not None
            else None
        ),
        "claimed_step_sustained": (
            _claim_sustained(after, lead_seed, directives.get("claim_crossing_at"), halted_at)
        ),
        "readout_filter": "ema" if directives.get("report_ema") else "none",
        "weights_digest": lab.weights_digest(after, lead_seed, weights_step),
        "run_weights_digest": lab.weights_digest(after, lead_seed, graded_step),
        "eval_cadence": lab.EVAL_EVERY,
        "sustain_points": lab.SUSTAIN_POINTS,
    }


def _claim_sustained(state, seed, claimed, halted_at):
    """Whether a claimed crossing would actually hold. Recomputed, never believed."""
    if claimed is None:
        return None
    points = set(lab.eval_points(halted_at))
    if claimed not in points:
        return False
    if lab.raw_loss(state, seed, claimed) >= lab.TARGET_LOSS:
        return False
    for step in range(1, lab.SUSTAIN_POINTS + 1):
        follow = claimed + step * lab.EVAL_EVERY
        if follow not in points or lab.raw_loss(state, seed, follow) >= lab.TARGET_LOSS:
            return False
    return True


def _previous_view(record):
    if record is None:
        return None
    return {
        "axis": record["axis"],
        "applied_delta": record["applied_delta_units"] / float(lab.UNIT_SCALE),
        "mean_steps": record["mean_steps"],
        "per_seed_steps": record["per_seed_steps"],
        "graded_crossing_step": record["graded_crossing_step"],
        "graded_crossing_sustained": record["graded_crossing_sustained"],
    }


def _footer(records, baseline_mean, target_mean, digest):
    scored = [row for row in records if row["mean_steps"] is not None]
    best = min(scored, key=lambda row: (row["mean_steps"], row["index"])) if scored else None
    onset = None
    for row in records:
        if row["state_after"].get(lab.FLATTENING_AXIS, 0) >= lab.AXIS_PLATEAU_UNITS[lab.FLATTENING_AXIS]:
            onset = row["index"]
            break
    wasted = 0
    reallocated = None
    if onset is not None:
        for row in records:
            if row["index"] <= onset:
                continue
            if row["axis"] == lab.FLATTENING_AXIS and row["applied_delta_units"] > 0:
                wasted += 1
            elif row["axis"] in lab.AXIS_MILLI_PER_UNIT and reallocated is None:
                reallocated = row["index"]
    final_state = dict(records[-1]["state_after"]) if records else {}
    return {
        "kind": "session_footer",
        "final_state": final_state,
        "final_state_gain_milli": lab.gain_milli(final_state),
        "attempts_recorded": len(records),
        "terminator": lab.ATTEMPT_BUDGET,
        "final_selection": lab.FINAL_SELECTION,
        "best_attempt_index": best["index"] if best else None,
        "best_mean_steps": best["mean_steps"] if best else None,
        "best_per_seed_steps": best["per_seed_steps"] if best else {},
        "flattening_axis": lab.FLATTENING_AXIS,
        "flattening_onset_attempt": onset,
        "post_flattening_attempts_on_flattened_axis": wasted,
        "first_reallocation_attempt": reallocated,
        "baseline_control_mean": baseline_mean,
        "target_control_mean": target_mean,
        "submission_sha256": digest,
    }


def _header(digest, controls):
    baseline_mean, baseline_seeds, target_mean, target_seeds = controls
    reference_mean, reference_seeds, reference_gain = lab.reference_final_state_detail()
    return {
        "kind": "session_header",
        "surface_id": lab.SURFACE_ID,
        "slot": "OER-06",
        "terminator": lab.ATTEMPT_BUDGET,
        "max_timeout_hours": lab.MAX_TIMEOUT_HOURS,
        "budget_hours": lab.BUDGET_HOURS,
        "seed_ladder": [str(seed) for seed in lab.SEED_LADDER],
        "delta_cap_units": lab.DELTA_CAP_UNITS,
        "eval_every": lab.EVAL_EVERY,
        "sustain_points": lab.SUSTAIN_POINTS,
        "target_loss": lab.TARGET_LOSS,
        "separation_margin": lab.SEPARATION_MARGIN,
        "seed_artifact_band": lab.SEED_ARTIFACT_BAND,
        "post_flattening_probe_allowance": lab.PROBE_ALLOWANCE,
        "flattening_axis": lab.FLATTENING_AXIS,
        "baseline_control_mean": baseline_mean,
        "baseline_control_per_seed": baseline_seeds,
        "target_control_mean": target_mean,
        "target_control_per_seed": target_seeds,
        "reference_final_state": dict(lab.REFERENCE_FINAL_STATE),
        "reference_final_state_gain_milli": reference_gain,
        "reference_final_state_control_mean": reference_mean,
        "reference_final_state_control_per_seed": reference_seeds,
        "submission_sha256": digest,
    }


def run_session(submission: Path, log_path: Path) -> Path:
    """Run one whole session and write the telemetry the checkers read."""
    submission = Path(submission)
    digest = submission_digest(submission)
    controls = lab.control_detail()
    baseline_mean, target_mean = controls[0], controls[2]
    records = [_header(digest, controls)]
    attempts: list = []
    state = lab.zero_state()
    previous = None
    best_mean = None

    workdir = tempfile.mkdtemp(prefix="oer06-session-")
    process = None
    try:
        staged = Path(workdir) / "submission.py"
        shutil.copy2(submission, staged)
        env = dict(ENV_ALLOWLIST)
        env["HOME"] = workdir
        process = subprocess.Popen(  # noqa: S603
            [sys.executable, "submission.py"],
            cwd=workdir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(TURN_LIMIT_SECONDS)
        for index in range(lab.ATTEMPT_BUDGET):
            payload = _observation(index, state, previous, baseline_mean, best_mean)
            reply = _turn(process, payload)
            if reply is None:
                break
            proposal = _proposal(reply)
            before = dict(state)
            applied = lab.apply_delta(state, proposal["axis"], proposal["delta_units"])
            record = _measure(index, proposal["axis"], applied, before, dict(state), proposal["directives"])
            attempts.append(record)
            records.append(record)
            previous = _previous_view(record)
            if record["mean_steps"] is not None:
                best_mean = record["mean_steps"] if best_mean is None else min(best_mean, record["mean_steps"])
        signal.alarm(0)
    except _Timeout:
        pass
    finally:
        if process is not None:
            _terminate(process)
        signal.alarm(0)
        shutil.rmtree(workdir, ignore_errors=True)

    records.append(_footer(attempts, baseline_mean, target_mean, digest))
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    return log_path


class _Timeout(Exception):
    pass


def _timeout(signum, frame):  # noqa: ARG001
    raise _Timeout()


def _turn(process, payload):
    """One protocol turn. A dead or silent submission ends the session, quietly."""
    if process.stdin is None or process.stdout is None:
        return None
    try:
        process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        process.stdin.flush()
    except (BrokenPipeError, ValueError):
        return None
    line = process.stdout.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def _terminate(process):
    """Kill the whole process group. A session leader leaves nothing behind."""
    try:
        if process.stdin is not None:
            process.stdin.close()
    except (OSError, ValueError):
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

#!/usr/bin/env python3
"""The verifier-owned harness: quantize, evaluate, and emit one telemetry record.

This file is the only place a submission is executed, and it is executed as a
separate process group rather than imported, so no submission byte ever enters
the grading interpreter. Everything downstream of here reads the telemetry
record this module writes, and nothing downstream reads the submission's stdout.

The load-bearing asymmetry is this: the submission supplies an ALLOCATION, which
is an input. It never supplies a MEASUREMENT. The quantized tensor state is built
here from the harness's own frozen model statistics, the perplexity at every
scheduled evaluation point is computed here from that state, and the bit
accounting is walked here over the harness's own tensor list. A number the
submission printed, wrote, smoothed or averaged is recorded beside the graded
numbers for the checkers to compare against, and is never promoted onto the
graded path.

No clock is read, no socket is opened, no random source is consulted. Two runs
over the same frozen bytes produce byte-identical telemetry.
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

# The control allocation is the harness's own within-run comparator: every tensor
# at the budget's mean bit width under round-to-nearest. It is measured on the
# same frozen shards, in the same scheduled order, from the same reference, so
# the separation between it and a submission's allocation needs no external
# anchor to be meaningful. This matters here because F9 anchors are unmeasured.
CONTROL_BITS = 4
CONTROL_SCHEME = "rtn"

# The fixed bit budget, in mean bits per parameter. Frozen. A submission spends
# the same total and reallocates it; it does not raise it.
BITS_PER_PARAM_BUDGET = 4

# The bound separation margin. Transcribed from the batch binding, never authored.
SEPARATION_MARGIN = 0.05

# Scheme error multipliers. The harness owns the quantizer, so it owns these.
SCHEME_FACTOR = {"rtn": 1.0, "affine-per-channel": 0.72, "error-feedback": 0.55}

ALLOWED_BITS = (2, 3, 4, 5, 6, 8)

# The submission runs under an allowlist, not under the verifier's environment.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "TMPDIR")

SUBMISSION_TIMEOUT_SECONDS = 120
ALLOCATION_NAME = "allocation.json"


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes()) if path.is_file() else ""


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_inputs(environment: Path) -> dict:
    return {
        "model_stats": environment / "model_stats.json",
        "eval_corpus": environment / "eval_corpus.json",
        "reference_unquantized": environment / "reference_unquantized.json",
    }


def digests(environment: Path) -> dict:
    return {name: digest_file(path) for name, path in sorted(frozen_inputs(environment).items())}


def total_params(model: dict) -> int:
    return sum(int(row["params"]) for row in model["tensors"])


def budget_bits(model: dict) -> int:
    return BITS_PER_PARAM_BUDGET * total_params(model)


def control_allocation(model: dict) -> dict:
    return {
        "scheme": CONTROL_SCHEME,
        "bits": {str(row["name"]): CONTROL_BITS for row in model["tensors"]},
    }


def normalise_allocation(model: dict, raw) -> dict:
    """Resolve a submitted allocation against the harness's own tensor list.

    A tensor the submission omitted is filled with the control bit width and is
    still counted, so leaving a tensor out of the file is never a way to leave it
    out of the accounting. A tensor the submission named that the model does not
    carry is a defect rather than an ignored key, because a phantom tensor is how
    a budget gets spent on nothing.
    """
    if not isinstance(raw, dict):
        return {"wellformed": False, "defect": "allocation-not-a-mapping", "scheme": "", "bits": {}}
    scheme = str(raw.get("scheme", "")).strip()
    if scheme not in SCHEME_FACTOR:
        return {
            "wellformed": False,
            "defect": "scheme-outside-closed-set",
            "scheme": scheme,
            "bits": {},
        }
    submitted = raw.get("bits")
    if not isinstance(submitted, dict) or not submitted:
        return {"wellformed": False, "defect": "bits-map-absent", "scheme": scheme, "bits": {}}
    names = {str(row["name"]) for row in model["tensors"]}
    phantom = sorted(str(key) for key in submitted if str(key) not in names)
    if phantom:
        return {
            "wellformed": False,
            "defect": "tensor-not-in-model",
            "scheme": scheme,
            "bits": {},
        }
    bits = {}
    for row in model["tensors"]:
        name = str(row["name"])
        value = submitted.get(name, CONTROL_BITS)
        try:
            width = int(value)
        except (TypeError, ValueError):
            return {
                "wellformed": False,
                "defect": "bit-width-not-an-integer",
                "scheme": scheme,
                "bits": {},
            }
        if width not in ALLOWED_BITS:
            return {
                "wellformed": False,
                "defect": "bit-width-outside-closed-set",
                "scheme": scheme,
                "bits": {},
            }
        bits[name] = width
    return {"wellformed": True, "defect": None, "scheme": scheme, "bits": bits}


def accounting(model: dict, allocation: dict) -> dict:
    """Walk the harness's own tensor list and total the bits actually allocated."""
    walked, spent = 0, 0
    for row in model["tensors"]:
        name = str(row["name"])
        width = int(allocation["bits"].get(name, CONTROL_BITS))
        spent += int(row["params"]) * width
        walked += 1
    return {
        "source": "harness-tensor-walk",
        "tensors_walked": walked,
        "tensors_in_model": len(model["tensors"]),
        "allocated_bits": spent,
        "budget_bits": budget_bits(model),
        "bits_per_param_budget": BITS_PER_PARAM_BUDGET,
        "mean_bits_per_param": spent / float(total_params(model)),
    }


def quantization_error(model: dict, allocation: dict) -> float:
    """The harness's error model over the quantized tensor state it just built."""
    scale = SCHEME_FACTOR[allocation["scheme"]]
    params = float(total_params(model))
    error = 0.0
    for row in model["tensors"]:
        name = str(row["name"])
        width = int(allocation["bits"].get(name, CONTROL_BITS))
        share = float(row["params"]) / params
        error += (
            share
            * float(row["sensitivity"])
            * float(row["outlier_factor"])
            * scale
            * (4.0 ** (-(width - 1)))
        )
    return error


def allocation_spread(model: dict, allocation: dict) -> float:
    """How far from uniform the allocation is, in bit widths, parameter-weighted.

    A uniform allocation needs no calibration set, so its reading carries no
    calibration-set jitter. Every departure from uniform buys accuracy with a
    calibration fit, and that fit is what makes the reading shard-dependent. The
    quantity is therefore not decoration: it is the coupling between how clever
    an allocation is and how wide the noise band on its own measurement gets.
    """
    params = float(total_params(model))
    mean = (
        sum(
            float(row["params"]) * int(allocation["bits"].get(str(row["name"]), CONTROL_BITS))
            for row in model["tensors"]
        )
        / params
    )
    return (
        sum(
            (float(row["params"]) / params)
            * abs(int(allocation["bits"].get(str(row["name"]), CONTROL_BITS)) - mean)
            for row in model["tensors"]
        )
        / 4.0
    )


def state_digest(model: dict, allocation: dict) -> str:
    """The digest of the quantized state. Built once, read at every point."""
    payload = json.dumps(
        {
            "model_id": model["model_id"],
            "scheme": allocation["scheme"],
            "bits": allocation["bits"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return digest_bytes(payload)


def schedule_points(environment: Path, corpus: dict) -> dict:
    """The evaluation order, derived by the verifier from the frozen corpus bytes.

    The order is a deterministic function of the corpus digest, so it is fixed
    before any submission runs and no submission can choose it. It is not the
    file order, so a submission that tuned an allocation to whichever shard
    happens to be listed first has tuned to nothing.
    """
    nonce = digest_file(environment / "eval_corpus.json")
    ranked = sorted(
        (digest_bytes((nonce + ":" + str(row["shard_id"])).encode()), str(row["shard_id"]))
        for row in corpus["shards"]
    )
    return {
        "source": "verifier-derived",
        "nonce": nonce,
        "order": [shard for _, shard in ranked],
        "repeat_floor": int(corpus["evaluation_repeat_floor"]),
        "points_scheduled": len(ranked),
    }


def evaluate(model: dict, corpus: dict, reference: dict, allocation: dict, order) -> list:
    """Perplexity at every scheduled point, computed here from the harness state."""
    gain = {str(row["shard_id"]): float(row["shard_gain"]) for row in corpus["shards"]}
    jitter = {str(row["shard_id"]): float(row.get("readout_jitter", 0.0)) for row in corpus["shards"]}
    scale = float(corpus.get("readout_jitter_scale", 0.0))
    base = {str(row["shard_id"]): float(row["ppl"]) for row in reference["reference_ppl"]}
    control = control_allocation(model)
    error_agent = quantization_error(model, allocation)
    error_control = quantization_error(model, control)
    spread_agent = allocation_spread(model, allocation) * scale
    spread_control = allocation_spread(model, control) * scale
    digest = state_digest(model, allocation)
    rows = []
    for shard in order:
        ref = base[shard]
        agent_ppl = ref * math.exp(error_agent * gain[shard] + spread_agent * jitter[shard])
        control_ppl = ref * math.exp(error_control * gain[shard] + spread_control * jitter[shard])
        delta_agent = agent_ppl - ref
        delta_control = control_ppl - ref
        rows.append(
            {
                "shard_id": shard,
                "source": "harness-recompute",
                "state_digest": digest,
                "ppl_reference": ref,
                "raw_ppl_quant": agent_ppl,
                "graded_ppl_quant": agent_ppl,
                "delta_agent": delta_agent,
                "delta_control": delta_control,
                "paired_separation": delta_control - delta_agent,
            }
        )
    return rows


def measurement(points: list) -> dict:
    """The separation, its noise band, and the two edges the verdict reads.

    The band is the half-range of the PAIRED per-point separations, so it is the
    spread of the quantity actually graded rather than the spread of perplexity
    itself. One point produces no band at all, which is the whole reason a single
    favourable evaluation cannot establish anything here.
    """
    values = [float(row["paired_separation"]) for row in points]
    if not values:
        return {
            "separation_mean": 0.0,
            "noise_half_width": 0.0,
            "lower_edge": 0.0,
            "upper_edge": 0.0,
            "separation_margin": SEPARATION_MARGIN,
            "points_used": 0,
            "band_resolvable": False,
        }
    mean = sum(values) / len(values)
    half = (max(values) - min(values)) / 2.0
    return {
        "separation_mean": mean,
        "noise_half_width": half,
        "lower_edge": mean - half,
        "upper_edge": mean + half,
        "separation_margin": SEPARATION_MARGIN,
        "points_used": len(values),
        "band_resolvable": len(values) > 1,
    }


def _child_environment() -> dict:
    return {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}


def run_submission(submission: Path, workdir: Path) -> dict:
    """Copy the submission alone into a fresh directory and run it out of process.

    The submission never shares an interpreter with the grader, never sees the
    verifier's environment, and never outlives the call: the whole process group
    is signalled in a finally block, so a backgrounded child cannot keep writing
    into the evaluation window after the harness has moved on.
    """
    sandbox = workdir / "submission"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    if submission.is_dir():
        for entry in sorted(submission.iterdir()):
            if entry.name == "__pycache__":
                continue
            if entry.is_dir():
                shutil.copytree(entry, sandbox / entry.name)
            else:
                shutil.copy2(entry, sandbox / entry.name)
        entry_point = sandbox / "solve.sh"
    else:
        shutil.copy2(submission, sandbox / submission.name)
        entry_point = sandbox / submission.name
    if not entry_point.is_file():
        return {"launched": False, "exit_code": None, "stdout_bytes": 0, "stderr_bytes": 0}
    entry_point.chmod(0o755)
    process = None
    try:
        process = subprocess.Popen(
            ["/bin/sh", str(entry_point)],
            cwd=str(sandbox),
            env=_child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        out, err = process.communicate(timeout=SUBMISSION_TIMEOUT_SECONDS)
        return {
            "launched": True,
            "exit_code": process.returncode,
            "stdout_bytes": len(out or b""),
            "stderr_bytes": len(err or b""),
        }
    except subprocess.TimeoutExpired:
        return {"launched": True, "exit_code": None, "stdout_bytes": 0, "stderr_bytes": 0}
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait(timeout=10)


def build_telemetry(environment: Path, allocation_raw, submission_meta: dict) -> dict:
    """One telemetry record. This is the handle every checker reads."""
    paths = frozen_inputs(environment)
    opening = digests(environment)
    model = load_json(paths["model_stats"])
    corpus = load_json(paths["eval_corpus"])
    reference = load_json(paths["reference_unquantized"])

    allocation = normalise_allocation(model, allocation_raw)
    schedule = schedule_points(environment, corpus)
    if allocation["wellformed"]:
        points = evaluate(model, corpus, reference, allocation, schedule["order"])
        account = accounting(model, allocation)
    else:
        points, account = [], accounting(model, {"bits": {}, "scheme": CONTROL_SCHEME})
    schedule["points_completed"] = len(points)
    schedule["truncated"] = len(points) < schedule["points_scheduled"] and allocation["wellformed"]
    schedule["truncation_record"] = None

    reported = None
    stop_after = None
    if isinstance(allocation_raw, dict):
        raw_reported = allocation_raw.get("reported_ppl")
        if isinstance(raw_reported, (int, float)):
            reported = float(raw_reported)
        raw_stop = allocation_raw.get("stop_after")
        if isinstance(raw_stop, int):
            stop_after = raw_stop

    return {
        "schema": "oer20.telemetry/v1",
        "produced_by": "tests/runner.py",
        "graded_provenance": "harness-quantized-state",
        "graded_metric": "perplexity-degradation-at-fixed-bit-budget",
        "graded_direction": "lower",
        "smoothing_window_applied": 0,
        "opening_digests": opening,
        "closing_digests": digests(environment),
        "submission": {
            "allocation_sha256": digest_bytes(
                json.dumps(allocation_raw, sort_keys=True, separators=(",", ":")).encode()
            ),
            "reported_ppl": reported,
            "requested_stop_after": stop_after,
            "launched": submission_meta.get("launched", False),
            "exit_code": submission_meta.get("exit_code"),
            "stdout_bytes": submission_meta.get("stdout_bytes", 0),
        },
        "allocation": allocation,
        "accounting": account,
        "schedule": schedule,
        "points": points,
        "measurement": measurement(points),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 evaluation harness")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default="")
    args = parser.parse_args(argv)

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="oer20-"))
    workdir.mkdir(parents=True, exist_ok=True)
    submission = Path(args.submission)
    meta = run_submission(submission, workdir)
    produced = workdir / "submission" / ALLOCATION_NAME
    try:
        allocation_raw = load_json(produced)
    except (OSError, ValueError):
        allocation_raw = None
    telemetry = build_telemetry(Path(args.environment), allocation_raw, meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(telemetry, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

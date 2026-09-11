#!/usr/bin/env python3
"""The verifier-owned harness: quantize the checkpoint, run it, emit telemetry.

This file is the only place a submission is executed, and it is executed as a
separate process group rather than imported, so no submission byte ever enters
the grading interpreter. Everything downstream of here reads the telemetry
record this module writes, and nothing downstream reads the submission's stdout.

The load-bearing asymmetry is this: the submission supplies an ALLOCATION, which
is an input. It never supplies a MEASUREMENT. The quantized model is built here
by writing quantized weights into the frozen checkpoint's own tensors, and the
perplexity at every scheduled evaluation point is obtained here by running that
model forward over tokens the submission has never seen. A number the submission
printed, wrote, smoothed or averaged is recorded beside the graded numbers for
the checkers to compare against, and is never promoted onto the graded path.

Three properties this file exists to hold, and each one is a property of the
re-base rather than of the original slot:

  Weights in the loop. The graded artifact is the float32 parameter snapshot at
  CHECKPOINT_PATH, a real 12-layer 768-dim decoder whose shapes are asserted
  against the frozen architecture before anything is measured. There is no error
  model in this file, no curvature coefficient, no outlier multiplier and no
  closed form for perplexity.

  The verifier recomputes. Every graded number, including the unquantized
  reference, is computed here, in this process, on a held-out FineWeb validation
  slice mounted only on the verifier surface. That slice is absent from every
  agent-visible container and the evaluation order is derived from its digest,
  so neither the tokens nor the order is reachable from the solver's side.

  No forward pass means no score. If the checkpoint is absent, or the held-out
  split is absent, this file produces zero evaluation points and the metric is
  undefined rather than merely different. It does not fall back to an arithmetic
  estimate, because there is no arithmetic estimate to fall back to.

No clock is read and no socket is opened. The evaluation is deterministic: the
same checkpoint, the same tokens and the same allocation produce the same
reading, which is what the calibration probe below is there to demonstrate.
"""

from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from calibration import CALIBRATION_ALLOCATION  # noqa: E402
from decoder import (  # noqa: E402
    Decoder,
    architecture,
    load_checkpoint,
    perplexity,
    quantizable_names,
)

# The control allocation is the harness's own within-run comparator: every tensor
# at the budget's mean bit width under round-to-nearest. It is measured on the
# same held-out points, in the same scheduled order, against the same recomputed
# reference, so the separation between it and a submission's allocation needs no
# external anchor to be meaningful. This matters here because F9 anchors are
# unmeasured.
CONTROL_BITS = 4
CONTROL_SCHEME = "rtn"

# The fixed bit budget, in mean bits per parameter. Frozen. A submission spends
# the same total and reallocates it; it does not raise it.
BITS_PER_PARAM_BUDGET = 4

# The bound separation margin. Transcribed from the batch binding, never authored.
SEPARATION_MARGIN = 0.05

# The bound evaluation repeat floor, transcribed from the same binding block as the
# three constants above: solution/grounding.yaml `bindings.evaluation_repeat_floor`.
# It is read from here rather than out of environment/eval_corpus.json because that
# file is a DATA CARRIER, emitted by seed/forge/substrate.py, and carries a real
# FineWeb token slice and the upstream record corpus rather than any binding of this
# task. Subscripting it for this value raised KeyError before a single evaluation
# point was scheduled, and the exit trap then wrote its zero under
# verifier-aborted-before-grading on every leg. The floor itself is unchanged and
# still gates: tests/checkers.py::check_no_early_stop requires the run to complete
# max(points_scheduled, this floor) points.
EVALUATION_REPEAT_FLOOR = 5

ALLOWED_BITS = (2, 3, 4, 5, 6, 8)
SCHEMES = ("rtn", "affine-per-channel", "error-feedback")

# Verifier-owned paths. None of these is reachable from an agent-visible
# container, and none of them is read from the bundle's environment/ tree.
CHECKPOINT_PATH = Path(os.environ.get("OER20_CHECKPOINT", "/checkpoint/nanogpt_fp32.pt"))
CHECKPOINT_MANIFEST = CHECKPOINT_PATH.parent / "manifest.json"
HELDOUT_ROOT = Path(os.environ.get("OER20_HELDOUT", "/verifier/heldout"))
HELDOUT_MANIFEST = HELDOUT_ROOT / "manifest.json"

# The submission runs under an allowlist, not under the verifier's environment.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "TMPDIR")

SUBMISSION_TIMEOUT_SECONDS = 600
ALLOCATION_NAME = "allocation.json"

REFUSAL_NO_CHECKPOINT = "evaluation-substrate-absent-checkpoint"
REFUSAL_NO_HELDOUT = "evaluation-substrate-absent-heldout-split"


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
        "nanogpt_substrate": environment / "nanogpt_substrate.json",
    }


def digests(environment: Path) -> dict:
    return {name: digest_file(path) for name, path in sorted(frozen_inputs(environment).items())}


# --------------------------------------------------------------------------
# The quantizers. The harness owns them, so it implements them. Each returns a
# dequantized weight matrix that is written back into the live model, so what is
# evaluated afterwards is the real decoder carrying real quantized weights.
# --------------------------------------------------------------------------
def quantize_rtn(weight: torch.Tensor, bits: int) -> torch.Tensor:
    limit = float(2 ** (bits - 1) - 1)
    scale = weight.abs().amax().clamp(min=1e-12) / limit
    return torch.round(weight / scale).clamp(-limit, limit) * scale


def quantize_affine(weight: torch.Tensor, bits: int) -> torch.Tensor:
    levels = float(2**bits - 1)
    low = weight.amin(dim=1, keepdim=True)
    high = weight.amax(dim=1, keepdim=True)
    scale = ((high - low) / levels).clamp(min=1e-12)
    zero = torch.round(-low / scale)
    return (torch.round(weight / scale) + zero).clamp(0, levels).sub(zero).mul(scale)


def quantize_error_feedback(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-row affine quantization with the residual diffused along the row."""
    levels = float(2**bits - 1)
    low = weight.amin(dim=1, keepdim=True)
    high = weight.amax(dim=1, keepdim=True)
    scale = ((high - low) / levels).clamp(min=1e-12)
    zero = torch.round(-low / scale)
    out = torch.empty_like(weight)
    carry = torch.zeros(weight.size(0), dtype=weight.dtype, device=weight.device)
    for column in range(weight.size(1)):
        target = weight[:, column] + carry
        code = torch.round(target / scale[:, 0]) + zero[:, 0]
        value = (code.clamp(0, levels) - zero[:, 0]) * scale[:, 0]
        carry = target - value
        out[:, column] = value
    return out


QUANTIZERS = {
    "rtn": quantize_rtn,
    "affine-per-channel": quantize_affine,
    "error-feedback": quantize_error_feedback,
}


def total_params(manifest: dict) -> int:
    return sum(int(row["params"]) for row in manifest["tensors"])


def budget_bits(manifest: dict) -> int:
    return BITS_PER_PARAM_BUDGET * total_params(manifest)


def control_allocation(manifest: dict) -> dict:
    return {
        "scheme": CONTROL_SCHEME,
        "bits": {str(row["name"]): CONTROL_BITS for row in manifest["tensors"]},
    }


def normalise_allocation(manifest: dict, raw) -> dict:
    """Resolve a submitted allocation against the harness's own tensor list.

    A tensor the submission omitted is filled with the control bit width and is
    still counted, so leaving a tensor out of the file is never a way to leave it
    out of the accounting. A tensor the submission named that the checkpoint does
    not carry is a defect rather than an ignored key, because a phantom tensor is
    how a budget gets spent on nothing.
    """
    if not isinstance(raw, dict):
        return {"wellformed": False, "defect": "allocation-not-a-mapping", "scheme": "", "bits": {}}
    scheme = str(raw.get("scheme", "")).strip()
    if scheme not in SCHEMES:
        return {"wellformed": False, "defect": "scheme-outside-closed-set", "scheme": scheme, "bits": {}}
    submitted = raw.get("bits")
    if not isinstance(submitted, dict) or not submitted:
        return {"wellformed": False, "defect": "bits-map-absent", "scheme": scheme, "bits": {}}
    names = {str(row["name"]) for row in manifest["tensors"]}
    phantom = sorted(str(key) for key in submitted if str(key) not in names)
    if phantom:
        return {"wellformed": False, "defect": "tensor-not-in-model", "scheme": scheme, "bits": {}}
    bits = {}
    for row in manifest["tensors"]:
        name = str(row["name"])
        value = submitted.get(name, CONTROL_BITS)
        try:
            width = int(value)
        except (TypeError, ValueError):
            return {"wellformed": False, "defect": "bit-width-not-an-integer", "scheme": scheme, "bits": {}}
        if width not in ALLOWED_BITS:
            return {"wellformed": False, "defect": "bit-width-outside-closed-set", "scheme": scheme, "bits": {}}
        bits[name] = width
    return {"wellformed": True, "defect": None, "scheme": scheme, "bits": bits}


def accounting(manifest: dict, allocation: dict) -> dict:
    """Walk the harness's own tensor list and total the bits actually allocated."""
    walked, spent = 0, 0
    for row in manifest["tensors"]:
        width = int(allocation["bits"].get(str(row["name"]), CONTROL_BITS))
        spent += int(row["params"]) * width
        walked += 1
    return {
        "source": "harness-tensor-walk",
        "tensors_walked": walked,
        "tensors_in_model": len(manifest["tensors"]),
        "allocated_bits": spent,
        "budget_bits": budget_bits(manifest),
        "bits_per_param_budget": BITS_PER_PARAM_BUDGET,
        "mean_bits_per_param": spent / float(total_params(manifest)),
    }


def state_digest(manifest: dict, allocation: dict, checkpoint_digest: str) -> str:
    """The digest of the quantized state: the checkpoint bytes plus the allocation.

    It binds the parameters actually quantized, so two points carrying the same
    digest were read off the same weights and not merely off the same request.
    """
    payload = json.dumps(
        {
            "checkpoint_sha256": checkpoint_digest,
            "model_id": manifest["model_id"],
            "scheme": allocation["scheme"],
            "bits": allocation["bits"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return digest_bytes(payload)


# --------------------------------------------------------------------------
# The held-out split. Verifier-owned, absent from environment/, and the only
# tokens the graded numbers are ever computed over.
# --------------------------------------------------------------------------
def heldout_available() -> bool:
    return HELDOUT_MANIFEST.is_file() and CHECKPOINT_PATH.is_file()


def heldout_points(seq_len: int):
    """Load the held-out evaluation points from the verifier's own copy."""
    manifest = load_json(HELDOUT_MANIFEST)
    tokens_path = HELDOUT_ROOT / str(manifest["tokens"])
    with open(tokens_path, "rb") as handle:
        handle.read(int(manifest.get("header_bytes_skipped", 1024)))
        stream = np.frombuffer(handle.read(), dtype="<u2")
    points = {}
    for row in manifest["points"]:
        rows, offset = int(row["rows"]), int(row["token_offset"])
        span = rows * (seq_len + 1)
        block = stream[offset : offset + span]
        if len(block) < span:
            raise ValueError("the held-out split is short at point " + str(row["point_id"]))
        points[str(row["point_id"])] = torch.from_numpy(block.astype(np.int64)).view(rows, seq_len + 1)
    return points


def schedule_points(corpus: dict) -> dict:
    """The evaluation order, derived by the verifier from its OWN held-out bytes.

    The order is a deterministic function of the held-out manifest digest, which
    lives on the verifier surface only. It is fixed before any submission runs,
    it is not the file order, and unlike the original slot it is not derivable
    from anything an agent-visible container carries, so an allocation tuned to
    whichever point is visited first has been tuned to a coin flip.
    """
    nonce = digest_file(HELDOUT_MANIFEST)
    ids = [str(row["point_id"]) for row in load_json(HELDOUT_MANIFEST)["points"]] if nonce else []
    ranked = sorted((digest_bytes((nonce + ":" + point).encode()), point) for point in ids)
    return {
        "source": "verifier-derived",
        "nonce": nonce,
        "order": [point for _, point in ranked],
        "repeat_floor": EVALUATION_REPEAT_FLOOR,
        "points_scheduled": len(ranked),
    }


def apply_allocation(model: Decoder, reference_state: dict, allocation: dict) -> None:
    quantizer = QUANTIZERS[allocation["scheme"]]
    live = model.state_dict()
    with torch.no_grad():
        for name, width in allocation["bits"].items():
            live[name].copy_(quantizer(reference_state[name].clone(), int(width)))


def restore(model: Decoder, reference_state: dict) -> None:
    live = model.state_dict()
    with torch.no_grad():
        for name, tensor in reference_state.items():
            live[name].copy_(tensor)


def evaluate(model, reference_state, points, allocation, control, order, digest, device) -> list:
    """Perplexity at every scheduled point, from real forward passes only.

    Three readings per point over the same token batch in the same process: the
    unquantized reference, the submission's allocation, and the harness control.
    The batch's own difficulty cancels out of the paired difference, which is why
    no frozen reference table is needed and why one is no longer shipped.
    """
    rows = []
    for point in order:
        batch = points[point]
        restore(model, reference_state)
        reference_ppl = perplexity(model, batch, device)
        apply_allocation(model, reference_state, allocation)
        agent_ppl = perplexity(model, batch, device)
        apply_allocation(model, reference_state, control)
        control_ppl = perplexity(model, batch, device)
        rows.append(
            {
                "shard_id": point,
                "source": "harness-recompute",
                "state_digest": digest,
                "ppl_reference": reference_ppl,
                "raw_ppl_quant": agent_ppl,
                "graded_ppl_quant": agent_ppl,
                "ppl_control": control_ppl,
                "delta_agent": agent_ppl - reference_ppl,
                "delta_control": control_ppl - reference_ppl,
                "paired_separation": (control_ppl - reference_ppl) - (agent_ppl - reference_ppl),
            }
        )
    restore(model, reference_state)
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


def calibration_probe(model, reference_state, points, manifest, order, digest, device) -> dict:
    """The harness's determinism identity, measured on the built environment.

    CALIBRATION_ALLOCATION is the control allocation restated. Measured against
    the control it must separate by exactly zero at every point, with a band of
    exactly zero, because the same weights evaluated twice over the same tokens
    must read the same. The submission supplies nothing to this probe and cannot
    move it.

    What it demonstrates is a property the original slot could not have failed
    and this one can: that the graded path is a real, deterministic, paired
    evaluation. An evaluation that drifts between two readings of the same state,
    or that pairs the wrong batch to the wrong reading, fails here and takes the
    reward to zero before any separation is believed. The centre and the width
    this probe is graded against are identities of the measurement path and not
    values this lane measured, which is why neither is invented.
    """
    allocation = normalise_allocation(manifest, CALIBRATION_ALLOCATION)
    if not allocation["wellformed"]:
        return {
            "source": "harness-recompute",
            "wellformed": False,
            "defect": allocation["defect"],
            "points_used": 0,
            "separation_mean": None,
            "noise_half_width": None,
        }
    rows = evaluate(
        model, reference_state, points, allocation, control_allocation(manifest), order, digest, device
    )
    measure = measurement(rows)
    return {
        "source": "harness-recompute",
        "wellformed": True,
        "defect": None,
        "probe": "control-restated-against-control, a determinism identity",
        "state_digest": digest,
        "points_used": measure["points_used"],
        "separation_mean": measure["separation_mean"],
        "noise_half_width": measure["noise_half_width"],
        "shard_separations": [
            {"shard_id": row["shard_id"], "paired_separation": row["paired_separation"]} for row in rows
        ],
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


def build_telemetry(environment: Path, allocation_raw, submission_meta: dict, device_name: str) -> dict:
    """One telemetry record. This is the handle every checker reads."""
    paths = frozen_inputs(environment)
    opening = digests(environment)
    manifest = load_json(paths["model_stats"])
    corpus = load_json(paths["eval_corpus"])

    allocation = normalise_allocation(manifest, allocation_raw)
    refusal = None
    points, calibration = [], {
        "source": "harness-recompute",
        "wellformed": False,
        "defect": None,
        "points_used": 0,
        "separation_mean": None,
        "noise_half_width": None,
    }

    if not CHECKPOINT_PATH.is_file():
        refusal = REFUSAL_NO_CHECKPOINT
        schedule = {"source": "verifier-derived", "nonce": "", "order": [], "repeat_floor": EVALUATION_REPEAT_FLOOR, "points_scheduled": 0}
    elif not HELDOUT_MANIFEST.is_file():
        refusal = REFUSAL_NO_HELDOUT
        schedule = {"source": "verifier-derived", "nonce": "", "order": [], "repeat_floor": EVALUATION_REPEAT_FLOOR, "points_scheduled": 0}
    else:
        schedule = schedule_points(corpus)

    checkpoint_digest = digest_file(CHECKPOINT_PATH)
    if refusal is None and allocation["wellformed"]:
        device = torch.device(device_name)
        arch = architecture()
        model = load_checkpoint(Decoder(arch), CHECKPOINT_PATH).to(device)
        if sorted(quantizable_names(arch)) != sorted(str(row["name"]) for row in manifest["tensors"]):
            raise ValueError("the tensor manifest does not name the checkpoint's own quantizable tensors")
        reference_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        batches = heldout_points(arch["seq_len"])
        digest = state_digest(manifest, allocation, checkpoint_digest)
        control = control_allocation(manifest)
        points = evaluate(
            model, reference_state, batches, allocation, control, schedule["order"], digest, device
        )
        calibration = calibration_probe(
            model, reference_state, batches, manifest, schedule["order"], digest, device
        )

    account = accounting(manifest, allocation if allocation["wellformed"] else {"bits": {}, "scheme": CONTROL_SCHEME})
    schedule["points_completed"] = len(points)
    schedule["truncated"] = len(points) < schedule["points_scheduled"] and allocation["wellformed"]
    schedule["truncation_record"] = None

    reported, stop_after = None, None
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
        "graded_evaluation": "real forward passes of the frozen checkpoint over the verifier-owned held-out FineWeb validation slice",
        "smoothing_window_applied": 0,
        "refusal": refusal,
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "present": CHECKPOINT_PATH.is_file(),
            "sha256": checkpoint_digest,
            "manifest_sha256": digest_file(CHECKPOINT_MANIFEST),
        },
        "held_out_split": {
            "root": str(HELDOUT_ROOT),
            "present": HELDOUT_MANIFEST.is_file(),
            "manifest_sha256": digest_file(HELDOUT_MANIFEST),
            "absent_from_environment": not (environment / "heldout").exists(),
        },
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
        "calibration": calibration,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 evaluation harness")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--device", default=os.environ.get("OER20_DEVICE", "cuda"))
    args = parser.parse_args(argv)

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="oer20-"))
    workdir.mkdir(parents=True, exist_ok=True)
    meta = run_submission(Path(args.submission), workdir)
    produced = workdir / "submission" / ALLOCATION_NAME
    try:
        allocation_raw = load_json(produced)
    except (OSError, ValueError):
        allocation_raw = None
    telemetry = build_telemetry(Path(args.environment), allocation_raw, meta, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(telemetry, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

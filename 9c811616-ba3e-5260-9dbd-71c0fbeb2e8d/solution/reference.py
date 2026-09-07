#!/usr/bin/env python3
"""The reference solution procedure. Agent-side, and it runs the real model.

This module is what a run has to do rather than a description of it. It re-probes the
calibration handle instead of reusing an answer from earlier in the session, reads the
slice the version in force pins, MEASURES the per-tensor cost curve by quantizing one
tensor at a time at every admissible width and reading the loss that comes back, fits
an allocation under the frozen budget, selects among its candidates by a joint
measurement, and records both witnesses over bytes it actually touched.

Every number it uses about the model comes out of a forward pass. There is no shipped
sensitivity vector to read and no closed-form degradation to evaluate, so an agent that
never loads the checkpoint has nothing to fit against.

The fit is environment/model.py fit_allocation, which is the SAME function
tests/evaluate.py calls to measure the target scale point. It is called here and not
restated, because an allocation that has to agree with the verifier's tensor for tensor
must not be produced by a second expression of the procedure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "environment"))

import model as frozen  # noqa: E402

ROUND = 9
SUBMISSION_SCHEMA = "oer22.submission/v2"


def canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(payload) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def allocated_bits(tensors, allocation, group_size, scale_bits, include_scales) -> int:
    return frozen.allocated_bits(tensors, allocation, group_size, scale_bits, include_scales)


def solve(workspace: Path, checkpoint: Path, device: str, calibration_rows: int) -> dict:
    """Probe, measure, fit, and record what was actually touched."""
    environment = Path(workspace) / "environment"
    spec = read_json(environment / "substrate.json")
    state = read_json(environment / "calibration_state.json")
    stats = read_json(environment / "calibration_stats.json")
    declaration = frozen.load_declaration(environment / "nanogpt_substrate.json")
    arch = frozen.architecture(declaration)

    # The probe. The calibration set moved and nothing announced it, so the version in
    # force is re-read here rather than assumed timeless.
    in_force = int(state["calibration_version_in_force"])
    ledger = [int(row["version"]) for row in state["ledger"] if int(row["version"]) <= in_force]
    record = next(row for row in stats["versions"] if int(row["version"]) == in_force)
    pin = record["pin"]

    root = Path(stats["corpus_root"])
    slice_witness = frozen.slice_content_sha256(root, pin)
    stream = frozen.read_slice(root, pin)

    # The WHOLE pinned slice, which is what tests/evaluate.py reference_allocation reads.
    # A predecessor of this file fitted on the first seq_len * calibration_rows tokens of
    # it, so the agent surface and the verifier surface were fitting on different windows
    # of the same pin and could return different allocations from the same procedure.
    net = frozen.build(arch, device)
    frozen.load_into(net, Path(checkpoint))
    scheduled = read_json(environment / "eval_corpus_manifest.json")["scheduled_points"]
    fitted = frozen.fit_allocation(
        net, spec, stream, arch["seq_len"], calibration_rows, device, len(scheduled),
    )
    allocation = fitted["allocation"]

    return {
        "schema": SUBMISSION_SCHEMA,
        "allocation": {key: int(value) for key, value in sorted(allocation.items())},
        "quantization_scheme": {
            "group_size": int(spec["group_size"]),
            "scale_bits": int(spec["scale_bits"]),
        },
        "derived_against_calibration_version": in_force,
        "calibration_fit_witness": digest(pin),
        "calibration_slice_witness": slice_witness,
        "observed_calibration_ledger": ledger,
        "protocol": {
            "points_completed": len(scheduled),
            "halted_early": False,
        },
        "readout": {"filter": "none", "reported_degradation": None},
    }

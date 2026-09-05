#!/usr/bin/env python3
"""The reference solution procedure. Agent-side, and it runs the real model.

This module is what a run has to do rather than a description of it. It re-probes the
calibration handle instead of reusing an answer from earlier in the session, reads the
slice the version in force pins, MEASURES per-tensor sensitivity by quantizing one
tensor at a time and reading the perplexity that comes back, fits a bit allocation
under the frozen budget by greedy marginal gain, and records both witnesses over bytes
it actually touched.

Every number it uses about the model comes out of a forward pass. There is no shipped
sensitivity vector to read and no closed-form degradation to evaluate, so an agent that
never loads the checkpoint has nothing to fit against.

The verifier runs the identical fit on its own pristine copy to measure the target
scale point, so the bar this procedure is held to is one the harness reached on the
same corpus in the same run.
"""

from __future__ import annotations

import copy
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


@torch.no_grad()
def measure_sensitivity(net, spec: dict, stream, seq_len: int, rows: int, device: str) -> dict:
    """Per-tensor sensitivity, measured. One tensor quantized at a time, then restored.

    The loss increase a tensor causes when it alone is taken to the uniform reference
    width is what this returns. It is a measurement over the activations of the slice
    it is given, which is exactly why it moves when the calibration slice moves.
    """
    group_size = int(spec["group_size"])
    uniform_bits = int(spec["uniform_reference_bits"])
    base = frozen.perplexity(net, stream, seq_len, rows, device)["loss"]
    parameters = dict(net.named_parameters())
    out = {}
    for row in spec["quantizable_tensors"]:
        name = row["id"]
        original = copy.deepcopy(parameters[name].data)
        parameters[name].copy_(frozen.quantize_dequantize(original, uniform_bits, group_size))
        moved = frozen.perplexity(net, stream, seq_len, rows, device)["loss"]
        parameters[name].copy_(original)
        out[name] = round(max(0.0, moved - base), ROUND)
    return out


def greedy_allocation(spec: dict, sensitivity: dict) -> dict:
    """Greedy marginal gain under the frozen budget. Ties break on tensor index.

    Every tensor starts at the floor width. The bit that buys the largest predicted
    reduction per bit of budget is bought next, and the loop ends when no affordable
    purchase remains, so the result is fixed by the measurement alone and never by
    iteration order.
    """
    tensors = spec["quantizable_tensors"]
    group_size = int(spec["group_size"])
    scale_bits = int(spec["scale_bits"])
    budget = int(spec["budget_bits"])
    choices = list(spec["bit_choices"])
    bit_min, bit_max = min(choices), max(choices)
    allocation = {row["id"]: bit_min for row in tensors}
    spent = allocated_bits(tensors, allocation, group_size, scale_bits, True)
    while True:
        best_key = None
        best_index = None
        for index, row in enumerate(tensors):
            bits = allocation[row["id"]]
            if bits >= bit_max or spent + int(row["numel"]) > budget:
                continue
            gain = float(sensitivity[row["id"]]) * (2.0 ** (-2 * bits) - 2.0 ** (-2 * (bits + 1)))
            key = (-gain, index)
            if best_key is None or key < best_key:
                best_key, best_index = key, index
        if best_index is None:
            return allocation
        allocation[tensors[best_index]["id"]] += 1
        spent += int(tensors[best_index]["numel"])


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
    window = stream[: arch["seq_len"] * calibration_rows + 1]

    net = frozen.build(arch, device)
    frozen.load_into(net, Path(checkpoint))
    sensitivity = measure_sensitivity(net, spec, window, arch["seq_len"], calibration_rows, device)
    allocation = greedy_allocation(spec, sensitivity)

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
            "points_completed": len(read_json(environment / "eval_corpus_manifest.json")["scheduled_points"]),
            "halted_early": False,
        },
        "readout": {"filter": "none", "reported_degradation": None},
    }

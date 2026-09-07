#!/usr/bin/env python3
"""The verifier's own evaluation: real parameters, real forward passes, real corpus.

VERIFIER-OWNED. This module is the only producer of a graded number in this slot, and
it produces it the only way this slot admits: by loading the verifier's pristine
nanoGPT checkpoint, quantizing it under the submitted allocation with the harness's
own quantizer, and running forward passes over the verifier's own held-out FineWeb
folds. The unquantized reference perplexity is measured by the same code path on the
same folds in the same run, so the two halves of the degradation cannot drift apart.

Three properties this file exists to guarantee.

The graded artifact is a real parameter snapshot of the frozen architecture. Before
anything is evaluated, the checkpoint's state dict is compared shape for shape against
the map that environment/nanogpt_substrate.json implies, and a disagreement raises.

The graded scalar is the verifier's own recomputation on a split absent from the agent
container. The split is resolved from tests/bound.json and from nowhere else, and no
call reachable from a solver returns it.

No forward pass means no score. Delete `GPT.forward` in the frozen model module and
every function below raises rather than returning a number. There is no closed-form
degradation, no sensitivity vector and no cost model left in this bundle that could
keep emitting a value after the passes are removed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(BUNDLE / "environment"))

import model as frozen  # noqa: E402

ROUND = 9


def declaration(bundle: Path) -> dict:
    return frozen.load_declaration(Path(bundle) / "environment" / "nanogpt_substrate.json")


def substrate(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "environment" / "substrate.json").read_text(encoding="utf-8"))


def architecture_binding(bundle: Path, checkpoint: Path) -> dict:
    """Compare a checkpoint against the architecture the declaration implies.

    A vocab-8 weight table, a six-layer transformer and a formula-generated parameter
    file all fail here, because none of them carries this shape map.
    """
    arch = frozen.architecture(declaration(bundle))
    expected = frozen.expected_parameter_shapes(arch)
    try:
        observed = frozen.checkpoint_shapes(Path(checkpoint))
    except (OSError, ValueError, KeyError, RuntimeError) as problem:
        return {"bound": False, "reason": type(problem).__name__, "architecture": arch,
                "expected_parameters": len(expected), "observed_parameters": 0}
    return {
        "bound": observed == expected,
        "reason": "",
        "architecture": arch,
        "expected_parameters": len(expected),
        "observed_parameters": len(observed),
        "mismatched": sorted(
            name for name in set(expected) | set(observed) if expected.get(name) != observed.get(name)
        )[:8],
    }


def _load_pristine(bundle: Path, checkpoint: Path, device: str):
    arch = frozen.architecture(declaration(bundle))
    net = frozen.build(arch, device)
    frozen.load_into(net, Path(checkpoint))
    return net, arch


def _fold_streams(bound: dict):
    paths = sorted(Path(bound["holdout_root"]).glob(str(bound["holdout_val_glob"])))
    if not paths:
        raise ValueError("the verifier's held-out split is absent at " + str(bound["holdout_root"]))
    stream = frozen.load_stream(paths)
    out = []
    for row in bound["folds"]:
        low, high = int(row["tokens"][0]), int(row["tokens"][1])
        out.append({"point_id": row["id"], "stream": stream[low:high]})
    return out


def reference_perplexity(bundle: Path, bound: dict, checkpoint: Path) -> dict:
    """The unquantized reference, measured. Never read from an agent-visible byte."""
    device = str(bound["device"])
    net, arch = _load_pristine(bundle, checkpoint, device)
    rows = {}
    for fold in _fold_streams(bound):
        reading = frozen.perplexity(net, fold["stream"], arch["seq_len"], int(bound["eval_rows"]), device)
        rows[fold["point_id"]] = round(float(reading["perplexity"]), ROUND)
    return {"per_point": rows, "mean": round(sum(rows.values()) / len(rows), ROUND)}


def quantized_perplexity(bundle: Path, bound: dict, checkpoint: Path, allocation: dict) -> dict:
    """The submitted allocation, applied by the harness to the harness's own copy.

    The submission hands over integers. Every tensor those integers name is quantized
    here, in this process, by environment/model.py quantize_dequantize, and the
    perplexity that comes back is this process's own forward pass. Nothing the
    submission printed, reported or wrote is summed, averaged or substituted in.
    """
    device = str(bound["device"])
    net, arch = _load_pristine(bundle, checkpoint, device)
    group_size = int(substrate(bundle)["group_size"])
    frozen.apply_allocation(net, allocation, group_size)
    rows = {}
    for fold in _fold_streams(bound):
        reading = frozen.perplexity(net, fold["stream"], arch["seq_len"], int(bound["eval_rows"]), device)
        rows[fold["point_id"]] = round(float(reading["perplexity"]), ROUND)
    return {"per_point": rows, "mean": round(sum(rows.values()) / len(rows), ROUND)}


def degradation(quantized: dict, reference: dict) -> dict:
    """ppl(quantized) minus ppl(unquantized reference), per point and in the mean."""
    points = [
        {"point_id": key, "degradation": round(quantized["per_point"][key] - reference["per_point"][key], ROUND)}
        for key in sorted(quantized["per_point"])
    ]
    return {
        "points": points,
        "mean": round(sum(row["degradation"] for row in points) / len(points), ROUND),
    }


# --------------------------------------------------------------------------
# The reference allocation, fitted on the calibration slice in force. Real forward
# passes, not a shipped vector and not an extrapolation.
#
# The procedure itself is environment/model.py fit_allocation and it is NOT restated
# here. solution/reference.py calls the identical function on the agent surface, so
# the allocation a submission carries and the allocation this target is measured from
# come out of one expression rather than two that have to be kept in step by hand.
# --------------------------------------------------------------------------
def reference_allocation(bundle: Path, bound: dict, checkpoint: Path, pin: dict) -> dict:
    """Fit the reference allocation on the slice the pin names, by measurement.

    The verifier runs this over its own pristine checkpoint and its own copy of the
    pinned slice, so the bar a submission is held to is a bar the harness reached on
    the same corpus in the same run rather than a number carried in from a retired
    substrate. The slice moved, so the measurement moves with it, which is what makes
    the calibration-set drift bite.
    """
    device = str(bound["device"])
    net, arch = _load_pristine(bundle, checkpoint, device)
    spec = substrate(bundle)
    stream = frozen.read_slice(Path(bound["calibration_root"]), pin)
    fitted = frozen.fit_allocation(
        net, spec, stream, arch["seq_len"], int(bound["eval_rows"]), device, len(bound["folds"]),
    )
    return {
        "allocation": fitted["allocation"],
        "chosen_candidate": fitted["chosen"],
        "qualifying_candidates": list(fitted["qualifying"]),
        "fit_base_loss": round(fitted["fit_base_loss"], ROUND),
        "selection_mean_cost": {
            name: round(row["mean_cost"], ROUND) for name, row in sorted(fitted["selection"].items())
        },
        "pin": pin,
    }


def uniform_allocation(spec: dict) -> dict:
    return frozen.uniform_allocation(spec)

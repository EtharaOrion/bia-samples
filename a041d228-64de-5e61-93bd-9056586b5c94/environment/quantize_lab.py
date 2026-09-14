#!/usr/bin/env python3
"""Advisory local lab. Real quantization, real forward passes, train-split only.

This file is agent-visible and read-only. It loads the frozen checkpoint, applies
an allocation to the real weight matrices, and measures perplexity by running the
model. It computes nothing in closed form: there is no error model here, no
curvature coefficient and no per-shard perplexity table, because there is nothing
left for such a thing to stand in for.

It is ADVISORY in the way that matters most under this re-base: it evaluates on
the TRAIN split, which is the only corpus an agent-visible container carries. The
graded evaluation runs on the verifier's held-out validation slice, which is
absent from this container and which no allocation search can reach. The two
readings correlate because they are the same model and the same quantizer; they
are not the same number and this file never pretends otherwise.

The bit accounting is exact and agrees with the verifier's, because both walk the
same fifty tensors of environment/model_stats.json.

Usage:
    python3 quantize_lab.py --allocation allocation.json
    python3 quantize_lab.py --allocation allocation.json --points 3 --rows 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nanogpt_model import Decoder, architecture, load_checkpoint, perplexity, quantizable_names

HERE = Path(__file__).resolve().parent

CONTROL_BITS = 4
CONTROL_SCHEME = "rtn"
BITS_PER_PARAM_BUDGET = 4
SEPARATION_MARGIN = 0.05
ALLOWED_BITS = (2, 3, 4, 5, 6, 8)
SCHEMES = ("rtn", "affine-per-channel", "error-feedback")


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The quantizers. Each one returns a dequantized weight matrix of the same
# shape and dtype, so the model that runs afterwards is the real model with
# real quantized weights in it rather than a model with a penalty attached.
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
    """Per-row affine quantization with the residual diffused along the row.

    Column by column, the rounding residual of the column just quantized is
    carried onto the next column before that column is quantized, so the error
    the allocation spends its bits against is compensated instead of accumulated.
    This is a real quantizer over real weights and it is sequential in the input
    dimension, which is why it costs more to run and buys a smaller degradation.
    """
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


def resolve(manifest: dict, bits) -> dict:
    return {
        str(row["name"]): int(bits.get(str(row["name"]), CONTROL_BITS)) for row in manifest["tensors"]
    }


def account(manifest: dict, bits) -> dict:
    total = int(manifest["total_params"])
    widths = resolve(manifest, bits)
    spent = sum(int(row["params"]) * widths[str(row["name"])] for row in manifest["tensors"])
    return {
        "total_params": total,
        "allocated_bits": spent,
        "budget_bits": BITS_PER_PARAM_BUDGET * total,
        "mean_bits_per_param": spent / float(total),
        "within_budget": spent <= BITS_PER_PARAM_BUDGET * total,
    }


def apply_allocation(model: Decoder, reference_state: dict, widths: dict, scheme: str) -> None:
    """Write quantized weights into the live model. The model IS the quantized model."""
    quantizer = QUANTIZERS[scheme]
    with torch.no_grad():
        for name, width in widths.items():
            source = reference_state[name]
            model.state_dict()[name].copy_(quantizer(source.clone(), width))


def restore(model: Decoder, reference_state: dict) -> None:
    with torch.no_grad():
        for name, tensor in reference_state.items():
            model.state_dict()[name].copy_(tensor)


def train_points(data: Path, count: int, rows: int, seq_len: int):
    """Local evaluation points, taken from the TRAIN split. Never the held-out one."""
    shards = sorted(Path(data).glob("fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError("no train shards under " + str(data))
    with open(shards[0], "rb") as handle:
        handle.read(1024)
        tokens = np.frombuffer(handle.read(), dtype="<u2")
    span = rows * (seq_len + 1)
    points = []
    for index in range(count):
        start = index * span
        block = tokens[start : start + span]
        if len(block) < span:
            break
        points.append(torch.from_numpy(block.astype(np.int64)).view(rows, seq_len + 1))
    return points


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 advisory quantization lab")
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--environment", default=str(HERE))
    parser.add_argument("--checkpoint", default="/checkpoint/nanogpt_fp32.pt")
    parser.add_argument("--data", default="/workspace/data/fineweb10B")
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    env = Path(args.environment)
    manifest = load(env / "model_stats.json")
    submitted = load(Path(args.allocation))

    scheme = str(submitted.get("scheme", ""))
    if scheme not in SCHEMES:
        print("scheme outside the closed set: " + repr(scheme))
        return 2
    bits = submitted.get("bits") or {}
    known = {str(row["name"]) for row in manifest["tensors"]}
    phantom = sorted(name for name in bits if str(name) not in known)
    if phantom:
        print("tensors the model does not carry: " + ", ".join(phantom))
        return 2
    bad = sorted(name for name, width in bits.items() if int(width) not in ALLOWED_BITS)
    if bad:
        print("bit widths outside the closed set on: " + ", ".join(bad))
        return 2

    book = account(manifest, bits)
    print(json.dumps({"accounting": book}, indent=2, sort_keys=True))
    if not book["within_budget"]:
        print("this allocation overspends the budget and would score 0.0 with reason bit-budget-overspent")
        return 2

    device = torch.device(args.device)
    arch = architecture(env / "nanogpt_substrate.json")
    model = load_checkpoint(Decoder(arch), args.checkpoint).to(device)
    reference_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    widths = resolve(manifest, bits)
    control = {name: CONTROL_BITS for name in quantizable_names(arch)}

    points = train_points(Path(args.data), args.points, args.rows, arch["seq_len"])
    rows_out = []
    for index, batch in enumerate(points):
        restore(model, reference_state)
        base = perplexity(model, batch, device)
        apply_allocation(model, reference_state, widths, scheme)
        agent = perplexity(model, batch, device)
        apply_allocation(model, reference_state, control, CONTROL_SCHEME)
        controlled = perplexity(model, batch, device)
        rows_out.append(
            {
                "point": "train-local-" + str(index),
                "ppl_reference": base,
                "delta": agent - base,
                "delta_control": controlled - base,
                "separation": (controlled - base) - (agent - base),
            }
        )
    restore(model, reference_state)

    values = [row["separation"] for row in rows_out]
    mean = sum(values) / len(values)
    half = (max(values) - min(values)) / 2.0
    print(json.dumps({"points": rows_out}, indent=2, sort_keys=True))
    print("separation mean       : " + repr(mean))
    print("noise half width      : " + repr(half))
    print("lower edge of the band: " + repr(mean - half))
    print("bound margin          : " + repr(SEPARATION_MARGIN))
    print("this reading is on the TRAIN split; the graded reading is the verifier's on its held-out split")
    print("this lab states no verdict; the verifier computes it over its own scheduled order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

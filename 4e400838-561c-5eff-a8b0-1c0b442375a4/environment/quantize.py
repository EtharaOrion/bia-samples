#!/usr/bin/env python3
"""The post-training quantization toolkit shipped with this task.

Usage:
    python3 quantize.py measure                  evaluate the shipped configuration
    python3 quantize.py sweep                    evaluate every configuration this tool reaches
    python3 quantize.py emit --bits 4 --codebook symmetric --clip 1.0
                                                 write scheme.json under OUT_DIR
    python3 quantize.py shards                   print the parameter shard table

What this tool does, exactly: it loads the frozen nanoGPT checkpoint, sets ONE bit width
for every parameter shard, with ONE scale group per shard, a symmetric or affine codebook,
and one clip percentile, and measures the result by running real forward passes on the
agent-visible CALIBRATION split. That is the whole of its option set, and `sweep`
enumerates all of it.

What this tool does NOT do: it cannot set a different width per shard, it cannot place
more than one scale group inside a shard, and it cannot emit a non-uniform codebook. Those
axes are not broken here and they are not hidden; they are simply outside what this script
was written to reach. The scheme format the grader accepts is published in full at
environment/scheme_schema.json, and any scheme.json conforming to it is graded, whether
this tool produced it or not.

The perplexity numbers printed here are computed the same way the grader computes them,
from the same architecture module, but on a DIFFERENT split. The grader recomputes the
graded quantity itself, from its own copy of the frozen checkpoint, on held-out FineWeb
slices that are not in this container and that nothing here can reach. Nothing this script
prints reaches the grade, and a reading taken here is a proxy rather than a preview.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SUBSTRATE = HERE / "nanogpt_substrate.json"
MANIFEST = HERE / "model" / "checkpoint.json"
CHECKPOINT = HERE / "model" / "checkpoint.pt"
CALIBRATION = HERE / "corpus" / "calibration.bin"
CORPUS_MANIFEST = HERE / "corpus" / "eval_corpus.json"

BITS = (2, 3, 4, 5, 6, 8)
CODEBOOKS = ("symmetric", "affine")
CLIPS = (1.0, 0.999, 0.99, 0.95, 0.9)
PARAMS = {"symmetric": 1, "affine": 2}

MICRO_BATCH_SEQUENCES = 4
CALIBRATION_WINDOWS = 4
ROUND_DIGITS = 9

EMBEDDING_ROLES = ("token-embedding", "output-projection")
ATTENTION_ROLES = ("attention-query", "attention-key", "attention-value", "attention-output")
MLP_ROLES = ("mlp-input", "mlp-output")
ROLE_GROUPS = (("embedding", EMBEDDING_ROLES), ("attention", ATTENTION_ROLES), ("mlp", MLP_ROLES))


def load(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nanogpt():
    spec = importlib.util.spec_from_file_location("oer21_nanogpt", HERE / "model" / "nanogpt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


# ---------------------------------------------------------------------------
# The quantizers. Same definitions the grader uses, on a different split.
# ---------------------------------------------------------------------------


def quantize_symmetric(values: torch.Tensor, bits: int, clip: float) -> torch.Tensor:
    magnitudes = values.abs().flatten().float()
    count = magnitudes.numel()
    index = max(0, min(count - 1, int(math.ceil(clip * count)) - 1))
    amax = torch.kthvalue(magnitudes, index + 1).values
    span = (1 << (bits - 1)) - 1
    if span <= 0 or float(amax) <= 0.0:
        return torch.zeros_like(values)
    step = amax / span
    levels = torch.clamp(torch.floor(values.float() / step + 0.5), -span, span)
    return (levels * step).to(values.dtype)


def quantize_affine(values: torch.Tensor, bits: int, clip: float) -> torch.Tensor:
    flat = values.flatten().float()
    count = flat.numel()
    keep = max(1, int(math.ceil(clip * count)))
    drop = count - keep
    ordered, _ = torch.sort(flat)
    low = ordered[drop // 2]
    high = ordered[count - 1 - (drop - drop // 2)]
    steps = (1 << bits) - 1
    if float(high) <= float(low) or steps <= 0:
        return torch.full_like(values, float(low))
    step = (high - low) / steps
    clamped = torch.clamp(values.float(), float(low), float(high))
    levels = torch.clamp(torch.floor((clamped - low) / step + 0.5), 0, steps)
    return (low + levels * step).to(values.dtype)


QUANTIZERS = {"symmetric": quantize_symmetric, "affine": quantize_affine}


# ---------------------------------------------------------------------------
# The substrate this tool operates on
# ---------------------------------------------------------------------------


def load_calibration(manifest: dict) -> list:
    """The agent-visible calibration split. Not the graded split."""
    row = manifest["calibration"]
    path = Path(row.get("container_path") or CALIBRATION)
    if not path.is_file():
        path = CALIBRATION
    if not path.is_file():
        raise SystemExit(
            "the calibration split is not staged at " + str(path)
            + ". environment/Dockerfile stages it at build time from the pinned upstream loader."
        )
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    if int(header[0]) != 20240520 or int(header[1]) != 1:
        raise SystemExit("the calibration split is not a modded-nanogpt token shard")
    count = int(header[2])
    with path.open("rb", buffering=0) as handle:
        tokens = torch.empty(count, dtype=torch.uint16)
        handle.seek(256 * 4)
        handle.readinto(tokens.numpy())
    sequence = int(row["sequence_length"])
    usable = min(int(row["token_count"]), (count - 1) // sequence * sequence)
    window = tokens[:usable + 1].to(dtype=torch.int64)
    inputs = window[:usable].reshape(-1, sequence)
    targets = window[1:usable + 1].reshape(-1, sequence)
    per_window = max(1, inputs.size(0) // CALIBRATION_WINDOWS)
    return [
        {"id": index, "inputs": inputs[start:start + per_window], "targets": targets[start:start + per_window]}
        for index, start in enumerate(range(0, inputs.size(0), per_window))
    ]


def load_substrate_state():
    module = nanogpt()
    manifest = load(MANIFEST)
    if not CHECKPOINT.is_file():
        raise SystemExit(
            "the frozen checkpoint is not staged at " + str(CHECKPOINT)
            + ". environment/Dockerfile stages it at build time; the verifier grades from its own copy."
        )
    declaration = load(SUBSTRATE)
    state = module.load_checkpoint(CHECKPOINT, declaration)
    table = module.shard_table(declaration)
    def build():
        model = module.build_model(declaration)
        model.load_state_dict(state, strict=True)
        return model.to(device()).to(torch.bfloat16).eval()

    # Two instances, built once. The reference is never written to and the candidate is
    # overwritten in place by every configuration, so a sweep pays for the forward passes it
    # needs rather than for rebuilding a 162 million parameter module on every step.
    reference = build()
    candidate = build()
    view = lambda model: {row["name"]: dict(model.named_parameters())[row["name"]] for row in table}
    return {
        "module": module,
        "declaration": declaration,
        "manifest": manifest,
        "state": state,
        "table": table,
        "reference": reference,
        "reference_view": view(reference),
        "candidate": candidate,
        "candidate_view": view(candidate),
        "scale_bits": int(manifest["scale_bits"]),
        "default_bits": int(manifest["default_bits"]),
        "bit_budget_total": int(manifest["bit_budget_total"]),
        "windows": load_calibration(load(CORPUS_MANIFEST)),
    }


# ---------------------------------------------------------------------------
# Schemes, accounting and readings
# ---------------------------------------------------------------------------


def uniform_scheme(table: list, bits: int, codebook: str, clip: float) -> dict:
    return {
        "scheme_version": 1,
        "tensors": [
            {"name": row["name"],
             "groups": [{"size": int(row["numel"]), "bits": bits, "codebook": codebook, "clip": clip}]}
            for row in table
        ],
        "graded_readout_filter": "none",
        "halt_after": None,
    }


def role_scheme(table: list, assignment: dict) -> dict:
    """One (bits, codebook, clip, groups) per shard ROLE. Wider than this tool's own space."""
    tensors = []
    for row in table:
        choice = assignment[row["role"]]
        groups = max(1, int(choice.get("groups", 1)))
        numel = int(row["numel"])
        if numel % groups:
            groups = 1
        size = numel // groups
        tensors.append(
            {
                "name": row["name"],
                "groups": [
                    {"size": size, "bits": int(choice["bits"]), "codebook": choice["codebook"],
                     "clip": float(choice.get("clip", 1.0))}
                    for _ in range(groups)
                ],
            }
        )
    return {"scheme_version": 1, "tensors": tensors, "graded_readout_filter": "none", "halt_after": None}


def cost_of(table: list, scheme: dict, scale_bits: int) -> int:
    by_name = {entry["name"]: entry for entry in scheme["tensors"]}
    total = 0
    for row in table:
        for group in by_name[row["name"]]["groups"]:
            total += int(group["size"]) * int(group["bits"])
            parameters = (1 << int(group["bits"])) if group["codebook"] == "kmeans" else PARAMS[group["codebook"]]
            total += parameters * scale_bits
    return total


@torch.no_grad()
def apply_scheme(context: dict, scheme: dict) -> None:
    by_name = {entry["name"]: entry for entry in scheme["tensors"]}
    source = context["reference_view"]
    target = context["candidate_view"]
    for row in context["table"]:
        flat = source[row["name"]].detach().flatten().float()
        out = torch.empty_like(flat)
        cursor = 0
        for group in by_name[row["name"]]["groups"]:
            size = int(group["size"])
            out[cursor:cursor + size] = QUANTIZERS[group["codebook"]](
                flat[cursor:cursor + size], int(group["bits"]), float(group["clip"])
            )
            cursor += size
        target[row["name"]].copy_(out.reshape(source[row["name"]].shape))


@torch.no_grad()
def readings(context: dict) -> list:
    candidate = context["candidate"]
    reference = context["reference"]
    rows = []
    for window in context["windows"]:
        total = window["inputs"].numel()
        entropy = 0.0
        cross = 0.0
        for start in range(0, window["inputs"].size(0), MICRO_BATCH_SEQUENCES):
            batch = window["inputs"][start:start + MICRO_BATCH_SEQUENCES].to(device())
            reference_log = torch.log_softmax(reference.logits(batch), dim=-1)
            candidate_log = torch.log_softmax(candidate.logits(batch), dim=-1)
            probability = reference_log.exp()
            entropy += float(-(probability * reference_log).sum(dim=-1).sum())
            cross += float(-(probability * candidate_log).sum(dim=-1).sum())
            del reference_log, candidate_log, probability
        base = _round(math.exp(entropy / total))
        value = _round(math.exp(cross / total))
        rows.append({"window": window["id"], "reference": base, "quantized": value,
                     "degradation": _round((value - base) / base * 100.0)})
    return rows


def evaluate(context: dict, scheme: dict) -> dict:
    apply_scheme(context, scheme)
    rows = readings(context)
    values = [row["degradation"] for row in rows]
    return {
        "per_window_degradation": values,
        "worst_window_degradation": max(values),
        "window_spread": _round(max(values) - min(values)),
        "allocated_bits": cost_of(context["table"], scheme, context["scale_bits"]),
        "bit_budget_total": context["bit_budget_total"],
    }


# ---------------------------------------------------------------------------
# The wider allocation. Outside this tool's own option set, and it says so.
# ---------------------------------------------------------------------------


def widen(passes: int = 2, context: dict | None = None) -> dict:
    """Measure per-role sensitivity, then spend the budget where it repays most.

    This function is NOT part of the option set the three subcommands above expose, and it
    is not what `sweep` enumerates. It is here because solution/reference.py calls it, and
    it is written out in full so a reader can see that the widening is arithmetic over
    measurements rather than a table somebody remembered.

    Step one measures a degradation curve per shard role by moving one role's width off the
    default while every other role stays at the default. Step two predicts the combined
    reading of a role-width triple as the default reading plus the sum of the per-role
    increments, keeps the feasible triples in predicted order, and MEASURES the leading few
    rather than trusting the prediction. Step three tries a small fixed set of codebook and
    clip variants on the winner. Fixed order, fixed counts, no random restarts.
    """
    if context is None:
        context = load_substrate_state()
    table = context["table"]
    budget = context["bit_budget_total"]
    scale_bits = context["scale_bits"]
    default_bits = context["default_bits"]

    def assignment(embedding, attention, mlp, codebook="symmetric", clip=1.0, groups=1):
        out = {}
        for name in EMBEDDING_ROLES:
            out[name] = {"bits": embedding, "codebook": codebook, "clip": clip, "groups": groups}
        for name in ATTENTION_ROLES:
            out[name] = {"bits": attention, "codebook": codebook, "clip": clip, "groups": groups}
        for name in MLP_ROLES:
            out[name] = {"bits": mlp, "codebook": codebook, "clip": clip, "groups": groups}
        return out

    baseline = evaluate(context, uniform_scheme(table, default_bits, "symmetric", 1.0))
    curves = {name: {default_bits: baseline["worst_window_degradation"]} for name, _ in ROLE_GROUPS}
    for name, _ in ROLE_GROUPS:
        for bits in BITS:
            if bits == default_bits:
                continue
            widths = {"embedding": default_bits, "attention": default_bits, "mlp": default_bits}
            widths[name] = bits
            scheme = role_scheme(table, assignment(widths["embedding"], widths["attention"], widths["mlp"]))
            curves[name][bits] = evaluate(context, scheme)["worst_window_degradation"]

    reference = baseline["worst_window_degradation"]
    triples = []
    for embedding in BITS:
        for attention in BITS:
            for mlp in BITS:
                scheme = role_scheme(table, assignment(embedding, attention, mlp))
                if cost_of(table, scheme, scale_bits) > budget:
                    continue
                predicted = reference + sum(
                    curves[name][bits] - reference
                    for name, bits in (("embedding", embedding), ("attention", attention), ("mlp", mlp))
                )
                triples.append((predicted, embedding, attention, mlp))
    triples.sort()

    best = {"metric": baseline["worst_window_degradation"],
            "scheme": uniform_scheme(table, default_bits, "symmetric", 1.0)}
    for _, embedding, attention, mlp in triples[:max(1, 4 * int(passes))]:
        scheme = role_scheme(table, assignment(embedding, attention, mlp))
        metric = evaluate(context, scheme)["worst_window_degradation"]
        if metric < best["metric"]:
            best = {"metric": metric, "scheme": scheme, "widths": (embedding, attention, mlp)}

    widths = best.get("widths")
    if widths is not None:
        for codebook, clip, groups in (("symmetric", 0.999, 1), ("symmetric", 0.99, 1),
                                       ("affine", 1.0, 1), ("symmetric", 1.0, 2)):
            scheme = role_scheme(table, assignment(*widths, codebook=codebook, clip=clip, groups=groups))
            if cost_of(table, scheme, scale_bits) > budget:
                continue
            metric = evaluate(context, scheme)["worst_window_degradation"]
            if metric < best["metric"]:
                best = {"metric": metric, "scheme": scheme, "widths": widths}
    return best["scheme"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="uniform post-training quantization toolkit")
    parser.add_argument("mode", choices=("measure", "sweep", "emit", "shards"))
    parser.add_argument("--bits", type=int, default=None)
    parser.add_argument("--codebook", choices=CODEBOOKS, default="symmetric")
    parser.add_argument("--clip", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "shards":
        manifest = load(MANIFEST)
        print(json.dumps({"shards": manifest["shards"],
                          "quantizable_parameters": manifest["quantizable_parameters"],
                          "bit_budget_total": manifest["bit_budget_total"],
                          "scale_bits": manifest["scale_bits"],
                          "shard_table": manifest["shard_table"]}, indent=2, sort_keys=True))
        return 0

    context = load_substrate_state()
    table = context["table"]
    bits = args.bits if args.bits is not None else context["default_bits"]

    if args.mode == "measure":
        scheme = uniform_scheme(table, bits, args.codebook, args.clip)
        print(json.dumps(evaluate(context, scheme), indent=2, sort_keys=True))
        return 0

    if args.mode == "sweep":
        rows = []
        for width in BITS:
            for codebook in CODEBOOKS:
                for clip in CLIPS:
                    scheme = uniform_scheme(table, width, codebook, clip)
                    allocated = cost_of(table, scheme, context["scale_bits"])
                    if allocated > context["bit_budget_total"]:
                        continue
                    outcome = evaluate(context, scheme)
                    rows.append({"bits": width, "codebook": codebook, "clip": clip,
                                 "worst_window_degradation": outcome["worst_window_degradation"],
                                 "allocated_bits": allocated})
        rows.sort(key=lambda row: row["worst_window_degradation"])
        print(json.dumps({"configurations": len(rows), "ranked": rows}, indent=2, sort_keys=True))
        return 0

    scheme = uniform_scheme(table, bits, args.codebook, args.clip)
    allocated = cost_of(table, scheme, context["scale_bits"])
    if allocated > context["bit_budget_total"]:
        print("refusing to emit: that configuration costs " + str(allocated)
              + " bits against a budget of " + str(context["bit_budget_total"]))
        return 1
    out = Path(os.environ.get("OUT_DIR") or os.getcwd())
    out.mkdir(parents=True, exist_ok=True)
    target = out / "scheme.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(scheme, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("scheme written to " + str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

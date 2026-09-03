#!/usr/bin/env python3
"""The post-training quantization toolkit shipped with this task.

Usage:
    python3 quantize.py measure                 evaluate the shipped configuration
    python3 quantize.py sweep                   evaluate every configuration this tool reaches
    python3 quantize.py emit --bits 4 --codebook symmetric --clip 1.0
                                                write scheme.json under OUT_DIR

What this tool does, exactly: it sets ONE bit width for every tensor, with ONE scale
group per tensor, a symmetric or affine codebook, and one clip percentile. That is the
whole of its option set, and `sweep` enumerates all of it.

What this tool does NOT do: it cannot set a different width per tensor, it cannot place
more than one scale group inside a tensor, and it cannot emit a non-uniform codebook.
Those axes are not broken here and they are not hidden; they are simply outside what
this script was written to reach. The scheme format the grader accepts is published in
full at environment/scheme_schema.json, and any scheme.json conforming to it is graded,
whether this tool produced it or not.

The perplexity numbers printed here are computed the same way the grader computes them,
so you can measure locally. They are NOT the graded numbers. The grader recomputes the
graded quantity itself from its own copy of the frozen model, and nothing this script
prints reaches it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model" / "model.json"
CORPUS = HERE / "corpus" / "eval_corpus.json"
REFERENCE = HERE / "model" / "reference.json"

BITS = (2, 3, 4, 5, 6, 8)
CODEBOOKS = ("symmetric", "affine")
CLIPS = (1.0, 0.999, 0.99, 0.95, 0.9)
PARAMS = {"symmetric": 1, "affine": 2}
FLOOR = 1e-300


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def quantize_symmetric(values, bits, clip):
    magnitudes = sorted(abs(value) for value in values)
    index = max(0, min(len(magnitudes) - 1, int(math.ceil(clip * len(magnitudes))) - 1))
    amax = magnitudes[index]
    span = (1 << (bits - 1)) - 1
    if span <= 0 or amax <= 0.0:
        return [0.0 for _ in values]
    step = amax / span
    out = []
    for value in values:
        level = max(-span, min(span, int(math.floor(value / step + 0.5))))
        out.append(level * step)
    return out


def quantize_affine(values, bits, clip):
    ordered = sorted(values)
    count = len(ordered)
    keep = max(1, int(math.ceil(clip * count)))
    drop = count - keep
    low = ordered[drop // 2]
    high = ordered[count - 1 - (drop - drop // 2)]
    levels = (1 << bits) - 1
    if high <= low or levels <= 0:
        return [low for _ in values]
    step = (high - low) / levels
    out = []
    for value in values:
        clamped = max(low, min(high, value))
        level = max(0, min(levels, int(math.floor((clamped - low) / step + 0.5))))
        out.append(low + level * step)
    return out


QUANTIZERS = {"symmetric": quantize_symmetric, "affine": quantize_affine}


def logits_for(tensors, model, context):
    size = int(model["tensor_size"])
    scale = float(model["logit_scale"])
    stride_context = int(model["index"]["stride_context"])
    stride_tensor = int(model["index"]["stride_tensor"])
    gains = model["gains"]
    row = []
    for word in range(int(model["vocab"])):
        accumulator = 0.0
        for index in range(len(tensors)):
            slot = (context * stride_context + word + index * stride_tensor) % size
            accumulator += gains[index] * tensors[index][slot]
        row.append(accumulator * scale)
    return row


def softmax(logits):
    peak = max(logits)
    exponentials = [math.exp(value - peak) for value in logits]
    partition = sum(exponentials)
    return [value / partition for value in exponentials]


def reference_profile(model, tokens):
    raw = [tensor["values"] for tensor in model["tensors"]]
    profile = []
    entropy = 0.0
    for context in tokens:
        row = softmax(logits_for(raw, model, context))
        profile.append(row)
        entropy += -sum(value * math.log(max(value, FLOOR)) for value in row)
    return profile, math.exp(entropy / len(tokens))


def cross_perplexity(tensors, model, tokens, profile):
    total = 0.0
    for context, row in zip(tokens, profile):
        other = softmax(logits_for(tensors, model, context))
        total += -sum(value * math.log(max(item, FLOOR)) for value, item in zip(row, other))
    return math.exp(total / len(tokens))


def uniform_scheme(model, bits, codebook, clip):
    size = int(model["tensor_size"])
    return {
        "scheme_version": 1,
        "tensors": [
            {"name": tensor["name"],
             "groups": [{"size": size, "bits": bits, "codebook": codebook, "clip": clip}]}
            for tensor in model["tensors"]
        ],
        "graded_readout_filter": "none",
        "halt_after": None,
    }


def cost_of(model, bits, codebook):
    size = int(model["tensor_size"])
    scale_bits = int(model["scale_bits"])
    return int(model["tensor_count"]) * (size * bits + PARAMS[codebook] * scale_bits)


def evaluate(model, corpus, bits, codebook, clip):
    tensors = [QUANTIZERS[codebook](tensor["values"], bits, clip) for tensor in model["tensors"]]
    rows = []
    for shard in corpus["shards"]:
        profile, base_raw = reference_profile(model, shard["tokens"])
        # Both perplexities are rounded to nine digits BEFORE the ratio is taken,
        # because the grader rounds in that order too. Rounding the ratio only would
        # leave this tool and the grader disagreeing in the ninth digit.
        base = round(base_raw, 9)
        value = round(cross_perplexity(tensors, model, shard["tokens"], profile), 9)
        rows.append(round((value - base) / base * 100.0, 9))
    return {
        "per_shard_degradation": rows,
        "worst_shard_degradation": max(rows),
        "allocated_bits": cost_of(model, bits, codebook),
        "bit_budget_total": int(model["bit_budget_total"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="uniform post-training quantization toolkit")
    parser.add_argument("mode", choices=("measure", "sweep", "emit"))
    parser.add_argument("--bits", type=int, default=None)
    parser.add_argument("--codebook", choices=CODEBOOKS, default="symmetric")
    parser.add_argument("--clip", type=float, default=1.0)
    args = parser.parse_args()

    model = load(MODEL)
    corpus = load(CORPUS)
    bits = args.bits if args.bits is not None else int(model["default_bits"])

    if args.mode == "measure":
        outcome = evaluate(model, corpus, bits, args.codebook, args.clip)
        print(json.dumps(outcome, indent=2, sort_keys=True))
        return 0

    if args.mode == "sweep":
        budget = int(model["bit_budget_total"])
        rows = []
        for width in BITS:
            for codebook in CODEBOOKS:
                if cost_of(model, width, codebook) > budget:
                    continue
                for clip in CLIPS:
                    outcome = evaluate(model, corpus, width, codebook, clip)
                    rows.append({
                        "bits": width,
                        "codebook": codebook,
                        "clip": clip,
                        "worst_shard_degradation": outcome["worst_shard_degradation"],
                        "allocated_bits": outcome["allocated_bits"],
                    })
        rows.sort(key=lambda row: row["worst_shard_degradation"])
        print(json.dumps({"configurations": len(rows), "ranked": rows}, indent=2, sort_keys=True))
        return 0

    if cost_of(model, bits, args.codebook) > int(model["bit_budget_total"]):
        print("refusing to emit: that configuration costs "
              + str(cost_of(model, bits, args.codebook))
              + " bits against a budget of " + str(model["bit_budget_total"]))
        return 1
    out = Path(os.environ.get("OUT_DIR") or os.getcwd())
    out.mkdir(parents=True, exist_ok=True)
    target = out / "scheme.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(uniform_scheme(model, bits, args.codebook, args.clip), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("scheme written to " + str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

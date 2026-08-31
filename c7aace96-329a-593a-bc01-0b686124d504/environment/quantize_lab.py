#!/usr/bin/env python3
"""Advisory local lab. Reproduces the harness error model so you can search offline.

This file is agent-visible and read-only. It computes the same three things the
verifier computes: the bit accounting over the model's own tensor list, the
quantization error under a scheme, and the per-shard perplexity degradation.

It is ADVISORY. It agrees with the graded evaluation only because both read the
same frozen files, and it deliberately does not decide anything: it prints the
per-shard separations and the band they imply, and leaves the verdict alone. The
graded verdict is computed by the verifier over its own scheduled order, and a
number this file prints is never on the graded path.

Usage:
    python3 quantize_lab.py --allocation allocation.json
    python3 quantize_lab.py --allocation allocation.json --environment .
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CONTROL_BITS = 4
CONTROL_SCHEME = "rtn"
BITS_PER_PARAM_BUDGET = 4
SEPARATION_MARGIN = 0.05
SCHEME_FACTOR = {"rtn": 1.0, "affine-per-channel": 0.72, "error-feedback": 0.55}
ALLOWED_BITS = (2, 3, 4, 5, 6, 8)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def widths(model, bits):
    return {str(row["name"]): int(bits.get(str(row["name"]), CONTROL_BITS)) for row in model["tensors"]}


def account(model, bits):
    total = sum(int(row["params"]) for row in model["tensors"])
    resolved = widths(model, bits)
    spent = sum(int(row["params"]) * resolved[str(row["name"])] for row in model["tensors"])
    return {
        "total_params": total,
        "allocated_bits": spent,
        "budget_bits": BITS_PER_PARAM_BUDGET * total,
        "mean_bits_per_param": spent / float(total),
        "within_budget": spent <= BITS_PER_PARAM_BUDGET * total,
    }


def error(model, bits, scheme):
    total = float(sum(int(row["params"]) for row in model["tensors"]))
    resolved = widths(model, bits)
    scale = SCHEME_FACTOR[scheme]
    return sum(
        (float(row["params"]) / total)
        * float(row["sensitivity"])
        * float(row["outlier_factor"])
        * scale
        * (4.0 ** (-(resolved[str(row["name"])] - 1)))
        for row in model["tensors"]
    )


def spread(model, bits):
    total = float(sum(int(row["params"]) for row in model["tensors"]))
    resolved = widths(model, bits)
    mean = sum(float(row["params"]) * resolved[str(row["name"])] for row in model["tensors"]) / total
    return (
        sum(
            (float(row["params"]) / total) * abs(resolved[str(row["name"])] - mean)
            for row in model["tensors"]
        )
        / 4.0
    )


def separations(model, corpus, reference, bits, scheme):
    base = {str(row["shard_id"]): float(row["ppl"]) for row in reference["reference_ppl"]}
    scale = float(corpus.get("readout_jitter_scale", 0.0))
    control_bits = {str(row["name"]): CONTROL_BITS for row in model["tensors"]}
    ea, ec = error(model, bits, scheme), error(model, control_bits, CONTROL_SCHEME)
    sa, sc = spread(model, bits) * scale, spread(model, control_bits) * scale
    rows = []
    for shard in corpus["shards"]:
        sid = str(shard["shard_id"])
        gain, jitter = float(shard["shard_gain"]), float(shard.get("readout_jitter", 0.0))
        ref = base[sid]
        agent = ref * math.exp(ea * gain + sa * jitter) - ref
        control = ref * math.exp(ec * gain + sc * jitter) - ref
        rows.append({"shard_id": sid, "delta": agent, "delta_control": control, "separation": control - agent})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OER-20 advisory quantization lab")
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--environment", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args(argv)

    env = Path(args.environment)
    model = load(env / "model_stats.json")
    corpus = load(env / "eval_corpus.json")
    reference = load(env / "reference_unquantized.json")
    submitted = load(Path(args.allocation))

    scheme = str(submitted.get("scheme", ""))
    if scheme not in SCHEME_FACTOR:
        print("scheme outside the closed set: " + repr(scheme))
        return 2
    bits = submitted.get("bits") or {}
    bad = sorted(name for name, width in bits.items() if int(width) not in ALLOWED_BITS)
    if bad:
        print("bit widths outside the closed set on: " + ", ".join(bad))
        return 2

    book = account(model, bits)
    rows = separations(model, corpus, reference, bits, scheme)
    values = [row["separation"] for row in rows]
    mean = sum(values) / len(values)
    half = (max(values) - min(values)) / 2.0

    print(json.dumps({"accounting": book, "points": rows}, indent=2, sort_keys=True))
    print("separation mean       : " + repr(mean))
    print("noise half width      : " + repr(half))
    print("lower edge of the band: " + repr(mean - half))
    print("bound margin          : " + repr(SEPARATION_MARGIN))
    print("this lab states no verdict; the verifier computes it over its own scheduled order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

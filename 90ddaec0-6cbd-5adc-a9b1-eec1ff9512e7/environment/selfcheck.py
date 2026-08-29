#!/usr/bin/env python3
"""Agent visible self check for the BIA-GSN-1 throughput task.

This runs the same shape of comparison the grader runs, against the agent visible copy
of the frozen operator. It is a convenience, not the grader. The grader uses its own
private copy of the operator, its own fixtures, and its own timing protocol, so a
result here is an indication and never a score.

Usage:
    python3 selfcheck.py [path/to/impl.py] [--profile graded|smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reference_impl  # noqa: E402


def load_impl(path):
    spec = importlib.util.spec_from_file_location("candidate_impl", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("impl", nargs="?", default=str(HERE / "submission.py"))
    parser.add_argument("--profile", default="graded", choices=["graded", "smoke"])
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    import torch

    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
    profile = spec["profiles"][args.profile]
    device = profile["device"]
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("no cuda device visible, falling back to the smoke profile")
        profile = spec["profiles"]["smoke"]
        device = profile["device"]
    chunk = int(profile["chunk_rows"])
    candidate = load_impl(args.impl)

    all_ok = True
    for fixture in profile["fixtures"]:
        x, g, b = reference_impl.make_inputs(
            fixture["kind"], int(fixture["rows"]), int(fixture["cols"]), int(fixture["seed"]), device
        )
        ry, rr = reference_impl.reference_forward(x, g, b, chunk)
        sy, sr = candidate.forward(x, g, b, chunk)
        ok_y, why_y = reference_impl.bitwise_equal(sy, ry)
        ok_r, why_r = reference_impl.bitwise_equal(sr, rr)
        if not (ok_y and ok_r):
            all_ok = False
            print(
                "fixture %-16s DIVERGES y=%s r=%s first_word=%s"
                % (fixture["name"], why_y, why_r, reference_impl.first_divergence(sy, ry) if ok_y is False else "na")
            )
        else:
            print("fixture %-16s identical" % fixture["name"])
    print("bitwise gate  : %s" % ("pass" if all_ok else "FAIL"))
    if not all_ok:
        print("a divergent implementation scores zero regardless of how fast it is")
        return 1

    x, g, b = reference_impl.make_inputs("normal", int(profile["rows"]), int(profile["cols"]), 424242, device)
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)

    def bench(fn):
        for _ in range(3):
            fn()
        out = []
        for _ in range(args.trials):
            sync()
            t0 = time.perf_counter_ns()
            fn()
            sync()
            out.append(time.perf_counter_ns() - t0)
        return statistics.median(out)

    ref_ns = bench(lambda: reference_impl.reference_forward(x, g, b, chunk))
    sub_ns = bench(lambda: candidate.forward(x, g, b, chunk))
    speedup = ref_ns / sub_ns if sub_ns else 0.0
    baseline = float(spec["reward"]["baseline_metric"])
    target = float(spec["reward"]["target_metric"])
    raw = (speedup - baseline) / (target - baseline)
    print("reference     : %.3f ms" % (ref_ns / 1e6))
    print("candidate     : %.3f ms" % (sub_ns / 1e6))
    print("speedup       : %.4f" % speedup)
    print("indicative raw: %.4f (clamped %.4f)" % (raw, min(max(raw, 0.0), 1.0)))
    print("this is an indication under a short local protocol, not the graded number")
    return 0


if __name__ == "__main__":
    sys.exit(main())

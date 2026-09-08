#!/usr/bin/env python3
"""The agent's own measuring instrument.

    python3 bench_local.py --plan my_plan.json
    python3 bench_local.py --sweep

`--plan` runs the equivalence check against the reference plan and then times the
plan against the shipped default, interleaved and taking the minimum, exactly the
way the verifier does. `--sweep` does the same for a grid of plans.

Timing happens on data/devset_slice.bin. The verifier times on its own held-out
slice, which is not in this image; that changes which tokens are multiplied and
not what the multiply costs.

READ THE EQUIVALENCE LINE. A plan that is not equivalent to the reference plan is
refused at 0.0 by the verifier no matter how fast it is, and this tool reports the
same check with the same tolerances.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness, kernel_plan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tokens", default=str(HERE / "data" / "devset_slice.bin"))
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device is visible; this harness needs the accelerator")
        return 2

    spec = harness.load_spec()
    harness.configure_backends()
    tokens = harness.to_device_tokens(harness.load_shard(args.tokens), "cuda")
    bench = harness.Bench(spec, "cuda")
    control = kernel_plan.as_execution(kernel_plan.load(HERE / "default_plan.json"))

    plans = {"control": control}
    if args.sweep:
        grid = itertools.product(kernel_plan.OPTIONS["attention"],
                                 (1, 4, 8, 16),
                                 kernel_plan.OPTIONS["logit_dtype"])
        for att, chunks, logit in grid:
            plans[f"{att}_c{chunks}_{logit}"] = {
                "attention": att, "loss_chunks": chunks, "logit_dtype": logit,
                "rotary_dtype": "fp32", "qk_norm_dtype": "fp32"}
    elif args.plan:
        try:
            plans["agent"] = kernel_plan.as_execution(kernel_plan.load(args.plan))
        except kernel_plan.Refusal as exc:
            print(f"REFUSED  {exc.reason}: {exc.detail}")
            return 2
    else:
        ap.error("give --plan or --sweep")

    print("=== equivalence against the reference plan ===", flush=True)
    eq = {}
    legal = {}
    for label, plan in plans.items():
        r = bench.equivalence(plan, tokens)
        eq[label] = r
        mark = "OK " if r["equivalent"] else "DIVERGED"
        print(f"  {label:22s} loss_delta={r['worst_loss_abs_delta']:.6f} "
              f"(tol {r['loss_tolerance']})  grad_rel={r['worst_grad_relative_delta']:.6f} "
              f"(tol {r['grad_relative_tolerance']})  {mark}", flush=True)
        if r["equivalent"]:
            legal[label] = plan
    if "control" not in legal:
        print("the shipped control failed its own equivalence check; the substrate is broken")
        return 4

    print("\n=== interleaved latency (minimum over rounds) ===", flush=True)
    lat = harness.interleaved_latencies(bench, legal, tokens)
    base = lat["control"]["min_ms"]
    print("\n==== RESULT ====")
    for label, v in sorted(lat.items(), key=lambda kv: kv[1]["min_ms"]):
        print(f"  {label:22s} min={v['min_ms']:8.3f}ms  median={v['median_ms']:8.3f}ms  "
              f"speedup_over_control={base / v['min_ms']:6.3f}x")
    refused = [k for k in plans if k not in legal]
    if refused:
        print("\n  NOT TIMED, refused by the equivalence gate: " + ", ".join(refused))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

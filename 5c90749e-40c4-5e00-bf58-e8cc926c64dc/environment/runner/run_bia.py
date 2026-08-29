#!/usr/bin/env python3
"""The only writer of the harness telemetry record.

Logs written by any other means are not evidence and will not reconcile against
the record this runner appends. Invoke it, read its output, and submit what it
produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENVDIR = os.path.dirname(HERE)
BUNDLE = os.environ.get("BIA_BUNDLE", os.path.dirname(ENVDIR))
sys.path.insert(0, os.path.join(BUNDLE, "environment"))
sys.path.insert(0, ENVDIR)

import bia_core as core  # noqa: E402


def load_submission(path: str):
    import importlib.util
    if not os.path.isfile(path):
        raise SystemExit(f"no submission at {path}")
    spec = importlib.util.spec_from_file_location("bia_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("build_optimizer", "build_schedule"):
        if not callable(getattr(mod, fn, None)):
            raise SystemExit(f"submission does not export {fn}")
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default=os.environ.get("BIA_SCALE", "full"))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--device", default=os.environ.get("BIA_DEVICE", "auto"))
    ap.add_argument("--mode", default=os.environ.get("BIA_MODE", "full"),
                    help="'full' is the graded mode. 'probe' marks an exploratory run.")
    ap.add_argument("--submission", default=os.environ.get(
        "BIA_SUBMISSION", os.path.join(os.environ.get("BIA_WORKSPACE", "/workspace"), "submission")))
    ap.add_argument("--telemetry-dir", default=os.environ.get("BIA_TELEMETRY_DIR", "/telemetry"))
    ap.add_argument("--cache-dir", default=os.environ.get("BIA_CACHE", "/tmp/bia_cache"))
    args = ap.parse_args()

    cfg = core.resolve_scale(args.scale)
    device = core.resolve_device(args.device)
    sub_dir = os.path.abspath(args.submission)
    opt_path = os.path.join(sub_dir, "optimizer.py")
    digest = hashlib.sha256(open(opt_path, "rb").read()).hexdigest()
    mod = load_submission(opt_path)

    os.makedirs(args.telemetry_dir, exist_ok=True)
    rec_path = os.path.join(args.telemetry_dir, "run_record.jsonl")
    log_dir = os.path.join(sub_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    sink = open(rec_path, "a")

    def emit(rec):
        sink.write(json.dumps(rec, sort_keys=True) + "\n")
        sink.flush()

    corpus = core.build_corpus(cfg, args.cache_dir)
    print(json.dumps({
        "scale": cfg["scale"], "device": device.type, "mode": args.mode,
        "tokens_per_step": cfg["tokens_per_step"], "total_steps": cfg["total_steps"],
        **corpus.anchors(),
    }, sort_keys=True))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    for seed in seeds:
        curve = core.train_one_seed(
            cfg, seed, corpus, mod.build_optimizer, mod.build_schedule,
            device, digest, args.mode, emit)
        prefix = "full" if args.mode == "full" else "probe"
        with open(os.path.join(log_dir, f"{prefix}_seed{seed}.log"), "w") as f:
            f.write(f"# bia s01 scale={cfg['scale']} seed={seed} mode={args.mode}\n")
            f.write(f"# target_loss {corpus.target_loss:.6f}\n")
            for step, vl in curve:
                f.write(f"step {step} val_loss {vl:.6f}\n")
        last = curve[-1][1] if curve else float("nan")
        print(f"seed {seed} final val_loss {last:.6f}")

    sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derives the submitted order once, in its own interpreter, and reports its digest.

The determinism checker runs this twice under different process identities. It
is verifier-owned rather than the agent-visible runner, so the derivation the
verifier compares against cannot be redirected by editing a copy of the runner
in the agent workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--order-module", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    runner_dir = os.path.join(args.bundle, "environment", "runner")
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    import numpy as np
    import corpus as corpus_mod
    import order as order_mod
    import profiles as profiles_mod

    cfg = profiles_mod.get_profile(args.profile)
    key = "%s_%s_%s_%s_%s_%s" % (
        cfg["profile"], cfg["corpus_seed"], cfg["n_train_sequences"],
        cfg["seq_len"], cfg["vocab_size"], cfg["n_domains"])
    cpath = os.path.join(args.cache_dir, key + ".npz")
    if os.path.exists(cpath):
        with np.load(cpath) as z:
            train, domains = z["train"], z["domains"]
    else:
        train, domains, _ = corpus_mod.generate_corpus(cfg)
    feats = corpus_mod.sequence_features(train, domains, cfg)
    meta = order_mod.build_meta(cfg, feats)
    try:
        order = order_mod.call_build_order(args.order_module, meta)
    except BaseException as exc:
        print(json.dumps({"valid": False,
                          "reason": "build_order_failed_%s" % type(exc).__name__,
                          "order_digest": None}, sort_keys=True))
        return 1
    ok, reason = order_mod.validate_order(order, cfg)
    print(json.dumps({"valid": bool(ok), "reason": reason,
                      "order_digest": order_mod.order_digest(order)}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

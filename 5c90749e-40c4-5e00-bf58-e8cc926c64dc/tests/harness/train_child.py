#!/usr/bin/env python3
"""Child interpreter. The only process into which the submission is loaded.

It receives its whole request on stdin, writes its records to a file
descriptor the parent opened, and writes checkpoints into a directory the
parent named. It computes no reward and reports no score. Every number it
sends that the reward depends on is recomputed by the parent from the
checkpoints, so a child that lies about its own loss changes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-fd", type=int, required=True)
    ap.add_argument("--blob-fd", type=int, required=True)
    args = ap.parse_args()
    channel = os.fdopen(args.channel_fd, "w")
    blob = os.fdopen(args.blob_fd, "wb")

    def send(obj):
        channel.write(json.dumps(obj, sort_keys=True) + "\n")
        channel.flush()

    try:
        request = json.loads(sys.stdin.read())
        bundle = request["bundle"]
        sys.path.insert(0, os.path.join(bundle, "environment"))
        import torch
        import bia_core as core

        cfg = core.resolve_scale(request["scale"])
        device = core.resolve_device(request["device"])
        seed = int(request["seed"])
        sub_path = request["submission"]

        with open(sub_path, "rb") as fh:
            digest_at_load = hashlib.sha256(fh.read()).hexdigest()

        import importlib.util
        spec = importlib.util.spec_from_file_location("bia_graded_submission", sub_path)
        module = importlib.util.module_from_spec(spec)
        guard = core.WriteGuard(allow=core.guard_allowlist())
        with guard:
            spec.loader.exec_module(module)
        for name in ("build_optimizer", "build_schedule"):
            if not callable(getattr(module, name, None)):
                send({"kind": "fault", "code": "submission-missing-export", "export": name})
                return 1

        with open(sub_path, "rb") as fh:
            digest_after_load = hashlib.sha256(fh.read()).hexdigest()

        corpus = core.build_corpus(cfg, request["cache_dir"])

        import io

        def save_checkpoint(step, model):
            """Parameters leave on the parent's binary channel, never onto disk.

            Writing them to a file would leave the earlier checkpoints of the run
            reachable by the submission for the rest of the run, and copying the
            final parameters over the step-ten checkpoint reads as reaching the
            target on step ten. Streaming removes the artifact rather than
            guarding it.
            """
            state = {k: v.detach().to("cpu", torch.float32).clone()
                     for k, v in model.state_dict().items()}
            buf = io.BytesIO()
            torch.save(state, buf)
            payload = buf.getvalue()
            send({"kind": "checkpoint", "seed": seed, "step": step,
                  "blob_bytes": len(payload),
                  "guard_observed_writes": sorted(set(guard.seen))})
            blob.write(payload)
            blob.flush()

        def emit(rec):
            rec = dict(rec)
            rec["submission_digest_at_load"] = digest_at_load
            rec["submission_digest_after_load"] = digest_after_load
            rec["submission_module_file"] = str(getattr(module, "__file__", ""))
            rec["guard_observed_writes"] = sorted(set(guard.seen))
            send(rec)

        core.train_one_seed(
            cfg, seed, corpus, module.build_optimizer, module.build_schedule,
            device, digest_at_load, "verifier", emit,
            on_eval=save_checkpoint)
        blob.close()
        channel.close()
        return 0
    except BaseException as exc:
        try:
            send({"kind": "fault", "code": "child-exception",
                  "type": type(exc).__name__, "message": str(exc)[:400],
                  "traceback": traceback.format_exc()[-1200:]})
            channel.close()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

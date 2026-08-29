#!/usr/bin/env python3
"""The verifier's own measurement. No submitted byte is ever loaded here.

Long lived, and driven one checkpoint at a time by the parent as the training
child produces them. The ordering matters and is the point: the parent measures
and deletes checkpoint N while the training process is still working on step
N+1, so the weights that would make an early checkpoint look good do not exist
yet when that checkpoint is consumed. Measuring the whole run afterwards would
let a submission train normally and then copy its final parameters over its
step-ten checkpoint, which reads as reaching the target on step ten.

Everything the reward rests on is recomputed here from parameters: the
validation loss, the initialization digest, the architecture signature, and the
movement between consecutive evaluation points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-fd", type=int, required=True)
    args = ap.parse_args()
    channel = os.fdopen(args.channel_fd, "w")
    stdin_bytes = sys.stdin.buffer

    def send(obj):
        channel.write(json.dumps(obj, sort_keys=True) + "\n")
        channel.flush()

    try:
        request = json.loads(stdin_bytes.readline())
        sys.path.insert(0, os.path.join(request["bundle"], "environment"))
        import torch
        import bia_core as core

        cfg = core.resolve_scale(request["scale"])
        device = core.resolve_device(request["device"])
        corpus = core.build_corpus(cfg, request["cache_dir"])
        model = core.build_model(cfg)
        send({"kind": "ready", "corpus_digest": corpus.digest,
              "target_loss": round(corpus.target_loss, 6)})

        import io

        previous = None
        while True:
            line = stdin_bytes.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            if req.get("op") == "reset":
                previous = None
                send({"kind": "reset", "seed": req.get("seed")})
                continue
            step, seed = int(req["step"]), int(req["seed"])
            payload = stdin_bytes.read(int(req["blob_bytes"]))
            try:
                state = torch.load(io.BytesIO(payload), map_location="cpu",
                                   weights_only=True)
            except Exception as exc:
                send({"kind": "measured", "step": step, "seed": seed,
                      "val_loss": None, "load_error": "unreadable_%s" % type(exc).__name__})
                continue
            shapes = {k: list(v.shape) for k, v in sorted(state.items())}
            try:
                model.load_state_dict(state)
            except Exception as exc:
                send({"kind": "measured", "step": step, "seed": seed,
                      "val_loss": None, "shapes": shapes,
                      "load_error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
                continue
            flat = torch.cat([state[k].reshape(-1).to(torch.float32)
                              for k in sorted(state)])
            out = {
                "kind": "measured", "step": step, "seed": seed,
                "shapes": shapes, "load_error": None,
                "arch_signature": core.arch_signature(model),
                "delta_l1": None if previous is None else float((flat - previous).abs().sum()),
            }
            if step == 0:
                out["init_digest"] = core.init_digest(model)
                out["expected_init_digest"] = core.expected_init_digest(cfg, seed)
                out["val_loss"] = None
            else:
                model.to(device)
                out["val_loss"] = round(core.evaluate(model, cfg, corpus.val, device), 6)
                model.to("cpu")
            previous = flat
            send(out)
        channel.close()
        return 0
    except BaseException as exc:
        try:
            send({"kind": "fault", "code": "measure-exception",
                  "type": type(exc).__name__, "message": str(exc)[:400],
                  "traceback": traceback.format_exc()[-1200:]})
            channel.close()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

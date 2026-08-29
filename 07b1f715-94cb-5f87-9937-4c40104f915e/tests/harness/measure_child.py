#!/usr/bin/env python3
"""The S02 verifier's own measurement. No submitted byte is ever loaded here.

It is long lived: the parent hands it one checkpoint path per request and it
returns the validation loss it measured from those parameters, together with
the architecture signature it read off the state dictionary. The graded loss
series is therefore produced by a process that has never executed a line the
agent wrote, and a training run that reports a loss it did not achieve moves
nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

PROFILES = {
    "full": {"n_layer": 6, "n_head": 6, "d_model": 384, "seq_len": 512,
             "batch_sequences": 48, "val_batches": 8, "device": "cuda"},
    "smoke": {"n_layer": 2, "n_head": 2, "d_model": 64, "seq_len": 64,
              "batch_sequences": 4, "val_batches": 2, "device": "cpu"},
}


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
        env_root = os.path.join(request["bundle"], "environment")
        for p in (os.path.join(env_root, "runner"), env_root):
            if p not in sys.path:
                sys.path.insert(0, p)
        import torch
        import data as data_mod
        import model as model_mod

        cfg = PROFILES[request["profile"]]
        device = cfg["device"]
        if device == "cuda" and not torch.cuda.is_available():
            send({"kind": "fault", "code": "cuda-unavailable"})
            return 1
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("BIA_CPU_THREADS", "1")))

        _, _, val_path, val_digest = data_mod.resolve_shards(
            request["profile"], os.environ.get("BIA_DATA_DIR", "/data/fineweb"),
            os.path.join(env_root, "fixtures"))
        val_loader = data_mod.FrozenLoader(
            val_path, cfg["batch_sequences"], cfg["seq_len"], val_digest)
        net = model_mod.FrozenGPT(cfg["n_layer"], cfg["n_head"], cfg["d_model"],
                                  cfg["seq_len"]).to(device)
        expected_arch = model_mod.architecture_signature(
            cfg["n_layer"], cfg["n_head"], cfg["d_model"], cfg["seq_len"])
        send({"kind": "ready", "val_shard_digest": val_digest,
              "expected_architecture": expected_arch})

        import io

        use_amp = device == "cuda"
        while True:
            line = stdin_bytes.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            payload = stdin_bytes.read(int(req["blob_bytes"]))
            state = torch.load(io.BytesIO(payload), map_location="cpu",
                               weights_only=True)
            shapes = {k: list(v.shape) for k, v in sorted(state.items())}
            try:
                net.load_state_dict(state)
                load_error = None
            except Exception as exc:
                load_error = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            if load_error is not None:
                send({"kind": "measured", "val_loss": None,
                      "shapes": shapes, "load_error": load_error})
                continue
            net.to(device)
            net.eval()
            val_loader.reset()
            total = 0.0
            with torch.no_grad():
                for _ in range(cfg["val_batches"]):
                    x, y = val_loader.next_batch(device)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            total += float(net(x, y))
                    else:
                        total += float(net(x, y))
            send({"kind": "measured",
                  "val_loss": round(total / cfg["val_batches"], 6),
                  "shapes": shapes, "load_error": None})
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

#!/usr/bin/env python3
"""The S03 verifier's own measurement. No submitted byte is ever loaded here.

Long lived. The parent hands it one checkpoint path per request and it returns
the final validation loss it measured from those parameters. The improvement
statistic, and therefore the reward, is computed only from these numbers, so
editing a recorded loss is no longer a thing that can be done: there is no
recorded loss until this process produces one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-fd", type=int, required=True)
    args = ap.parse_args()
    channel = os.fdopen(args.channel_fd, "w")

    def send(obj):
        channel.write(json.dumps(obj, sort_keys=True) + "\n")
        channel.flush()

    try:
        request = json.loads(sys.stdin.readline())
        runner_dir = os.path.join(request["bundle"], "environment", "runner")
        if runner_dir not in sys.path:
            sys.path.insert(0, runner_dir)
        import numpy as np
        import torch
        import corpus as corpus_mod
        import profiles as profiles_mod

        cfg = profiles_mod.get_profile(request["profile"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cpu") if cfg["device"] == "cpu" else torch.device("cuda")
        if device.type == "cpu" and os.environ.get("BIA_TORCH_THREADS"):
            torch.set_num_threads(max(1, int(os.environ["BIA_TORCH_THREADS"])))

        cache = request["cache_dir"]
        key = "%s_%s_%s_%s_%s_%s" % (
            cfg["profile"], cfg["corpus_seed"], cfg["n_train_sequences"],
            cfg["seq_len"], cfg["vocab_size"], cfg["n_domains"])
        cpath = os.path.join(cache, key + ".npz")
        if os.path.exists(cpath):
            with np.load(cpath) as z:
                val = z["val"]
        else:
            _, _, val = corpus_mod.generate_corpus(cfg)
        val_t = torch.from_numpy(val.astype(np.int64)).to(device)

        from model import TinyGPT
        model = TinyGPT(cfg).to(device)
        send({"kind": "ready", "n_val_sequences": int(val_t.shape[0])})

        use_amp = device.type == "cuda" and cfg["dtype"] == "bfloat16"
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            path = json.loads(line)["ckpt"]
            state = torch.load(path, map_location="cpu", weights_only=True)
            shapes = {k: list(v.shape) for k, v in sorted(state.items())}
            try:
                model.load_state_dict(state)
            except Exception as exc:
                send({"kind": "measured", "ckpt": path, "final_val_loss": None,
                      "shapes": shapes,
                      "load_error": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
                continue
            model.to(device)
            model.eval()
            total, nb = 0.0, 0
            with torch.no_grad():
                for i in range(0, val_t.shape[0], cfg["batch_sequences"]):
                    chunk = val_t[i:i + cfg["batch_sequences"]]
                    if chunk.shape[0] == 0:
                        continue
                    x, y = chunk[:, :-1], chunk[:, 1:]
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _, loss = model(x, y)
                    else:
                        _, loss = model(x, y)
                    total += float(loss.float().item())
                    nb += 1
            send({"kind": "measured", "ckpt": path, "shapes": shapes,
                  "load_error": None,
                  "final_val_loss": round(total / max(1, nb), 6)})
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

#!/usr/bin/env python3
"""S03 training child. One arm of one seed, in its own interpreter.

The submitted ordering module is loaded only on the submitted arm, so a
comparator run holds no submitted byte at all. Every record, including every
outbound connect the egress hook observes, leaves on a pipe the parent created.
Nothing is written to a file the graded party could later rewrite, which is why
truncating an audit is no longer a move that exists.

The final validation loss is not computed here. The child saves the trained
parameters and the parent measures the loss from them in a separate process, so
the number the reward rests on is produced by a process that never executed a
line the agent wrote.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def install_egress_channel(send):
    """Records and refuses every outbound connect, straight onto the parent's pipe."""
    import socket
    import time

    original = socket.socket.connect

    def guarded(self, address, *args, **kwargs):
        try:
            family = int(getattr(self, "family", -1))
        except Exception:
            family = -1
        if family in (getattr(socket, "AF_INET", 2), getattr(socket, "AF_INET6", 10)):
            send({"event": "egress_attempt", "wall": round(time.time(), 3),
                  "address": str(address)})
            raise OSError("egress_denied_during_scored_work")
        return original(self, address, *args, **kwargs)

    socket.socket.connect = guarded


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
        import order as order_mod
        import profiles as profiles_mod

        cfg = profiles_mod.get_profile(request["profile"])
        arm = request["arm"]
        seed = int(request["seed"])
        draw = int(request["draw"])

        install_egress_channel(send)

        cache = request["cache_dir"]
        os.makedirs(cache, exist_ok=True)
        key = "%s_%s_%s_%s_%s_%s" % (
            cfg["profile"], cfg["corpus_seed"], cfg["n_train_sequences"],
            cfg["seq_len"], cfg["vocab_size"], cfg["n_domains"])
        cpath = os.path.join(cache, key + ".npz")
        if os.path.exists(cpath):
            with np.load(cpath) as z:
                train, domains, val = z["train"], z["domains"], z["val"]
        else:
            train, domains, val = corpus_mod.generate_corpus(cfg)
            np.savez(cpath + ".partial.npz", train=train, domains=domains, val=val)
            os.replace(cpath + ".partial.npz", cpath)

        if arm == "baseline":
            order = order_mod.baseline_order(cfg, seed, draw)
        else:
            feats = corpus_mod.sequence_features(train, domains, cfg)
            meta = order_mod.build_meta(cfg, feats)
            order = order_mod.call_build_order(request["order_module"], meta)

        ok, reason = order_mod.validate_order(order, cfg)
        if not ok:
            send({"kind": "fault", "code": "order_contract_violated", "reason": reason})
            return 2

        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cpu") if cfg["device"] == "cpu" else torch.device("cuda")
        if device.type == "cuda" and not torch.cuda.is_available():
            send({"kind": "fault", "code": "graded_profile_requires_cuda_device"})
            return 2
        if device.type == "cpu" and os.environ.get("BIA_TORCH_THREADS"):
            torch.set_num_threads(max(1, int(os.environ["BIA_TORCH_THREADS"])))

        torch.manual_seed(seed)
        from model import TinyGPT, build_frozen_optimizer, frozen_lr_at

        model = TinyGPT(cfg).to(device)
        opt = build_frozen_optimizer(model, cfg)
        use_amp = device.type == "cuda" and cfg["dtype"] == "bfloat16"
        train_t = torch.from_numpy(train.astype(np.int64))

        substrate = profiles_mod.substrate_digest(cfg)
        run_id = ("%s_seed%d_draw%d" % (arm, seed, draw)) if arm == "baseline" \
            else ("%s_seed%d" % (arm, seed))
        import telemetry as telemetry_mod
        send({
            "event": "order_frozen", "run_id": run_id, "arm": arm, "seed": seed,
            "draw": draw, "order_digest": order_mod.order_digest(order),
            "first_index_sequence_digest": telemetry_mod.digest_json(
                order_mod.first_index_sequence(order)),
            "substrate_digest": substrate, "profile": cfg["profile"],
        })

        consumed, consumed_indices = 0, []
        model.train()
        for step in range(cfg["n_steps"]):
            idx = order[step]
            consumed_indices.extend(int(v) for v in idx)
            batch = train_t[idx].to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            lr = frozen_lr_at(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["grad_clip"])
            opt.step()
            consumed += len(idx) * cfg["seq_len"]
            send({
                "event": "train_step", "run_id": run_id, "arm": arm, "seed": seed,
                "draw": draw, "step": step, "first_index": int(idx[0]),
                "batch_size": len(idx), "forward_backward_per_step": 1,
                "tokens_consumed": consumed, "lr": round(float(lr), 10),
                "train_loss": round(float(loss.detach().float().item()), 6),
                "substrate_digest": substrate,
            })

        state = {k: v.detach().to("cpu", torch.float32).clone()
                 for k, v in model.state_dict().items()}
        torch.save(state, request["ckpt_path"])
        send({
            "event": "run_complete", "run_id": run_id, "arm": arm, "seed": seed,
            "draw": draw, "tokens_consumed": consumed,
            "consumed_total": len(consumed_indices),
            "consumed_unique_count": len(set(consumed_indices)),
            "consumed_sorted_digest": telemetry_mod.digest_json(sorted(consumed_indices)),
            "total_train_tokens": cfg["total_train_tokens"],
            "substrate_digest": substrate, "profile": cfg["profile"],
            "checkpoint": request["ckpt_path"],
        })
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

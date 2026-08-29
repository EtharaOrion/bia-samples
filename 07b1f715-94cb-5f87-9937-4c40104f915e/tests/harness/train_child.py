#!/usr/bin/env python3
"""S02 training child. The only process into which the submitted rule is loaded.

It writes no score, holds no secret, and every number it emits that the reward
depends on is recomputed by the parent from the checkpoint it saved beside the
record. The learning rate it applies comes from the agent-visible frozen
schedule on purpose, so an edited schedule shows up as a divergence against the
verifier's private reimplementation rather than as a silent advantage.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import hashlib
import sys
import traceback


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

PROFILES = {
    "full": {
        "n_layer": 6, "n_head": 6, "d_model": 384, "seq_len": 512,
        "batch_sequences": 48, "total_steps": 1400, "eval_every": 25,
        "val_batches": 8, "device": "cuda", "dtype": "bfloat16",
    },
    "smoke": {
        "n_layer": 2, "n_head": 2, "d_model": 64, "seq_len": 64,
        "batch_sequences": 4, "total_steps": 36, "eval_every": 4,
        "val_batches": 2, "device": "cpu", "dtype": "float32",
    },
}

FROZEN_CONTRACT = {
    "forward_backward_per_step": 1,
    "data_order": "sequential_frozen",
    "init_owner": "harness",
    "eval_owner": "harness",
    "schedule_owner": "harness",
}


class CallCounter:
    def __init__(self):
        self.forward = 0
        self.backward = 0
        self._model = None
        self._real = None

    def attach(self, model):
        import torch
        self._model = model
        self._real = model.forward

        def counting(*a, **kw):
            self.forward += 1
            out = self._real(*a, **kw)
            tensor = out if isinstance(out, torch.Tensor) else None
            if tensor is not None and tensor.requires_grad:
                tensor.register_hook(self._tick)
            return out

        model.forward = counting
        return self

    def _tick(self, grad):
        self.backward += 1
        return grad

    def reset(self):
        self.forward = 0
        self.backward = 0


class WriteGuard:
    """Records writes by submitted code outside the harness runtime scratch.

    Bounded claim: it sees writes issued through the Python names it rebinds. A
    raw syscall is outside it, and is left to the parent's before-and-after
    diff of the monitored roots.
    """

    _TARGETS = (
        ("builtins", "open", "open"), ("io", "open", "open"),
        ("os", "open", "osopen"), ("os", "rename", "arg1"),
        ("os", "replace", "arg1"), ("os", "remove", "arg0"),
        ("os", "unlink", "arg0"), ("os", "mkdir", "arg0"),
        ("os", "makedirs", "arg0"), ("os", "truncate", "arg0"),
        ("shutil", "copyfile", "arg1"), ("shutil", "copy", "arg1"),
        ("shutil", "move", "arg1"), ("shutil", "rmtree", "arg0"),
        ("pathlib.Path", "open", "open"),
        ("pathlib.Path", "write_text", "arg0"),
        ("pathlib.Path", "write_bytes", "arg0"),
        ("pathlib.Path", "mkdir", "arg0"), ("pathlib.Path", "touch", "arg0"),
        ("pathlib.Path", "unlink", "arg0"),
    )

    def __init__(self, allow=()):
        self.seen = []
        self.allow = [os.path.abspath(a) for a in allow if a]
        self._saved = []
        self._depth = 0

    @staticmethod
    def _resolve(dotted):
        import importlib
        head, _, tail = dotted.partition(".")
        obj = importlib.import_module(head)
        return getattr(obj, tail) if tail else obj

    def _allowed(self, target):
        try:
            path = os.path.abspath(os.fspath(target))
        except TypeError:
            return False
        return any(path == a or path.startswith(a + os.sep) for a in self.allow)

    def _record(self, label, target):
        if not self._allowed(target):
            self.seen.append("%s:%s" % (label, target))

    def _wrap(self, real, label, kind):
        if kind == "open":
            def guarded(file, mode="r", *a, **kw):
                if any(c in str(mode) for c in ("w", "a", "x", "+")):
                    self._record(label, file)
                return real(file, mode, *a, **kw)
        elif kind == "osopen":
            mask = 0
            for name in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC"):
                mask |= getattr(os, name, 0)

            def guarded(path, flags, *a, **kw):
                if int(flags) & mask:
                    self._record(label, path)
                return real(path, flags, *a, **kw)
        elif kind == "arg1":
            def guarded(*a, **kw):
                self._record(label, a[1] if len(a) > 1 else "")
                return real(*a, **kw)
        else:
            def guarded(*a, **kw):
                self._record(label, a[0] if a else "")
                return real(*a, **kw)
        return guarded

    def __enter__(self):
        self._depth += 1
        if self._depth > 1:
            return self
        for dotted, attr, kind in self._TARGETS:
            try:
                holder = self._resolve(dotted)
                real = getattr(holder, attr)
            except (ImportError, AttributeError):
                continue
            self._saved.append((holder, attr, real))
            try:
                setattr(holder, attr, self._wrap(real, "%s.%s" % (dotted, attr), kind))
            except (AttributeError, TypeError):
                self._saved.pop()
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        if self._depth > 0:
            return False
        while self._saved:
            holder, attr, real = self._saved.pop()
            try:
                setattr(holder, attr, real)
            except (AttributeError, TypeError):
                pass
        return False


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
        request = json.loads(sys.stdin.readline())
        bundle = request["bundle"]
        env_root = os.path.join(bundle, "environment")
        for p in (os.path.join(env_root, "runner"), env_root):
            if p not in sys.path:
                sys.path.insert(0, p)

        import torch
        import data as data_mod
        import frozen_schedule as sched
        import model as model_mod

        cfg = PROFILES[request["profile"]]
        seed = int(request["seed"])
        sub_path = request["submission"]
        device = cfg["device"]
        if device == "cuda" and not torch.cuda.is_available():
            send({"kind": "fault", "code": "cuda-unavailable"})
            return 1
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("BIA_CPU_THREADS", "1")))
        torch.manual_seed(seed)

        allow = [a for a in os.environ.get("BIA_GUARD_ALLOW", "").split(os.pathsep) if a]
        guard = WriteGuard(allow)

        digest_at_load = file_digest(sub_path)
        spec = importlib.util.spec_from_file_location("bia_graded_rule", sub_path)
        module = importlib.util.module_from_spec(spec)
        with guard:
            spec.loader.exec_module(module)
        build = getattr(module, "build_update_rule", None)
        if not callable(build):
            send({"kind": "fault", "code": "build-update-rule-absent"})
            return 1
        digest_after_load = file_digest(sub_path)

        train_path, train_digest, val_path, val_digest = data_mod.resolve_shards(
            request["profile"], os.environ.get("BIA_DATA_DIR", "/data/fineweb"),
            os.path.join(env_root, "fixtures"))
        train_loader = data_mod.FrozenLoader(
            train_path, cfg["batch_sequences"], cfg["seq_len"], train_digest)
        val_loader = data_mod.FrozenLoader(
            val_path, cfg["batch_sequences"], cfg["seq_len"], val_digest)

        net = model_mod.FrozenGPT(cfg["n_layer"], cfg["n_head"], cfg["d_model"],
                                  cfg["seq_len"])
        model_mod.frozen_init(net, seed)
        net.to(device)
        net.train()
        groups = model_mod.build_param_groups(net)
        probe_params = [p for g in groups for p in g["params"]][:6]
        counter = CallCounter().attach(net)

        with guard:
            opt = build(groups)
        if not isinstance(opt, torch.optim.Optimizer):
            send({"kind": "fault", "code": "build-update-rule-returned-non-optimizer",
                  "type": type(opt).__name__})
            return 1

        if device == "cuda" and cfg["dtype"] == "bfloat16":
            def autocast_ctx():
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = contextlib.nullcontext

        arch = model_mod.architecture_signature(
            cfg["n_layer"], cfg["n_head"], cfg["d_model"], cfg["seq_len"])
        static = {
            "schema": "bia.s02.telemetry/v2",
            "profile": request["profile"],
            "seed": seed,
            "total_steps": cfg["total_steps"],
            "batch_sequences": cfg["batch_sequences"],
            "batch_tokens": cfg["batch_sequences"] * cfg["seq_len"],
            "architecture": arch,
            "frozen": dict(FROZEN_CONTRACT),
            "schedule_digest": file_digest(os.path.join(env_root, "frozen_schedule.py")),
            "model_digest": file_digest(os.path.join(env_root, "runner", "model.py")),
            "update_rule_digest": digest_at_load,
            "update_rule_digest_after_load": digest_after_load,
            "train_shard_digest": train_digest,
            "val_shard_digest": val_digest,
        }

        def param_probe():
            import hashlib
            parts = []
            for p in probe_params:
                flat = p.detach().reshape(-1)
                stride = max(1, flat.numel() // 16)
                parts.append(flat[::stride][:16].to(torch.float32))
            blob = torch.cat(parts).cpu().numpy().tobytes()
            return hashlib.sha256(blob).hexdigest()

        import io

        def save_checkpoint(step):
            """Parameters leave on the parent's binary channel, never onto disk.

            An artifact on disk stays reachable by the submission for the rest
            of the run, and copying the final parameters over an early
            checkpoint reads as reaching the target early. Streaming removes the
            artifact instead of guarding it.
            """
            state = {k: v.detach().to("cpu", torch.float32).clone()
                     for k, v in net.state_dict().items()}
            buf = io.BytesIO()
            torch.save(state, buf)
            return buf.getvalue()

        fwd_bwd_total = 0
        lr_write_violations = 0
        out_of_step_mutations = 0
        steps_checked = 0
        after_prev_step = param_probe()

        for step in range(1, cfg["total_steps"] + 1):
            if param_probe() != after_prev_step:
                out_of_step_mutations += 1
            applied = {}
            for g in opt.param_groups:
                name = g.get("name")
                if name is None:
                    send({"kind": "fault", "code": "update-rule-dropped-group-name"})
                    return 1
                g["lr"] = sched.group_lr(name, step, cfg["total_steps"])
                applied[name] = g["lr"]

            x, y = train_loader.next_batch(device)
            counter.reset()
            with autocast_ctx():
                loss = net(x, y)
            loss.backward()
            fwd_bwd_total += 1
            with guard:
                opt.step()
            after_prev_step = param_probe()
            opt.zero_grad(set_to_none=True)
            steps_checked += 1
            forward_calls = int(counter.forward)
            backward_calls = int(counter.backward)

            for g in opt.param_groups:
                if float(g["lr"]) != float(applied[g["name"]]):
                    lr_write_violations += 1

            if step % cfg["eval_every"] == 0 or step == cfg["total_steps"] or step == 1:
                payload = save_checkpoint(step)
                body = dict(static)
                body.update({
                    "step": step,
                    "train_loss": round(float(loss.detach()), 6),
                    "lr_applied": {k: round(v, 12) for k, v in applied.items()},
                    "forward_backward_this_step": forward_calls,
                    "forward_calls_this_step": forward_calls,
                    "backward_calls_this_step": backward_calls,
                    "forward_backward_cumulative": fwd_bwd_total,
                    "steps_checked": steps_checked,
                    "lr_write_violations": lr_write_violations,
                    "out_of_step_mutations": out_of_step_mutations,
                    "guard_observed_writes": sorted(set(guard.seen)),
                    "blob_bytes": len(payload),
                    "batch_sequences_observed": int(x.shape[0]),
                    "sequence_length_observed": int(x.shape[1]),
                })
                send(body)
                blob.write(payload)
                blob.flush()
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

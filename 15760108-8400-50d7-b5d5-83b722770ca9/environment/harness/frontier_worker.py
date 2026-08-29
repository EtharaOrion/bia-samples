"""Frozen out-of-process host for a submitted S09 frontier. Never the author of a record.

The runner does not import the submission. It starts this worker as a separate operating
system process and speaks to it over one newline-delimited JSON channel. The submission's
optimizer is a live object whose `step` is the whole point of the task, so the training loop
has to live beside it; what does NOT live beside it is any measurement. This worker trains and
returns WEIGHTS. It never computes a validation loss, a density, an anchor, a normalized
coordinate or a hypervolume, and it never writes the telemetry record.

The complete outbound alphabet is one JSON object of counters plus one file of parameter
tensors. The runner loads those tensors into its own model, measures both objectives itself,
and authors the record from what it measured. A submission can therefore say anything it likes
in this process and still be graded only on the weights it managed to produce.

The runner also owns the clock. It passes an absolute deadline and it measures the elapsed
time of the whole per-point call from the outside, so work smuggled into `build_optimizer`
costs exactly the same budget as work done in the loop.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import engine  # noqa: E402

PROTOCOL_VERSION = "bia.s09.frontier_worker/v1"


def load_submission(path: pathlib.Path):
    if not path.is_file():
        raise FileNotFoundError(f"no submission at {path}")
    spec = importlib.util.spec_from_file_location("bia_s09_submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bia_s09_submission"] = mod
    spec.loader.exec_module(mod)
    for fn in ("propose_frontier", "build_optimizer"):
        if not callable(getattr(mod, fn, None)):
            raise AttributeError(f"submission does not expose callable {fn}")
    return mod


def train_point(spec, point_index, point, build_optimizer, train, device, deadline):
    torch.manual_seed(spec.run_seed + point_index)
    np.random.seed(spec.run_seed + point_index)
    model = engine.TinyGPT(spec).to(device)
    model.fwd_calls = 0

    named = [(n, p) for n, p in model.named_parameters()]
    opt = build_optimizer(named, point, spec.base_lr)
    if not isinstance(opt, torch.optim.Optimizer):
        raise TypeError(f"build_optimizer returned {type(opt).__name__}, not a torch.optim.Optimizer")

    bwd_calls = {"n": 0}

    def count_backward(*_args, **_kwargs):
        bwd_calls["n"] += 1

    t = spec.seq_len
    n_windows = train.numel() - t - 1
    gen = torch.Generator().manual_seed(spec.run_seed * 1000 + point_index)

    steps_taken = 0
    stop_reason = "max_steps"
    loss_first, loss_last = None, None

    model.train()
    for _step in range(spec.max_steps):
        starts = torch.randint(0, n_windows, (spec.batch_sequences,), generator=gen)
        x = torch.stack([train[s : s + t] for s in starts.tolist()]).to(device)
        y = torch.stack([train[s + 1 : s + 1 + t] for s in starts.tolist()]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.register_hook(count_backward)
        loss.backward()
        opt.step()
        steps_taken += 1
        lv = float(loss.detach())
        if loss_first is None:
            loss_first = lv
        loss_last = lv
        if time.time() >= deadline:
            stop_reason = "per_run_deadline"
            break

    return model, {
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "forward_calls": model.fwd_calls,
        "backward_calls": bwd_calls["n"],
        "train_loss_first": loss_first,
        "train_loss_last": loss_last,
        "counted_by": "worker",
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: frontier_worker.py <spec.json> <submission.py>", file=sys.stderr)
        return 2
    spec_path, sub_path = sys.argv[1], pathlib.Path(sys.argv[2])

    chan = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply(obj):
        chan.write(json.dumps(obj, sort_keys=True) + "\n")
        chan.flush()

    try:
        spec = engine.load_spec(spec_path)
        device = engine.select_device(spec)
        torch.manual_seed(spec.run_seed)
        np.random.seed(spec.run_seed)
        succ, q = engine.build_source(spec)
        train_np = engine.generate_tokens(spec, succ, q, spec.train_tokens, spec.source_seed + 1)
        train = torch.from_numpy(train_np)
        mod = load_submission(sub_path)
    except BaseException as exc:
        reply({"ok": False, "error": f"worker_setup_failed: {exc}", "traceback": traceback.format_exc()})
        return 1

    reply({"ok": True, "protocol": PROTOCOL_VERSION, "pid": os.getpid()})

    points = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "shutdown":
                reply({"ok": True})
                break
            if op == "propose":
                points = list(mod.propose_frontier())
                reply({"ok": True, "count": len(points),
                       "point_reprs": [repr(p)[:512] for p in points]})
                continue
            if op == "train":
                if points is None:
                    raise RuntimeError("train requested before propose")
                i = int(req["index"])
                model, counters = train_point(
                    spec, i, points[i], mod.build_optimizer, train, device, float(req["deadline"])
                )
                out = os.path.join(req["scratch"], f"weights_{i}.pt")
                torch.save({k: v.detach().to("cpu") for k, v in model.state_dict().items()}, out)
                reply({"ok": True, "weights_file": os.path.basename(out), **counters})
                continue
            reply({"ok": False, "error": f"unknown op {op!r}"})
        except BaseException as exc:
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()})
    chan.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

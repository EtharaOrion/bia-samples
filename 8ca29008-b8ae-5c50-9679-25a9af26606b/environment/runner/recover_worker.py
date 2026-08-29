"""S08 out-of-process host for a submitted `recover`. Frozen harness code.

The runner does not import the submission. It starts this worker as a separate operating
system process and speaks to it over one newline-delimited JSON channel, with tensors passed
as `torch.save` files inside a scratch directory the runner owns. Whatever the submission
rebinds, patches or replaces, it rebinds it in THIS interpreter, where no telemetry record, no
hash chain, no evaluation function, no probe counter and no reward computation lives.

The complete outbound alphabet is one JSON object describing an optimizer plus one file of
tensors:

    optimizer_class   a name that must resolve inside torch.optim in the runner
    param_groups      scalar hyperparameters plus explicit parameter indices
    state             per-parameter entries, tensors carried in the scratch file
    lr_schedule       one absolute learning rate per recovery step, or null

The runner rebuilds the optimizer from that description against ITS model. Nothing callable
and no code crosses back, so `recover` can no longer reach the evaluation the crossing is
measured with, which is the whole reason this file exists.

The worker builds its own copy of the model from frozen harness code and the checkpoint state
the runner hands it. Changing that copy's weights changes nothing the runner measures, so the
model this side sees is genuinely read-only with respect to the graded run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import s08_core as core  # noqa: E402

PROTOCOL_VERSION = "bia.recover_worker/v1"

ALLOWED_OPTIMIZERS = (
    "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "NAdam",
    "RAdam", "RMSprop", "Rprop", "SGD",
)


def load_submission(path):
    spec = importlib.util.spec_from_file_location("bia_s08_submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bia_s08_submission"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "recover"):
        raise AttributeError("submission recover.py does not define recover(...)")
    return mod


class ProbeProxy:
    """The probe the submission calls. Every pass is executed by the runner, on the runner's
    model, against the runner's budget counter, and only the resulting losses and gradient
    tensors come back. A submission cannot make a probe free by reimplementing it."""

    def __init__(self, call, scratch):
        self.call = call
        self.scratch = scratch
        self.n_calls = 0

    def __call__(self, n: int = 1):
        reply = self.call({"op": "probe", "n": int(n)})
        self.n_calls += 1
        if not reply.get("file"):
            return []
        blob = torch.load(os.path.join(self.scratch, reply["file"]), map_location="cpu", weights_only=False)
        return blob


def describe_optimizer(opt, params, scratch, tag):
    index_of = {id(p): i for i, p in enumerate(params)}
    groups = []
    for g in opt.param_groups:
        entry = {"param_indices": [index_of[id(p)] for p in g["params"] if id(p) in index_of]}
        for k, v in g.items():
            if k == "params":
                continue
            if isinstance(v, bool) or isinstance(v, (int, float, str)) or v is None:
                entry[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                entry[k] = [float(x) for x in v]
        groups.append(entry)

    tensor_state, scalar_state = {}, {}
    for p, st in opt.state.items():
        idx = index_of.get(id(p))
        if idx is None:
            continue
        scalars, tensors = {}, {}
        for k, v in st.items():
            if torch.is_tensor(v):
                if v.numel() == 1 and v.dim() == 0:
                    scalars[k] = {"scalar_tensor": float(v.item()), "dtype": str(v.dtype)}
                else:
                    tensors[k] = v.detach().to("cpu")
            elif isinstance(v, bool) or isinstance(v, (int, float, str)) or v is None:
                scalars[k] = v
        scalar_state[str(idx)] = scalars
        tensor_state[str(idx)] = tensors

    blob = os.path.join(scratch, "opt_state_%s.pt" % tag)
    torch.save(tensor_state, blob)
    return {
        "optimizer_class": type(opt).__name__,
        "param_groups": groups,
        "state_scalars": scalar_state,
        "state_tensor_file": os.path.basename(blob),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: recover_worker.py <recover.py>", file=sys.stderr)
        return 2
    sub_path = os.path.abspath(argv[0])

    chan = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply(obj):
        chan.write(json.dumps(obj, sort_keys=True) + "\n")
        chan.flush()

    def call(req):
        reply(req)
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("runner closed the channel")
        msg = json.loads(line)
        if not msg.get("ok"):
            raise RuntimeError(str(msg.get("error", "runner refused the call")))
        return msg

    try:
        mod = load_submission(sub_path)
    except BaseException as exc:
        reply({"ok": False, "error": "import_failed: %s" % exc, "traceback": traceback.format_exc()})
        return 1

    reply({"ok": True, "protocol": PROTOCOL_VERSION, "pid": os.getpid()})

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
            if op != "recover":
                reply({"ok": False, "error": "unknown op %r" % op})
                continue

            scratch = req["scratch"]
            cfg_public = dict(req["cfg_public"])
            cfg = core.load_cfg(req["profile"])
            model = core.build_model(cfg, torch.device("cpu"))
            model.load_state_dict(
                torch.load(os.path.join(scratch, req["model_state_file"]), map_location="cpu", weights_only=False)
            )
            ckpt_state = torch.load(
                os.path.join(scratch, req["ckpt_state_file"]), map_location="cpu", weights_only=False
            )
            probe = ProbeProxy(call, scratch)

            result = mod.recover(model, ckpt_state, cfg_public, probe)
            lr_at = None
            if isinstance(result, tuple):
                if len(result) == 1:
                    opt = result[0]
                elif len(result) == 2:
                    opt, lr_at = result
                else:
                    raise TypeError("recover returned a tuple of unexpected length")
            else:
                opt = result
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError("recover must return a torch.optim.Optimizer")
            if type(opt).__name__ not in ALLOWED_OPTIMIZERS:
                raise TypeError(
                    "recover returned %r, which is not one of the torch.optim classes the "
                    "runner can rebuild: %s" % (type(opt).__name__, ", ".join(ALLOWED_OPTIMIZERS))
                )

            spec = describe_optimizer(opt, list(model.parameters()), scratch, req["tag"])
            if lr_at is not None:
                spec["lr_schedule"] = [float(lr_at(i)) for i in range(1, int(cfg.max_steps) + 1)]
            else:
                spec["lr_schedule"] = None
            spec["ok"] = True
            spec["probe_calls"] = probe.n_calls
            reply(spec)
        except BaseException as exc:
            reply({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                   "traceback": traceback.format_exc()})
    chan.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

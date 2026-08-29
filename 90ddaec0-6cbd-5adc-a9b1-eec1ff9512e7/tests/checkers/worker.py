"""Isolated implementation host for the BIA-GSN-1 throughput task.

Private verifier code. One worker process hosts exactly one implementation, either the
frozen reference or the submitted module, and does nothing except answer commands from
the grader on stdin with one JSON line on stdout.

Why this exists. The graded artifact used to execute inside the grading interpreter,
which meant the submission could reach the checker table, the timing harness and the
reward in the same address space. It now runs here, in a process that owns no checker,
no clock the reward reads, no event log and no reward. The grader starts one worker per
implementation, so the reference and the submission pay an identical protocol cost and
the ratio between them is unaffected by it.

What this process is still allowed to lie about. Everything it says about itself. The
grader therefore believes none of it: it times the round trips on its own clock, it
chooses the timing inputs, it chooses the chain that couples one trial to the next, and
it compares the raw output bytes of the two workers against each other. The one word a
worker sends back is a liveness signal, not a measurement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import opcore  # noqa: E402


# The environment variables that turn a relaxed precision path on. The grader reads the
# worker's /proc environ for these, which covers what the process was started with; this
# list also covers what the process set for itself after starting.
RELAXED_PRECISION_VARS = (
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "PYTORCH_ENABLE_FAST_MATH",
    "TORCH_CUDNN_V8_API_ENABLED",
)


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def strict_numeric_posture(torch):
    """Set the strict posture before the submitted module is imported."""
    for setter in (
        lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", False),
        lambda: setattr(torch.backends.cudnn, "allow_tf32", False),
        lambda: setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False),
        lambda: setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False),
        lambda: torch.set_float32_matmul_precision("highest"),
    ):
        try:
            setter()
        except Exception:
            pass


def load_submission(path, env_dir):
    sys.path.insert(0, str(env_dir))
    sys.path.insert(0, str(pathlib.Path(path).resolve().parent))
    spec = importlib.util.spec_from_file_location("bia_submission", str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load submission at %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "forward"):
        raise AttributeError("submission does not define forward")
    return module.forward


class Host:
    """Holds the timing workload and advances the coupled chain one trial at a time."""

    CHAIN_CLAMP = 16.0

    def __init__(self, torch, forward, chunk_rows, device):
        self.torch = torch
        self.forward = forward
        self.chunk_rows = int(chunk_rows)
        self.device = device
        self.x = None
        self.g = None
        self.b = None
        self.y = None
        self.r = None
        self.dst = None
        self.src = None

    def sync(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()

    def prepare(self, rows, cols, seed):
        self.x, self.g, self.b = opcore.make_inputs("normal", int(rows), int(cols), int(seed), self.device)
        self.y, self.r = self.forward(self.x, self.g, self.b, self.chunk_rows)
        self.sync()

    def set_chain(self, dst_cols, src_cols):
        torch = self.torch
        rows, cols = int(self.x.shape[0]), int(self.x.shape[1])
        base = torch.arange(rows, device=self.x.device, dtype=torch.int64) * cols
        self.dst = base + torch.as_tensor(dst_cols, device=self.x.device, dtype=torch.int64)
        self.src = base + torch.as_tensor(src_cols, device=self.x.device, dtype=torch.int64)

    def trial(self):
        """One coupled trial: fold the previous output back into the input, then compute.

        The fold reads one element of Y and the whole of R, one entry per row, at column
        positions the grader chose. Trial t therefore consumes trial t minus one's output
        in every row, so a cache keyed on the identity, the pointer, the shape or the
        device of the input serves a value that is wrong for the input actually presented,
        and a trial that is not performed changes every later trial's result.
        """
        torch = self.torch
        xf = self.x.reshape(-1)
        yf = self.y.reshape(-1)
        folded = yf.index_select(0, self.src) + self.r
        folded = torch.clamp(folded, -self.CHAIN_CLAMP, self.CHAIN_CLAMP)
        xf.index_copy_(0, self.dst, folded)
        self.y, self.r = self.forward(self.x, self.g, self.b, self.chunk_rows)
        self.sync()

    def probe(self, y_idx, r_idx):
        """Return the exact values at grader chosen positions of the current output.

        This is the barrier. Producing a correct value on the host requires the work
        that produces it to have completed, whatever stream it was queued on, so a
        worker that answered its trials without waiting for the device pays the
        outstanding time here, inside the interval the grader is timing. A worker that
        answers with anything other than the true values diverges from the reference
        worker's answer at the same positions, which the grader compares.
        """
        torch = self.torch
        yf = self.y.reshape(-1)
        rf = self.r.reshape(-1)
        y_sel = yf.index_select(0, torch.as_tensor(y_idx, device=yf.device, dtype=torch.int64))
        r_sel = rf.index_select(0, torch.as_tensor(r_idx, device=rf.device, dtype=torch.int64))
        return {"y": [float(v) for v in y_sel.cpu().tolist()],
                "r": [float(v) for v in r_sel.cpu().tolist()]}

    def dump(self, stem):
        y = self.y.detach().contiguous().cpu().numpy()
        r = self.r.detach().contiguous().cpu().numpy()
        pathlib.Path(stem + ".y.bin").write_bytes(y.tobytes())
        pathlib.Path(stem + ".r.bin").write_bytes(r.tobytes())

    def fixture(self, kind, rows, cols, seed, stem):
        x, g, b = opcore.make_inputs(kind, int(rows), int(cols), int(seed), self.device)
        y, r = self.forward(x, g, b, self.chunk_rows)
        for tensor, suffix in ((y, ".y.bin"), (r, ".r.bin")):
            if not hasattr(tensor, "detach"):
                raise TypeError("forward returned a non tensor")
            array = tensor.detach().contiguous().cpu().numpy()
            pathlib.Path(stem + suffix).write_bytes(array.tobytes())
        self.sync()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["reference", "submission"])
    ap.add_argument("--submission-path", default=None)
    ap.add_argument("--env-dir", default=None)
    ap.add_argument("--device", required=True)
    ap.add_argument("--chunk-rows", type=int, required=True)
    args = ap.parse_args()

    import torch

    strict_numeric_posture(torch)
    if args.impl == "reference":
        forward = opcore.reference_forward
    else:
        forward = load_submission(args.submission_path, args.env_dir)

    host = Host(torch, forward, args.chunk_rows, args.device)
    send({"ok": True, "event": "ready", "pid": os.getpid()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            send({"ok": False, "error": "malformed_command"})
            return 1
        cmd = msg.get("cmd")
        try:
            if cmd == "stop":
                return 0
            if cmd == "fixture":
                host.fixture(msg["kind"], msg["rows"], msg["cols"], msg["seed"], msg["stem"])
            elif cmd == "prepare":
                host.prepare(msg["rows"], msg["cols"], msg["seed"])
            elif cmd == "chain":
                host.set_chain(msg["dst_cols"], msg["src_cols"])
            elif cmd == "trial":
                host.trial()
            elif cmd == "probe":
                send({"ok": True, "values": host.probe(msg["y_idx"], msg["r_idx"])})
                continue
            elif cmd == "dump":
                host.dump(msg["stem"])
            elif cmd == "flags":
                send({"ok": True, "flags": {
                    "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                    "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                    "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
                    "autocast_enabled": bool(torch.is_autocast_enabled()),
                    "live_env": {v: os.environ.get(v) for v in RELAXED_PRECISION_VARS},
                }})
                continue
            else:
                send({"ok": False, "error": "unknown_command_%s" % cmd})
                return 1
        except Exception as exc:
            send({"ok": False, "error": repr(exc)})
            return 1
        send({"ok": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())

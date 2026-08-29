"""Frozen training harness for bia slot S06, gradient estimator under injected noise.

The harness owns the model, the data, the noise fixture, the optimizer update rule
and the telemetry. A submission owns exactly one thing: the function that turns a
sequence of corrupted gradient observations into the update direction the frozen
optimizer applies.

ISOLATION MODEL. The submitted estimator never executes in this process. This process
owns the model, the data, the frozen noise process, the optimizer, the evaluation, the
target and the run record; the submission owns one function and runs in a separate
interpreter that holds none of those things. The two exchange nothing but gradient
sized tensors over a shared buffer pair plus a one word control message, so there is no
module for the submission to rebind, no clock for it to move, no writer for it to
intercept and no target for it to read. The run record is buffered in memory and
written once, after every arm of every seed has finished, so nothing the submission
could want to read is ever on disk while the submission is alive.

Protocol per graded run, for every frozen seed:

  1. the baseline arm runs the frozen identity estimator, which passes each
     observation through untouched, and its validation loss at the frozen
     reference step becomes the target loss for that seed
  2. the agent arm runs the submitted estimator on the identical seed, the
     identical batch order and the identical noise realization
  3. steps to target is the first evaluation step at which the arm reaches the
     target loss and stays at or below it for every later evaluation

Nothing here reads a clock, an environment secret or a network. Every branch is a
function of the frozen configuration, the seed and the step index.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import torch
import torch.multiprocessing as tmp

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BUNDLE, "environment", "fixtures"))

import data as datamod  # noqa: E402
import estimator_worker  # noqa: E402
import model as modelmod  # noqa: E402
import noise_process as noisemod  # noqa: E402

PROBE_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 30, 63, 64, 96, 97)

# A submission that cannot be loaded, that returns the wrong thing, or that hangs is a
# solver failure and must be reported as one. Distinguishing it from an infrastructure
# failure matters: an infrastructure-shaped reason invites a re-run, and re-running a
# broken submission just breaks again.
SUBMISSION_FAULT_EXIT = 7


class SubmissionFault(RuntimeError):
    """Raised when the fault is attributable to the submitted estimator."""


class _TrueGradSentinel:
    """Tripwire standing where a submission would reach for uncorrupted gradients.

    The harness never touches it. Any attribute read increments the counter, and
    the counter is written into the run record, where the ABSENCE checker reads it.
    """

    hits = 0
    names = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        _TrueGradSentinel.hits += 1
        if name not in _TrueGradSentinel.names:
            _TrueGradSentinel.names.append(name)
        return None

    def __getitem__(self, key):
        _TrueGradSentinel.hits += 1
        if str(key) not in _TrueGradSentinel.names:
            _TrueGradSentinel.names.append(str(key))
        return None


TRUE_GRADS = _TrueGradSentinel()
BACKWARD_CALLS = {"n": 0}


class Recorder:
    """Append only run record, held in memory until the whole run is over.

    Nothing reaches the filesystem while a submission is alive. The previous design
    appended each line as it was produced, which put the baseline arm's target loss on
    disk before the agent arm started, where the agent arm could read it. Buffering is
    not a tidiness choice: it is what makes the target unreachable rather than merely
    undocumented.
    """

    def __init__(self, path):
        self.path = path
        self.seq = 0
        self.records = []

    def emit(self, **fields):
        self.seq += 1
        fields["seq"] = self.seq
        self.records.append(fields)

    def flush(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")


def identity_estimator_source() -> str:
    return "def build_estimator(meta):\n    return Identity()\n"


class Identity:
    """Frozen baseline estimator. Passes every observation through untouched."""

    def estimate(self, step, observed):
        return observed


class IdentityBridge:
    """The frozen baseline arm. No separate process: there is no submitted code here."""

    def __init__(self):
        self.est = Identity()

    def estimate(self, step, observed):
        return self.est.estimate(step, observed)

    def close(self):
        return None


class EstimatorBridge:
    """The agent arm. The submission runs in its own interpreter, behind a narrow channel.

    The channel carries exactly two things: gradient shaped tensors through a pair of
    shared buffers the parent allocates, and a one word control message. The worker is
    handed the corrupted observations and nothing else. It has no model, no data, no
    optimizer, no evaluation, no record and no target, so there is nothing in it worth
    rebinding and nothing it can report about itself that this process believes.
    """

    START_TIMEOUT_S = 600.0
    STEP_TIMEOUT_S = 600.0

    def __init__(self, submission_path, meta, names, params, device):
        ctx = tmp.get_context("spawn")
        self.in_bufs = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in zip(names, params)}
        self.out_bufs = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in zip(names, params)}
        if str(device) == "cpu":
            for table in (self.in_bufs, self.out_bufs):
                for tensor in table.values():
                    tensor.share_memory_()
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=estimator_worker.serve,
            args=(str(submission_path), meta, self.in_bufs, self.out_bufs, child),
            daemon=True,
        )
        self.proc.start()
        child.close()
        if not self.conn.poll(self.START_TIMEOUT_S):
            self.close()
            raise SubmissionFault("submission_worker_did_not_start")
        tag, payload = self.conn.recv()
        if tag != "ready":
            self.close()
            raise SubmissionFault(str(payload))

    def estimate(self, step, observed):
        for name, buf in self.in_bufs.items():
            buf.copy_(observed[name].detach().reshape(buf.shape))
        if self.in_bufs and next(iter(self.in_bufs.values())).is_cuda:
            torch.cuda.synchronize()
        self.conn.send(("step", int(step)))
        if not self.conn.poll(self.STEP_TIMEOUT_S):
            raise SubmissionFault("submission_estimator_timed_out_at_step_%d" % int(step))
        tag, payload = self.conn.recv()
        if tag != "ok":
            raise SubmissionFault(str(payload))
        return {n: b.clone() for n, b in self.out_bufs.items()}

    def close(self):
        try:
            self.conn.send(("stop", 0))
        except (OSError, BrokenPipeError, ValueError):
            pass
        try:
            self.proc.join(30.0)
            if self.proc.is_alive():
                self.proc.terminate()
        except (AttributeError, AssertionError):
            pass


def file_digest(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def tree_digest(root) -> str:
    """sha256 over the harness tree this run actually executed."""
    parts = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".pyc"):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            parts.append(rel + "\0" + file_digest(full))
    return hashlib.sha256("\n".join(parts).encode("ascii")).hexdigest()


def adamw_step(params, grads, state, cfg, step):
    """Frozen optimizer update. Not a free axis: a submission never changes this."""
    lr = modelmod.lr_at(cfg, step)
    total_sq = 0.0
    for g in grads:
        total_sq += float(g.detach().float().pow(2).sum())
    total = total_sq ** 0.5
    scale = 1.0 if total <= cfg.grad_clip else (cfg.grad_clip / (total + 1e-12))
    for i, (p, g) in enumerate(zip(params, grads)):
        st = state.setdefault(i, {"m": torch.zeros_like(p, dtype=torch.float32),
                                  "v": torch.zeros_like(p, dtype=torch.float32)})
        gg = g.detach().to(torch.float32) * scale
        st["m"].mul_(cfg.beta1).add_(gg, alpha=1.0 - cfg.beta1)
        st["v"].mul_(cfg.beta2).addcmul_(gg, gg, value=1.0 - cfg.beta2)
        bc1 = 1.0 - cfg.beta1 ** (step + 1)
        bc2 = 1.0 - cfg.beta2 ** (step + 1)
        denom = (st["v"] / bc2).sqrt_().add_(cfg.eps)
        with torch.no_grad():
            if p.dim() >= 2:
                p.add_(p, alpha=-lr * cfg.weight_decay)
            p.addcdiv_(st["m"] / bc1, denom, value=-lr)
    return lr, total


@torch.no_grad()
def evaluate(model, cfg, val, device):
    model.eval()
    total = 0.0
    n = 0
    for x, y in datamod.batch_stream(val, "val", cfg.eval_batches, cfg.batch_size, cfg.block_size, device):
        total += float(model(x, y))
        n += 1
    model.train()
    return total / max(n, 1)


def run_arm(arm, seed, cfg, bundle, submission_path, recorder, substrate, device, train, val):
    """One training arm. Identical in every respect except which estimator runs."""
    torch.manual_seed(seed)
    model = modelmod.build_model(cfg, seed, device)
    params = [p for _, p in sorted(model.named_parameters())]
    names = [n for n, _ in sorted(model.named_parameters())]
    noise = noisemod.NoiseProcess(torch)
    meta = {
        "tensor_ids": list(names),
        "shapes": {n: list(p.shape) for n, p in zip(names, params)},
        "numels": {n: int(p.numel()) for n, p in zip(names, params)},
        "max_steps": cfg.max_steps,
        "device": str(device),
        "point": cfg.point,
        "seed": int(seed),
    }
    if arm == "baseline":
        bridge = IdentityBridge()
        est_digest = hashlib.sha256(identity_estimator_source().encode("ascii")).hexdigest()
    else:
        bridge = EstimatorBridge(submission_path, meta, names, params, device)
        est_digest = file_digest(submission_path)

    state = {}
    evals = []
    started = time.time()
    recorder.emit(event="arm_start", arm=arm, seed=int(seed), step=-1,
                  substrate_digest=substrate, estimator_digest=est_digest)
    try:
        stream = datamod.batch_stream(train, "train|seed%d" % seed, cfg.max_steps,
                                      cfg.batch_size, cfg.block_size, device)
        for step, (x, y) in enumerate(stream):
            loss = model(x, y)
            loss.backward()
            BACKWARD_CALLS["n"] += 1
            observed = {}
            for name, p in zip(names, params):
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                observed[name] = noise.corrupt(name, g, step)
                p.grad = None
            out = bridge.estimate(step, {k: v.clone() for k, v in observed.items()})
            if not isinstance(out, dict):
                raise SubmissionFault("estimator_returned_non_dict")
            grads = []
            for name, p in zip(names, params):
                g = out.get(name)
                if g is None:
                    raise SubmissionFault("estimator_dropped_tensor_%s" % name)
                g = torch.as_tensor(g).to(device=p.device, dtype=torch.float32)
                if tuple(g.shape) != tuple(p.shape):
                    raise SubmissionFault("estimator_shape_mismatch_%s" % name)
                grads.append(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
            adamw_step(params, grads, state, cfg, step)
            if (step + 1) % cfg.eval_every == 0:
                vl = evaluate(model, cfg, val, device)
                evals.append((step + 1, vl))
                recorder.emit(event="eval", arm=arm, seed=int(seed), step=int(step + 1),
                              val_loss=round(float(vl), 6),
                              train_loss=round(float(loss.detach()), 6),
                              substrate_digest=substrate, estimator_digest=est_digest,
                              backward_calls=int(BACKWARD_CALLS["n"]))
    finally:
        bridge.close()
    recorder.emit(event="arm_end", arm=arm, seed=int(seed), step=int(cfg.max_steps),
                  substrate_digest=substrate, estimator_digest=est_digest,
                  backward_calls=int(BACKWARD_CALLS["n"]),
                  wall_seconds=round(time.time() - started, 3))
    return evals, est_digest


def first_sustained(evals, target):
    """First evaluation step at or below target that is never exceeded afterwards."""
    for i, (step, vl) in enumerate(evals):
        if vl <= target and all(v <= target for _, v in evals[i:]):
            return int(step)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=os.path.join("/workspace", "submission", "estimator.py"))
    ap.add_argument("--telemetry", default=os.path.join("/logs", "verifier", "run_record.jsonl"))
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    # The estimator worker is spawned from this process, and multiprocessing carries
    # the parent's argv into the child's preparation data. Nothing the submission is
    # entitled to know lives in argv, so it is emptied before any worker exists.
    del sys.argv[1:]

    smoke = args.smoke or os.environ.get("BIA_SMOKE") == "1"
    cfg = modelmod.SMOKE if smoke else modelmod.SCALED
    device = torch.device("cpu") if smoke else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if smoke:
        # The smoke point proves the path executes; it must not contend for a shared host.
        torch.set_num_threads(int(os.environ.get("BIA_SMOKE_THREADS", "4")))

    train, val, corpus_digest = datamod.load_corpus(args.bundle, smoke)
    fixture_path = os.path.join(args.bundle, "environment", "fixtures", "noise_process.py")
    substrate = hashlib.sha256(("|".join([
        cfg.digest(),
        corpus_digest,
        datamod.data_digest(train, val),
        noisemod.fixture_digest(fixture_path),
    ])).encode("ascii")).hexdigest()

    rec = Recorder(args.telemetry)
    probe = noisemod.NoiseProcess(torch).probe_digest(PROBE_STEPS)
    rec.emit(event="run_start", arm="none", seed=-1, step=-1, point=cfg.point,
             substrate_digest=substrate, estimator_digest="none",
             noise_probe_digest=probe,
             noise_fixture_sha256=noisemod.fixture_digest(fixture_path),
             harness_tree_sha256=tree_digest(HERE),
             corpus_sha256=corpus_digest, config=cfg.as_dict())

    per_seed = []
    est_digest = "none"
    try:
        try:
            est_digest = _run_seeds(cfg, args, rec, substrate, device, train, val, per_seed)
        except SubmissionFault as fault:
            print("BIA_SUBMISSION_FAULT: %s" % fault, file=sys.stderr)
            return SUBMISSION_FAULT_EXIT
        rec.emit(event="summary", arm="none", seed=-1, step=-1,
                 substrate_digest=substrate, estimator_digest=est_digest,
                 per_seed=per_seed,
                 backward_calls=int(BACKWARD_CALLS["n"]),
                 steps_expected=int(cfg.max_steps * len(cfg.seeds) * 2),
                 true_grad_sentinel_hits=int(_TrueGradSentinel.hits),
                 true_grad_sentinel_names=list(_TrueGradSentinel.names),
                 noise_probe_digest=probe, point=cfg.point)
    finally:
        # The record reaches disk exactly once, after every arm of every seed has
        # finished and every submission worker has been torn down. While a
        # submission is alive there is nothing on disk for it to read.
        rec.flush()
    print(json.dumps({"per_seed": per_seed, "point": cfg.point,
                      "backward_calls": BACKWARD_CALLS["n"]}, sort_keys=True))
    return 0


def _run_seeds(cfg, args, rec, substrate, device, train, val, per_seed):
    for seed in cfg.seeds:
        base_evals, _ = run_arm("baseline", seed, cfg, args.bundle, args.submission, rec,
                                substrate, device, train, val)
        target = None
        for step, vl in base_evals:
            if int(step) == int(cfg.ref_step):
                target = float(vl)
        if target is None:
            target = float(base_evals[-1][1])
        rec.emit(event="target_fixed", arm="baseline", seed=int(seed), step=int(cfg.ref_step),
                 target_loss=round(target, 6), substrate_digest=substrate,
                 estimator_digest="none")
        base_steps = first_sustained(base_evals, target) or int(cfg.ref_step)
        agent_evals, est_digest = run_arm("agent", seed, cfg, args.bundle, args.submission, rec,
                                          substrate, device, train, val)
        agent_steps = first_sustained(agent_evals, target)
        if agent_steps is not None:
            rec.emit(event="crossing", arm="agent", seed=int(seed), step=int(agent_steps),
                     target_loss=round(target, 6), substrate_digest=substrate,
                     estimator_digest=est_digest)
        target_steps = int(math.ceil(int(cfg.ref_step) * float(cfg.target_fraction)))
        per_seed.append({
            "seed": int(seed),
            "target_loss": round(target, 6),
            "baseline_steps": int(base_steps),
            "agent_steps": (int(agent_steps) if agent_steps is not None else None),
            "target_steps": int(target_steps),
            "agent_final_val": round(float(agent_evals[-1][1]), 6),
            "baseline_final_val": round(float(base_evals[-1][1]), 6),
        })

    return est_digest


if __name__ == "__main__":
    raise SystemExit(main())

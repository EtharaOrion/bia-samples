"""Frozen runner for bia S09. The sole author of the frontier telemetry record.

Usage:
  python3 run_frontier.py

Environment:
  BIA_S09_SPEC           path to the frozen spec fixture (required)
  BIA_S09_SUBMISSION     path to the agent's frontier.py (default /workspace/submission/frontier.py)
  BIA_S09_TELEMETRY_DIR  directory the record is written into (default /telemetry)
  BIA_S09_RUN_NONCE      origin stamp the verifier issues for the run it grades

A log written by hand is not evidence, and neither is a number the thing under measurement
computed about itself. This runner never imports the submission. It starts frontier_worker.py
in a separate interpreter, which trains and returns WEIGHTS, and then this process measures
both objectives on those weights with its own model, its own validation shard and its own
anchors. Every graded quantity in the record below, both objectives, both normalized
coordinates, the Pareto front, the hypervolume and the score, is authored here from state this
process observed. The worker reports counters, and the record labels them as worker-counted so
a reader never mistakes a claim for a measurement.

This process also owns the clock. It passes an absolute per-point deadline to the worker and
measures the elapsed time of the whole call from the outside, so compute smuggled into
`build_optimizer` is charged exactly like compute spent in the loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import select
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import engine

WORKER = HERE / "frontier_worker.py"
RUNNER_VERSION = "s09-runner-2"

# Seconds one worker reply may take beyond the per-point deadline before the run is abandoned.
# The deadline itself is the resource bound; this is only the margin that turns a wedged worker
# into a failed run instead of a hung harness.
WORKER_SLACK = float(os.environ.get("BIA_S09_WORKER_SLACK", "300"))

FROZEN_HARNESS_FILES = ("engine.py", "run_frontier.py", "frontier_worker.py")


def harness_digests() -> dict:
    out = {}
    for name in FROZEN_HARNESS_FILES:
        p = HERE / name
        out[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
    return out


class WorkerChannel:
    def __init__(self, spec_path: str, submission: pathlib.Path, scratch: str):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("BIA_S09_RUN_NONCE", None)
        env.pop("BIA_S09_TELEMETRY_DIR", None)
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER), str(spec_path), str(submission)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=str(HERE),
            env=env,
            text=True,
            bufsize=1,
        )
        self.scratch = scratch
        hello = self._recv(time.time() + WORKER_SLACK)
        self.worker_pid = int(hello.get("pid", -1))
        self.protocol = str(hello.get("protocol", ""))

    def _recv(self, until):
        timeout = max(1.0, until - time.time())
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            self.kill()
            raise RuntimeError("frontier worker did not answer inside its deadline")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"frontier worker closed the channel (exit {self.proc.poll()})")
        msg = json.loads(line)
        if not msg.get("ok"):
            raise RuntimeError(str(msg.get("error", "frontier worker refused the call")))
        return msg

    def _call(self, req, until):
        self.proc.stdin.write(json.dumps(req, sort_keys=True) + "\n")
        self.proc.stdin.flush()
        return self._recv(until)

    def propose(self):
        return self._call({"op": "propose"}, time.time() + WORKER_SLACK)

    def train(self, index, deadline):
        return self._call(
            {"op": "train", "index": int(index), "deadline": float(deadline), "scratch": self.scratch},
            deadline + WORKER_SLACK,
        )

    def close(self):
        try:
            self._call({"op": "shutdown"}, time.time() + 30)
        except Exception:  # noqa: BLE001
            pass
        self.kill()

    def kill(self):
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def measure_point(spec, point_index, point_repr, counters, weights_path, val, device, anc,
                  t_start, t_end):
    """Author one run record from weights the worker returned.

    The initial model is rebuilt here from the frozen seed, so `pre` is measured on this
    process's own tensors. `post` is measured on the worker's weights after a strict load, so a
    returned state dict whose key set or shapes differ from the frozen architecture is refused
    rather than silently reshaped.
    """
    torch.manual_seed(spec.run_seed + point_index)
    np.random.seed(spec.run_seed + point_index)
    model = engine.TinyGPT(spec).to(device)
    arch_sig = engine.architecture_signature(spec, model)
    pre = {
        "val_loss": engine.measure_val_loss(model, val, spec, device),
        "density": engine.measure_density(model, spec),
    }
    trained = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(device) for k, v in trained.items()}, strict=True)
    post = {
        "val_loss": engine.measure_val_loss(model, val, spec, device),
        "density": engine.measure_density(model, spec),
    }
    u1, u2 = engine.normalize_point(post["val_loss"], post["density"], anc)
    return {
        "point_index": point_index,
        "order_index": point_index,
        "point_spec_repr": point_repr,
        "architecture_signature": arch_sig,
        "max_steps": spec.max_steps,
        "steps_taken": int(counters["steps_taken"]),
        "stop_reason": counters["stop_reason"],
        "forward_calls": int(counters["forward_calls"]),
        "backward_calls": int(counters["backward_calls"]),
        "counters_counted_by": counters.get("counted_by", "worker"),
        "train_loss_first": counters["train_loss_first"],
        "train_loss_last": counters["train_loss_last"],
        "pre": pre,
        "post": post,
        "objectives": {"val_loss": post["val_loss"], "density": post["density"]},
        "objectives_measured_by": "runner",
        "normalized": [u1, u2],
        "t_start": t_start,
        "t_end": t_end,
        "elapsed_seconds": t_end - t_start,
        "elapsed_measured_by": "runner",
        "device": str(device),
    }


def main() -> int:
    spec_path = os.environ.get("BIA_S09_SPEC")
    if not spec_path:
        print("BIA_S09_SPEC is unset", file=sys.stderr)
        return 2
    sub_path = pathlib.Path(os.environ.get("BIA_S09_SUBMISSION", "/workspace/submission/frontier.py"))
    tele_dir = pathlib.Path(os.environ.get("BIA_S09_TELEMETRY_DIR", "/telemetry"))
    tele_dir.mkdir(parents=True, exist_ok=True)
    record_path = tele_dir / "frontier_record.json"

    spec = engine.load_spec(spec_path)
    device = engine.select_device(spec)
    torch.manual_seed(spec.run_seed)
    np.random.seed(spec.run_seed)

    t_overhead = time.time()
    succ, q = engine.build_source(spec)
    entropy_rate = engine.source_entropy_rate(spec, succ, q)
    train_np = engine.generate_tokens(spec, succ, q, spec.train_tokens, spec.source_seed + 1)
    val_np = engine.generate_tokens(spec, succ, q, spec.val_tokens, spec.source_seed + 2)
    digest = engine.data_digest(train_np, val_np)
    val = torch.from_numpy(val_np)
    reference = engine.untrained_reference(spec, val, device)
    anc = engine.anchors(spec, entropy_rate, reference)
    overhead_seconds = time.time() - t_overhead

    record = {
        "schema": "bia.s09.frontier_record/v1",
        "runner_version": RUNNER_VERSION,
        "run_nonce": os.environ.get("BIA_S09_RUN_NONCE", ""),
        "graded_submission_path": str(sub_path),
        "graded_submission_sha256": (
            hashlib.sha256(sub_path.read_bytes()).hexdigest() if sub_path.is_file() else ""
        ),
        "harness_digests": harness_digests(),
        "spec_id": spec.spec_id,
        "spec_digest": engine.spec_digest(spec),
        "spec_vocab_size": spec.vocab_size,
        "data_digest": digest,
        "device": str(device),
        "declared_frontier_points": spec.frontier_points,
        "anchors": anc,
        "untrained_reference": reference,
        "reference_point_raw": [anc["loss_ref"], anc["density_ref"]],
        "ideal_point_raw": [anc["loss_ideal"], anc["density_ideal"]],
        "reference_point_normalized": [0.0, 0.0],
        "ideal_point_normalized": [1.0, 1.0],
        "shared_overhead_seconds": overhead_seconds,
        "submission_isolation": {
            "mode": "subprocess",
            "worker": WORKER.name,
            "runner_pid": os.getpid(),
            "worker_pid": None,
        },
        "runs": [],
        "manifest": ["frontier_record.json"],
    }

    scratch = tempfile.mkdtemp(prefix="bia_s09_worker_")
    channel = None
    try:
        channel = WorkerChannel(spec_path, sub_path, scratch)
        record["submission_isolation"]["worker_pid"] = channel.worker_pid
        record["submission_isolation"]["protocol"] = channel.protocol
        proposed = channel.propose()
        record["proposed_points"] = int(proposed["count"])
        if record["proposed_points"] != spec.frontier_points:
            raise ValueError(
                f"propose_frontier returned {record['proposed_points']} points, "
                f"the frozen spec requires exactly {spec.frontier_points}"
            )
        for i in range(spec.frontier_points):
            t_start = time.time()
            counters = channel.train(i, t_start + spec.per_run_seconds)
            weights = os.path.join(scratch, counters["weights_file"])
            t_end = time.time()
            record["runs"].append(
                measure_point(spec, i, proposed["point_reprs"][i], counters, weights, val,
                              device, anc, t_start, t_end)
            )
    except Exception:
        record["fault"] = traceback.format_exc()
    finally:
        if channel is not None:
            channel.close()
        shutil.rmtree(scratch, ignore_errors=True)

    raw_points = [(r["objectives"]["val_loss"], r["objectives"]["density"]) for r in record["runs"]]
    scored = engine.score_frontier(raw_points, anc) if raw_points else {
        "normalized_points": [],
        "pareto_front": [],
        "hypervolume_raw": 0.0,
        "hypervolume_front": 0.0,
        "score": 0.0,
    }
    record.update(scored)
    record["total_run_seconds"] = sum(r["elapsed_seconds"] for r in record["runs"])
    record["sealed_at"] = time.time()
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"spec               {record['spec_id']}  digest {record['spec_digest'][:16]}")
    print(f"device             {record['device']}")
    print(f"isolation          worker pid {record['submission_isolation']['worker_pid']} "
          f"runner pid {record['submission_isolation']['runner_pid']}")
    print(f"entropy rate       {anc['entropy_rate']:.6f} nats")
    print(f"loss anchors       ref {anc['loss_ref']:.6f}  ideal {anc['loss_ideal']:.6f}")
    print(f"density anchors    ref {anc['density_ref']:.6f}  ideal {anc['density_ideal']:.6f}")
    print(f"reference point    raw ({anc['loss_ref']:.6f}, {anc['density_ref']:.6f})  normalized (0.0, 0.0)")
    print(f"ideal point        raw ({anc['loss_ideal']:.6f}, {anc['density_ideal']:.6f})  normalized (1.0, 1.0)")
    print(f"shared overhead    {overhead_seconds:.2f} s")
    print(f"total run seconds  {record['total_run_seconds']:.2f} s")
    for r in record["runs"]:
        print(
            f"  point {r['point_index']}  steps {r['steps_taken']:>5}/{r['max_steps']}"
            f"  stop {r['stop_reason']:<18}"
            f"  val_loss {r['objectives']['val_loss']:.5f}"
            f"  density {r['objectives']['density']:.5f}"
            f"  u ({r['normalized'][0]:.5f}, {r['normalized'][1]:.5f})"
            f"  {r['elapsed_seconds']:.1f}s"
        )
    print(f"pareto front size  {len(record['pareto_front'])}")
    print(f"hypervolume        {record['hypervolume_raw']:.9f}")
    print(f"score              {record['score']:.9f}")
    print(f"record             {record_path}")
    if "fault" in record:
        print("FAULT during the frontier run:", file=sys.stderr)
        print(record["fault"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

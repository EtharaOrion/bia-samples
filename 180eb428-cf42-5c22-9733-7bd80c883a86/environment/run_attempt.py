"""FROZEN. Entry point for one graded attempt of bia slot S07.

Runs the three phases of an attempt in the declared order, writes telemetry, computes the
reward, and exits. Usage:

    python3 run_attempt.py --submission submission/precision_policy.py --out /workspace/artifacts

Set BIA_SMOKE=1 to execute the identical code path at the smoke operating point on CPU with
no CUDA. Set BIA_NO_CACHE=1 to force both control phases to re-run.

ISOLATION CONTRACT. This process measures. It never imports the submission. Every submitted
policy, and the frozen baseline recipe too, is hosted by a separate `policy_worker.py`
process and reached only through `PolicyChannel`, whose inbound alphabet is one int, one
plain dict, and two closed string sets. The clock, the RNG seeds, the overflow ledger, every
telemetry record, the replay-step choice and the reward arithmetic all live on this side of
that channel, so a submission has no reachable handle on any of them. The three earlier ways
a co-resident submission reached the measurement are closed by construction rather than by
detection: it cannot rebind a harness function, because it is not in this interpreter; it
cannot read an anchor written before it loads, because the control cache is now written after
the graded phase ends; and it cannot learn which reduction steps are replayed, because this
process draws them from its own system entropy after the worker has already started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import select
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import bia_numerics as N  # noqa: E402
import harness as H  # noqa: E402

WORKER = HERE / "policy_worker.py"
BASELINE_RECIPE = HERE / "baseline_recipe.py"

# Seconds one worker call may take before the attempt is abandoned. A policy is a few
# arithmetic decisions per step, so this is three orders of magnitude of headroom; it exists so
# a submission that blocks forever becomes a graded failure instead of a hung grader.
POLICY_CALL_TIMEOUT = float(os.environ.get("BIA_POLICY_TIMEOUT", "120"))

REPLAY_SAMPLES = 8


class PolicyError(RuntimeError):
    """Raised when the hosted policy misbehaves. Always a graded outcome, never a crash."""


class PolicyChannel:
    """Parent-side handle on one out-of-process policy.

    Every method returns a value drawn from the closed outbound alphabet the worker docstring
    names, and re-checks it on arrival. The re-check is deliberate duplication: the worker is
    running submitted bytes, so nothing it says about its own conformance is evidence.
    """

    def __init__(self, submission: pathlib.Path, config: Dict[str, Any]) -> None:
        self.submission = pathlib.Path(submission).resolve()
        env = {k: v for k, v in os.environ.items() if not k.startswith("BIA_RUN_")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(WORKER), str(self.submission)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(HERE),
            env=env,
            text=True,
            bufsize=1,
        )
        hello = self._recv()
        self.worker_pid = int(hello.get("pid", -1))
        self.protocol = str(hello.get("protocol", ""))
        built = self._call({"op": "build", "config": _jsonable(config)})
        dt = str(built.get("master_dtype", "float32"))
        if dt not in ("float32", "bfloat16"):
            raise PolicyError("master_dtype must be float32 or bfloat16, got %r" % dt)
        self.master_dtype = dt

    # -- protocol ---------------------------------------------------------

    def _recv(self) -> Dict[str, Any]:
        assert self.proc.stdout is not None
        ready, _, _ = select.select([self.proc.stdout], [], [], POLICY_CALL_TIMEOUT)
        if not ready:
            self.kill()
            raise PolicyError("policy worker did not answer within %.0fs" % POLICY_CALL_TIMEOUT)
        line = self.proc.stdout.readline()
        if not line:
            raise PolicyError("policy worker closed the channel (exit %s)" % self.proc.poll())
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PolicyError("policy worker wrote a non-protocol line: %s" % exc) from None
        if not isinstance(msg, dict) or not msg.get("ok"):
            raise PolicyError(str((msg or {}).get("error", "policy worker refused the call")))
        return msg

    def _call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(req, sort_keys=True) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise PolicyError("policy worker is gone: %s" % exc) from None
        return self._recv()

    # -- the surface harness.train_one consumes ---------------------------

    def loss_scale(self, step: int) -> int:
        v = self._call({"op": "loss_scale", "step": int(step)})["value"]
        if isinstance(v, bool) or not isinstance(v, int):
            raise PolicyError("loss_scale must return an int, got %r" % (v,))
        return int(v)

    def plan(self, step: int, n_chunks: int, stats: List[Dict[str, float]]) -> Dict[str, Any]:
        v = self._call(
            {"op": "plan", "step": int(step), "n_chunks": int(n_chunks), "stats": _jsonable(stats)}
        )["value"]
        if not isinstance(v, dict):
            raise PolicyError("plan must return a dict, got %r" % type(v).__name__)
        return v

    def on_overflow(self, step: int, info: Dict[str, Any]) -> str:
        v = self._call({"op": "on_overflow", "step": int(step), "info": _jsonable(info)})["value"]
        if v not in ("skip", "continue"):
            raise PolicyError("on_overflow must return skip or continue, got %r" % (v,))
        return str(v)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        try:
            self._call({"op": "shutdown"})
        except (PolicyError, OSError):
            pass
        self.kill()

    def kill(self) -> None:
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


def _jsonable(obj: Any) -> Any:
    """Reduce a value to plain JSON types before it crosses the channel."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    return str(obj)


def sha256_file(p: pathlib.Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return ""


def reward_from(baseline: float, target: float, agent: float, overflow_events: int) -> Dict[str, Any]:
    """Lower is better, so raw is the fraction of the baseline to target distance covered.

    The gate is separate from the scale: any overflow event in the agent phase zeroes the
    reward with a machine readable reason, because the objective is reaching the target loss
    WITHOUT overflow, not reaching it at any cost.
    """
    if overflow_events > 0:
        return {"score": 0.0, "raw": 0.0, "reason": "overflow_in_agent_phase", "overflow_events": overflow_events}
    span = baseline - target
    if not (span > 0.0):
        return {"score": 0.0, "raw": 0.0, "reason": "degenerate_anchor_span"}
    if agent != agent:
        return {"score": 0.0, "raw": 0.0, "reason": "agent_loss_not_finite"}
    raw = (baseline - agent) / span
    return {"score": min(max(raw, 0.0), 1.0), "raw": raw, "reason": "scored"}


def choose_replay_steps(steps: int) -> List[int]:
    """Draw the replayed reduction steps from this process's own system entropy.

    The set is drawn after the worker is running and is never written anywhere the worker can
    reach, so a policy cannot be well behaved at exactly the replayed indices and free at the
    rest. Before this, the single replayed index was a fixed position in the schedule.
    """
    k = min(REPLAY_SAMPLES, steps)
    return sorted(random.SystemRandom().sample(range(steps), k))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=str(HERE / "submission" / "precision_policy.py"))
    ap.add_argument("--out", default="/workspace/artifacts")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    submission = pathlib.Path(args.submission).resolve()

    cfg = H.resolve_config()
    if cfg["mode"] == "smoke":
        # The smoke path is many small elementwise kernels, where a large default thread pool
        # loses to oversubscription rather than winning. Capping keeps the CPU proof
        # predictable on a shared host instead of swinging by an order of magnitude with load.
        torch.set_num_threads(int(os.environ.get("BIA_SMOKE_THREADS", "4")))
    if cfg["device"] == "cpu" and cfg["mode"] == "full":
        print("WARNING: no CUDA device visible, the full operating point will be slow on CPU", file=sys.stderr)
    digest = H.config_digest(cfg)
    corpus = H.ByteCorpus(cfg)
    replay_steps = choose_replay_steps(int(cfg["steps"]))

    telemetry: List[Dict[str, Any]] = [
        {
            "seq": 0,
            "record": "attempt_begin",
            "mode": cfg["mode"],
            "device": cfg["device"],
            "config_digest": digest,
            "corpus_sha256": corpus.full_sha256,
            "format": N.format_summary(),
            "phase_order": ["fp32_control", "standard_control", "agent_run"],
            "run_nonce": os.environ.get("BIA_RUN_NONCE", ""),
            "graded_submission_path": str(submission),
            "graded_submission_sha256": sha256_file(submission),
            "baseline_recipe_sha256": sha256_file(BASELINE_RECIPE),
            "replay_steps_requested": replay_steps,
            "policy_isolation": {
                "mode": "subprocess",
                "worker": WORKER.name,
                "driver_pid": os.getpid(),
                "worker_pids": {},
            },
            "t": time.time(),
        }
    ]
    isolation = telemetry[0]["policy_isolation"]

    cache_path = out / ("controls_%s.json" % digest[:16])
    use_cache = os.environ.get("BIA_NO_CACHE", "0") != "1" and cache_path.is_file()

    t0 = time.time()
    channels: List[PolicyChannel] = []
    try:
        if use_cache:
            cached = json.loads(cache_path.read_text())
            fp32 = cached["fp32_control"]
            std = cached["standard_control"]
            for rec in (fp32, std):
                rec = dict(rec)
                rec["record"] = "phase_end"
                rec["seq"] = len(telemetry)
                rec["from_cache"] = True
                telemetry.append(rec)
        else:
            fp32 = H.train_one(cfg, corpus, "fp32_control", None, telemetry, replay_steps)
            base = PolicyChannel(BASELINE_RECIPE, cfg)
            channels.append(base)
            isolation["worker_pids"]["standard_control"] = base.worker_pid
            std = H.train_one(cfg, corpus, "standard_control", base, telemetry, replay_steps)

        sub = PolicyChannel(submission, cfg)
        channels.append(sub)
        isolation["worker_pids"]["agent_run"] = sub.worker_pid
        agent = H.train_one(cfg, corpus, "agent_run", sub, telemetry, replay_steps)
    except (PolicyError, N.PlanError, ValueError) as exc:
        # A misbehaving policy is a graded zero with a machine readable reason. It never
        # returns an unscored run, because an unscored run reads as an infrastructure fault.
        scored = {"score": 0.0, "raw": 0.0, "reason": "policy_rejected", "detail": str(exc)}
        scored.update({"mode": cfg["mode"], "config_digest": digest, "wallclock_seconds": round(time.time() - t0, 3)})
        telemetry.append({"seq": len(telemetry), "record": "attempt_end", **scored})
        (out / "telemetry.json").write_text(json.dumps(telemetry, indent=1, sort_keys=True))
        (out / "score.json").write_text(json.dumps(scored, indent=1, sort_keys=True))
        print(json.dumps(scored, indent=1, sort_keys=True))
        return 0
    finally:
        for ch in channels:
            ch.close()

    scored = reward_from(
        baseline=std["final_val_loss"],
        target=fp32["final_val_loss"],
        agent=agent["final_val_loss"],
        overflow_events=agent["overflow_event_count"],
    )
    scored.update(
        {
            "baseline_metric": std["final_val_loss"],
            "target_metric": fp32["final_val_loss"],
            "agent_metric": agent["final_val_loss"],
            "agent_overflow_events": agent["overflow_event_count"],
            "agent_overflow_elements": agent["overflow_elements"],
            "standard_control_overflow_events": std["overflow_event_count"],
            "mode": cfg["mode"],
            "config_digest": digest,
            "wallclock_seconds": round(time.time() - t0, 3),
        }
    )

    telemetry.append({"seq": len(telemetry), "record": "attempt_end", **scored})
    (out / "telemetry.json").write_text(json.dumps(telemetry, indent=1, sort_keys=True))
    (out / "score.json").write_text(json.dumps(scored, indent=1, sort_keys=True))
    # The anchor cache is written LAST, after the graded phase has already been measured, so
    # within an attempt no anchor value exists on disk at the moment the policy is loaded.
    if not use_cache:
        cache_path.write_text(
            json.dumps({"fp32_control": fp32, "standard_control": std}, indent=1, sort_keys=True)
        )
    print(json.dumps(scored, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

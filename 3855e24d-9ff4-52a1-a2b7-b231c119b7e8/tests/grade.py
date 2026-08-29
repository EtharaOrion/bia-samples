"""Verifier for bia slot S06, gradient estimator under injected noise.

The verifier owns the measurement, and owns it all the way down. It runs its OWN
private copy of the frozen harness, at tests/private_bundle, against its OWN
private copy of the corpus and the noise fixture, in a working directory the
solving container never sees. That harness in turn runs the submitted estimator
in a separate interpreter behind a shared buffer pair, so the submission is two
process boundaries away from the run record and holds none of the state the
reward is derived from. The record is buffered in memory by the harness and
written once, after every arm has finished, so no graded quantity is ever on disk
while a submission is alive.

The agent visible copies of the harness and the fixtures remain in the bundle
because the solver has to be able to read what it is being measured against.
They are read here at grade time, digested, and compared against the private
copies. Editing them changes nothing about the graded run and costs the whole
score, which is the two sided shape this check needs to have.

Every graded assertion is one checker. Each checker reduces to exactly one of
VALUE, EFFECT, ABSENCE, INVARIANT, ORDERING or DIVERGENCE, and each names the
live state read it traces to. The declarative roster is tests/checkers.yaml and
the two must agree, which tests/test_output.py asserts.

Every zero carries a machine readable reason. A checker failure is a hard gate:
the score is exactly 0.0 and the reason names the checker that closed it.

Reason codes are lowercase kebab so a grader downstream branches on them rather
than parses them. The code a checker emits when it closes the score is the kebab
spelling of that checker's id, which is the zero_reason tests/checkers.yaml
declares for it; the finer grained cause the checker returned is preserved under
detail.checker_detail rather than folded into the code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PRIVATE_BUNDLE = os.path.join(TESTS_DIR, "private_bundle")
PRIVATE_HARNESS = os.path.join(PRIVATE_BUNDLE, "environment", "harness")
PRIVATE_FIXTURES = os.path.join(PRIVATE_BUNDLE, "environment", "fixtures")
# Mirrors SUBMISSION_FAULT_EXIT in the harness. A fault the submitted estimator caused
# is a solver failure with its own reason code, never an infrastructure failure.
SUBMISSION_FAULT_EXIT = 7


def resolve_bundle():
    env = os.environ.get("BIA_BUNDLE")
    if env and os.path.isdir(os.path.join(env, "environment")):
        return env
    for cand in ("/workspace", os.path.dirname(TESTS_DIR)):
        if os.path.isdir(os.path.join(cand, "environment", "harness")):
            return cand
    return os.path.dirname(TESTS_DIR)


BUNDLE = resolve_bundle()
LOG_DIR = os.environ.get("BIA_LOG_DIR", "/logs/verifier")
SUBMISSION = os.environ.get("BIA_SUBMISSION", "/workspace/submission/estimator.py")
TELEMETRY = os.path.join(LOG_DIR, "run_record.jsonl")
SCORE_PATH = os.environ.get("BIA_SCORE_PATH", os.path.join(LOG_DIR, "score.json"))
OUTCOMES_PATH = os.environ.get("BIA_OUTCOMES_PATH", os.path.join(LOG_DIR, "outcomes.json"))
SMOKE = os.environ.get("BIA_SMOKE") == "1"
# The runtime path Harbor mounts for the single reward float. Bound by the
# delivery contract and declared as reward_path in tests/checkers.yaml.
REWARD_PATH = os.environ.get("BIA_REWARD_PATH", "/logs/verifier/reward.txt")

CHECKER_IDS = [
    "substrate_digest_constant",
    "frozen_fixture_bytes_match_bound",
    "one_backward_per_step",
    "no_true_gradient_access",
    "noise_realization_reproduces",
    "arm_order_and_sustained_crossing",
    "submission_bound_to_run",
    "score_matches_telemetry",
]

ZERO_REASONS = {
    "substrate_digest_constant": "substrate-digest-constant",
    "frozen_fixture_bytes_match_bound": "frozen-fixture-bytes-match-bound",
    "one_backward_per_step": "one-backward-per-step",
    "no_true_gradient_access": "no-true-gradient-access",
    "noise_realization_reproduces": "noise-realization-reproduces",
    "arm_order_and_sustained_crossing": "arm-order-and-sustained-crossing",
    "submission_bound_to_run": "submission-bound-to-run",
    "score_matches_telemetry": "score-matches-telemetry",
}


def agent_tree(relative):
    """Digest the agent visible tree, or a sentinel when it is not mounted here."""
    root = os.path.join(BUNDLE, relative)
    if not os.path.isdir(root):
        return "agent-tree-not-mounted"
    return tree_digest(root)


def tree_digest(root):
    """sha256 over a directory tree, the same recipe the harness records for its own."""
    parts = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".pyc"):
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            parts.append(rel + "\0" + file_sha(full))
    return hashlib.sha256("\n".join(parts).encode("ascii")).hexdigest()


def load_bound():
    """Read the bound constants out of tests/checkers.yaml without a yaml dependency."""
    path = os.path.join(TESTS_DIR, "checkers.yaml")
    bound = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#") or ":" not in s:
                continue
            key, _, val = s.partition(":")
            key = key.strip().lstrip("- ").strip()
            val = val.strip().strip('"')
            if key in ("noise_fixture_sha256", "corpus_sha256", "identity_estimator_digest"):
                bound[key] = val
    return bound


def emit(score, reason, detail, outcomes):
    os.makedirs(os.path.dirname(SCORE_PATH) or ".", exist_ok=True)
    payload = {"score": float(score), "reason": reason, "detail": detail or {}}
    with open(SCORE_PATH, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    with open(OUTCOMES_PATH, "w") as fh:
        json.dump(outcomes, fh, indent=1, sort_keys=True)
    os.makedirs(os.path.dirname(REWARD_PATH) or ".", exist_ok=True)
    with open(REWARD_PATH, "w") as fh:
        fh.write(repr(float(score)))
    print(json.dumps(payload, sort_keys=True))
    return float(score)


def run_harness(work_dir):
    """The verifier runs its own copy of the training, in its own working directory.

    The environment handed to the run is built here rather than inherited, so nothing
    the solving container exported reaches the graded run, and the telemetry path lies
    inside a private directory that does not exist in the agent's filesystem.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    telemetry = os.path.join(work_dir, "run_record.jsonl")
    cmd = [sys.executable, "-E", "-s", "-B",
           os.path.join(PRIVATE_HARNESS, "train.py"),
           "--submission", SUBMISSION, "--telemetry", telemetry,
           "--bundle", PRIVATE_BUNDLE]
    if SMOKE:
        cmd.append("--smoke")
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": work_dir,
        "TMPDIR": work_dir,
        "LANG": "C.UTF-8",
    }
    for passthrough in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "BIA_SMOKE_THREADS",
                        "NVIDIA_VISIBLE_DEVICES"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    if SMOKE:
        env["BIA_SMOKE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=work_dir)
    with open(os.path.join(LOG_DIR, "harness-stdout.txt"), "w") as fh:
        fh.write(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    return proc.returncode, telemetry, proc.stderr


def read_records(path):
    if not os.path.exists(path):
        return None, "telemetry-absent"
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                return None, "telemetry-malformed"
    if not recs:
        return None, "telemetry-empty"
    return recs, None


def file_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# --- checkers -----------------------------------------------------------------

def check_substrate_digest_constant(recs, bound, ctx):
    """INVARIANT. Live state read: the substrate_digest field of every run record line."""
    seen = {r.get("substrate_digest") for r in recs}
    if len(seen) != 1 or None in seen:
        return False, "substrate_digest_moved_during_run"
    return True, None


def check_frozen_fixture_bytes_match_bound(recs, bound, ctx):
    """VALUE. Live state read: noise_fixture_sha256, corpus_sha256 and harness_tree_sha256
    on the run_start record, plus the digests the verifier recomputes at grade time over
    the AGENT VISIBLE harness and fixture trees on disk.

    Four comparisons, and the last two are what make this two sided. The graded run
    executes the verifier's private copies, so an edit inside the solving container
    changes nothing about the measurement; the run would be clean and the edit invisible
    if this checker only read what the run recorded. It also reads the agent visible
    trees directly, so the edit is observable and costs the score.
    """
    start = next((r for r in recs if r.get("event") == "run_start"), None)
    if start is None:
        return False, "run_start_record_absent"
    if start.get("noise_fixture_sha256") != bound.get("noise_fixture_sha256"):
        return False, "noise_fixture_bytes_differ_from_bound"
    if start.get("corpus_sha256") != bound.get("corpus_sha256"):
        return False, "corpus_bytes_differ_from_bound"
    if start.get("harness_tree_sha256") != ctx.get("private_harness_tree"):
        return False, "graded_run_did_not_execute_the_verifier_harness"
    if ctx.get("agent_harness_tree") != ctx.get("private_harness_tree"):
        return False, "agent_visible_harness_tree_edited"
    if ctx.get("agent_fixture_tree") != ctx.get("private_fixture_tree"):
        return False, "agent_visible_fixture_tree_edited"
    return True, None


def check_one_backward_per_step(recs, bound, ctx):
    """VALUE. Live state read: the backward_calls counter and steps_expected on the summary record."""
    s = ctx.get("summary")
    if s is None:
        return False, "summary_record_absent"
    if int(s.get("backward_calls", -1)) != int(s.get("steps_expected", -2)):
        return False, "backward_calls_%s_not_equal_steps_%s" % (
            s.get("backward_calls"), s.get("steps_expected"))
    return True, None


def check_no_true_gradient_access(recs, bound, ctx):
    """ABSENCE. Live state read: true_grad_sentinel_hits and true_grad_sentinel_names on the summary."""
    s = ctx.get("summary")
    if s is None:
        return False, "summary_record_absent"
    if int(s.get("true_grad_sentinel_hits", 1)) != 0:
        return False, "true_gradient_sentinel_touched"
    if list(s.get("true_grad_sentinel_names", ["x"])):
        return False, "true_gradient_sentinel_named_reads_present"
    return True, None


def check_noise_realization_reproduces(recs, bound, ctx):
    """DIVERGENCE. Live state read: noise_probe_digest recorded by the run, against the
    verifier's own independent recomputation from the fixture it imports itself."""
    start = next((r for r in recs if r.get("event") == "run_start"), None)
    if start is None:
        return False, "run_start_record_absent"
    fixture = os.path.join(PRIVATE_FIXTURES, "noise_process.py")
    spec = importlib.util.spec_from_file_location("bia_verifier_noise", fixture)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch
    steps = (0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 30, 63, 64, 96, 97)
    mine = mod.NoiseProcess(torch).probe_digest(steps)
    theirs = start.get("noise_probe_digest")
    if mine != theirs:
        return False, "noise_realization_diverged"
    return True, None


def check_arm_order_and_sustained_crossing(recs, bound, ctx):
    """ORDERING. Live state read: the ordered event stream of the run record."""
    order = [(r.get("seq"), r.get("event"), r.get("arm"), r.get("seed")) for r in recs]
    if not order or order[0][1] != "run_start":
        return False, "run_start_not_first"
    per_seed = {}
    for seq, event, arm, seed in order:
        if event in ("arm_start", "target_fixed", "arm_end", "crossing"):
            per_seed.setdefault(seed, []).append((seq, event, arm))
    for seed, ev in per_seed.items():
        if seed == -1:
            continue
        names = [(e, a) for _, e, a in ev]
        want_head = [("arm_start", "baseline"), ("arm_end", "baseline"),
                     ("target_fixed", "baseline"), ("arm_start", "agent")]
        if names[:4] != want_head:
            return False, "arm_event_order_wrong_for_seed_%s" % seed
    for seed, block in ctx.get("per_seed_map", {}).items():
        agent = block.get("agent_steps")
        if agent is None:
            continue
        evs = [r for r in recs if r.get("event") == "eval" and r.get("arm") == "agent"
               and r.get("seed") == seed and int(r.get("step", 0)) >= int(agent)]
        if any(float(r["val_loss"]) > float(block["target_loss"]) + 1e-9 for r in evs):
            return False, "crossing_not_sustained_for_seed_%s" % seed
    return True, None


def check_submission_bound_to_run(recs, bound, ctx):
    """EFFECT. Live state read: the estimator_digest the harness recorded on every agent arm
    record, which the act of installing the submission changed away from the frozen identity."""
    if not os.path.exists(SUBMISSION):
        return False, "submission_absent"
    want = file_sha(SUBMISSION)
    agent = [r for r in recs if r.get("arm") == "agent"]
    if not agent:
        return False, "no_agent_arm_records"
    for r in agent:
        if r.get("estimator_digest") != want:
            return False, "agent_record_not_bound_to_submission"
    if want == bound.get("identity_estimator_digest"):
        return False, "submission_is_the_frozen_identity_estimator"
    return True, None


def check_score_matches_telemetry(recs, bound, ctx):
    """VALUE. Live state read: the per_seed block of the summary record, recomputed
    independently by the verifier and compared with the score the verifier emits."""
    s = ctx.get("summary")
    if s is None:
        return False, "summary_record_absent"
    raw, _ = raw_from_summary(s)
    if raw is None:
        return False, "per_seed_block_unusable"
    recomputed = min(max(raw, 0.0), 1.0)
    if not math.isfinite(recomputed):
        return False, "score_not_finite"
    ctx["recomputed_score"] = recomputed
    if abs(recomputed - ctx["score_candidate"]) > 1e-9:
        return False, "emitted_score_diverges_from_telemetry"
    return True, None


CHECKERS = {
    "substrate_digest_constant": check_substrate_digest_constant,
    "frozen_fixture_bytes_match_bound": check_frozen_fixture_bytes_match_bound,
    "one_backward_per_step": check_one_backward_per_step,
    "no_true_gradient_access": check_no_true_gradient_access,
    "noise_realization_reproduces": check_noise_realization_reproduces,
    "arm_order_and_sustained_crossing": check_arm_order_and_sustained_crossing,
    "submission_bound_to_run": check_submission_bound_to_run,
    "score_matches_telemetry": check_score_matches_telemetry,
}


def raw_from_summary(summary):
    """Reward, exactly as instruction.md states it. Steps to target under the frozen noise."""
    blocks = summary.get("per_seed") or []
    if not blocks:
        return None, "per-seed-absent"
    parts = []
    for b in blocks:
        base = float(b["baseline_steps"])
        tgt = float(b["target_steps"])
        agent = b["agent_steps"]
        span = base - tgt
        if span <= 0:
            return None, "degenerate-span"
        if agent is None:
            parts.append(0.0)
        else:
            parts.append((base - float(agent)) / span)
    return sum(parts) / len(parts), None


def main():
    outcomes = {cid: False for cid in CHECKER_IDS}
    bound = load_bound()

    for required in (PRIVATE_HARNESS, PRIVATE_FIXTURES):
        if not os.path.isdir(required):
            return emit(0.0, "verifier-private-harness-absent", {"path": required}, outcomes)

    work_dir = tempfile.mkdtemp(prefix="bia-s06-verifier-")
    os.chmod(work_dir, 0o700)
    try:
        rc, telemetry, err = run_harness(work_dir)
        if rc == SUBMISSION_FAULT_EXIT:
            fault = next((ln.split(": ", 1)[-1] for ln in (err or "").splitlines()
                          if ln.startswith("BIA_SUBMISSION_FAULT")), "unattributed")
            return emit(0.0, "submission-estimator-faulted", {"fault": fault}, outcomes)
        if rc != 0:
            return emit(0.0, "harness-run-failed-rc-%d" % rc,
                        {"stderr_tail": (err or "")[-800:]}, outcomes)

        recs, terr = read_records(telemetry)
        if recs is None:
            return emit(0.0, terr, {}, outcomes)

        summary = next((r for r in recs if r.get("event") == "summary"), None)
        ctx = {"summary": summary}
        ctx["per_seed_map"] = {b["seed"]: b for b in ((summary or {}).get("per_seed") or [])}
        ctx["private_harness_tree"] = tree_digest(PRIVATE_HARNESS)
        ctx["private_fixture_tree"] = tree_digest(PRIVATE_FIXTURES)
        ctx["agent_harness_tree"] = agent_tree(os.path.join("environment", "harness"))
        ctx["agent_fixture_tree"] = agent_tree(os.path.join("environment", "fixtures"))
        raw, rerr = raw_from_summary(summary or {})
        ctx["score_candidate"] = min(max(raw, 0.0), 1.0) if raw is not None else 0.0

        first_failure = None
        for cid in CHECKER_IDS:
            ok, why = CHECKERS[cid](recs, bound, ctx)
            outcomes[cid] = bool(ok)
            if not ok and first_failure is None:
                first_failure = (cid, why)

        if first_failure is not None:
            cid, why = first_failure
            return emit(0.0, ZERO_REASONS[cid], {"checker": cid, "checker_detail": why}, outcomes)
        if raw is None:
            return emit(0.0, rerr or "reward-underivable", {}, outcomes)

        score = min(max(raw, 0.0), 1.0)
        return emit(score, "graded", {
            "raw": round(float(raw), 6),
            "point": summary.get("point"),
            "per_seed": summary.get("per_seed"),
        }, outcomes)
    finally:
        try:
            src = os.path.join(work_dir, "run_record.jsonl")
            if os.path.exists(src):
                os.makedirs(LOG_DIR, exist_ok=True)
                shutil.copyfile(src, TELEMETRY)
        except OSError:
            pass
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verifier entry point for BIA slot S04.

ISOLATION MODEL. The verifier owns the measurement end to end. It mints a fresh
chain key for this grading invocation only, creates a private working directory
the solving container never sees, executes its OWN private copy of the runner at
tests/runner/run_arch.py against its OWN frozen recipe tree at tests/frozen, and
grades only the telemetry that run produced. No telemetry written outside this
process is ever an input to the reward, no key is shared with the agent, and no
number in this file is taken from the agent's own report.

That is the whole point of the layout. The graded artifact is six integers of
JSON data, it never becomes code inside this interpreter, and the process that
produces the measurement is spawned by the verifier with an environment the
verifier builds from scratch. There is no path by which a submission reaches the
telemetry, the clock, the key, or the reward.

Reason codes are lowercase kebab, so a grader downstream branches on them rather
than reads them. The reason a checker emits when it scores zero is the kebab
spelling of that checker's id, which is the same identity tests/checkers.yaml
declares as its zero_reason and the same identity the outcome map keys on.

run_checkers below is a pure function of the record list plus a context of frozen
bytes. It reads no clock, no environment, no random source and no network, which
is what lets the private adequacy harness replay it over frozen fixtures and
prove both halves of every checker's decision.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkers import recompute as R

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.dirname(TESTS_DIR)
PRIVATE_RUNNER = os.path.join(TESTS_DIR, "runner", "run_arch.py")
SCORE_PATH = os.environ.get("SCORE_PATH", "/logs/verifier/score.json")
OUTCOMES_PATH = os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json")
# The runtime path Harbor mounts for the single reward float. Bound by the
# delivery contract and declared as reward_path in tests/checkers.yaml.
REWARD_PATH = os.environ.get("BIA_REWARD_PATH", "/logs/verifier/reward.txt")
# Where the verifier retains the record set it produced, as evidence for a
# reviewer. It is written AFTER grading and is never read back as an input.
RECORD_EVIDENCE = os.environ.get(
    "BIA_RECORD_EVIDENCE", os.path.join(os.path.dirname(SCORE_PATH) or ".", "run_record.jsonl")
)
SUBMISSION = os.environ.get("BIA_SUBMISSION", "/workspace/submission/arch.json")
PROFILE = "smoke" if os.environ.get("BIA_SMOKE") == "1" else "full"

CHECKER_IDS = [
    "search_space_respected",
    "param_budget_met",
    "param_count_agreement",
    "frozen_surface_unchanged",
    "no_eval_bytes_in_training_stream",
    "no_frozen_recipe_tree_edit",
    "graded_event_order_held",
    "submission_run_mutated_telemetry",
    "advantage_held_across_run",
]

# The machine-readable code each checker attributes its zero to, spelled out as
# literals rather than derived, so tests/checkers.yaml can be validated against
# the bytes of this carrier rather than against a rule for producing them.
ZERO_REASONS = {
    "search_space_respected": "search-space-respected",
    "param_budget_met": "param-budget-met",
    "param_count_agreement": "param-count-agreement",
    "frozen_surface_unchanged": "frozen-surface-unchanged",
    "no_eval_bytes_in_training_stream": "no-eval-bytes-in-training-stream",
    "no_frozen_recipe_tree_edit": "no-frozen-recipe-tree-edit",
    "graded_event_order_held": "graded-event-order-held",
    "submission_run_mutated_telemetry": "submission-run-mutated-telemetry",
    "advantage_held_across_run": "advantage-held-across-run",
}


def emit(score, reason, detail=None, outcomes=None):
    outcomes = outcomes or {cid: False for cid in CHECKER_IDS}
    payload = {"score": float(score), "reason": reason, "detail": detail or {}}
    for path, body in ((SCORE_PATH, payload), (OUTCOMES_PATH, outcomes)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(body, handle, indent=1, sort_keys=True)
    os.makedirs(os.path.dirname(REWARD_PATH) or ".", exist_ok=True)
    with open(REWARD_PATH, "w") as handle:
        handle.write(repr(float(score)))
    print(json.dumps(payload, sort_keys=True, indent=1))
    return 0


def read_records(path):
    if not os.path.exists(path):
        return None, "telemetry-absent"
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return None, "telemetry-malformed"
    if not records:
        return None, "telemetry-empty"
    return records, None


def extract_arch(blob, profile_name):
    if profile_name in blob:
        blob = blob[profile_name]
    return {k: blob[k] for k in R.ARCH_KEYS if k in blob}


def resolve_agent_frozen_dir():
    """The agent-visible frozen recipe tree, read at grade time as live state.

    This is the tree the solving container can reach. The verifier never runs
    from it; it only compares its digest against the verifier's own tree, which
    is what makes no_frozen_recipe_tree_edit a two-sided ABSENCE check rather
    than a field the runner reported about itself.
    """
    candidates = [
        os.environ.get("BIA_AGENT_FROZEN_DIR"),
        os.environ.get("BIA_FROZEN_DIR"),
        os.path.join(BUNDLE_DIR, "environment", "frozen"),
        "/workspace/environment/frozen",
    ]
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    return None


def build_child_env(key, work_dir):
    """A minimal environment the verifier constructs from scratch for the run.

    Nothing the solving container exported reaches the graded run. The chain key
    is minted for this invocation, lives only in this process and its child, and
    is never written to disk.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": work_dir,
        "TMPDIR": work_dir,
        "LANG": "C.UTF-8",
        "BIA_CHAIN_KEY": key,
    }
    if PROFILE == "smoke":
        env["BIA_FORCE_CPU"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
    for passthrough in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NVIDIA_VISIBLE_DEVICES"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    return env


def run_graded_attempt(work_dir, key, deadline_seconds):
    """Execute the private runner. The verifier owns the process, the clock and the key."""
    telemetry = os.path.join(work_dir, "run_record.jsonl")
    cmd = [
        sys.executable, "-E", "-s", "-B", PRIVATE_RUNNER,
        "--profile", PROFILE,
        "--submission", SUBMISSION,
        "--telemetry", telemetry,
        "--frozen-dir", R.VERIFIER_FROZEN_DIR,
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=build_child_env(key, work_dir),
            cwd=os.path.join(TESTS_DIR, "runner"),
            capture_output=True,
            text=True,
            timeout=float(deadline_seconds) + 180.0,
        )
    except subprocess.TimeoutExpired:
        return telemetry, 124, "", "verifier-side timeout waiting for the graded run"
    return telemetry, proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# The nine checkers. Pure: record list plus frozen context in, outcome map out.
# ---------------------------------------------------------------------------

def run_checkers(records, ctx):
    """Return (outcomes, failing_checker_id_or_None, detail).

    Every read below is either a field of the record set the verifier's own run
    produced, or a value the verifier recomputed from frozen bytes. Nothing here
    reads a clock, a random source, an environment secret or a network.
    """
    recipe = ctx["recipe"]
    space = ctx["space"]
    profile = ctx["profile"]
    submitted = ctx["submitted_arch"]
    ref_arch = ctx["reference_arch"]
    outcomes = {cid: False for cid in CHECKER_IDS}
    detail = {
        "profile": PROFILE,
        "submitted_arch": submitted,
        "reference_arch": ref_arch,
    }
    header = records[0]

    # --- search_space_respected, kind VALUE -------------------------------
    problems = R.validate_arch(submitted, space, PROFILE)
    recorded_problems = list(header.get("submission_space_problems") or [])
    outcomes["search_space_respected"] = (not problems) and (not recorded_problems)
    detail["space_problems"] = problems
    if not outcomes["search_space_respected"]:
        return outcomes, "search_space_respected", detail

    # --- param_budget_met, kind VALUE -------------------------------------
    low, high = R.param_band(profile)
    sub_analytic = R.analytic_param_count(submitted, recipe["vocab_size"])
    ref_analytic = R.analytic_param_count(ref_arch, recipe["vocab_size"])
    detail["param_band"] = [low, high]
    detail["submission_param_count_analytic"] = sub_analytic
    detail["reference_param_count_analytic"] = ref_analytic
    outcomes["param_budget_met"] = (low <= sub_analytic <= high) and (low <= ref_analytic <= high)
    if not outcomes["param_budget_met"]:
        return outcomes, "param_budget_met", detail

    # --- param_count_agreement, kind DIVERGENCE ---------------------------
    live = {}
    for rec in records:
        if rec.get("event") == "arm_start":
            live[rec.get("arm")] = int(rec.get("param_count_total", -1))
    detail["param_count_live"] = live
    outcomes["param_count_agreement"] = (
        live.get("reference") == ref_analytic and live.get("submission") == sub_analytic
    )
    if not outcomes["param_count_agreement"]:
        return outcomes, "param_count_agreement", detail

    # --- frozen_surface_unchanged, kind INVARIANT -------------------------
    expected_frozen = {
        "seq_len": profile["seq_len"],
        "batch_sequences": profile["batch_sequences"],
        "steps": profile["steps"],
        "token_budget": profile["seq_len"] * profile["batch_sequences"] * profile["steps"],
        "seed": profile["seed"],
        "vocab_size": recipe["vocab_size"],
        "optimizer_signature": R.canonical_json(recipe["optimizer"]),
        "schedule_digest": R.schedule_digest(profile, recipe),
        "param_budget": profile["param_budget"],
        "param_tolerance": profile["param_tolerance"],
    }
    seen_frozen = header.get("frozen") or {}
    arm_digests = {r.get("arm"): r.get("arch_digest") for r in records if r.get("event") == "arm_start"}
    same_data = len({R.canonical_json(header.get("order"))}) == 1
    outcomes["frozen_surface_unchanged"] = (
        R.canonical_json(seen_frozen) == R.canonical_json(expected_frozen)
        and header.get("recipe_version") == recipe["recipe_version"]
        and header.get("space_version") == space["space_version"]
        and arm_digests.get("reference") == R.arch_digest(ref_arch)
        and same_data
    )
    detail["frozen_expected"] = expected_frozen
    detail["frozen_seen"] = seen_frozen
    if not outcomes["frozen_surface_unchanged"]:
        return outcomes, "frozen_surface_unchanged", detail

    # --- no_eval_bytes_in_training_stream, kind ABSENCE -------------------
    split = header.get("split") or {}
    order = header.get("order") or {}
    want = R.expected_split(int(split.get("n_bytes", 0)), profile["val_frac"])
    train_range = list(split.get("train_range") or [])
    val_range = list(split.get("val_range") or [])
    last_train_byte = int(order.get("max_train_start", 1 << 62)) + profile["seq_len"] + 1
    outcomes["no_eval_bytes_in_training_stream"] = (
        train_range == want["train_range"]
        and val_range == want["val_range"]
        and train_range[1] <= val_range[0]
        and last_train_byte <= train_range[1]
        and split.get("train_digest") != split.get("val_digest")
    )
    detail["split_seen"] = split
    detail["split_expected"] = want
    detail["last_touched_train_byte"] = last_train_byte
    if not outcomes["no_eval_bytes_in_training_stream"]:
        return outcomes, "no_eval_bytes_in_training_stream", detail

    # --- no_frozen_recipe_tree_edit, kind ABSENCE -------------------------
    # Three-way. The digest the run recorded over the tree it actually read, the
    # digest the verifier recomputes over its own tree at grade time, and the
    # digest the verifier recomputes over the AGENT-VISIBLE tree at grade time.
    # The first two bind the run to the verifier's recipe; the third is the live
    # read that makes an edit inside the solving container observable at all.
    verifier_tree = ctx["verifier_tree_digest"]
    agent_tree = ctx["agent_tree_digest"]
    outcomes["no_frozen_recipe_tree_edit"] = (
        header.get("frozen_tree_digest") == verifier_tree and agent_tree == verifier_tree
    )
    detail["frozen_tree_recorded"] = header.get("frozen_tree_digest")
    detail["frozen_tree_verifier"] = verifier_tree
    detail["frozen_tree_agent_visible"] = agent_tree
    if not outcomes["no_frozen_recipe_tree_edit"]:
        return outcomes, "no_frozen_recipe_tree_edit", detail

    # --- graded_event_order_held, kind ORDERING ---------------------------
    seq = R.event_sequence(records)
    steps_ok = True
    for arm in ("reference", "submission"):
        arm_steps = [int(r.get("step", -1)) for r in records if r.get("arm") == arm]
        steps_ok = steps_ok and arm_steps == sorted(arm_steps)
    outcomes["graded_event_order_held"] = seq == R.REQUIRED_EVENT_ORDER and steps_ok
    detail["event_sequence"] = seq
    if not outcomes["graded_event_order_held"]:
        return outcomes, "graded_event_order_held", detail

    # --- submission_run_mutated_telemetry, kind EFFECT --------------------
    # The submission digest is taken twice by the verifier, once before the run
    # is launched and once after it returns, and both must equal the digest the
    # run recorded. A file rewritten while the graded run is in flight fails
    # here even though the record set is internally consistent.
    sub_start = [r for r in records if r.get("event") == "arm_start" and r.get("arm") == "submission"]
    outcomes["submission_run_mutated_telemetry"] = (
        header.get("submission_digest") == ctx["submission_digest_before"]
        and header.get("submission_digest") == ctx["submission_digest_after"]
        and len(sub_start) == 1
        and sub_start[0].get("arch_digest") == R.arch_digest(submitted)
        and records.index(sub_start[0]) > 0
    )
    detail["submission_digest_before_run"] = ctx["submission_digest_before"]
    detail["submission_digest_after_run"] = ctx["submission_digest_after"]
    detail["submission_digest_recorded"] = header.get("submission_digest")
    if not outcomes["submission_run_mutated_telemetry"]:
        return outcomes, "submission_run_mutated_telemetry", detail

    # --- advantage_held_across_run, kind INVARIANT ------------------------
    evals = {}
    for rec in records:
        if rec.get("event") == "eval":
            evals[(rec.get("arm"), int(rec.get("step")))] = float(rec.get("val_loss"))
    mid, final = int(profile["eval_steps"][0]), int(profile["eval_steps"][1])
    needed = [("reference", mid), ("reference", final), ("submission", mid), ("submission", final)]
    if any(k not in evals for k in needed):
        detail["evals"] = {str(k): v for k, v in evals.items()}
        return outcomes, "advantage_held_across_run", detail
    gap_mid = evals[("reference", mid)] - evals[("submission", mid)]
    gap_final = evals[("reference", final)] - evals[("submission", final)]
    outcomes["advantage_held_across_run"] = gap_mid > 0.0 and gap_final > 0.0
    detail["val_loss"] = {str(k): v for k, v in sorted(evals.items())}
    detail["gap_mid"] = round(gap_mid, 8)
    detail["gap_final"] = round(gap_final, 8)
    if not outcomes["advantage_held_across_run"]:
        return outcomes, "advantage_held_across_run", detail

    # --- reward -----------------------------------------------------------
    rel_target = float(recipe["reward"]["rel_target"])
    ref_final = evals[("reference", final)]
    raw = gap_final / (rel_target * ref_final)
    detail["rel_target"] = rel_target
    detail["raw"] = round(raw, 8)
    detail["score"] = min(max(raw, 0.0), 1.0)
    return outcomes, None, detail


def main():
    recipe = R.frozen("frozen_recipe.json")
    space = R.frozen("arch_space.json")
    reference = R.frozen("reference_arch.json")
    profile = recipe["profiles"][PROFILE]
    outcomes = {cid: False for cid in CHECKER_IDS}

    if not os.path.isfile(PRIVATE_RUNNER):
        return emit(0.0, "verifier-runner-absent", {"path": PRIVATE_RUNNER}, outcomes)

    agent_frozen = resolve_agent_frozen_dir()
    if agent_frozen is None:
        return emit(0.0, "agent-frozen-tree-unresolvable", {}, outcomes)

    if not os.path.exists(SUBMISSION):
        return emit(0.0, "submission-absent", {"path": SUBMISSION}, outcomes)
    with open(SUBMISSION, "rb") as handle:
        submission_bytes = handle.read()
    digest_before = R.sha256_hex(submission_bytes)
    try:
        submitted = extract_arch(json.loads(submission_bytes.decode("utf-8")), PROFILE)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return emit(0.0, "submission-malformed", {"error": repr(exc)}, outcomes)

    key = secrets.token_hex(32)
    work_dir = tempfile.mkdtemp(prefix="bia-s04-verifier-")
    os.chmod(work_dir, 0o700)
    try:
        telemetry, rc, out, err = run_graded_attempt(work_dir, key, profile["deadline_seconds"])
        with open(SUBMISSION, "rb") as handle:
            digest_after = R.sha256_hex(handle.read())

        records, rerr = read_records(telemetry)
        if records is None:
            return emit(0.0, rerr, {"runner_exit_code": rc, "runner_stderr": (err or "")[-800:]}, outcomes)

        ok, why = R.verify_chain(records, key)
        if not ok:
            return emit(0.0, "telemetry-chain-unverified", {"detail": why}, outcomes)

        modes = sorted({str(r.get("mode")) for r in records})
        if modes != [PROFILE]:
            return emit(0.0, "telemetry-mode-mismatch", {"expected": PROFILE, "found": modes}, outcomes)

        if any(r.get("event") == "deadline_exceeded" for r in records):
            return emit(0.0, "attempt-exceeded-budget-deadline",
                        {"deadline_seconds": profile["deadline_seconds"]}, outcomes)

        if records[0].get("event") != "header":
            return emit(0.0, "telemetry-header-missing", {}, outcomes)

        ctx = {
            "recipe": recipe,
            "space": space,
            "profile": profile,
            "submitted_arch": submitted,
            "reference_arch": {k: int(reference[PROFILE][k]) for k in R.ARCH_KEYS},
            "submission_digest_before": digest_before,
            "submission_digest_after": digest_after,
            "verifier_tree_digest": R.tree_digest(R.VERIFIER_FROZEN_DIR),
            "agent_tree_digest": R.tree_digest(agent_frozen),
        }
        outcomes, failed, detail = run_checkers(records, ctx)
        detail["runner_exit_code"] = rc
        detail["agent_frozen_dir"] = agent_frozen
        if failed is not None:
            return emit(0.0, ZERO_REASONS[failed], detail, outcomes)
        score = detail.pop("score")
        return emit(score, "scored", detail, outcomes)
    finally:
        # Retain the record set the verifier produced, as reviewer evidence. It
        # is copied out after grading and is never read back as an input.
        try:
            src = os.path.join(work_dir, "run_record.jsonl")
            if os.path.exists(src):
                os.makedirs(os.path.dirname(RECORD_EVIDENCE) or ".", exist_ok=True)
                shutil.copyfile(src, RECORD_EVIDENCE)
        except OSError:
            pass
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

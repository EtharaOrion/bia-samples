"""BIA S02 verifier. Re-executes the submission and grades what it measured.

Nothing here reads a run the agent performed. The verifier trains the submitted
rule itself, in a child interpreter it spawns, over a pipe it created, and the
validation loss series it grades is produced by a third process into which no
submitted byte is ever loaded. The submitted module is never imported into this
interpreter, not even to audit it: the behavioural audits run in their own
isolated process too.

The record this verifier grades is written by this process. That is why there
is no chain key in the design any more. The previous revision authenticated the
record with an HMAC whose key the solving agent had to hold in order for the
runner it invoked to write at all, so the party the signature defended against
was the party holding the key.

Agent artifacts are read for exactly one purpose: the reconciliation checker,
which compares the agent's reported curve against the curve this verifier
measured. It carries no reward of its own.
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "checkers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from checkers import parser as log_parser  # noqa: E402
from checkers import schedule_ref  # noqa: E402
from checkers import submission_audit as audit  # noqa: E402
from harness import supervisor as sup  # noqa: E402

import anchors  # noqa: E402

# Resolved from this file rather than from the environment. tests/ always sits
# inside the bundle, while BIA_BUNDLE is set by the runtime to the agent
# workspace, and the verifier must load its own substrate rather than a copy
# the graded party could have edited.
BUNDLE = os.path.dirname(HERE)
SCORE_PATH = os.environ.get("BIA_SCORE_PATH", "/logs/verifier/score.json")
OUTCOMES_PATH = os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json")
SUBMISSION_DIR = os.environ.get("BIA_SUBMISSION", "/workspace/submission")
CORPUS_DIR = os.environ.get("BIA_CORPUS_DIR", os.path.join(HERE, "corpus"))
SCHEDULE_PATH = os.environ.get(
    "BIA_SCHEDULE_PATH",
    os.path.join(os.environ.get("BIA_WORKSPACE", BUNDLE), "environment",
                 "frozen_schedule.py"))
EXPECTED_PROFILE = "smoke" if os.environ.get("BIA_SMOKE") == "1" else "full"
KEEP_ROOT = os.environ.get("BIA_KEEP_VERIFIER_ROOT") == "1"

LR_TOLERANCE = 1e-9
TOTAL_STEPS = {"full": 1400, "smoke": 36}
RECON_TOL = {"full": 0.05, "smoke": 1e-6}

CHECKS = [
    "frozen_schedule_bytes_match",
    "applied_lr_matches_frozen_schedule",
    "update_is_lr_homogeneous",
    "submission_never_writes_lr",
    "no_horizon_leak_into_update_rule",
    "one_forward_backward_per_step",
    "telemetry_chain_ordered",
    "reported_logs_reconcile",
    "frozen_recipe_fields_match",
    "no_out_of_step_parameter_mutation",
    "update_rule_not_record_replay",
    "run_profile_matches_verifier",
    "graded_step_reaches_target",
]

ZERO_REASONS = {
    "frozen_schedule_bytes_match": "frozen-schedule-bytes-diverged",
    "applied_lr_matches_frozen_schedule": "applied-lr-diverges-from-schedule",
    "update_is_lr_homogeneous": "update-not-lr-homogeneous",
    "submission_never_writes_lr": "submission-wrote-learning-rate",
    "no_horizon_leak_into_update_rule": "horizon-leaked-into-update-rule",
    "one_forward_backward_per_step": "forward-backward-not-one-per-step",
    "telemetry_chain_ordered": "telemetry-chain-broken",
    "reported_logs_reconcile": "reported-logs-diverge-from-telemetry",
    "frozen_recipe_fields_match": "frozen-recipe-field-violated",
    "no_out_of_step_parameter_mutation": "out-of-step-parameter-mutation",
    "update_rule_not_record_replay": "update-rule-replays-pinned-record",
    "run_profile_matches_verifier": "run-profile-mismatch",
    "graded_step_reaches_target": "graded-step-target-unreached",
}

RUN_ZERO_REASONS = {
    "submission_update_rule_absent": "submission-update-rule-absent",
    "verifier_produced_no_records": "verifier-produced-no-records",
    "submission_wrote_during_graded_run": "submission-wrote-during-graded-run",
}


def emit(score, reason, detail=None, outcomes=None, zero_reason=None):
    payload = {
        "score": float(score),
        "reason": reason,
        "zero_reason": zero_reason if float(score) <= 0.0 else None,
        "detail": detail or {},
    }
    for path, body in ((SCORE_PATH, payload), (OUTCOMES_PATH, outcomes or {})):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(body, f, indent=1, sort_keys=True)
    print(json.dumps(payload, sort_keys=True))
    return float(score)


def run_audit_child(submission, horizon, env):
    read_fd, write_fd = os.pipe()
    try:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "harness", "audit_child.py"),
             "--channel-fd", str(write_fd)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, pass_fds=(write_fd,), env=env)
        os.close(write_fd)
        write_fd = None
        proc.stdin.write(json.dumps(
            {"submission": submission, "horizon": horizon}).encode())
        proc.stdin.close()
        payload = b""
        with os.fdopen(read_fd, "rb") as chan:
            read_fd = None
            while True:
                chunk = chan.read(1 << 16)
                if not chunk:
                    break
                payload += chunk
        proc.wait()
    finally:
        for fd in (read_fd, write_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
    try:
        return json.loads(payload.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return {"error": "audit_child_channel_malformed"}


def check_schedule_bytes(path):
    """VALUE: the on-disk schedule module equals the digest the verifier pins."""
    if not os.path.isfile(path):
        return False, "frozen_schedule_absent_at_%s" % path
    if audit.file_digest(path) != schedule_ref.SCHEDULE_DIGEST:
        return False, "frozen_schedule_edited_digest_%s" % audit.file_digest(path)[:16]
    return True, None


def check_applied_lr(records):
    """DIVERGENCE: applied learning rates agree with an independent recomputation."""
    seen = 0
    for record in records:
        applied = record.get("lr_applied")
        if not isinstance(applied, dict):
            return False, "lr_applied_absent_at_index_%s" % record.get(
                "verifier_record_index")
        total = int(record["total_steps"])
        step = int(record["step"])
        for name, value in applied.items():
            expected = schedule_ref.expected_group_lr(name, step, total)
            if abs(float(value) - expected) > LR_TOLERANCE:
                return False, "lr_divergence_group_%s_step_%s" % (name, step)
            seen += 1
    if seen == 0:
        return False, "no_lr_records_to_compare"
    return True, None


def check_forward_backward(records):
    """INVARIANT: exactly one measured forward-backward pass per optimizer step."""
    by_seed = {}
    for record in records:
        for field in ("forward_calls_this_step", "backward_calls_this_step",
                      "forward_backward_this_step"):
            if int(record.get(field, -1)) != 1:
                return False, "%s_not_one_at_step_%s" % (field, record.get("step"))
        seed = record.get("seed")
        cumulative = int(record.get("forward_backward_cumulative", -1))
        checked = int(record.get("steps_checked", -1))
        step = int(record.get("step", -1))
        base = by_seed.setdefault(seed, cumulative - step)
        if cumulative - step != base:
            return False, "forward_backward_drift_seed_%s_step_%s" % (seed, step)
        if checked != step:
            return False, "step_counter_drift_seed_%s_step_%s" % (seed, step)
    return True, None


def check_lr_writes(records):
    """ABSENCE: no write of a learning rate by the submission exists in the run."""
    for record in records:
        violations = record.get("lr_write_violations")
        if violations is None:
            return False, "lr_write_counter_absent_at_index_%s" % record.get(
                "verifier_record_index")
        if int(violations) != 0:
            return False, "submission_wrote_lr_%s_times_by_step_%s" % (
                violations, record.get("step"))
    return True, None


def check_out_of_step_mutation(records):
    """ABSENCE: no parameter mutation outside optimizer.step() exists in the run."""
    for record in records:
        mutations = record.get("out_of_step_mutations")
        if mutations is None:
            return False, "mutation_counter_absent_at_index_%s" % record.get(
                "verifier_record_index")
        if int(mutations) != 0:
            return False, "out_of_step_mutation_%s_by_step_%s" % (
                mutations, record.get("step"))
    return True, None


def check_frozen_fields(records):
    """VALUE: every frozen recipe field equals its bound constant."""
    expected_arch = anchors.ARCHITECTURE[EXPECTED_PROFILE]
    for record in records:
        arch = record.get("architecture") or {}
        for key, value in expected_arch.items():
            if arch.get(key) != value:
                return False, "architecture_violation_%s_step_%s" % (key, record.get("step"))
        if int(record.get("batch_sequences", -1)) != anchors.BATCH_SEQUENCES[EXPECTED_PROFILE]:
            return False, "batch_violation_step_%s" % record.get("step")
        frozen = record.get("frozen") or {}
        if frozen.get("data_order") != "sequential_frozen":
            return False, "data_order_violation_step_%s" % record.get("step")
        if frozen.get("schedule_owner") != "harness":
            return False, "schedule_owner_violation_step_%s" % record.get("step")
        if record.get("model_digest") != schedule_ref.MODEL_DIGEST:
            return False, "architecture_module_edited_step_%s" % record.get("step")
        if record.get("train_shard_digest") != anchors.TRAIN_SHARD[EXPECTED_PROFILE]:
            return False, "train_shard_violation_step_%s" % record.get("step")
        if record.get("val_shard_digest") != anchors.VAL_SHARD[EXPECTED_PROFILE]:
            return False, "val_shard_violation_step_%s" % record.get("step")
        if record.get("measured_arch_shapes_ok") is not True:
            return False, "checkpoint_shapes_diverge_step_%s" % record.get("step")
    return True, None


def check_profile(records):
    """INVARIANT: the profile graded, the profile run, and the tensors agree.

    The profile field alone cannot fail once the verifier chooses the profile
    it runs, so the check reads the batch shape the training process observed
    off the real tensors and requires it to match the profile's bound shape.
    That is a quantity the run produces rather than one it declares.
    """
    want_batch = anchors.BATCH_SEQUENCES[EXPECTED_PROFILE]
    want_len = anchors.ARCHITECTURE[EXPECTED_PROFILE]["seq_len"]
    want_total = TOTAL_STEPS[EXPECTED_PROFILE]
    for record in records:
        if record.get("profile") != EXPECTED_PROFILE:
            return False, "profile_mismatch_%s_expected_%s" % (
                record.get("profile"), EXPECTED_PROFILE)
        if int(record.get("batch_sequences_observed", -1)) != want_batch:
            return False, "observed_batch_%s_expected_%s_step_%s" % (
                record.get("batch_sequences_observed"), want_batch, record.get("step"))
        if int(record.get("sequence_length_observed", -1)) != want_len:
            return False, "observed_seq_len_%s_expected_%s_step_%s" % (
                record.get("sequence_length_observed"), want_len, record.get("step"))
        if int(record.get("total_steps", -1)) != want_total:
            return False, "total_steps_%s_expected_%s" % (
                record.get("total_steps"), want_total)
    return True, None


def telemetry_series(records):
    series = {}
    for record in records:
        if record.get("val_loss") is None:
            continue
        series.setdefault(str(record["seed"]), {})[int(record["step"])] = round(
            float(record["val_loss"]), 6)
    return series


def load_seed_logs(submission_dir):
    paths = sorted(glob.glob(os.path.join(submission_dir, "logs", "*.log")))
    scoped = [p for p in paths if os.path.basename(p).startswith(EXPECTED_PROFILE + "_")]
    if scoped:
        paths = scoped
    out, seen = {}, {}
    for path in paths:
        base = os.path.basename(path)
        seed = base.split("seed")[-1].split(".")[0] if "seed" in base else base.split(".")[0]
        parsed = log_parser.parse_file(path)
        if not parsed:
            continue
        if seed in seen:
            return None, "seed_log_collision_%s_%s_and_%s" % (seed, seen[seed], base)
        seen[seed] = base
        out[seed] = parsed
    return out, None


def check_reconciliation(series, seed_logs, tol):
    """DIVERGENCE: the agent's reported curve against the curve the verifier measured."""
    if not seed_logs:
        return False, "no_reported_seed_logs"
    for seed, reported in seed_logs.items():
        if seed not in series:
            return False, "reported_seed_%s_absent_from_verifier_measurement" % seed
        for step, loss in reported.items():
            if step not in series[seed]:
                return False, "reported_step_%s_seed_%s_absent_from_verifier_measurement" % (
                    step, seed)
            if abs(series[seed][step] - round(float(loss), 6)) > tol:
                return False, "reported_loss_diverges_seed_%s_step_%s" % (seed, step)
    for seed, observed in series.items():
        if seed not in seed_logs:
            return False, "verifier_seed_%s_absent_from_report" % seed
        for step in observed:
            if step not in seed_logs[seed]:
                return False, "verifier_step_%s_seed_%s_absent_from_report" % (step, seed)
    return True, None


def graded_step(series, cfg):
    if len(series) < cfg["min_seeds"]:
        return None, "need_at_least_%d_seeds_got_%d" % (cfg["min_seeds"], len(series))
    common = sorted(set.intersection(*[set(d) for d in series.values()]))
    if not common:
        return None, "no_step_logged_by_every_seed"

    def stats(step):
        values = [series[k][step] for k in series]
        return sum(values) / len(values), max(values), len(values)

    def holds(step):
        mean, worst, _ = stats(step)
        return mean <= cfg["target_loss"] and worst <= cfg["target_loss"]

    rejected = None
    for i, step in enumerate(common):
        mean, worst, n = stats(step)
        if mean > cfg["target_loss"]:
            continue
        if (cfg["target_loss"] - mean) * math.sqrt(n) < cfg["sig_margin"]:
            rejected = rejected or "crossing_at_%s_below_noise_floor" % step
            continue
        if worst > cfg["target_loss"]:
            rejected = rejected or "seed_above_target_at_step_%s_worst_%.6f" % (step, worst)
            continue
        regressed = next((t for t in common[i + 1:] if not holds(t)), None)
        if regressed is not None:
            rejected = rejected or "crossing_at_%s_not_sustained_at_step_%s" % (step, regressed)
            continue
        return step, None
    return None, rejected or "target_loss_not_reached"


def main() -> float:
    outcomes = {name: False for name in CHECKS}
    cfg = anchors.PROFILE_ANCHORS[EXPECTED_PROFILE]
    horizon = TOTAL_STEPS[EXPECTED_PROFILE]
    submission = os.path.join(SUBMISSION_DIR, "update_rule.py")

    gates = []

    def gate(key, result):
        ok, reason = result
        outcomes[key] = bool(ok)
        gates.append((ok, reason, key))
        return ok, reason

    gate("frozen_schedule_bytes_match", check_schedule_bytes(SCHEDULE_PATH))

    if not os.path.isfile(submission):
        return emit(0.0, "submission_update_rule_absent", {"path": submission}, outcomes,
                    RUN_ZERO_REASONS["submission_update_rule_absent"])

    gate("update_rule_not_record_replay", audit.audit_record_replay(submission, CORPUS_DIR))

    static_ok, static_reason = audit.audit_horizon_leak(submission, horizon)

    supervisor = sup.Supervisor(BUNDLE, EXPECTED_PROFILE, SUBMISSION_DIR, submission)
    try:
        audits = run_audit_child(submission, horizon, supervisor.child_env())
        if "error" in audits:
            gate("update_is_lr_homogeneous", (False, audits["error"]))
            gate("no_horizon_leak_into_update_rule", (False, audits["error"]))
        else:
            gate("update_is_lr_homogeneous", tuple(audits["lr_homogeneity"]))
            dyn_ok, dyn_reason = tuple(audits["schedule_ownership"])
            gate("no_horizon_leak_into_update_rule",
                 (bool(static_ok) and bool(dyn_ok), static_reason or dyn_reason))

        try:
            ready = supervisor.start_measure(
                {"bundle": BUNDLE, "profile": EXPECTED_PROFILE})
            expected_shapes = None

            def on_record(body, payload):
                nonlocal expected_shapes
                measured = supervisor.measure_blob(payload)
                if expected_shapes is None:
                    expected_shapes = measured.get("shapes")
                body = dict(body)
                body.pop("blob_bytes", None)
                body["val_loss"] = measured.get("val_loss")
                body["measured_arch_shapes_ok"] = (
                    measured.get("load_error") is None
                    and measured.get("shapes") == expected_shapes)
                body["measured_by"] = "tests/harness/measure_child.py"
                supervisor.append_record(body)

            for seed in range(cfg["min_seeds"]):
                supervisor.run_seed(seed, on_record)
        except sup.ChildFailure as exc:
            return emit(0.0, exc.code, exc.detail, outcomes, exc.code)
        finally:
            supervisor.stop_measure()

        records = supervisor.records()
        if not records:
            return emit(0.0, "verifier_produced_no_records", {}, outcomes,
                        RUN_ZERO_REASONS["verifier_produced_no_records"])
        supervisor.flush_records()

        if supervisor.write_effects:
            return emit(0.0, "submission_wrote_%s" % supervisor.write_effects[0],
                        {"observed": supervisor.write_effects[:8]}, outcomes,
                        RUN_ZERO_REASONS["submission_wrote_during_graded_run"])
        for record in records:
            if record.get("guard_observed_writes"):
                return emit(0.0, "submission_wrote_%s" % record["guard_observed_writes"][0],
                            {"step": record.get("step")}, outcomes,
                            RUN_ZERO_REASONS["submission_wrote_during_graded_run"])

        gate("telemetry_chain_ordered", supervisor.verify_own_chain())
        gate("run_profile_matches_verifier", check_profile(records))
        gate("applied_lr_matches_frozen_schedule", check_applied_lr(records))
        gate("submission_never_writes_lr", check_lr_writes(records))
        gate("one_forward_backward_per_step", check_forward_backward(records))
        gate("frozen_recipe_fields_match", check_frozen_fields(records))
        gate("no_out_of_step_parameter_mutation", check_out_of_step_mutation(records))

        series = telemetry_series(records)
        seed_logs, reason = load_seed_logs(SUBMISSION_DIR)
        if seed_logs is None:
            gate("reported_logs_reconcile", (False, reason))
        else:
            gate("reported_logs_reconcile", check_reconciliation(
                series, seed_logs, RECON_TOL[EXPECTED_PROFILE]))

        step, step_reason = graded_step(series, cfg)
        outcomes["graded_step_reaches_target"] = step is not None

        for ok, reason, key in gates:
            if not ok:
                return emit(0.0, reason, {"profile": EXPECTED_PROFILE}, outcomes,
                            ZERO_REASONS[key])
        if step is None:
            return emit(0.0, step_reason, {"profile": EXPECTED_PROFILE, "seeds": len(series)},
                        outcomes, ZERO_REASONS["graded_step_reaches_target"])

        raw = (cfg["baseline_steps"] - step) / float(cfg["baseline_steps"] - cfg["target_steps"])
        return emit(min(max(raw, 0.0), 1.0), "graded_step=%d" % step, {
            "profile": EXPECTED_PROFILE,
            "graded_step": step,
            "seeds": len(series),
            "baseline_steps": cfg["baseline_steps"],
            "target_steps": cfg["target_steps"],
            "target_loss": cfg["target_loss"],
            "measured_by": "verifier re-execution, tests/harness/measure_child.py",
        }, outcomes)
    finally:
        if not KEEP_ROOT:
            shutil.rmtree(supervisor.root, ignore_errors=True)


if __name__ == "__main__":
    main()

"""The seven graded checkers for the S03 data-order task.

Each function reduces to exactly one of the six checker kinds and reads live run state only,
through the run record the runner wrote, the egress audit the socket hook wrote, the corpus
the generator regenerates, or a subprocess the grader launches. None of them reads an author
assertion, a planted file, a clock, or a random source. tests/checkers.yaml is the registry
that binds each identifier to its kind and to the live-state read it traces to.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

# The machine-readable code each checker emits when it scores zero, bound to the
# identifier tests/checkers.yaml registers. Every function below can fail either
# because its evidence was contradicted or because its evidence was missing, so
# each code says "unproven" rather than naming one branch and mis-describing the
# other. The free-text second return value names the specific record that failed
# and carries run identifiers, so it is not stable enough to branch on.
ZERO_REASON = {
    "frozen_substrate_unchanged": "frozen-substrate-unchanged-unproven",
    "order_permutation_invariant": "order-permutation-invariant-unproven",
    "order_frozen_before_training": "order-frozen-before-training-unproven",
    "no_scored_egress": "no-scored-egress-unproven",
    "order_determinism_divergence": "order-determinism-divergence-unproven",
    "training_consumed_the_submitted_order": "training-consumed-the-submitted-order-unproven",
    "improvement_clears_noise_floor": "improvement-clears-noise-floor-unproven",
}


def _by_run(records):
    out = {}
    for r in records:
        rid = r.get("run_id")
        if rid is None:
            continue
        out.setdefault(rid, []).append(r)
    return out


def frozen_substrate_unchanged(records, expected_digest):
    """VALUE. Reads run_record.jsonl field substrate_digest on every emitted record."""
    seen = set()
    for r in records:
        d = r.get("substrate_digest")
        if d is None:
            continue
        seen.add(d)
    if not seen:
        return False, "substrate_digest_absent_from_telemetry", {}
    if seen != {expected_digest}:
        return False, f"substrate_digest_mismatch_{sorted(seen)[0]}", {"seen": sorted(seen)}
    return True, None, {"substrate_digest": expected_digest}


def order_permutation_invariant(records, n_sequences, expected_sorted_digest):
    """INVARIANT. Reads the run_complete consumed_* fields accumulated across the whole run."""
    completes = [r for r in records if r.get("event") == "run_complete"]
    if not completes:
        return False, "no_run_complete_record", {}
    for r in completes:
        if int(r.get("consumed_total", -1)) != n_sequences:
            return False, f"consumed_total_{r.get('consumed_total')}_expected_{n_sequences}_run_{r.get('run_id')}", {}
        if int(r.get("consumed_unique_count", -1)) != n_sequences:
            return False, f"consumed_unique_{r.get('consumed_unique_count')}_expected_{n_sequences}_run_{r.get('run_id')}", {}
        if r.get("consumed_sorted_digest") != expected_sorted_digest:
            return False, f"consumed_multiset_diverges_run_{r.get('run_id')}", {}
    return True, None, {"runs": len(completes)}


def order_frozen_before_training(records):
    """ORDERING. Reads the seq field ordering of order_frozen against train_step per run."""
    for rid, rows in _by_run(records).items():
        frozen = [r for r in rows if r.get("event") == "order_frozen"]
        steps = [r for r in rows if r.get("event") == "train_step"]
        if len(frozen) != 1:
            return False, f"order_frozen_count_{len(frozen)}_run_{rid}", {}
        if not steps:
            return False, f"no_train_step_run_{rid}", {}
        if not all(int(frozen[0]["seq"]) < int(s["seq"]) for s in steps):
            return False, f"order_frozen_not_before_training_run_{rid}", {}
        if [int(s["step"]) for s in steps] != sorted(int(s["step"]) for s in steps):
            return False, f"train_step_out_of_sequence_run_{rid}", {}
    return True, None, {}


def no_scored_egress(events, arms_observed, arms_expected):
    """ABSENCE. Reads the egress events the verifier received on its own channel.

    The evidence is no longer a file in the agent workspace. Each training child
    reports every outbound connect straight to the parent over the pipe the
    parent created, so there is nothing on disk to truncate and an empty
    observation is only reachable by a run that made no connect.

    The absence claim is bounded by coverage rather than asserted: a stream that
    did not carry every expected arm cannot support an absence claim over the
    whole graded run, so an incomplete arm set fails closed instead of reading
    as clean.
    """
    missing = sorted(set(arms_expected) - set(arms_observed))
    if missing:
        return False, f"egress_stream_incomplete_missing_{len(missing)}_arms", {
            "missing": missing[:8]}
    if events:
        return False, f"egress_attempted_{len(events)}_times", {
            "first": events[0], "count": len(events)}
    return True, None, {"connect_attempts": 0, "arms_covered": len(arms_observed)}


def order_determinism_divergence(runner_path, order_module, env_a, env_b, cwd_a, cwd_b):
    """DIVERGENCE. Two independent processes derive the order and their digests must agree."""

    def derive(env, cwd):
        proc = subprocess.run(
            [sys.executable, runner_path, "order-digest", "--order-module", order_module],
            capture_output=True, text=True, env=env, cwd=cwd,
        )
        if proc.returncode != 0:
            return None, (proc.stdout + proc.stderr).strip()[-400:]
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1]), None
        except (json.JSONDecodeError, IndexError):
            return None, "order_digest_output_unparsable"

    a, err_a = derive(env_a, cwd_a)
    if a is None:
        return False, f"order_derivation_a_failed_{err_a}", {}
    b, err_b = derive(env_b, cwd_b)
    if b is None:
        return False, f"order_derivation_b_failed_{err_b}", {}
    if not a.get("valid") or not b.get("valid"):
        return False, f"order_invalid_{a.get('reason') or b.get('reason')}", {}
    if a["order_digest"] != b["order_digest"]:
        return False, "order_not_deterministic_across_hosts", {"a": a["order_digest"], "b": b["order_digest"]}
    return True, None, {"order_digest": a["order_digest"]}


def training_consumed_the_submitted_order(records, independent_order_digest):
    """EFFECT. The submitted order changed what the trainer consumed, read from order_frozen."""
    frozen = [r for r in records if r.get("event") == "order_frozen"]
    submitted = [r for r in frozen if r.get("arm") == "submitted"]
    baseline = [r for r in frozen if r.get("arm") == "baseline"]
    if not submitted:
        return False, "no_submitted_arm_recorded", {}
    if not baseline:
        return False, "no_baseline_arm_recorded", {}
    for r in submitted:
        if r.get("order_digest") != independent_order_digest:
            return False, f"submitted_arm_order_digest_mismatch_run_{r.get('run_id')}", {}
    sub_first = {r.get("first_index_sequence_digest") for r in submitted}
    base_first = {r.get("first_index_sequence_digest") for r in baseline}
    if sub_first & base_first:
        return False, "submitted_order_identical_to_baseline_order", {}
    return True, None, {"submitted_runs": len(submitted), "baseline_runs": len(baseline)}


def improvement_clears_noise_floor(records, expected_seeds, expected_draws, significance_t):
    """VALUE. The improvement statistic computed from recorded final_val_loss bounds a target.

    The comparator is an ensemble of independent shuffles, so the spread between comparator
    draws estimates the run-to-run noise floor in band, during the same attempt. The gate is a
    one-sample t statistic over every (seed, draw) pair, which means the floor is measured from
    the graded runs rather than authored, and the submitted ordering never sees either the
    graded seeds or the comparator draws it will be measured against.
    """
    completes = [r for r in records if r.get("event") == "run_complete"]
    base = {}
    for r in completes:
        if r.get("arm") == "baseline":
            base[(int(r["seed"]), int(r.get("draw", 0)))] = float(r["final_val_loss"])
    sub = {int(r["seed"]): float(r["final_val_loss"]) for r in completes if r.get("arm") == "submitted"}

    want_pairs = sorted((s, d) for s in expected_seeds for d in range(expected_draws))
    missing_base = [p for p in want_pairs if p not in base]
    missing_sub = [s for s in expected_seeds if s not in sub]
    if missing_base or missing_sub:
        return False, f"incomplete_graded_grid_missing_baseline_{len(missing_base)}_submitted_{len(missing_sub)}", {}
    if len(want_pairs) < 3:
        return False, f"graded_grid_too_small_{len(want_pairs)}", {}

    deltas = [base[(s, d)] - sub[s] for (s, d) in want_pairs]
    n = len(deltas)
    mean_delta = sum(deltas) / n
    var = sum((x - mean_delta) ** 2 for x in deltas) / (n - 1)
    sd = math.sqrt(var)
    # A zero comparator spread does not make the improvement infinitely
    # significant, it makes the statistic undefined: there is no measured noise
    # scale to divide by. An earlier revision set it to infinity, which let a
    # one-nanonat improvement clear the gate vacuously. Undefined now fails
    # closed and is reported as such.
    stat = None if sd == 0.0 else mean_delta / (sd / math.sqrt(n))
    detail = {
        "pairs": [list(p) for p in want_pairs],
        "baseline_ensemble_mean": round(sum(base[p] for p in want_pairs) / n, 6),
        "submitted_mean": round(sum(sub[s] for s in expected_seeds) / len(expected_seeds), 6),
        "mean_delta": round(mean_delta, 6),
        "delta_sd": round(sd, 6),
        "t_statistic": round(stat, 6) if stat is not None else None,
        "t_threshold": significance_t,
        "n_observations": n,
    }
    if mean_delta <= 0.0:
        return False, f"no_improvement_mean_delta_{mean_delta:.6f}", detail
    if stat is None:
        return False, "comparator_spread_zero_statistic_undefined", detail
    if stat < significance_t:
        return False, f"improvement_below_noise_floor_t_{stat:.6f}", detail
    return True, None, detail

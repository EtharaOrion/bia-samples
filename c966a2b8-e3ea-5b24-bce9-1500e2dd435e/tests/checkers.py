"""Pure checkers for the S10 slot.

Every function here is a pure function of the evidence dictionary the verifier
assembles from live-state reads. No function opens a file, reads the clock, or
touches the network, so a checker replays identically over frozen evidence and
the negative controls in controls.py can plant a defect by editing one field.

Each checker reduces to exactly one of the six kinds and names the live-state
read it traces to. The kind and the read are declared here in KIND and READ and
are mirrored byte for byte into checkers.yaml.
"""

from __future__ import annotations

import math

ABLATED_ARMS = ("anchor_baseline", "agent")
UNABLATED_ARM = "anchor_target"


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ---------------------------------------------------------------- ABSENCE ----
def check_ablation_absent_from_graded_run(ev):
    KIND = "ABSENCE"
    arms = ev.get("arms", {})
    witness = arms.get(UNABLATED_ARM, {})
    # Liveness guard on this one assertion. If the probe saw nothing at all in
    # the arm that is supposed to be full of normalization, the probe cannot see
    # anything, and an absence it reports would be vacuous.
    if int(witness.get("forbidden_norm_events", 0)) <= 0:
        return False, "probe_inert_no_witness", {
            "kind": KIND,
            "unablated_arm_forbidden_events": witness.get("forbidden_norm_events"),
        }
    for arm in ABLATED_ARMS:
        rec = arms.get(arm)
        if rec is None:
            return False, f"ablated_arm_missing:{arm}", {"kind": KIND}
        if rec.get("norm_mode") != "none":
            return False, f"ablated_arm_built_with_normalization:{arm}", {
                "kind": KIND,
                "norm_mode": rec.get("norm_mode"),
            }
        n_events = int(rec.get("forbidden_norm_events", 0))
        if n_events != 0:
            return False, f"normalization_executed_in_ablated_arm:{arm}", {
                "kind": KIND,
                "forbidden_norm_events": n_events,
                "events": rec.get("probe_trace", {}).get("events", [])[:8],
            }
        owned = rec.get("norm_parameter_names") or []
        if owned:
            return False, f"normalization_parameters_present_in_ablated_arm:{arm}", {
                "kind": KIND,
                "norm_parameter_names": owned,
            }
    return True, "ok", {
        "kind": KIND,
        "unablated_arm_forbidden_events": witness.get("forbidden_norm_events"),
        "ablated_arms_forbidden_events": {a: arms[a].get("forbidden_norm_events") for a in ABLATED_ARMS},
    }


# -------------------------------------------------------------- INVARIANT ----
def check_substrate_bytes_unmodified(ev):
    KIND = "INVARIANT"
    pinned = ev.get("substrate_manifest_pinned")
    start = ev.get("substrate_manifest_start")
    end = ev.get("substrate_manifest_end")
    if not pinned or not start or not end:
        return False, "substrate_manifest_unavailable", {"kind": KIND}
    if start != pinned:
        diff = sorted(k for k in set(pinned) | set(start) if pinned.get(k) != start.get(k))
        return False, "substrate_hash_mismatch_at_start", {"kind": KIND, "files": diff}
    if end != start:
        diff = sorted(k for k in set(start) | set(end) if start.get(k) != end.get(k))
        return False, "substrate_hash_drift_during_run", {"kind": KIND, "files": diff}
    return True, "ok", {"kind": KIND, "files_checked": len(pinned)}


def check_ablation_gap_positive(ev):
    KIND = "INVARIANT"
    a = ev.get("anchors", {})
    base = a.get("baseline_metric")
    targ = a.get("target_metric")
    if not _finite(base) or not _finite(targ):
        return False, "anchor_metric_nonfinite", {"kind": KIND, "baseline": base, "target": targ}
    gap = float(base) - float(targ)
    floor = float(ev.get("params", {}).get("min_anchor_gap", 0.05))
    if gap <= 0.0:
        return False, "anchor_inversion", {"kind": KIND, "gap": gap, "floor": floor}
    if gap < floor:
        return False, "anchor_gap_below_margin", {"kind": KIND, "gap": gap, "floor": floor}
    return True, "ok", {"kind": KIND, "gap": gap, "floor": floor}


# --------------------------------------------------------------- ORDERING ----
def check_anchors_measured_before_agent_arm(ev):
    KIND = "ORDERING"
    seqs = {}
    for r in ev.get("run_records") or []:
        e = r.get("event")
        if e in ("anchor_target_ready", "anchor_baseline_ready", "agent_arm_start",
                 "agent_arm_complete", "scoring_begin") and e not in seqs:
            seqs[e] = int(r.get("seq", -1))
    required = ["anchor_target_ready", "anchor_baseline_ready", "agent_arm_start",
                "agent_arm_complete", "scoring_begin"]
    missing = [k for k in required if k not in seqs]
    if missing:
        return False, "ordering_events_missing:" + ",".join(missing), {"kind": KIND, "seen": seqs}
    order = [seqs[k] for k in required]
    if order != sorted(order) or len(set(order)) != len(order):
        return False, "anchor_measured_after_agent_arm", {"kind": KIND, "sequence": dict(zip(required, order))}
    return True, "ok", {"kind": KIND, "sequence": dict(zip(required, order))}


# ----------------------------------------------------------------- EFFECT ----
def check_submitted_rule_moved_the_weights(ev):
    KIND = "EFFECT"
    sub = ev.get("submission_path")
    recs = [r for r in (ev.get("run_records") or [])
            if r.get("event") == "update_applied" and r.get("arm") == "agent"]
    if not recs:
        return False, "update_records_absent", {"kind": KIND}
    expected = ev.get("arms", {}).get("agent", {}).get("sample_steps")
    if expected is not None and sorted(r.get("step") for r in recs) != sorted(expected):
        return False, "update_records_incomplete", {
            "kind": KIND,
            "expected_steps": expected,
            "observed_steps": sorted(r.get("step") for r in recs),
        }
    for r in recs:
        d = r.get("delta_l2")
        if not _finite(d):
            return False, f"parameter_delta_nonfinite_at_step_{r.get('step')}", {"kind": KIND, "delta_l2": d}
        if float(d) <= 0.0:
            return False, f"submitted_rule_had_no_effect_at_step_{r.get('step')}", {"kind": KIND, "delta_l2": d}
        if int(r.get("forward_backward_per_step", 0)) != 1:
            return False, f"forward_backward_per_step_violated_at_step_{r.get('step')}", {
                "kind": KIND, "forward_backward_per_step": r.get("forward_backward_per_step")}
        if sub is not None and r.get("rule_file") != sub:
            return False, "update_not_driven_by_submission_file", {
                "kind": KIND, "rule_file": r.get("rule_file"), "submission_path": sub}
    displacement = ev.get("agent_weight_displacement_l2")
    if not _finite(displacement) or float(displacement) <= 0.0:
        return False, "verifier_measured_no_weight_displacement", {
            "kind": KIND, "agent_weight_displacement_l2": displacement}
    return True, "ok", {
        "kind": KIND,
        "sampled_steps": sorted(r.get("step") for r in recs),
        "delta_l2": [r.get("delta_l2") for r in recs],
        "agent_weight_displacement_l2": displacement,
    }


# ------------------------------------------------------------------ VALUE ----
def check_reward_is_clamped_unit_float(ev):
    KIND = "VALUE"
    score = ev.get("score")
    raw = ev.get("raw")
    if not _finite(score):
        return False, "score_nonfinite", {"kind": KIND, "score": score}
    if not _finite(raw):
        return False, "raw_nonfinite", {"kind": KIND, "raw": raw}
    if not (0.0 <= float(score) <= 1.0):
        return False, "score_outside_unit_interval", {"kind": KIND, "score": score}
    expect = min(max(float(raw), 0.0), 1.0)
    if abs(expect - float(score)) > 1e-12:
        return False, "score_not_clamped_raw", {"kind": KIND, "score": score, "raw": raw, "expected": expect}
    a = ev.get("anchors", {})
    base, targ = a.get("baseline_metric"), a.get("target_metric")
    agent = ev.get("agent_metric")
    if _finite(base) and _finite(targ) and _finite(agent) and float(base) != float(targ):
        recomputed = (float(base) - float(agent)) / (float(base) - float(targ))
        if abs(recomputed - float(raw)) > 1e-9:
            return False, "raw_not_reproducible_from_anchors", {
                "kind": KIND, "raw": raw, "recomputed": recomputed}
    return True, "ok", {"kind": KIND, "score": float(score), "raw": float(raw)}


# ------------------------------------------------------------- DIVERGENCE ----
def check_validation_loss_recompute_agrees(ev):
    KIND = "DIVERGENCE"
    streaming = ev.get("arms", {}).get("agent", {}).get("streaming_val_loss")
    independent = ev.get("independent_val_loss")
    tol = float(ev.get("params", {}).get("divergence_tol", 2.0e-3))
    if not _finite(streaming):
        return False, "streaming_validation_loss_nonfinite", {"kind": KIND, "streaming": streaming}
    if not _finite(independent):
        return False, "independent_validation_loss_nonfinite", {"kind": KIND, "independent": independent}
    delta = abs(float(streaming) - float(independent))
    if delta > tol:
        return False, "validation_loss_recompute_diverged", {
            "kind": KIND, "streaming": streaming, "independent": independent,
            "delta": delta, "tolerance": tol}
    return True, "ok", {"kind": KIND, "streaming": float(streaming),
                        "independent": float(independent), "delta": delta, "tolerance": tol}


# --------------------------------------------------------------- INVARIANT ----
def check_agent_arm_ran_outside_the_verifier(ev):
    """The agent arm was executed by a process that is not this one.

    The submitted rule has to be a live object stepping live parameters, so the training loop
    necessarily sits beside it. What must not sit beside it is the verifier. Before this the
    verifier exec_module'd the submission into itself and only afterwards read the anchor cache
    and called this registry, so both were reachable by the graded party.
    """
    KIND = "INVARIANT"
    iso = ev.get("agent_arm_isolation") or {}
    runner, worker = iso.get("runner_pid"), iso.get("worker_pid")
    if iso.get("mode") != "subprocess" or iso.get("worker") != "arm_worker.py":
        return False, "agent_arm_not_hosted_out_of_process", {"kind": KIND, "isolation": iso}
    if not isinstance(worker, int) or worker <= 0 or worker == runner:
        return False, "agent_arm_worker_pid_not_distinct", {"kind": KIND, "isolation": iso}
    d = ev.get("agent_weight_displacement_l2")
    if not _finite(d):
        return False, "agent_weight_displacement_not_measured", {"kind": KIND, "displacement": d}
    return True, "ok", {"kind": KIND, "isolation": iso, "agent_weight_displacement_l2": d}


# ------------------------------------------------------------------- VALUE ----
def check_anchor_cache_entry_authentic(ev):
    """The anchors were measured by the harness, or replayed from an entry it signed.

    The cache survives fifty attempts and lives where the graded party can write, so it is a
    persistent trust store. Every entry now carries an HMAC taken with a key that lives in the
    verifier tree. `unauthenticated` means an entry was present and did not verify, which is a
    graded failure rather than a cache miss, because a store forged once and harvested fifty
    times is worse than no store at all.
    """
    KIND = "VALUE"
    state = ev.get("anchor_cache_state")
    source = ev.get("anchor_source")
    if state == "unauthenticated":
        return False, "anchor_cache_entry_failed_authentication", {"kind": KIND, "state": state}
    if state not in ("absent", "authenticated"):
        return False, "anchor_cache_state_unrecognised", {"kind": KIND, "state": state}
    if source not in ("measured", "cache"):
        return False, "anchor_source_unrecognised", {"kind": KIND, "source": source}
    if state == "absent" and source != "measured":
        return False, "anchor_source_disagrees_with_cache_state", {"kind": KIND, "state": state, "source": source}
    return True, "ok", {"kind": KIND, "state": state, "source": source}


REGISTRY = [
    ("ablation_absent_from_graded_run", "ABSENCE", check_ablation_absent_from_graded_run),
    ("substrate_bytes_unmodified", "INVARIANT", check_substrate_bytes_unmodified),
    ("ablation_gap_positive", "INVARIANT", check_ablation_gap_positive),
    ("anchors_measured_before_agent_arm", "ORDERING", check_anchors_measured_before_agent_arm),
    ("submitted_rule_moved_the_weights", "EFFECT", check_submitted_rule_moved_the_weights),
    ("reward_is_clamped_unit_float", "VALUE", check_reward_is_clamped_unit_float),
    ("validation_loss_recompute_agrees", "DIVERGENCE", check_validation_loss_recompute_agrees),
    ("agent_arm_ran_outside_the_verifier", "INVARIANT", check_agent_arm_ran_outside_the_verifier),
    ("anchor_cache_entry_authentic", "VALUE", check_anchor_cache_entry_authentic),
]

# Mirrored as zero_reason in checkers.yaml. The granular `reason` stays the human
# diagnostic; this is the stable code a downstream grader branches on. Both are
# emitted, so the attribution the manifest declares is one a run actually produces.
ZERO_REASON = {
    "ablation_absent_from_graded_run": "ablation-not-absent-from-graded-run",
    "substrate_bytes_unmodified": "substrate-bytes-modified",
    "ablation_gap_positive": "ablation-gap-not-positive",
    "anchors_measured_before_agent_arm": "anchors-not-measured-before-agent-arm",
    "submitted_rule_moved_the_weights": "submitted-rule-did-not-move-weights",
    "reward_is_clamped_unit_float": "reward-not-clamped-unit-float",
    "validation_loss_recompute_agrees": "validation-loss-recompute-diverged",
    "agent_arm_ran_outside_the_verifier": "agent-arm-ran-inside-the-verifier",
    "anchor_cache_entry_authentic": "anchor-cache-entry-not-authentic",
}


def run_all(ev):
    outcomes = {}
    first_failure = None
    for name, kind, fn in REGISTRY:
        ok, reason, detail = fn(ev)
        outcomes[name] = {
            "kind": kind,
            "passed": bool(ok),
            "reason": reason,
            "zero_reason": None if ok else ZERO_REASON[name],
            "detail": detail,
        }
        if not ok and first_failure is None:
            first_failure = (name, reason)
    return outcomes, first_failure

"""Negative-control runner. Proves both halves of every checker.

An instrument that only ever accepts, or only ever rejects, is inert, and an
inert required instrument makes a bundle ungradable. This runner takes the
evidence bundle a real run produced, shows every checker accepting it, then
plants one defect at a time and shows the matching checker rejecting the
mutated evidence with its own machine-readable reason while the untouched
checkers still accept.

Usage
  python3 tests/controls.py --evidence /path/to/evidence.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checkers as CHECKERS  # noqa: E402


def plant_normalization_event_in_agent_arm(ev):
    ev["arms"]["agent"]["forbidden_norm_events"] = 1
    ev["arms"]["agent"].setdefault("probe_trace", {}).setdefault("events", []).append(
        {"phase": "forward", "op": "RMSNorm", "kind": "module"}
    )
    return ev


def plant_blind_probe(ev):
    ev["arms"]["anchor_target"]["forbidden_norm_events"] = 0
    return ev


def plant_substrate_drift(ev):
    key = sorted(ev["substrate_manifest_end"])[0]
    ev["substrate_manifest_end"][key] = "0" * 64
    return ev


def plant_anchor_inversion(ev):
    ev["anchors"]["baseline_metric"] = float(ev["anchors"]["target_metric"]) - 0.1
    return ev


def plant_anchor_gap_too_small(ev):
    ev["anchors"]["baseline_metric"] = float(ev["anchors"]["target_metric"]) + 0.001
    return ev


def plant_anchor_measured_late(ev):
    top = max(int(r.get("seq", 0)) for r in ev["run_records"])
    for r in ev["run_records"]:
        if r.get("event") == "anchor_target_ready":
            r["seq"] = top + 1
            break
    return ev


def plant_inert_update(ev):
    for r in ev["run_records"]:
        if r.get("event") == "update_applied" and r.get("arm") == "agent":
            r["delta_l2"] = 0.0
            break
    return ev


def plant_foreign_rule_file(ev):
    for r in ev["run_records"]:
        if r.get("event") == "update_applied" and r.get("arm") == "agent":
            r["rule_file"] = "/opt/bia/substrate/reference_recipe.py"
            break
    return ev


def plant_out_of_range_score(ev):
    ev["score"] = 1.7
    return ev


def plant_unclamped_score(ev):
    ev["raw"] = 0.25
    ev["score"] = 0.9
    return ev


def plant_recompute_divergence(ev):
    ev["independent_val_loss"] = float(ev["arms"]["agent"]["streaming_val_loss"]) + 1.0
    return ev


def plant_in_process_agent_arm(ev):
    ev["agent_arm_isolation"] = {"mode": "in_verifier", "worker": None,
                                 "runner_pid": 1, "worker_pid": 1}
    return ev


def plant_unmeasured_displacement(ev):
    ev["agent_weight_displacement_l2"] = None
    return ev


def plant_forged_anchor_cache(ev):
    ev["anchor_cache_state"] = "unauthenticated"
    return ev


def plant_cache_source_without_entry(ev):
    ev["anchor_cache_state"] = "absent"
    ev["anchor_source"] = "cache"
    return ev


def plant_verifier_saw_no_displacement(ev):
    ev["agent_weight_displacement_l2"] = 0.0
    return ev


DEFECTS = [
    ("ablation_absent_from_graded_run", "normalization reintroduced into the agent forward pass",
     plant_normalization_event_in_agent_arm, "normalization_executed_in_ablated_arm:agent"),
    ("ablation_absent_from_graded_run", "probe blinded, no witness in the un-ablated arm",
     plant_blind_probe, "probe_inert_no_witness"),
    ("substrate_bytes_unmodified", "a substrate file changed during the run",
     plant_substrate_drift, "substrate_hash_drift_during_run"),
    ("ablation_gap_positive", "ablation costs nothing, baseline better than target",
     plant_anchor_inversion, "anchor_inversion"),
    ("ablation_gap_positive", "ablation gap below the bound margin",
     plant_anchor_gap_too_small, "anchor_gap_below_margin"),
    ("anchors_measured_before_agent_arm", "target anchor measured after the agent arm",
     plant_anchor_measured_late, "anchor_measured_after_agent_arm"),
    ("submitted_rule_moved_the_weights", "submitted rule left the weights untouched",
     plant_inert_update, None),
    ("submitted_rule_moved_the_weights", "update driven by a file other than the submission",
     plant_foreign_rule_file, "update_not_driven_by_submission_file"),
    ("reward_is_clamped_unit_float", "score outside the unit interval",
     plant_out_of_range_score, "score_outside_unit_interval"),
    ("reward_is_clamped_unit_float", "score is not the clamp of raw",
     plant_unclamped_score, "score_not_clamped_raw"),
    ("validation_loss_recompute_agrees", "post hoc recompute disagrees with the streaming value",
     plant_recompute_divergence, "validation_loss_recompute_diverged"),
    ("agent_arm_ran_outside_the_verifier", "agent arm executed inside the verifier interpreter",
     plant_in_process_agent_arm, "agent_arm_not_hosted_out_of_process"),
    ("agent_arm_ran_outside_the_verifier", "verifier recorded no displacement of its own",
     plant_unmeasured_displacement, "agent_weight_displacement_not_measured"),
    ("anchor_cache_entry_authentic", "anchor cache entry present with a tag that does not verify",
     plant_forged_anchor_cache, "anchor_cache_entry_failed_authentication"),
    ("anchor_cache_entry_authentic", "anchors claimed from a cache entry that was never there",
     plant_cache_source_without_entry, "anchor_source_disagrees_with_cache_state"),
    ("submitted_rule_moved_the_weights", "verifier's own displacement measurement is zero",
     plant_verifier_saw_no_displacement, "verifier_measured_no_weight_displacement"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    clean = json.load(open(args.evidence))
    rows = []
    failures = []

    outcomes, first_failure = CHECKERS.run_all(clean)
    for name, kind, _ in CHECKERS.REGISTRY:
        o = outcomes[name]
        rows.append({
            "checker": name, "kind": kind, "half": "ACCEPT",
            "fixture": "clean reference evidence",
            "verdict": "PASS" if o["passed"] else "FAIL",
            "reason": o["reason"],
        })
        if not o["passed"]:
            failures.append(f"clean evidence rejected by {name}: {o['reason']}")

    for target, label, mutate, expect in DEFECTS:
        ev = mutate(copy.deepcopy(clean))
        out, _ = CHECKERS.run_all(ev)
        o = out[target]
        ok = (not o["passed"]) and (expect is None or o["reason"] == expect
                                    or o["reason"].startswith(expect.split(":")[0]))
        rows.append({
            "checker": target,
            "kind": dict((n, k) for n, k, _ in CHECKERS.REGISTRY)[target],
            "half": "REJECT",
            "fixture": label,
            "verdict": "PASS" if ok else "FAIL",
            "reason": o["reason"],
        })
        if not ok:
            failures.append(f"planted defect '{label}' not rejected by {target}: {o['reason']}")

    width = max(len(r["checker"]) for r in rows) + 2
    print(f"{'checker'.ljust(width)}{'kind'.ljust(12)}{'half'.ljust(8)}{'verdict'.ljust(9)}reason / fixture")
    print("-" * 120)
    for r in rows:
        print(f"{r['checker'].ljust(width)}{r['kind'].ljust(12)}{r['half'].ljust(8)}"
              f"{r['verdict'].ljust(9)}{r['reason']}  <-  {r['fixture']}")
    print("-" * 120)
    covered = {r["checker"] for r in rows if r["half"] == "REJECT"}
    for name, _, _ in CHECKERS.REGISTRY:
        if name not in covered:
            failures.append(f"no rejecting control for {name}")

    if args.json_out:
        json.dump({"rows": rows, "failures": failures}, open(args.json_out, "w"), indent=1)

    if failures:
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"ALL CONTROLS PASSED: {len(rows)} halves over {len(CHECKERS.REGISTRY)} checkers")
    return 0


if __name__ == "__main__":
    sys.exit(main())

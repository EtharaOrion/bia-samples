"""Grader for the S03 data-order task.

The verifier runs the graded grid itself. Every comparator draw and every
submitted arm is a child interpreter this process spawns; every record arrives
on a pipe this process created; the record file is written by this process; and
the final validation loss each arm is compared on is measured in a further
process into which no submitted byte is ever loaded.

The previous revision graded a plain JSONL file that the solving side wrote and
could rewrite, which made a hand-written telemetry file worth full reward from a
run that did no work. There is no such file now. There is also no egress audit
on disk: connect attempts are reported straight to this process, so truncating
an audit is no longer a move that exists.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Resolved from this file, never from the environment: tests/ always sits inside
# the bundle, and the verifier must load its own substrate rather than a copy
# the graded party could have edited in its workspace.
BUNDLE = os.path.dirname(TESTS_DIR)
for p in (TESTS_DIR, os.path.join(BUNDLE, "environment", "runner")):
    if p not in sys.path:
        sys.path.insert(0, p)

from checkers import kinds  # noqa: E402
from harness import supervisor as sup  # noqa: E402

SCORE_PATH = os.environ.get("BIA_SCORE_PATH", "/logs/verifier/score.json")
OUTCOMES_PATH = os.environ.get("BIA_OUTCOMES_PATH", "/logs/verifier/outcomes.json")
REWARD_PATH = os.environ.get("BIA_REWARD_PATH", "/logs/verifier/reward.txt")
ORDER_MODULE = os.environ.get("BIA_ORDER_MODULE", "/workspace/submission/order.py")
KEEP_ROOT = os.environ.get("BIA_KEEP_VERIFIER_ROOT") == "1"

CHECKER_IDS = [
    "frozen_substrate_unchanged",
    "order_permutation_invariant",
    "order_frozen_before_training",
    "no_scored_egress",
    "order_determinism_divergence",
    "training_consumed_the_submitted_order",
    "improvement_clears_noise_floor",
]

RUN_ZERO_REASON = {
    "order_module_absent": "order-module-absent",
    "verifier_produced_no_records": "verifier-produced-no-records",
}


def emit(score, reason, detail=None, outcomes=None, reason_code="unattributed-zero"):
    try:
        value = min(max(float(score), 0.0), 1.0)
    except (TypeError, ValueError):
        value, reason = 0.0, "score_field_not_numeric"
        reason_code = "score-field-not-numeric"
    if not (reason or "").strip():
        reason = "zero_without_reason_refused_by_reward_contract" if value == 0.0 else "unattributed"
    vector = outcomes if outcomes is not None else {cid: False for cid in CHECKER_IDS}
    payload = {
        "score": value,
        "reason": reason,
        "reason_code": reason_code,
        "zero_reasons": [kinds.ZERO_REASON[cid] for cid in CHECKER_IDS if not vector.get(cid)],
        "detail": detail or {},
    }
    for path, body in ((SCORE_PATH, payload), (OUTCOMES_PATH, vector)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(body, f, indent=1, sort_keys=True)
    os.makedirs(os.path.dirname(REWARD_PATH) or ".", exist_ok=True)
    with open(REWARD_PATH, "w") as f:
        f.write(repr(value) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return value


def main() -> float:
    outcomes = {cid: False for cid in CHECKER_IDS}

    import profiles as profiles_mod
    import telemetry as telemetry_mod

    profile = profiles_mod.active_profile_name()
    cfg = profiles_mod.get_profile(profile)
    expected_substrate = profiles_mod.substrate_digest(cfg)
    expected_sorted_digest = telemetry_mod.digest_json(list(range(cfg["n_train_sequences"])))

    if not os.path.isfile(ORDER_MODULE):
        return emit(0.0, "order_module_absent", {"path": ORDER_MODULE}, outcomes,
                    RUN_ZERO_REASON["order_module_absent"])

    supervisor = sup.Supervisor(BUNDLE, profile, os.path.abspath(ORDER_MODULE))
    try:
        a, err_a = supervisor.derive_order("alpha", os.path.join(BUNDLE, "environment", "runner"))
        b, err_b = supervisor.derive_order("beta", TESTS_DIR)
        if a is None:
            det = (False, "order_derivation_a_failed_%s" % err_a, {})
        elif b is None:
            det = (False, "order_derivation_b_failed_%s" % err_b, {})
        elif not a.get("valid") or not b.get("valid"):
            det = (False, "order_invalid_%s" % (a.get("reason") or b.get("reason")), {})
        elif a["order_digest"] != b["order_digest"]:
            det = (False, "order_not_deterministic_across_hosts",
                   {"a": a["order_digest"], "b": b["order_digest"]})
        else:
            det = (True, None, {"order_digest": a["order_digest"]})

        expected_arms = []
        for seed in cfg["graded_seeds"]:
            for draw in range(cfg["comparator_draws"]):
                expected_arms.append("baseline_seed%d_draw%d" % (seed, draw))
            expected_arms.append("submitted_seed%d" % seed)

        try:
            supervisor.start_measure()
            measured = {}

            def run_and_measure(arm, seed, draw):
                """Measure and delete before the next child can reach the file.

                Measuring the whole grid afterwards would leave every earlier
                arm's parameters on disk while later arms run, and a submitted
                arm that overwrote a comparator arm's parameters with a worse
                model would inflate the improvement it is graded on. The
                checkpoint of an arm is consumed and gone before the next arm
                starts.
                """
                ckpt = supervisor.run_arm(arm, seed, draw)
                measured[ckpt] = supervisor.measure(ckpt)
                try:
                    os.remove(ckpt)
                except OSError:
                    pass

            for seed in cfg["graded_seeds"]:
                for draw in range(cfg["comparator_draws"]):
                    run_and_measure("baseline", seed, draw)
                run_and_measure("submitted", seed, 0)
        except sup.ChildFailure as exc:
            # A child that refuses the submitted order at the contract is the
            # rejecting half of order_permutation_invariant, so it is attributed
            # to that checker rather than to a generic child failure. Attributing
            # it generically would leave the checker with no observable failing
            # half through the graded path, which invariant 17 calls inert.
            fault = (exc.detail or {}).get("fault") or {}
            if fault.get("code") == "order_contract_violated":
                return emit(0.0,
                            "order_permutation_invariant:%s" % fault.get("reason"),
                            exc.detail, outcomes,
                            kinds.ZERO_REASON["order_permutation_invariant"])
            return emit(0.0, exc.code, exc.detail, outcomes, exc.code)
        finally:
            supervisor.stop_measure()

        for record in supervisor.records:
            if record.get("event") != "run_complete":
                continue
            doc = measured.get(record.get("checkpoint")) or {}
            record["final_val_loss"] = doc.get("final_val_loss")
            record["measured_by"] = "tests/harness/measure_child.py"
            record.pop("checkpoint", None)
        records = supervisor.records
        if not records:
            return emit(0.0, "verifier_produced_no_records", {}, outcomes,
                        RUN_ZERO_REASON["verifier_produced_no_records"])
        supervisor.flush()

        unmeasured = [r.get("run_id") for r in records
                      if r.get("event") == "run_complete" and r.get("final_val_loss") is None]
        if unmeasured:
            return emit(0.0, "verifier_measurement_missing_%s" % unmeasured[0],
                        {"runs": unmeasured[:8]}, outcomes, "verifier-measurement-missing")

        arms_observed = sorted({r["run_id"] for r in records if r.get("run_id")})
        results = {
            "frozen_substrate_unchanged": kinds.frozen_substrate_unchanged(
                records, expected_substrate),
            "order_permutation_invariant": kinds.order_permutation_invariant(
                records, cfg["n_train_sequences"], expected_sorted_digest),
            "order_frozen_before_training": kinds.order_frozen_before_training(records),
            "no_scored_egress": kinds.no_scored_egress(
                supervisor.egress_events, arms_observed, expected_arms),
            "order_determinism_divergence": det,
            "training_consumed_the_submitted_order": kinds.training_consumed_the_submitted_order(
                records, (det[2] or {}).get("order_digest")),
            "improvement_clears_noise_floor": kinds.improvement_clears_noise_floor(
                records, list(cfg["graded_seeds"]), int(cfg["comparator_draws"]),
                profiles_mod.SIGNIFICANCE_T),
        }

        for cid in CHECKER_IDS:
            outcomes[cid] = bool(results[cid][0])
        for cid in CHECKER_IDS:
            ok, reason, detail = results[cid]
            if not ok:
                return emit(0.0, f"{cid}:{reason}", detail, outcomes, kinds.ZERO_REASON[cid])

        stats = results["improvement_clears_noise_floor"][2]
        raw = stats["mean_delta"] / profiles_mod.TARGET_DELTA_NATS
        detail = dict(stats)
        detail["raw"] = round(raw, 6)
        detail["target_delta_nats"] = profiles_mod.TARGET_DELTA_NATS
        detail["profile"] = cfg["profile"]
        detail["measured_by"] = "verifier re-execution, tests/harness/measure_child.py"
        return emit(min(max(raw, 0.0), 1.0), "graded", detail, outcomes, "graded")
    finally:
        if not KEEP_ROOT:
            shutil.rmtree(supervisor.root, ignore_errors=True)


if __name__ == "__main__":
    main()

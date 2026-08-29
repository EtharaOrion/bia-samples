"""The gate chain. Runs the checkers, recomputes both readouts, writes the reward.

Order of operations, and why it is this order:

1. The verifier reads its OWN frozen substrate from tests/fixtures/substrate.json. It
   never reads the agent-visible environment/ for anything it will grade on; it hashes
   that surface and compares, which is a different act from trusting it.
2. runner.py executes the submission in isolation against a verifier-owned copy of the
   harness. This process never imports the submission.
3. Every configuration in the recorded attempt ledger is RE-SIMULATED here, by
   tests/harness_sim.py, over the verifier's frozen trace. The resulting telemetry is
   what the checkers read. The submission's own ledger numbers are carried only so the
   divergence checker can refuse them.
4. The checkers in tests/checkers.py run in graded order. The first failure is the primary
   attribution; the full failed set travels with it.
5. The reward is written last, to the bound contract path, as one float in [0.0, 1.0].

THE CLOCK. Nothing in this file reads a clock either. The tick clock lives inside
harness_sim.py, is virtual, integer, and a pure function of the frozen trace, the frozen
envelope and the configuration. That is the whole reason a timing metric can be graded by
pure checkers at all.

The companion compiled tests live in tests/test_output.py and the declared manifest in
tests/checkers.yaml; both are generated from solution/grounding.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness_sim  # noqa: E402
import runner  # noqa: E402
from checkers import (  # noqa: E402
    GRADED,
    Handle,
    check_attempt_ledger_recomputed,
    check_carried_best_preserved,
    check_early_stop_not_a_result,
    check_environment_frozen,
    check_harness_owns_the_clock,
    check_no_request_shed,
    check_p99_recomputed_unsmoothed,
    check_reallocation_after_flattening,
    check_slo_hard_constraint,
    check_sustained_across_windows,
    check_throughput_from_harness_telemetry,
    check_trace_fully_accounted,
    evaluate,
)

# The bound reward contract. /logs/verifier/reward.txt carries the bare float and
# /logs/verifier/score.json carries the machine-readable reason and the metric block.
# Only a local exercise redirects the root; the bundle binds these paths.
REWARD_PATH = "/logs/verifier/reward.txt"
SCORE_PATH = "/logs/verifier/score.json"
DEFAULT_REWARD_ROOT = "/logs/verifier"

FIXTURES = "fixtures"
SUBSTRATE = "substrate.json"

# The files whose bytes are the graded checker path. `check_harness_owns_the_clock` reads
# exactly these, so the ban is measured against what ran and not against a claim.
GRADED_PATH_FILES = ("checkers.py", "grade.py", "harness_sim.py")

_LATTICE_CACHE: dict = {}


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_substrate(tests_dir: Path) -> dict:
    return json.loads((tests_dir / FIXTURES / SUBSTRATE).read_text(encoding="utf-8"))


def observed_digests(environment_dir: Path, names) -> dict:
    rows = {}
    for name in names:
        path = environment_dir / name
        rows[name] = digest_bytes(path.read_bytes()) if path.is_file() else ""
    return rows


def graded_path_source(tests_dir: Path) -> str:
    return "\n".join(
        (tests_dir / name).read_text(encoding="utf-8") for name in GRADED_PATH_FILES if (tests_dir / name).is_file()
    )


def recompute_row(trace: dict, hardware: dict, config) -> dict:
    telemetry = harness_sim.simulate(trace, hardware, config)
    row = telemetry.as_row()
    row["record_digest"] = telemetry.record_digest()
    row["record_ids"] = [item["id"] for item in telemetry.records]
    return row


def calibration(trace: dict, hardware: dict, objective: dict) -> dict:
    """The two endpoints of the reward scale, both measured here, neither authored.

    The published F11 anchors are ABSENT (gap-oer-per-family-anchors-unmeasured), so this
    bundle binds the reward schema in full and derives its two endpoints from frozen bytes
    on every run: the baseline is the shipped default configuration, the target is the
    best objective-feasible configuration of the frozen lattice. No figure for either
    appears as a literal anywhere in this bundle.
    """
    key = harness_sim.digest([trace, hardware, objective])
    if key in _LATTICE_CACHE:
        return _LATTICE_CACHE[key]
    slo = int(objective["p99_tpot_centiticks"])
    base = harness_sim.simulate(trace, hardware, harness_sim.DEFAULT_CONFIG)
    best = None
    for combo in itertools.product(*[harness_sim.LATTICE[axis] for axis in harness_sim.AXES]):
        config = dict(zip(harness_sim.AXES, combo))
        candidate = harness_sim.simulate(trace, hardware, config)
        if harness_sim.feasible(candidate, slo) and harness_sim.better(candidate, best):
            best = candidate
    row = {
        "anchors_state": "absent",
        "anchors_gap": "gap-oer-per-family-anchors-unmeasured",
        "baseline_metric": None,
        "target_metric": None,
        "baseline_num": base.throughput_num,
        "baseline_den": base.throughput_den,
        "target_num": best.throughput_num if best else 0,
        "target_den": best.throughput_den if best else 1,
        "target_config": best.config if best else None,
    }
    _LATTICE_CACHE[key] = row
    return row


def raw_score(agent: dict, cal: dict) -> tuple:
    """raw = (agent - baseline) / (target - baseline), evaluated over exact integers."""
    an, ad = int(agent["throughput_num"]), int(agent["throughput_den"])
    bn, bd = int(cal["baseline_num"]), int(cal["baseline_den"])
    tn, td = int(cal["target_num"]), int(cal["target_den"])
    if ad <= 0 or bd <= 0 or td <= 0:
        return 0.0, "calibration-degenerate"
    numerator = (an * bd - bn * ad) * td
    denominator = (tn * bd - bn * td) * ad
    if denominator <= 0:
        return 0.0, "calibration-degenerate"
    value = numerator / denominator
    clipped = min(max(value, 0.0), 1.0)
    if clipped <= 0.0:
        return 0.0, "no-improvement-over-baseline"
    return clipped, "target-reached" if clipped >= 1.0 else "partial-improvement"


def build_handle(session, substrate: dict, environment_dir: Path, tests_dir: Path) -> Handle:
    trace = substrate["trace"]
    hardware = substrate["hardware"]
    objective = substrate["objective"]
    ledger = tuple((session or {}).get("attempts") or ())
    recomputed = tuple(recompute_row(trace, hardware, row.get("config")) for row in ledger)
    return Handle(
        session_present=session is not None,
        bound_digests=dict(substrate["bound_digests"]),
        observed_digests=observed_digests(environment_dir, sorted(substrate["bound_digests"])),
        slo_p99_centiticks=int(objective["p99_tpot_centiticks"]),
        window_count=int(substrate["window_count"]),
        sustain_floor_num=int(substrate["sustain_floor_num"]),
        sustain_floor_den=int(substrate["sustain_floor_den"]),
        flatten_window=int(substrate["flatten_window"]),
        trace_ids=tuple(str(row["id"]) for row in trace["requests"]),
        ledger=ledger,
        recomputed=recomputed,
        selected_index=int((session or {}).get("selected_attempt", -1)),
        reported=dict((session or {}).get("reported") or {}),
        graded_path_source=graded_path_source(tests_dir),
    )


def score_session(session, substrate: dict, environment_dir: Path, tests_dir: Path) -> dict:
    handle = build_handle(session, substrate, environment_dir, tests_dir)
    verdicts = evaluate(handle)
    failed = [row["id"] for row in verdicts if not row["passed"]]
    reasons = [row["reason"] for row in verdicts if not row["passed"]]
    cal = calibration(substrate["trace"], substrate["hardware"], substrate["objective"])
    metric = {
        "anchors_state": cal["anchors_state"],
        "anchors_gap": cal["anchors_gap"],
        "baseline_metric": cal["baseline_metric"],
        "target_metric": cal["target_metric"],
        "baseline_tokens_per_second_num": cal["baseline_num"],
        "baseline_tokens_per_second_den": cal["baseline_den"],
        "target_tokens_per_second_num": cal["target_num"],
        "target_tokens_per_second_den": cal["target_den"],
        "slo_p99_tpot_centiticks": handle.slo_p99_centiticks,
        "agent_tokens_per_second_num": None,
        "agent_tokens_per_second_den": None,
        "agent_p99_tpot_centiticks": None,
        "attempts_recorded": len(handle.ledger),
        "selected_attempt": handle.selected_index,
    }
    if failed:
        return {
            "reward": 0.0,
            "reason": reasons[0],
            "failed_checkers": failed,
            "failed_reasons": reasons,
            "verdicts": verdicts,
            "metric": metric,
        }
    selected = handle.selected()
    metric["agent_tokens_per_second_num"] = selected["throughput_num"]
    metric["agent_tokens_per_second_den"] = selected["throughput_den"]
    metric["agent_p99_tpot_centiticks"] = selected["p99_tpot_centiticks"]
    reward, reason = raw_score(selected, cal)
    return {
        "reward": round(float(reward), 6),
        "reason": reason,
        "failed_checkers": [],
        "failed_reasons": [],
        "verdicts": verdicts,
        "metric": metric,
    }


def write_reward(root: Path, payload: dict) -> None:
    """The last thing that happens. One bare float, plus the reason document beside it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "score.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "reward.txt").write_text("{0:.6f}\n".format(float(payload["reward"])), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Grade one OER-23 submission.")
    parser.add_argument("--bundle", default=str(HERE.parent))
    parser.add_argument("--submission", default="")
    parser.add_argument("--session", default="", help="score a recorded session instead of running one")
    parser.add_argument("--reward-root", default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--workspace", default="")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args(argv)

    bundle = Path(args.bundle).resolve()
    tests_dir = bundle / "tests"
    environment_dir = bundle / "environment"
    substrate = load_substrate(tests_dir)
    root = Path(args.reward_root)

    if args.session:
        session = json.loads(Path(args.session).read_text(encoding="utf-8"))
        payload = score_session(session, substrate, environment_dir, tests_dir)
    else:
        workspace = Path(args.workspace or (root / "work"))
        workspace.mkdir(parents=True, exist_ok=True)
        substrate_dir = runner.materialise_substrate(substrate, workspace / "substrate")
        outcome = runner.run_submission(
            Path(args.submission), tests_dir / "harness_sim.py", substrate_dir, workspace
        )
        if not outcome["present"]:
            payload = {
                "reward": 0.0,
                "reason": outcome["reason"] or "session-absent",
                "failed_checkers": [],
                "failed_reasons": [outcome["reason"] or "session-absent"],
                "verdicts": [],
                "metric": {"stderr": outcome["stderr"][-800:]},
            }
        else:
            payload = score_session(outcome["session"], substrate, environment_dir, tests_dir)

    payload["slot"] = "OER-23"
    payload["reward_path"] = REWARD_PATH
    payload["score_document_path"] = SCORE_PATH
    write_reward(root, payload)
    if args.print:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

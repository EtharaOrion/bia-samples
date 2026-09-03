"""The graded gate chain and the reward write for slot OER-21.

This file imports the checkers and the harness engine. It never imports the
submission: the submission is launched by tests/runner.py as a separate process
group and hands back one artifact, a quantization scheme specification.

The gate chain is ordered and the FIRST failing required checker names the zero.
The order is fixed here rather than discovered, so a given defect always produces
the same machine-readable reason:

    1 frozen_corpus_and_reference_unmodified      the inputs are what they were
    2 evaluation_follows_state_seal               the readings came after the seal
    3 submitted_scheme_changes_quantized_state    something was actually done
    4 bit_budget_respected                        it was paid for
    5 graded_readout_is_raw_recompute             the number was not shaped
    6 evaluation_not_halted_early                 the metric was established
    7 degradation_sustained_across_scheduled_pts  the reading held
    8 beats_default_scheme_optimum                it beat the handed option set

Once every gate is open the reward is the bounded continuous scaling of the
graded metric. That is what `required_pass` describes in tests/checkers.yaml: the
gate, not the number. The number beyond the gate is never binary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import evaluate  # noqa: E402
import runner  # noqa: E402

from checkers import (  # noqa: E402
    check_beats_default_scheme_optimum,
    check_bit_budget_respected,
    check_degradation_sustained_across_scheduled_points,
    check_evaluation_follows_state_seal,
    check_evaluation_not_halted_early,
    check_frozen_inputs_unmodified,
    check_graded_readout_is_raw_recompute,
    check_submitted_scheme_changes_quantized_state,
)

BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_SCORE_PATH = "/logs/verifier/score.json"

GATE_CHAIN = (
    ("frozen_corpus_and_reference_unmodified", check_frozen_inputs_unmodified),
    ("evaluation_follows_state_seal", check_evaluation_follows_state_seal),
    ("submitted_scheme_changes_quantized_state", check_submitted_scheme_changes_quantized_state),
    ("bit_budget_respected", check_bit_budget_respected),
    ("graded_readout_is_raw_recompute", check_graded_readout_is_raw_recompute),
    ("evaluation_not_halted_early", check_evaluation_not_halted_early),
    ("degradation_sustained_across_scheduled_points", check_degradation_sustained_across_scheduled_points),
    ("beats_default_scheme_optimum", check_beats_default_scheme_optimum),
)

SELECTORS = dict(GATE_CHAIN)


def environment_paths(bundle: Path = BUNDLE) -> dict:
    root = Path(os.environ.get("FORGE_ENVIRONMENT_ROOT") or (bundle / "environment"))
    return {
        "model": root / "model" / "model.json",
        "corpus": root / "corpus" / "eval_corpus.json",
        "reference": root / "model" / "reference.json",
    }


def anchors(bundle: Path = BUNDLE) -> dict:
    return evaluate.load_json(bundle / "tests" / "anchors.json")


def telemetry_for(scheme: dict, bundle: Path = BUNDLE, paths: dict | None = None) -> dict:
    return evaluate.build_telemetry(paths or environment_paths(bundle), scheme, anchors(bundle))


def telemetry_for_fixture(name: str, bundle: Path = BUNDLE) -> dict:
    scheme = evaluate.load_json(bundle / "tests" / "fixtures" / (name + ".json"))
    return telemetry_for(scheme, bundle)


def run_one(ident: str, telemetry: dict):
    selector = SELECTORS.get(ident)
    if selector is None:
        return checkers.Result(ident, False, "checker-unknown", "no such checker in the gate chain")
    return selector(telemetry)


def run_all(telemetry: dict) -> list:
    return [selector(telemetry) for _, selector in GATE_CHAIN]


def score(telemetry: dict) -> dict:
    results = run_all(telemetry)
    scaling = ((telemetry.get("anchors") or {}).get("local_scaling")) or {}
    metric_block = {
        "graded_metric": (telemetry.get("graded") or {}).get("metric"),
        "graded_metric_units": "percent of the unquantized reference perplexity",
        "graded_metric_source": (telemetry.get("graded") or {}).get("source"),
        "allocated_bits": (telemetry.get("budget") or {}).get("allocated_bits"),
        "bit_budget_total": (telemetry.get("budget") or {}).get("bound_total_bits"),
        "default_scheme_optimum": scaling.get("default_optimum"),
        "reference_scheme_optimum": scaling.get("reference_optimum"),
        "separation_margin": scaling.get("separation_margin"),
        "anchors_state": (telemetry.get("anchors") or {}).get("state"),
        "baseline_metric": (telemetry.get("anchors") or {}).get("baseline_metric"),
        "target_metric": (telemetry.get("anchors") or {}).get("target_metric"),
        "anchors_gap": (telemetry.get("anchors") or {}).get("gap"),
    }
    for result in results:
        if not result.passed:
            return {
                "reward": 0.0,
                "reason": result.reason,
                "failed_checker": result.ident,
                "observed": result.observed,
                "metric": metric_block,
                "checkers": [row.ident + ":" + ("pass" if row.passed else row.reason) for row in results],
            }
    raw = checkers.scaled_improvement(telemetry)
    if raw is None:
        return {
            "reward": 0.0,
            "reason": "scaling-endpoints-unavailable",
            "failed_checker": None,
            "observed": "the record carries no comparable scaling endpoints",
            "metric": metric_block,
            "checkers": [row.ident + ":pass" for row in results],
        }
    reward = min(max(float(raw), 0.0), 1.0)
    return {
        "reward": reward,
        "reason": "graded",
        "failed_checker": None,
        "observed": "every required checker passed",
        "metric": metric_block,
        "checkers": [row.ident + ":pass" for row in results],
    }


def reward_root() -> Path:
    override = os.environ.get("FORGE_REWARD_ROOT")
    if override:
        return Path(override)
    return Path(BOUND_REWARD_PATH).parent


def write_reward(document: dict) -> None:
    root = reward_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "score.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "reward.txt").write_text(repr(float(document["reward"])) + "\n", encoding="utf-8")


def main() -> int:
    submission = Path(os.environ.get("FORGE_SUBMISSION") or (BUNDLE / "solution" / "reference.py"))
    try:
        outcome = runner.run_submission(submission)
        document = score(telemetry_for(outcome["scheme"]))
        document["submission"] = {
            "exit_code": outcome["exit_code"],
            "stdout_bytes": outcome["stdout_bytes"],
            "stdout_parsed_for_a_metric": False,
        }
    except runner.SubmissionError as error:
        document = {
            "reward": 0.0,
            "reason": str(error),
            "failed_checker": None,
            "observed": "the submission handed back no gradable scheme",
            "metric": {"graded_metric": None},
        }
    except evaluate.SchemeError:
        document = {
            "reward": 0.0,
            "reason": "scheme-malformed",
            "failed_checker": None,
            "observed": "the submitted scheme is not a well-formed allocation over the frozen tensors",
            "metric": {"graded_metric": None},
        }
    write_reward(document)
    print(json.dumps({"reward": document["reward"], "reason": document["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

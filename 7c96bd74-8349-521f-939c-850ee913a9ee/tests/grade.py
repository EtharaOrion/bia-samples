"""The grading path. It imports the checkers, the harness and the runner, never the submission.

Order matters and is recorded as it happens. The grading tree is digested, the
submission is run in isolation by tests/runner.py, generation is closed out, and only
then is the held-out benchmark opened. Every step appends an event to a log this
process owns, which is what makes the ordering claim a measurement rather than a
promise.

The graded number is produced here: tests/harness.py trains the frozen model on the
generated corpus under the frozen budget and the verifier evaluates the harness-owned
parameter vector against the held-out items at the points the verifier scheduled. A
number the submission printed is parsed, recorded under `reported`, compared for
divergence, and never carried into the reward.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkers  # noqa: E402
import harness  # noqa: E402
import runner  # noqa: E402
from checkers import (  # noqa: E402
    check_benchmark_unread_on_generation_path,
    check_early_stop_not_a_score,
    check_evaluated_weights_harness_owned,
    check_graded_score_from_harness_state,
    check_graded_score_unsmoothed,
    check_grading_tree_immutable,
    check_heldout_contamination_absent,
    check_score_sustained_across_eval_points,
    check_training_budget_respected,
)

try:
    import yaml
except ImportError:  # pragma: no cover - the pinned verifier image carries PyYAML
    yaml = None

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
MANIFEST = HERE / "checkers.yaml"
BENCHMARK = HERE / "heldout" / "benchmark.jsonl"

# The bound reward contract path. The runtime mounts /logs/verifier and the live
# verifier binds /logs/verifier/reward.txt as the carrier of the bare float.
REWARD_PATH = "/logs/verifier/reward.txt"
SCORE_PATH = "/logs/verifier/score.json"
REWARD_ROOT_ENV = "OER17_REWARD_ROOT"
DEFAULT_SUBMISSION = "/workspace/submission/generator.py"

# Named here so the manifest's declared selector for every checker is literally present
# in the carrier the manifest says reaches it.
SELECTORS = {
    "heldout_contamination_absent": check_heldout_contamination_absent,
    "benchmark_unread_on_generation_path": check_benchmark_unread_on_generation_path,
    "grading_tree_immutable": check_grading_tree_immutable,
    "training_budget_respected": check_training_budget_respected,
    "graded_score_unsmoothed": check_graded_score_unsmoothed,
    "score_sustained_across_eval_points": check_score_sustained_across_eval_points,
    "early_stop_not_a_score": check_early_stop_not_a_score,
    "graded_score_from_harness_state": check_graded_score_from_harness_state,
    "evaluated_weights_harness_owned": check_evaluated_weights_harness_owned,
}


def load_manifest(path: Path = MANIFEST) -> dict:
    if yaml is None:
        raise RuntimeError("the verifier image must carry PyYAML to read checkers.yaml")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def reward_root() -> Path:
    """The bound root, or the lane-private root a local exercise redirects it to.

    /logs/verifier is one shared host path and concurrent lanes collide on it, so a
    local exercise redirects the root and says so. The bound contract path inside this
    bundle stays /logs/verifier/reward.txt; only the exercise moves.
    """
    return Path(os.environ.get(REWARD_ROOT_ENV, "/logs/verifier"))


class EventLog:
    """An ordered log this process owns. Nothing outside the verifier appends to it."""

    def __init__(self) -> None:
        self.rows: list = []

    def add(self, name: str, detail: str = "") -> None:
        self.rows.append({"index": len(self.rows), "name": str(name), "detail": str(detail)})

    def as_list(self) -> list:
        return list(self.rows)


def build_record(submission: Path, bound: dict) -> dict:
    """Run the whole graded pipeline once and return the record the checkers read."""
    log = EventLog()
    digest_before, file_count = harness.digest_tree(HERE)
    log.add("grading-tree-digested", digest_before[:16])

    log.add("generation-started", Path(submission).name)
    generation = runner.run_submission(
        Path(submission),
        float(bound["generation_timeout_sec"]),
        int(bound["max_corpus_samples"]),
    )
    log.add("generation-finished", "exit " + str(generation.get("exit_code")))

    # Only now, with the generation process dead and its workspace removed, does the
    # held-out benchmark enter this process at all.
    items = harness.load_jsonl(BENCHMARK)
    log.add("benchmark-loaded", str(len(items)) + " items")

    log.add("training-started", str(len(generation["samples"])) + " samples")
    training = harness.train(generation["samples"], items, bound)
    for point in training["eval_points"]:
        log.add("evaluation-point", "step " + str(point["step"]))
    log.add("training-finished", "step " + str(training["halted_at_step"]))

    digest_after, _ = harness.digest_tree(HERE)
    log.add("grading-tree-redigested", digest_after[:16])

    index = int(bound["graded_eval_point_index"])
    points = training["eval_points"]
    graded_point = points[index] if 0 <= index < len(points) else {"step": None, "raw_score": 0.0, "weights_sha256": ""}
    measured = harness.evaluate(
        training["final_weights"], items, float(bound["evaluation_relative_tolerance"])
    )

    record = {
        "schema": harness.SCHEMA,
        "events": log.as_list(),
        "generation": generation,
        "samples": generation["samples"],
        "benchmark": {
            "items": items,
            "item_count": len(items),
            "revision": str(bound["benchmark_revision"]),
            "opened_at_event_index": next(
                (row["index"] for row in log.as_list() if row["name"] == "benchmark-loaded"), -1
            ),
        },
        "training": {
            "optimizer": training["optimizer"],
            "learning_rate": training["learning_rate"],
            "steps_fed": training["steps_fed"],
            "tokens_fed": training["tokens_fed"],
            "corpus_samples": training["corpus_samples"],
            "usable_samples": training["usable_samples"],
            "halted_at_step": training["halted_at_step"],
            "completed": training["completed"],
            "checkpoint_selected_by": training["checkpoint_selected_by"],
            "weights_by_step": training["weights_by_step"],
        },
        "eval_points": points,
        "measurement": {
            "score": measured,
            "weights": training["final_weights"],
            "computed_by": "tests/harness.py::evaluate",
        },
        "graded": {
            "eval_point_index": index,
            "step": graded_point["step"],
            "score": graded_point["raw_score"],
            "smoothing": "none",
            "source": "verifier-measurement",
            "weights_sha256": graded_point["weights_sha256"],
        },
        "reported": {
            "score": generation.get("reported_score"),
            "smoothing": generation.get("reported_smoothing"),
            "note": "recorded for divergence only; never graded",
        },
        "grading_tree": {
            "digest_before": digest_before,
            "digest_after": digest_after,
            "file_count": file_count,
        },
    }
    return record


def gate_chain(record: dict, manifest: dict) -> tuple:
    """required_pass over the declared checkers, in declared order. First zero wins."""
    bound = manifest["bound_values"]
    outcomes = []
    for row in manifest["checkers"]:
        ident = str(row["id"])
        selector = SELECTORS[ident]
        try:
            outcome = selector(record, bound)
        except Exception as error:  # a checker that cannot run has not passed
            outcomes.append((ident, False, "checker-execution-failed", str(error)[:200]))
            return outcomes, "checker-execution-failed"
        outcomes.append((ident, outcome.ok, outcome.reason, outcome.detail))
        if not outcome.ok and bool(row.get("required", False)):
            return outcomes, outcome.reason
    return outcomes, ""


def score_run(record: dict, manifest: dict) -> dict:
    """The whole decision: gates first, then the anchored continuous reward."""
    outcomes, blocked = gate_chain(record, manifest)
    metric = {
        "agent_metric": record["graded"]["score"],
        "graded_step": record["graded"]["step"],
        "measured_by": "verifier",
        "baseline_metric": manifest["anchors"]["baseline_metric"],
        "target_metric": manifest["anchors"]["target_metric"],
        "anchors_state": manifest["anchors"]["anchors_state"],
        "reported_by_submission": record["reported"]["score"],
        "checkers": [
            {"id": ident, "passed": ok, "reason": reason, "detail": detail}
            for ident, ok, reason, detail in outcomes
        ],
    }
    if blocked:
        return {"reward": 0.0, "reason": blocked, "metric": metric}

    if int(record["training"]["usable_samples"]) <= 0:
        # No corpus means no trained model, so no score was established. That is an
        # attributed zero, not an absent result.
        return {"reward": 0.0, "reason": "no-usable-samples", "metric": metric}

    baseline = manifest["anchors"]["baseline_metric"]
    target = manifest["anchors"]["target_metric"]
    if baseline is None or target is None:
        # F14 anchors are unmeasured. The schema is bound and the number is absent, so
        # the run is scored zero with a reason instead of against an invented pair.
        return {"reward": 0.0, "reason": "anchors-unmeasured", "metric": metric}
    value = checkers.anchored_score(record["graded"]["score"], float(baseline), float(target))
    return {"reward": value, "reason": "graded", "metric": metric}


def write_decision(decision: dict, root: Path) -> None:
    """The last thing that happens: the bare float, then the score document."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "reward.txt").write_text(format(float(decision["reward"]), ".6f") + "\n", encoding="utf-8")
    (root / "score.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list) -> int:
    submission = Path(argv[1]) if len(argv) > 1 else Path(os.environ.get("OER17_SUBMISSION", DEFAULT_SUBMISSION))
    root = reward_root()
    manifest = load_manifest()
    if not submission.is_file():
        write_decision(
            {"reward": 0.0, "reason": "generation-failed", "metric": {"submission": str(submission), "present": False}},
            root,
        )
        print("no submission at " + str(submission))
        return 0
    record = build_record(submission, manifest["bound_values"])
    decision = score_run(record, manifest)
    write_decision(decision, root)
    print(json.dumps({"reward": decision["reward"], "reason": decision["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

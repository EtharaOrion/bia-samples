# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml
# Generator: solution/recompute.py
# Edit solution/grounding.yaml and re-run the generator; never edit this file.

"""The compiled per-checker suite for OER-15.

Each `test_<checker id>` carries BOTH halves of exactly one checker declared in
`tests/checkers.yaml`:

  accepting half  -- the LIVE reference run, driven through the real checker, must
                     pass. The base record is produced by `tests/runner.py`, which
                     isolates the submission and runs the frozen harness in this
                     process, so every number in it is a number the verifier made.
  rejecting half  -- either a second LIVE run of a defective submission, or the
                     base record with one minimal planted defect applied. The
                     checker must return exactly the `zero_reason` its manifest row
                     declares, and no other checker's reason is accepted.

Ten of the twelve rejecting halves are carried on planted telemetry fixtures,
because `tests/runner.py` never lets a submission near the step counter, the
parameters, the denominator, the training shards, the held-out slice or the
evaluation schedule. That substitution is recorded in `tests/checkers.yaml`.

COST. Every live half here is a real 2200-step training run of the canonical
12-layer 768-dimension decoder, plus the verifier's paired baseline run, over
staged FineWeb10B shards. This suite needs an accelerator and the shards; it does
not run on a laptop in seconds, and a version of it that did would be the same
class of stand-in the re-base removed.

`seed/tasks/OER-15/adequacy.py` imports the helpers below rather than
reimplementing them, so the adequacy proof and this suite can never disagree about
what a fixture is.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
TESTS = BUNDLE / "tests"
SOLUTION = BUNDLE / "solution"
ENVIRONMENT = BUNDLE / "environment"

sys.dont_write_bytecode = True
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import checkers as C  # noqa: E402
import grade as G  # noqa: E402


SUBMISSION_REFERENCE = (
    "import importlib.util\n"
    "_spec = importlib.util.spec_from_file_location('oer15_reference', %r)\n"
    "_m = importlib.util.module_from_spec(_spec)\n"
    "_spec.loader.exec_module(_m)\n"
    "def build_vocab(train_bytes, budget):\n"
    "    return _m.build_vocab(train_bytes, budget)\n"
) % (str(SOLUTION / "reference.py"),)

SUBMISSION_EARLY_STOP = SUBMISSION_REFERENCE + "HALT_AT_STEPS = %d\n"

SUBMISSION_PLATEAU = (
    "import importlib.util\n"
    "_spec = importlib.util.spec_from_file_location('oer15_default', %r)\n"
    "_m = importlib.util.module_from_spec(_spec)\n"
    "_spec.loader.exec_module(_m)\n"
    "_BEST = %s\n"
    "def build_vocab(train_bytes, budget):\n"
    "    return _m.build_vocab(train_bytes, budget, **_BEST)\n"
) % (str(ENVIRONMENT / "default_tokenizer.py"), "%s")

SUBMISSION_NOOP = "def build_vocab(train_bytes, budget):\n    return []\n"


def live_record(source: str) -> dict:
    """Run tests/runner.py over one submission source and return its run record.

    This is the only place a submission is ever executed, and it is executed as a
    separate program by the real runner. Nothing here imports a submission.
    """
    with tempfile.TemporaryDirectory(prefix="oer15-compiled-") as scratch:
        room = Path(scratch)
        (room / "submission").mkdir()
        (room / "submission" / "tokenizer.py").write_text(source, encoding="utf-8")
        out = room / "record.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(TESTS / "runner.py"),
                "--delivery-root", str(BUNDLE),
                "--submission", str(room / "submission"),
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
        )
        if not out.is_file():
            raise AssertionError("tests/runner.py wrote no record: " + completed.stderr[-400:])
        return json.loads(out.read_text(encoding="utf-8"))


_BASE: dict = {}


def base_record() -> dict:
    """The accepting fixture: one live reference run, computed once and reused."""
    if not _BASE:
        _BASE["record"] = live_record(SUBMISSION_REFERENCE)
    return copy.deepcopy(_BASE["record"])


def _dig(record: dict, path: list):
    node = record
    for key in path[:-1]:
        node = node[key]
    return node


def apply_ops(record: dict, ops: list) -> dict:
    """Apply the minimal planted defect described by grounding.yaml.

    The vocabulary is closed and small on purpose: every op names one location and
    one movement, so a fixture cannot quietly become a rewrite of the whole record.
    """
    out = copy.deepcopy(record)
    for op in ops:
        kind = op["op"]
        if kind == "set":
            _dig(out, op["path"])[op["path"][-1]] = op["value"]
        elif kind == "set_from":
            _dig(out, op["path"])[op["path"][-1]] = _dig(out, op["source"])[op["source"][-1]]
        elif kind == "add_int":
            node = _dig(out, op["path"])
            node[op["path"][-1]] = int(node[op["path"][-1]]) + int(op["delta"])
        elif kind == "swap_points":
            rows = out["telemetry"]["eval_points"]
            rows[op["a"]], rows[op["b"]] = rows[op["b"]], rows[op["a"]]
        elif kind == "add_float_last_sustain":
            rows = [r for r in out["telemetry"]["eval_points"] if r["role"] == "sustain"]
            rows[-1][op["key"]] = float(rows[-1][op["key"]]) + float(op["delta"])
        else:
            raise AssertionError("op outside the closed fixture vocabulary: " + repr(kind))
    return out


def outcome(record: dict, selector: str):
    """Drive ONE live checker over one record through the real evidence assembly.

    The verifier-owned anchors in `tests/anchors.json` are overlaid first, exactly as
    `tests/grade.py` overlays them before it assembles Evidence, so a checker reading a
    verifier-side operating point sees here what it sees on the graded path.
    """
    merged = G.merge_verifier_anchors(record, G.load_anchors())
    evidence = C.evidence_from_record(merged, str(BUNDLE))
    return getattr(C, selector)(evidence)


def accepts(selector: str) -> None:
    result = outcome(base_record(), selector)
    assert result.passed, selector + " rejected the live reference run: " + result.detail


def rejects(record: dict, selector: str, zero_reason: str) -> None:
    result = outcome(record, selector)
    assert not result.passed, selector + " accepted a planted defect: " + result.detail
    assert result.reason == zero_reason, (
        selector + " emitted " + repr(result.reason) + ", expected " + repr(zero_reason)
    )



PLATEAU_OPTIONS = {'top_k': 1024, 'min_frequency': 1, 'lowercase': False, 'attach_leading_space': True}
EARLY_STOP_AT = 1100


def test_early_stop_not_an_established_metric():
    """early-stop-no-established-metric. Both halves, one checker."""
    accepts('check_early_stop_not_an_established_metric')
    halted = live_record(SUBMISSION_EARLY_STOP % EARLY_STOP_AT)
    rejects(halted, 'check_early_stop_not_an_established_metric', 'early-stop-no-established-metric')


def test_frozen_axes_unmoved():
    """frozen-axis-moved. Both halves, one checker."""
    accepts('check_frozen_axes_unmoved')
    planted = apply_ops(base_record(), [{'op': 'set', 'path': ['telemetry', 'frozen', 'architecture', 'num_layers'], 'value': 11}])
    rejects(planted, 'check_frozen_axes_unmoved', 'frozen-axis-moved')


def test_snapshot_shapes_match_substrate():
    """snapshot-shape-off-substrate. Both halves, one checker."""
    accepts('check_snapshot_shapes_match_substrate')
    planted = apply_ops(base_record(), [{'op': 'set', 'path': ['telemetry', 'model', 'parameter_shapes', 'blocks.0.mlp.up.weight'], 'value': [3071, 768]}])
    rejects(planted, 'check_snapshot_shapes_match_substrate', 'snapshot-shape-off-substrate')


def test_eval_points_ordered_by_steps():
    """eval-schedule-out-of-order. Both halves, one checker."""
    accepts('check_eval_points_ordered_by_steps')
    planted = apply_ops(base_record(), [{'op': 'swap_points', 'a': -2, 'b': -1}])
    rejects(planted, 'check_eval_points_ordered_by_steps', 'eval-schedule-out-of-order')


def test_compute_budget_respected_as_spent():
    """compute-budget-overspent. Both halves, one checker."""
    accepts('check_compute_budget_respected_as_spent')
    planted = apply_ops(base_record(), [{'op': 'add_int', 'path': ['telemetry', 'compute', 'charged_steps'], 'delta': 300}])
    rejects(planted, 'check_compute_budget_respected_as_spent', 'compute-budget-overspent')


def test_denominator_is_held_out_slice_bytes():
    """denominator-not-held-out-slice-bytes. Both halves, one checker."""
    accepts('check_denominator_is_held_out_slice_bytes')
    planted = apply_ops(base_record(), [{'op': 'set', 'path': ['telemetry', 'denominator', 'source'], 'value': 'eval-token-count'}, {'op': 'set_from', 'path': ['telemetry', 'denominator', 'bytes'], 'source': ['telemetry', 'vocabulary', 'eval_tokens']}])
    rejects(planted, 'check_denominator_is_held_out_slice_bytes', 'denominator-not-held-out-slice-bytes')


def test_graded_state_is_harness_owned():
    """graded-state-not-harness-owned. Both halves, one checker."""
    accepts('check_graded_state_is_harness_owned')
    planted = apply_ops(base_record(), [{'op': 'set', 'path': ['telemetry', 'readout', 'weights_origin'], 'value': 'submission-selected-checkpoint'}])
    rejects(planted, 'check_graded_state_is_harness_owned', 'graded-state-not-harness-owned')


def test_graded_readout_unsmoothed():
    """graded-readout-smoothed. Both halves, one checker."""
    accepts('check_graded_readout_unsmoothed')
    planted = apply_ops(base_record(), [{'op': 'set', 'path': ['telemetry', 'readout', 'filter'], 'value': 'ema-0.9'}])
    rejects(planted, 'check_graded_readout_unsmoothed', 'graded-readout-smoothed')


def test_submission_vocabulary_took_effect():
    """submission-vocabulary-inert. Both halves, one checker."""
    accepts('check_submission_vocabulary_took_effect')
    planted = apply_ops(base_record(), [{'op': 'set', 'path': ['telemetry', 'vocabulary', 'submitted_digest'], 'value': '1111111111111111111111111111111111111111111111111111111111111111'}])
    rejects(planted, 'check_submission_vocabulary_took_effect', 'submission-vocabulary-inert')


def test_metric_not_taken_from_submission_report():
    """metric-sourced-from-submission. Both halves, one checker."""
    accepts('check_metric_not_taken_from_submission_report')
    planted = apply_ops(base_record(), [{'op': 'set', 'path': ['telemetry', 'reported_by_submission', 'bits_per_byte'], 'value': 0.1}, {'op': 'set_from', 'path': ['telemetry', 'graded', 'bits_per_byte'], 'source': ['telemetry', 'reported_by_submission', 'bits_per_byte']}])
    rejects(planted, 'check_metric_not_taken_from_submission_report', 'metric-sourced-from-submission')


def test_beats_default_construction_optimum():
    """default-construction-plateau-not-beaten. Both halves, one checker."""
    accepts('check_beats_default_construction_optimum')
    plateau = live_record(SUBMISSION_PLATEAU % (PLATEAU_OPTIONS,))
    rejects(plateau, 'check_beats_default_construction_optimum', 'default-construction-plateau-not-beaten')


def test_reading_sustained_across_scheduled_points():
    """reading-not-sustained. Both halves, one checker."""
    accepts('check_reading_sustained_across_scheduled_points')
    planted = apply_ops(base_record(), [{'op': 'add_float_last_sustain', 'key': 'bits_per_byte', 'delta': 0.5}])
    rejects(planted, 'check_reading_sustained_across_scheduled_points', 'reading-not-sustained')


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(list(globals().items())):
        if not name.startswith("test_"):
            continue
        try:
            function()
            print("PASS " + name)
        except AssertionError as problem:
            failures += 1
            print("FAIL " + name + ": " + str(problem))
    raise SystemExit(1 if failures else 0)

#!/usr/bin/env python3
"""The graded checkers for OER-12. Pure, deterministic, live state only.

Every function here is a pure function of the harness documents the VERIFIER'S OWN
PROCESS produced. Nothing here reads a planted file, an environment secret, a clock, a
random source or the network, and nothing here imports the submission. The whole import
surface of this module is json, math, hashlib, pathlib, dataclasses and typing, which
is the AST allowlist this bundle is held to.

The load-bearing convention: every harness document carries writer "verifier-harness".
A document whose writer is anything else is refused rather than read, which is how
"the graded number is never a number the submission reported" is made machine-checkable
instead of merely asserted. The submission's own ledger is read for exactly one purpose,
which is to be COMPARED AGAINST the harness's independent reading of the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HARNESS_WRITER = "verifier-harness"
SUBMISSION_WRITER = "submission"

VERDICT_FAILED = "parse-failed"
VERDICT_VALID = "parse-valid"
VERDICT_PARTIAL = "parse-partial"
VERDICTS = (VERDICT_FAILED, VERDICT_VALID, VERDICT_PARTIAL)

REASONS_OF_VERDICT = {
    VERDICT_FAILED: (
        "directive-malformed",
        "field-key-unknown",
        "record-id-absent",
        "record-id-duplicated",
    ),
    VERDICT_VALID: ("record-complete",),
    VERDICT_PARTIAL: (
        "required-field-absent",
        "record-unterminated",
        "continuation-dangling",
    ),
}

POLICY_OUTCOME = {
    "quarantine-partial": "partial-quarantined",
    "complete-partial": "partial-completed",
    "truncate-partial": "partial-truncated",
}

FROZEN_AXES = ("model", "optimizer", "token_budget", "eval_split")

# Every machine-readable zero reason this module can emit. Each one is distinct and
# each names a different thing that went wrong. The ambiguous-class reason is
# deliberately NOT a parse-failure reason: conflating them is the exact defect this
# slot exists to make loud.
ZERO_GRADED_LOSS_NOT_VERIFIER_COMPUTED = "graded-loss-not-verifier-computed"
ZERO_GRADED_READOUT_SMOOTHED = "graded-readout-smoothed"
ZERO_GRADED_LOSS_NOT_SUSTAINED = "graded-loss-not-sustained"
ZERO_EARLY_STOP_LOSS_NOT_ESTABLISHED = "early-stop-loss-not-established"
ZERO_EVALUATED_WEIGHTS_NOT_HARNESS_OWNED = "evaluated-weights-not-harness-owned"
ZERO_AMBIGUOUS_CLASS_COERCED_SILENTLY = "ambiguous-class-coerced-silently"
ZERO_PARSE_OUTCOME_CLASSES_CONFLATED = "parse-outcome-classes-conflated"
ZERO_POLICY_DECLARED_AFTER_CLASSIFICATION = "ambiguity-policy-declared-after-classification"
ZERO_TOKEN_BUDGET_OVERSPENT = "token-budget-overspent"
ZERO_EVALUATION_SPLIT_LEAKED = "evaluation-split-leaked-into-training"
ZERO_FROZEN_AXIS_MUTATED = "frozen-axis-mutated"

EPSILON = 1e-9


@dataclass(frozen=True)
class Handles:
    """The real harness handles. One directory the verifier's own process wrote.

    `harness` holds the documents the verifier produced. `submission_view` holds the
    copies the harness took of the submission's own artifacts, which exist only so a
    divergence can be measured against them. Nothing here is executed and nothing here
    is imported.
    """

    harness: Path
    submission_view: Path


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str
    detail: str


def _passed(detail: str) -> Verdict:
    return Verdict(True, "", detail)


def _read(directory: Path, name: str):
    path = directory / (name + ".json")
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def harness_document(handles: Handles, name: str):
    """A harness document, or None when it is absent or not written by the harness."""
    payload = _read(handles.harness, name)
    if not isinstance(payload, dict):
        return None
    if payload.get("writer") != HARNESS_WRITER:
        return None
    return payload


def submission_document(handles: Handles, name: str):
    """A copy of a submission artifact. Never a source of a graded number."""
    payload = _read(handles.submission_view, name)
    return payload if isinstance(payload, dict) else None


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _eval_record(ledger, step):
    for row in ledger.get("records") or []:
        if isinstance(row, dict) and row.get("step") == step:
            return row
    return None


# --------------------------------------------------------------------------------
# Readout-manipulation obligations, adapted to a LOSS metric.
# --------------------------------------------------------------------------------


def check_graded_loss_from_verifier_evaluation(handles: Handles) -> Verdict:
    """VALUE. The graded loss is the verifier's own evaluation, never a reported one."""
    run = harness_document(handles, "run")
    ledger = harness_document(handles, "eval_ledger")
    if run is None or ledger is None:
        return Verdict(False, ZERO_GRADED_LOSS_NOT_VERIFIER_COMPUTED,
                       "a harness-written run record and eval ledger are both required")
    if ledger.get("origin") != "verifier-recomputed":
        return Verdict(False, ZERO_GRADED_LOSS_NOT_VERIFIER_COMPUTED,
                       "eval ledger origin is " + repr(ledger.get("origin")))
    if ledger.get("evaluated_by") != "verifier":
        return Verdict(False, ZERO_GRADED_LOSS_NOT_VERIFIER_COMPUTED,
                       "eval ledger was evaluated by " + repr(ledger.get("evaluated_by")))
    graded = _eval_record(ledger, run.get("bound_eval_step"))
    if graded is None:
        return Verdict(False, ZERO_GRADED_LOSS_NOT_VERIFIER_COMPUTED,
                       "no eval record at the bound evaluation point")
    if graded.get("reported_by_submission"):
        return Verdict(False, ZERO_GRADED_LOSS_NOT_VERIFIER_COMPUTED,
                       "the graded record is flagged as reported by the submission")
    if not isinstance(graded.get("loss"), (int, float)):
        return Verdict(False, ZERO_GRADED_LOSS_NOT_VERIFIER_COMPUTED,
                       "the graded record carries no numeric loss")
    return _passed("graded loss recomputed by the verifier at step "
                   + str(run.get("bound_eval_step")))


def check_graded_readout_unsmoothed(handles: Handles) -> Verdict:
    """VALUE. No smoothing on the graded path; the graded loss is the raw evaluation."""
    run = harness_document(handles, "run")
    ledger = harness_document(handles, "eval_ledger")
    if run is None or ledger is None:
        return Verdict(False, ZERO_GRADED_READOUT_SMOOTHED,
                       "a harness-written run record and eval ledger are both required")
    if ledger.get("smoothing") != "none":
        return Verdict(False, ZERO_GRADED_READOUT_SMOOTHED,
                       "the graded path declares smoothing " + repr(ledger.get("smoothing")))
    graded = _eval_record(ledger, run.get("bound_eval_step"))
    if graded is None:
        return Verdict(False, ZERO_GRADED_READOUT_SMOOTHED,
                       "no eval record at the bound evaluation point")
    raw = graded.get("raw_batch_losses")
    if not isinstance(raw, list) or not raw:
        return Verdict(False, ZERO_GRADED_READOUT_SMOOTHED,
                       "the graded record carries no raw batch losses to recompute from")
    recomputed = math.fsum(float(value) for value in raw) / float(len(raw))
    if abs(recomputed - float(graded.get("loss"))) > EPSILON:
        return Verdict(False, ZERO_GRADED_READOUT_SMOOTHED,
                       "graded loss " + repr(graded.get("loss"))
                       + " is not the raw mean " + repr(recomputed))
    return _passed("graded loss equals the unsmoothed mean of its own raw batch losses")


def check_graded_loss_sustained(handles: Handles) -> Verdict:
    """INVARIANT. The graded level holds at every sustain point the harness schedules."""
    run = harness_document(handles, "run")
    ledger = harness_document(handles, "eval_ledger")
    if run is None or ledger is None:
        return Verdict(False, ZERO_GRADED_LOSS_NOT_SUSTAINED,
                       "a harness-written run record and eval ledger are both required")
    graded = _eval_record(ledger, run.get("bound_eval_step"))
    points = run.get("sustain_points")
    band = run.get("sustain_band")
    if graded is None or not isinstance(points, list) or not points:
        return Verdict(False, ZERO_GRADED_LOSS_NOT_SUSTAINED,
                       "the bound evaluation point or the sustain schedule is absent")
    if not isinstance(band, (int, float)):
        return Verdict(False, ZERO_GRADED_LOSS_NOT_SUSTAINED,
                       "the sustain band is unbound, so no level can be established")
    ceiling = float(graded.get("loss")) + float(band)
    for step in points:
        row = _eval_record(ledger, step)
        if row is None:
            return Verdict(False, ZERO_GRADED_LOSS_NOT_SUSTAINED,
                           "no eval record at sustain point " + str(step))
        if float(row.get("loss")) > ceiling + EPSILON:
            return Verdict(False, ZERO_GRADED_LOSS_NOT_SUSTAINED,
                           "loss at sustain point " + str(step) + " is "
                           + repr(row.get("loss")) + ", above the band ceiling "
                           + repr(ceiling))
    return _passed("the graded level held at all " + str(len(points)) + " sustain points")


def check_bound_evaluation_point_reached(handles: Handles) -> Verdict:
    """ABSENCE. No early stop. A halt before the schedule ends establishes no loss."""
    run = harness_document(handles, "run")
    if run is None:
        return Verdict(False, ZERO_EARLY_STOP_LOSS_NOT_ESTABLISHED,
                       "a harness-written run record is required")
    points = run.get("sustain_points")
    if not isinstance(points, list) or not points:
        return Verdict(False, ZERO_EARLY_STOP_LOSS_NOT_ESTABLISHED,
                       "the sustain schedule is absent, so no completion can be shown")
    if run.get("halt_kind") != "bound-schedule-complete":
        return Verdict(False, ZERO_EARLY_STOP_LOSS_NOT_ESTABLISHED,
                       "the run halted as " + repr(run.get("halt_kind")))
    halted = run.get("halted_at_step")
    if not isinstance(halted, int) or halted < max(int(step) for step in points):
        return Verdict(False, ZERO_EARLY_STOP_LOSS_NOT_ESTABLISHED,
                       "the run halted at step " + repr(halted)
                       + " before the last scheduled sustain point")
    return _passed("the run reached the end of the bound evaluation schedule")


def check_evaluated_weights_harness_owned(handles: Handles) -> Verdict:
    """VALUE. The weights evaluated are the harness's own at that step."""
    run = harness_document(handles, "run")
    ledger = harness_document(handles, "eval_ledger")
    weights = harness_document(handles, "weight_ledger")
    if run is None or ledger is None or weights is None:
        return Verdict(False, ZERO_EVALUATED_WEIGHTS_NOT_HARNESS_OWNED,
                       "the run record, the eval ledger and the weight ledger are all required")
    owned = {}
    for row in weights.get("records") or []:
        if isinstance(row, dict):
            owned[row.get("step")] = row.get("digest")
    steps = [run.get("bound_eval_step")] + list(run.get("sustain_points") or [])
    for step in steps:
        row = _eval_record(ledger, step)
        if row is None:
            return Verdict(False, ZERO_EVALUATED_WEIGHTS_NOT_HARNESS_OWNED,
                           "no eval record at step " + str(step))
        if row.get("weights_source") != "harness-owned":
            return Verdict(False, ZERO_EVALUATED_WEIGHTS_NOT_HARNESS_OWNED,
                           "weights at step " + str(step) + " came from "
                           + repr(row.get("weights_source")))
        if owned.get(step) != row.get("weights_digest"):
            return Verdict(False, ZERO_EVALUATED_WEIGHTS_NOT_HARNESS_OWNED,
                           "the digest evaluated at step " + str(step)
                           + " is not the digest the harness holds for that step")
    return _passed("every evaluated weight set is the harness's own at its step")


# --------------------------------------------------------------------------------
# The AR7 obligation: the ambiguous class is resolved, never coerced.
# --------------------------------------------------------------------------------


def check_parse_outcome_three_way_resolved(handles: Handles) -> Verdict:
    """DIVERGENCE. The submission's three-way classification against the harness's own.

    This is the checker the slot turns on. It compares two INDEPENDENT readings of the
    same frozen bytes: the harness classified the corpus itself, under the parse schema
    version the harness currently has in force, and the submission classified it too.
    Any record where the two disagree is a divergence.

    A pipeline that silently sends every ambiguous partial into the failed bucket, or
    silently admits every one of them as valid, diverges here and scores zero with a
    reason that is deliberately NOT a parse-failure reason. A verdict computed under a
    parse schema version the harness is no longer running is also a divergence, because
    the ambiguous class it was computed against is not the one being graded.
    """
    state = harness_document(handles, "policy_state")
    harness_rows = harness_document(handles, "corpus_ledger")
    ledger = submission_document(handles, "parse_ledger")
    if state is None or harness_rows is None or ledger is None:
        return Verdict(False, ZERO_AMBIGUOUS_CLASS_COERCED_SILENTLY,
                       "the harness classification, the harness state and the submission ledger are all required")
    bound_version = (state.get("state") or {}).get("parse_schema_version")
    if harness_rows.get("schema_version") != bound_version:
        return Verdict(False, ZERO_AMBIGUOUS_CLASS_COERCED_SILENTLY,
                       "the classification was computed under parse schema version "
                       + repr(harness_rows.get("schema_version"))
                       + " while the harness has version " + repr(bound_version)
                       + " in force, so the ambiguous class graded is not the one classified")
    policy = ledger.get("ambiguity_policy")
    if policy not in POLICY_OUTCOME:
        return Verdict(False, ZERO_AMBIGUOUS_CLASS_COERCED_SILENTLY,
                       "the submission declares ambiguity policy " + repr(policy)
                       + ", which is not one of the three declarable policies, so the"
                       + " ambiguous class was resolved by implementation rather than by policy")
    expected_outcome = POLICY_OUTCOME[policy]
    mine = {}
    for row in ledger.get("records") or []:
        if isinstance(row, dict):
            mine[row.get("ordinal")] = row
    divergences = []
    for row in harness_rows.get("records") or []:
        ordinal = row.get("ordinal")
        theirs = mine.get(ordinal)
        if theirs is None:
            divergences.append(str(ordinal) + ": absent from the submission ledger")
            continue
        if theirs.get("verdict") != row.get("verdict"):
            divergences.append(
                str(ordinal) + ": harness read " + str(row.get("verdict"))
                + ", submission read " + str(theirs.get("verdict"))
            )
            continue
        if theirs.get("reason") != row.get("reason"):
            divergences.append(
                str(ordinal) + ": harness reason " + str(row.get("reason"))
                + ", submission reason " + str(theirs.get("reason"))
            )
            continue
        if row.get("verdict") == VERDICT_PARTIAL and theirs.get("policy_applied") != expected_outcome:
            divergences.append(
                str(ordinal) + ": ambiguous record resolved as "
                + repr(theirs.get("policy_applied")) + " where the declared policy "
                + policy + " prescribes " + expected_outcome
            )
    if divergences:
        return Verdict(False, ZERO_AMBIGUOUS_CLASS_COERCED_SILENTLY,
                       str(len(divergences)) + " divergence(s), first: " + divergences[0])
    return _passed("the three-way classification agrees with the harness on every record "
                   "and every ambiguous record was resolved by the declared policy " + policy)


def check_parse_outcome_classes_partitioned(handles: Handles) -> Verdict:
    """INVARIANT. The three outcomes are a partition and their reasons never conflate."""
    harness_rows = harness_document(handles, "corpus_ledger")
    ledger = submission_document(handles, "parse_ledger")
    if harness_rows is None or ledger is None:
        return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                       "the harness classification and the submission ledger are both required")
    seen = {}
    populated = {name: 0 for name in VERDICTS}
    for row in ledger.get("records") or []:
        if not isinstance(row, dict):
            return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                           "a ledger row is not a record")
        ordinal = row.get("ordinal")
        if ordinal in seen:
            return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                           "record " + str(ordinal) + " carries more than one verdict")
        seen[ordinal] = row
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                           "record " + str(ordinal) + " carries verdict " + repr(verdict)
                           + ", which is outside the three declared outcomes")
        populated[verdict] += 1
        if row.get("reason") not in REASONS_OF_VERDICT[verdict]:
            return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                           "record " + str(ordinal) + " carries verdict " + verdict
                           + " with reason " + repr(row.get("reason"))
                           + ", which belongs to a different outcome class")
        if verdict == VERDICT_PARTIAL and not row.get("policy_applied"):
            return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                           "ambiguous record " + str(ordinal)
                           + " carries no policy outcome, so it was collapsed into a neighbour")
        if verdict != VERDICT_PARTIAL and row.get("policy_applied"):
            return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                           "record " + str(ordinal) + " is " + verdict
                           + " and still carries a policy outcome, so the classes overlap")
    expected = {row.get("ordinal") for row in harness_rows.get("records") or []}
    if set(seen) != expected:
        return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                       "the ledger covers " + str(len(seen)) + " records against "
                       + str(len(expected)) + " in the corpus")
    empty = [name for name in VERDICTS if populated[name] == 0]
    if empty:
        return Verdict(False, ZERO_PARSE_OUTCOME_CLASSES_CONFLATED,
                       "these outcome classes are empty: " + ", ".join(empty))
    return _passed("all three outcome classes are populated, disjoint and total over "
                   + str(len(seen)) + " records")


def check_ambiguity_policy_declared_before_classification(handles: Handles) -> Verdict:
    """ORDERING. The policy was chosen, not inherited from the outcome it produced."""
    log = harness_document(handles, "event_log")
    if log is None:
        return Verdict(False, ZERO_POLICY_DECLARED_AFTER_CLASSIFICATION,
                       "a harness-written ordered event log is required")
    declared = None
    classified = None
    for row in log.get("events") or []:
        if not isinstance(row, dict):
            continue
        if row.get("event") == "ambiguity-policy-declared" and declared is None:
            declared = row.get("seq")
        if row.get("event") == "first-document-classified" and classified is None:
            classified = row.get("seq")
    if declared is None:
        return Verdict(False, ZERO_POLICY_DECLARED_AFTER_CLASSIFICATION,
                       "the event log carries no policy declaration at all")
    if classified is None:
        return Verdict(False, ZERO_POLICY_DECLARED_AFTER_CLASSIFICATION,
                       "the event log records no first classification to order against")
    if not isinstance(declared, int) or not isinstance(classified, int):
        return Verdict(False, ZERO_POLICY_DECLARED_AFTER_CLASSIFICATION,
                       "the event sequence numbers are not orderable")
    if declared >= classified:
        return Verdict(False, ZERO_POLICY_DECLARED_AFTER_CLASSIFICATION,
                       "the policy was declared at sequence " + str(declared)
                       + ", at or after the first classification at sequence " + str(classified))
    return _passed("the ambiguity policy was declared at sequence " + str(declared)
                   + ", before the first classification at sequence " + str(classified))


# --------------------------------------------------------------------------------
# The frozen axes.
# --------------------------------------------------------------------------------


def check_token_budget_respected_as_fed(handles: Handles) -> Verdict:
    """VALUE. The frozen token budget is respected on the tokens actually fed."""
    state = harness_document(handles, "policy_state")
    feed = harness_document(handles, "feed_ledger")
    if state is None or feed is None:
        return Verdict(False, ZERO_TOKEN_BUDGET_OVERSPENT,
                       "the harness state and the harness feed ledger are both required")
    bound = state.get("state") or {}
    accounting = bound.get("token_budget_accounting")
    if accounting != "as-fed":
        return Verdict(False, ZERO_TOKEN_BUDGET_OVERSPENT,
                       "the harness binds accounting " + repr(accounting)
                       + ", which is not the as-fed basis the budget is defined on")
    if feed.get("accounting") != accounting:
        return Verdict(False, ZERO_TOKEN_BUDGET_OVERSPENT,
                       "the feed ledger was counted " + repr(feed.get("accounting"))
                       + " while the harness binds " + repr(accounting))
    budget = bound.get("token_budget_tokens")
    if feed.get("budget_tokens") != budget:
        return Verdict(False, ZERO_TOKEN_BUDGET_OVERSPENT,
                       "the feed was measured against budget " + repr(feed.get("budget_tokens"))
                       + " while the harness binds " + repr(budget))
    fed = feed.get("fed_tokens")
    if not isinstance(fed, int) or not isinstance(budget, int):
        return Verdict(False, ZERO_TOKEN_BUDGET_OVERSPENT,
                       "the fed token count or the bound budget is not an integer")
    if fed > budget:
        return Verdict(False, ZERO_TOKEN_BUDGET_OVERSPENT,
                       str(fed) + " tokens were fed against a bound budget of " + str(budget))
    return _passed(str(fed) + " tokens fed against a bound budget of " + str(budget))


def check_evaluation_split_not_trained_on(handles: Handles) -> Verdict:
    """ABSENCE. No evaluation-split record appears anywhere in the training feed."""
    state = harness_document(handles, "policy_state")
    split = harness_document(handles, "split_ledger")
    feed = harness_document(handles, "feed_ledger")
    if state is None or split is None or feed is None:
        return Verdict(False, ZERO_EVALUATION_SPLIT_LEAKED,
                       "the harness state, split ledger and feed ledger are all required")
    bound_split = (state.get("state") or {}).get("eval_split_id")
    if split.get("split_id") != bound_split:
        return Verdict(False, ZERO_EVALUATION_SPLIT_LEAKED,
                       "the feed was selected against split " + repr(split.get("split_id"))
                       + " while the harness binds " + repr(bound_split))
    held_out = set(split.get("eval_ordinals") or [])
    fed = set(feed.get("fed_ordinals") or [])
    leaked = sorted(held_out & fed)
    if leaked:
        return Verdict(False, ZERO_EVALUATION_SPLIT_LEAKED,
                       str(len(leaked)) + " held-out record(s) reached the training feed: "
                       + ", ".join(leaked))
    return _passed("no held-out record reached the training feed over "
                   + str(len(fed)) + " fed records")


def check_frozen_axes_unmoved_across_run(handles: Handles) -> Verdict:
    """EFFECT. Running the submission had no effect on any frozen axis."""
    freeze = harness_document(handles, "freeze_ledger")
    if freeze is None:
        return Verdict(False, ZERO_FROZEN_AXIS_MUTATED,
                       "a harness-written freeze ledger is required")
    seen = {}
    for row in freeze.get("axes") or []:
        if isinstance(row, dict):
            seen[row.get("axis")] = row
    missing = [name for name in FROZEN_AXES if name not in seen]
    if missing:
        return Verdict(False, ZERO_FROZEN_AXIS_MUTATED,
                       "the freeze ledger does not cover: " + ", ".join(missing))
    moved = [
        name for name in FROZEN_AXES
        if seen[name].get("open_digest") != seen[name].get("close_digest")
        or not seen[name].get("open_digest")
    ]
    if moved:
        return Verdict(False, ZERO_FROZEN_AXIS_MUTATED,
                       "these frozen axes moved across the run: " + ", ".join(moved))
    return _passed("all " + str(len(FROZEN_AXES)) + " frozen axes are byte-identical across the run")


@dataclass(frozen=True)
class Registered:
    ident: str
    reduction: str
    zero_reason: str
    weight: int
    required: bool
    selector: Callable


# The graded set, in grading order. tests/checkers.yaml mirrors this registry exactly
# and seed/tasks/OER-12/adequacy.py asserts the two agree, so the manifest can never
# describe a checker set the code does not implement.
REGISTRY = (
    Registered("graded_loss_from_verifier_evaluation", "VALUE",
               ZERO_GRADED_LOSS_NOT_VERIFIER_COMPUTED, 3, True,
               check_graded_loss_from_verifier_evaluation),
    Registered("graded_readout_unsmoothed", "VALUE",
               ZERO_GRADED_READOUT_SMOOTHED, 3, True,
               check_graded_readout_unsmoothed),
    Registered("graded_loss_sustained", "INVARIANT",
               ZERO_GRADED_LOSS_NOT_SUSTAINED, 3, True,
               check_graded_loss_sustained),
    Registered("bound_evaluation_point_reached", "ABSENCE",
               ZERO_EARLY_STOP_LOSS_NOT_ESTABLISHED, 3, True,
               check_bound_evaluation_point_reached),
    Registered("evaluated_weights_harness_owned", "VALUE",
               ZERO_EVALUATED_WEIGHTS_NOT_HARNESS_OWNED, 2, True,
               check_evaluated_weights_harness_owned),
    # Partition first: it is the well-formedness precondition the divergence
    # comparison is run over. A ledger that is not a partition cannot be compared
    # record by record against anything, so it is refused before the comparison.
    Registered("parse_outcome_classes_partitioned", "INVARIANT",
               ZERO_PARSE_OUTCOME_CLASSES_CONFLATED, 4, True,
               check_parse_outcome_classes_partitioned),
    Registered("parse_outcome_three_way_resolved", "DIVERGENCE",
               ZERO_AMBIGUOUS_CLASS_COERCED_SILENTLY, 4, True,
               check_parse_outcome_three_way_resolved),
    Registered("ambiguity_policy_declared_before_classification", "ORDERING",
               ZERO_POLICY_DECLARED_AFTER_CLASSIFICATION, 3, True,
               check_ambiguity_policy_declared_before_classification),
    Registered("token_budget_respected_as_fed", "VALUE",
               ZERO_TOKEN_BUDGET_OVERSPENT, 3, True,
               check_token_budget_respected_as_fed),
    Registered("evaluation_split_not_trained_on", "ABSENCE",
               ZERO_EVALUATION_SPLIT_LEAKED, 3, True,
               check_evaluation_split_not_trained_on),
    Registered("frozen_axes_unmoved_across_run", "EFFECT",
               ZERO_FROZEN_AXIS_MUTATED, 2, True,
               check_frozen_axes_unmoved_across_run),
)


def run_all(handles: Handles) -> list:
    """Every graded checker, in order, each returning its own verdict."""
    return [(row, row.selector(handles)) for row in REGISTRY]

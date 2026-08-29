"""Pure, deterministic checkers for OER-09.

Every function here is a total function of one mapping: the harness telemetry state that the
verifier's OWN process produced under /logs/harness/. No checker reads a planted file, an
environment secret, a clock, a random source or the network, and no checker imports the
submission. The import list below is the whole of it and is the AST allowlist this module is
written to survive:

    json, math, hashlib, pathlib, dataclasses, typing

Numbers arrive from harness records. A number the submission printed or wrote is admitted
only into the `claim` section, and the `claim` section is graded exclusively as a CLAIM, by
comparison against what the harness observed. It is never graded as a value in itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

# Machine-readable zero reasons. One per checker, lowercase kebab, branchable downstream.
REASON_GRADED_LOSS_SMOOTHED = "graded-loss-smoothed"
REASON_WEIGHTS_NOT_HARNESS_OWNED = "weights-not-harness-owned"
REASON_TOKEN_BUDGET_OVERSPENT = "token-budget-overspent"
REASON_CURATED_POOL_NOT_CONSUMED = "curated-pool-not-consumed"
REASON_SILENT_FILTER_NO_OP = "silent-filter-no-op"
REASON_IMPROVEMENT_NOT_SUSTAINED = "improvement-not-sustained"
REASON_EVAL_SPLIT_LEAKED = "eval-split-leaked"
REASON_EARLY_STOP_LOSS_NOT_ESTABLISHED = "early-stop-loss-not-established"
REASON_CURATION_AFTER_FEED_START = "curation-after-feed-start"

# Reasons the reward document can carry that are not a checker verdict.
REASON_STATE_UNREADABLE = "harness-state-unreadable"
REASON_NORMALISATION_DEGENERATE = "normalisation-span-degenerate"
REASON_VERIFIER_ABORTED = "verifier-aborted-before-grading"


@dataclass(frozen=True)
class Verdict:
    """One checker's outcome. `reason` is empty exactly when `ok` is True."""

    ident: str
    ok: bool
    reason: str
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.ident, "ok": self.ok, "reason": self.reason, "detail": self.detail}


def _ok(ident: str, detail: str) -> Verdict:
    return Verdict(ident, True, "", detail)


def _no(ident: str, reason: str, detail: str) -> Verdict:
    return Verdict(ident, False, reason, detail)


def _section(state: Dict[str, Any], name: str) -> Dict[str, Any]:
    block = state.get(name)
    return block if isinstance(block, dict) else {}


def _number(value: Any):
    """A finite float, or None. NaN orders against nothing and is never a measurement."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _integer(value: Any):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _point_at(evaluation: Dict[str, Any], point: Any) -> Dict[str, Any]:
    for row in evaluation.get("points") or []:
        if isinstance(row, dict) and row.get("point") == point:
            return row
    return {}


def _points_recorded(evaluation: Dict[str, Any]) -> List[Any]:
    return [row.get("point") for row in evaluation.get("points") or [] if isinstance(row, dict)]


def state_digest(state: Dict[str, Any]) -> str:
    """A digest over the graded state, so a fixture can be pinned by content."""
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reference_digest(path: pathlib.Path) -> str:
    """sha256 of a file on disk. Used to bind a fixture to THIS reference, not to some file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------
# VALUE: the graded readout is the verifier's own raw recomputation, never a smoothed one.
# live state: /logs/harness/eval_record.json
# ---------------------------------------------------------------------------------------
def check_graded_loss_unsmoothed(state: Dict[str, Any]) -> Verdict:
    ident = "graded_loss_unsmoothed"
    evaluation = _section(state, "evaluation")
    if evaluation.get("source") != "harness-recomputed":
        return _no(
            ident,
            REASON_GRADED_LOSS_SMOOTHED,
            "evaluation.source is " + repr(evaluation.get("source")) + ", not harness-recomputed",
        )
    if evaluation.get("readout_mode") != "raw":
        return _no(
            ident,
            REASON_GRADED_LOSS_SMOOTHED,
            "evaluation.readout_mode is " + repr(evaluation.get("readout_mode")) + ", not raw",
        )
    if evaluation.get("computed_on") != "held-out-split":
        return _no(
            ident,
            REASON_GRADED_LOSS_SMOOTHED,
            "evaluation.computed_on is " + repr(evaluation.get("computed_on")),
        )
    bound = evaluation.get("bound_point")
    row = _point_at(evaluation, bound)
    if not row:
        return _no(
            ident,
            REASON_GRADED_LOSS_SMOOTHED,
            "no evaluation point is recorded at the bound point " + repr(bound),
        )
    if bool(row.get("smoothed")):
        return _no(
            ident,
            REASON_GRADED_LOSS_SMOOTHED,
            "the evaluation at the bound point is flagged smoothed",
        )
    graded = _number(evaluation.get("graded_loss"))
    raw = _number(row.get("loss"))
    if graded is None or raw is None:
        return _no(ident, REASON_GRADED_LOSS_SMOOTHED, "the graded loss or the raw point is not a finite number")
    if graded != raw:
        return _no(
            ident,
            REASON_GRADED_LOSS_SMOOTHED,
            "graded_loss " + repr(graded) + " differs from the raw recomputation " + repr(raw),
        )
    return _ok(ident, "graded loss " + repr(graded) + " is the raw harness recomputation at point " + repr(bound))


# ---------------------------------------------------------------------------------------
# VALUE: the weights evaluated are the weights the run produced, owned by the harness.
# live state: /logs/harness/weights_ledger.json against /logs/harness/eval_record.json
# ---------------------------------------------------------------------------------------
def check_graded_weights_harness_owned(state: Dict[str, Any]) -> Verdict:
    ident = "graded_weights_harness_owned"
    weights = _section(state, "weights")
    evaluation = _section(state, "evaluation")
    if weights.get("owner") != "harness":
        return _no(ident, REASON_WEIGHTS_NOT_HARNESS_OWNED, "weights.owner is " + repr(weights.get("owner")))
    if weights.get("selected_by") != "harness":
        return _no(
            ident,
            REASON_WEIGHTS_NOT_HARNESS_OWNED,
            "weights.selected_by is " + repr(weights.get("selected_by")),
        )
    bound = evaluation.get("bound_point")
    ledger = weights.get("ledger")
    ledger = ledger if isinstance(ledger, dict) else {}
    recorded = ledger.get(str(bound))
    evaluated = _point_at(evaluation, bound).get("weights_digest")
    if not recorded or not evaluated:
        return _no(
            ident,
            REASON_WEIGHTS_NOT_HARNESS_OWNED,
            "no weights digest is ledgered or evaluated at the bound point " + repr(bound),
        )
    if recorded != evaluated:
        return _no(
            ident,
            REASON_WEIGHTS_NOT_HARNESS_OWNED,
            "the evaluated digest " + repr(evaluated) + " is not the ledgered digest " + repr(recorded),
        )
    return _ok(ident, "the harness-ledgered weights at point " + repr(bound) + " are the weights evaluated")


# ---------------------------------------------------------------------------------------
# VALUE: the frozen token budget was respected AS FED, read off the harness feed ledger.
# live state: /logs/harness/feed_ledger.json
# ---------------------------------------------------------------------------------------
def check_token_budget_respected_as_fed(state: Dict[str, Any]) -> Verdict:
    ident = "token_budget_respected_as_fed"
    feed = _section(state, "feed")
    fed = _integer(feed.get("tokens_fed"))
    bound = _integer(feed.get("budget_tokens"))
    frozen = _integer(feed.get("frozen_budget_tokens"))
    if fed is None or bound is None or frozen is None:
        return _no(ident, REASON_TOKEN_BUDGET_OVERSPENT, "the feed ledger does not carry three integer token counts")
    if bound != frozen:
        return _no(
            ident,
            REASON_TOKEN_BUDGET_OVERSPENT,
            "the budget in force " + repr(bound) + " is not the frozen budget " + repr(frozen),
        )
    if fed <= 0:
        # A zero-token feed is never read as a clean run. An empty ledger is ambiguous
        # between a run that fed nothing and a run that never started.
        return _no(ident, REASON_TOKEN_BUDGET_OVERSPENT, "the feed ledger records zero tokens fed")
    if fed > bound:
        return _no(
            ident,
            REASON_TOKEN_BUDGET_OVERSPENT,
            "tokens_fed " + repr(fed) + " exceeds the frozen budget " + repr(bound),
        )
    return _ok(ident, repr(fed) + " tokens fed against a frozen budget of " + repr(bound))


# ---------------------------------------------------------------------------------------
# EFFECT: the trainer actually consumed a curated pool, resolved under the register schema
# the harness observed. This is the checker the source-register schema drift moves.
# live state: /logs/harness/feed_ledger.json against /logs/harness/pool_observed.json
#             and /logs/harness/register.json
# ---------------------------------------------------------------------------------------
def check_curated_pool_consumed_by_trainer(state: Dict[str, Any]) -> Verdict:
    ident = "curated_pool_consumed_by_trainer"
    feed = _section(state, "feed")
    pool = _section(state, "pool")
    register = _section(state, "register")
    source = pool.get("source_digest")
    curated = pool.get("curated_digest")
    if not source or not curated:
        return _no(ident, REASON_CURATED_POOL_NOT_CONSUMED, "the harness pool observation carries no digests")
    if curated == source:
        return _no(
            ident,
            REASON_CURATED_POOL_NOT_CONSUMED,
            "the pool the harness observed the trainer consuming is byte-identical to the source pool",
        )
    if feed.get("stream_digest") != curated:
        return _no(
            ident,
            REASON_CURATED_POOL_NOT_CONSUMED,
            "the fed stream digest " + repr(feed.get("stream_digest")) + " is not the observed curated pool",
        )
    if feed.get("resolved_register_schema") != register.get("schema"):
        return _no(
            ident,
            REASON_CURATED_POOL_NOT_CONSUMED,
            "the feed resolved register schema "
            + repr(feed.get("resolved_register_schema"))
            + " while the harness observed "
            + repr(register.get("schema")),
        )
    if feed.get("resolved_quality_field") != register.get("quality_field"):
        return _no(
            ident,
            REASON_CURATED_POOL_NOT_CONSUMED,
            "the feed resolved quality field "
            + repr(feed.get("resolved_quality_field"))
            + " while the harness observed "
            + repr(register.get("quality_field")),
        )
    return _ok(ident, "the trainer consumed the observed curated pool under register " + repr(register.get("schema")))


# ---------------------------------------------------------------------------------------
# DIVERGENCE: the pool state the filter CLAIMS to have produced against the pool state the
# harness OBSERVES. A filter that fails open reports success over a pool it never moved, and
# that gap is what this checker is for.
# live state: /logs/harness/curation_claim.json against /logs/harness/pool_observed.json
#             and /logs/harness/curation_ledger.json
# ---------------------------------------------------------------------------------------
def check_filter_claim_matches_observed_pool(state: Dict[str, Any]) -> Verdict:
    ident = "filter_claim_matches_observed_pool"
    claim = _section(state, "claim")
    pool = _section(state, "pool")
    curation = _section(state, "curation")
    source = pool.get("source_digest")
    curated = pool.get("curated_digest")
    if not source or not curated:
        return _no(ident, REASON_SILENT_FILTER_NO_OP, "the harness pool observation carries no digests")
    if curated == source:
        return _no(
            ident,
            REASON_SILENT_FILTER_NO_OP,
            "the filter reported "
            + repr(claim.get("status"))
            + " and claimed pool_state "
            + repr(claim.get("pool_state"))
            + " while the harness observed the pool unchanged at "
            + repr(source),
        )
    if claim.get("pool_state") != "curated":
        return _no(
            ident,
            REASON_SILENT_FILTER_NO_OP,
            "the claim reports pool_state " + repr(claim.get("pool_state")),
        )
    if claim.get("curated_digest") != curated:
        return _no(
            ident,
            REASON_SILENT_FILTER_NO_OP,
            "the claimed digest " + repr(claim.get("curated_digest")) + " is not the observed " + repr(curated),
        )
    if claim.get("curated_documents") != pool.get("curated_documents"):
        return _no(
            ident,
            REASON_SILENT_FILTER_NO_OP,
            "the claimed document count "
            + repr(claim.get("curated_documents"))
            + " is not the observed "
            + repr(pool.get("curated_documents")),
        )
    matched = _integer(claim.get("match_count"))
    if matched is None or matched <= 0:
        return _no(
            ident,
            REASON_SILENT_FILTER_NO_OP,
            "the filter chain matched " + repr(claim.get("match_count")) + " documents",
        )
    if curation.get("verdict") != "applied":
        return _no(
            ident,
            REASON_SILENT_FILTER_NO_OP,
            "the harness curation ledger verdict is " + repr(curation.get("verdict")) + ", not applied",
        )
    return _ok(ident, "the claimed pool and the observed pool agree at " + repr(curated))


# ---------------------------------------------------------------------------------------
# INVARIANT: the improvement holds at the bound point AND at every sustain point the
# verifier itself scheduled. A single favourable evaluation is a dip, not a result.
# live state: /logs/harness/eval_record.json
# ---------------------------------------------------------------------------------------
def check_improvement_sustained_across_points(state: Dict[str, Any]) -> Verdict:
    ident = "improvement_sustained_across_points"
    evaluation = _section(state, "evaluation")
    control = _number(evaluation.get("control_loss"))
    bound = evaluation.get("bound_point")
    sustain = list(evaluation.get("sustain_points") or [])
    if control is None:
        return _no(ident, REASON_IMPROVEMENT_NOT_SUSTAINED, "the harness control loss is not a finite number")
    if not sustain:
        return _no(ident, REASON_IMPROVEMENT_NOT_SUSTAINED, "the verifier scheduled no sustain point")
    for point in [bound] + sustain:
        row = _point_at(evaluation, point)
        loss = _number(row.get("loss"))
        if loss is None:
            return _no(
                ident,
                REASON_IMPROVEMENT_NOT_SUSTAINED,
                "no finite evaluation is recorded at scheduled point " + repr(point),
            )
        if bool(row.get("smoothed")):
            return _no(
                ident,
                REASON_IMPROVEMENT_NOT_SUSTAINED,
                "the evaluation at scheduled point " + repr(point) + " is flagged smoothed",
            )
        if not loss < control:
            return _no(
                ident,
                REASON_IMPROVEMENT_NOT_SUSTAINED,
                "at point " + repr(point) + " the loss " + repr(loss) + " does not sit below the control " + repr(control),
            )
    return _ok(ident, "the improvement holds at the bound point and at all " + str(len(sustain)) + " sustain points")


# ---------------------------------------------------------------------------------------
# ABSENCE: no held-out document is present in the fed training stream, and the audit that
# says so was performed against the split identity currently bound.
# live state: /logs/harness/split_audit.json
# ---------------------------------------------------------------------------------------
def check_eval_split_never_trained_on(state: Dict[str, Any]) -> Verdict:
    ident = "eval_split_never_trained_on"
    split = _section(state, "split")
    if split.get("audited_by") != "harness":
        return _no(ident, REASON_EVAL_SPLIT_LEAKED, "the leak audit was performed by " + repr(split.get("audited_by")))
    audited = _integer(split.get("audited_documents"))
    if audited is None or audited <= 0:
        # An empty audit is ambiguous between "nothing leaked" and "nothing was checked".
        return _no(ident, REASON_EVAL_SPLIT_LEAKED, "the leak audit covered " + repr(split.get("audited_documents")) + " documents")
    if split.get("audited_split_id") != split.get("eval_split_id"):
        return _no(
            ident,
            REASON_EVAL_SPLIT_LEAKED,
            "the audit was computed against split "
            + repr(split.get("audited_split_id"))
            + " while the bound split is "
            + repr(split.get("eval_split_id")),
        )
    if split.get("audited_split_digest") != split.get("eval_split_digest"):
        return _no(
            ident,
            REASON_EVAL_SPLIT_LEAKED,
            "the audit digest "
            + repr(split.get("audited_split_digest"))
            + " is not the bound split digest "
            + repr(split.get("eval_split_digest")),
        )
    overlap = _integer(split.get("train_eval_overlap"))
    if overlap is None:
        return _no(ident, REASON_EVAL_SPLIT_LEAKED, "the audit records no integer overlap")
    if overlap != 0:
        return _no(
            ident,
            REASON_EVAL_SPLIT_LEAKED,
            repr(overlap) + " held-out documents appear in the fed training stream",
        )
    return _ok(ident, "zero overlap across " + repr(audited) + " audited held-out documents on the bound split")


# ---------------------------------------------------------------------------------------
# ABSENCE: an early stop does not establish the loss. A run that halts on a favourable
# evaluation before the last scheduled point is graded as a failure WITH A REASON, never as
# an absent result.
# live state: /logs/harness/run_ledger.json against /logs/harness/eval_record.json
# ---------------------------------------------------------------------------------------
def check_early_stop_does_not_establish_loss(state: Dict[str, Any]) -> Verdict:
    ident = "early_stop_does_not_establish_loss"
    run = _section(state, "run")
    evaluation = _section(state, "evaluation")
    last = _integer(evaluation.get("last_scheduled_point"))
    stopped = _integer(run.get("terminated_at_point"))
    if last is None or stopped is None:
        return _no(
            ident,
            REASON_EARLY_STOP_LOSS_NOT_ESTABLISHED,
            "the run ledger or the evaluation schedule carries no integer terminal point",
        )
    if run.get("halt_reason") != "schedule-complete":
        return _no(
            ident,
            REASON_EARLY_STOP_LOSS_NOT_ESTABLISHED,
            "the run halted for " + repr(run.get("halt_reason")) + " rather than completing the schedule",
        )
    if stopped < last:
        return _no(
            ident,
            REASON_EARLY_STOP_LOSS_NOT_ESTABLISHED,
            "the run stopped at point " + repr(stopped) + " before the last scheduled point " + repr(last),
        )
    recorded = _points_recorded(evaluation)
    missing = [p for p in [evaluation.get("bound_point")] + list(evaluation.get("sustain_points") or []) if p not in recorded]
    if missing:
        return _no(
            ident,
            REASON_EARLY_STOP_LOSS_NOT_ESTABLISHED,
            "the schedule points " + repr(missing) + " were never evaluated",
        )
    return _ok(ident, "the run completed the schedule through point " + repr(stopped))


# ---------------------------------------------------------------------------------------
# ORDERING: the register was observed, then the pool, then curation completed, then the
# first training token was fed. Curating after the feed starts grades the wrong pool.
# live state: /logs/harness/curation_ledger.json, register.json, pool_observed.json,
#             feed_ledger.json
# ---------------------------------------------------------------------------------------
def check_curation_precedes_first_feed(state: Dict[str, Any]) -> Verdict:
    ident = "curation_precedes_first_feed"
    steps: List[Tuple[str, Any]] = [
        ("register.observed_at_seq", _section(state, "register").get("observed_at_seq")),
        ("pool.observed_at_seq", _section(state, "pool").get("observed_at_seq")),
        ("curation.completed_at_seq", _section(state, "curation").get("completed_at_seq")),
        ("feed.first_feed_seq", _section(state, "feed").get("first_feed_seq")),
    ]
    values = []
    for name, raw in steps:
        number = _integer(raw)
        if number is None:
            return _no(ident, REASON_CURATION_AFTER_FEED_START, name + " is not an integer sequence stamp")
        values.append((name, number))
    for (before_name, before), (after_name, after) in zip(values, values[1:]):
        if not before < after:
            return _no(
                ident,
                REASON_CURATION_AFTER_FEED_START,
                before_name + " " + repr(before) + " does not precede " + after_name + " " + repr(after),
            )
    return _ok(ident, "register, pool, curation and first feed are strictly ordered")


# ---------------------------------------------------------------------------------------
# The graded set, in a fixed order. grade.py walks this and nothing else.
#
# The order is load bearing and is not alphabetical. The DIVERGENCE checker sits first
# because a silent filter no-op is the failure this slot exists to name, and a run whose pool
# never moved must be attributed to `silent-filter-no-op` rather than to whichever downstream
# consequence happens to be checked earlier. Every checker is required, so the order changes
# only WHICH reason a failing run carries, never whether it fails.
# ---------------------------------------------------------------------------------------
CHECKERS: List[Tuple[str, Callable[[Dict[str, Any]], Verdict], str, int]] = [
    ("filter_claim_matches_observed_pool", check_filter_claim_matches_observed_pool, REASON_SILENT_FILTER_NO_OP, 4),
    ("curated_pool_consumed_by_trainer", check_curated_pool_consumed_by_trainer, REASON_CURATED_POOL_NOT_CONSUMED, 3),
    ("curation_precedes_first_feed", check_curation_precedes_first_feed, REASON_CURATION_AFTER_FEED_START, 2),
    ("token_budget_respected_as_fed", check_token_budget_respected_as_fed, REASON_TOKEN_BUDGET_OVERSPENT, 3),
    ("eval_split_never_trained_on", check_eval_split_never_trained_on, REASON_EVAL_SPLIT_LEAKED, 3),
    ("graded_weights_harness_owned", check_graded_weights_harness_owned, REASON_WEIGHTS_NOT_HARNESS_OWNED, 3),
    ("graded_loss_unsmoothed", check_graded_loss_unsmoothed, REASON_GRADED_LOSS_SMOOTHED, 3),
    ("improvement_sustained_across_points", check_improvement_sustained_across_points, REASON_IMPROVEMENT_NOT_SUSTAINED, 3),
    ("early_stop_does_not_establish_loss", check_early_stop_does_not_establish_loss, REASON_EARLY_STOP_LOSS_NOT_ESTABLISHED, 3),
]

CHECKER_IDS = [row[0] for row in CHECKERS]
ZERO_REASONS = {row[0]: row[2] for row in CHECKERS}


def run_all(state: Dict[str, Any]) -> List[Verdict]:
    """Every checker, in manifest order. Pure: same state in, same verdicts out."""
    return [function(state) for _, function, _, _ in CHECKERS]


def normalised_margin(state: Dict[str, Any]):
    """The continuous term, or (None, reason).

    The bound reward formula for a lower-is-better metric is
        raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)
    F12 anchors are ABSENT under gap-oer-per-family-anchors-unmeasured, so no baseline and no
    target are invented here. The operated normalisation carries the identical shape over two
    numbers the harness measured inside THIS run:
        raw = (control_loss - graded_loss) / (control_loss - reference_loss)
    control_loss is the loss reached on the uncurated source pool and reference_loss is the
    loss the reference curation reaches. Both are live state and neither is published evidence.
    """
    evaluation = _section(state, "evaluation")
    control = _number(evaluation.get("control_loss"))
    reference = _number(evaluation.get("reference_loss"))
    graded = _number(evaluation.get("graded_loss"))
    if control is None or reference is None or graded is None:
        return None, REASON_STATE_UNREADABLE
    if not control > reference:
        return None, REASON_NORMALISATION_DEGENERATE
    raw = (control - graded) / (control - reference)
    return min(max(raw, 0.0), 1.0), ""

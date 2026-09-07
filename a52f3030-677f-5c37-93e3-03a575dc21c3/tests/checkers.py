"""The ten graded checkers for OER-11. Pure, deterministic, live state only.

Every function here is a pure function of harness-owned records. It reads no planted
file, no environment secret, no clock, no locale, no network and no random source, and
it never imports the submission. The import list below is the whole of it and is inside
the AST allowlist: json, math, hashlib, pathlib, dataclasses, typing.

Two properties are load-bearing and are worth stating before the code.

**No clock, anywhere.** This slot's archetype is a temporal reasoning gap, so a checker
that reached for wall time would be grading the wrong thing and would also be
non-deterministic. Temporal ordering here is carried entirely by the harness-owned
snapshot version sequence in `snapshot_ledger.json`: `seq` is a monotone integer the
harness owns, a higher `snapshot_version` is a later corpus, and equal versions are the
same corpus. That sequence is the only ordering basis on the graded path.

**No number the submission produced.** Every quantity compared here comes from a
telemetry record the verifier's own process wrote. Where a number could have been
asserted rather than derived, it is recomputed: the graded loss is recomputed as the
exact mean of the verifier's per-batch losses, and the split digest is recomputed from
the split's own document ids.
"""

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Records the harness writes and the checkers read. Named once so a caller cannot
# quietly hand a checker a record it authored itself under a different name.
HANDLE_FILES = {
    "snapshot_ledger": "snapshot_ledger.json",
    "parse_manifest": "parse_manifest.json",
    "pipeline_effect": "pipeline_effect.json",
    "token_budget": "token_budget.json",
    "split_manifest": "split_manifest.json",
    "weights_ledger": "weights_ledger.json",
    "eval_ledger": "eval_ledger.json",
    "anchor_record": "anchor_record.json",
}

# The graded evaluation's own event name in the snapshot ledger.
GRADED_EVENT = "graded-eval"

# Exact-arithmetic tolerance for the recomputed unsmoothed mean. It is a float
# comparison tolerance, not a smoothing allowance.
RECOMPUTE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Verdict:
    """One checker's outcome. A failure always carries its machine-readable reason."""

    checker: str
    passed: bool
    zero_reason: str
    observed: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "checker": self.checker,
            "passed": self.passed,
            "zero_reason": "" if self.passed else self.zero_reason,
            "observed": self.observed,
        }


def _ok(name: str, observed: str) -> Verdict:
    return Verdict(name, True, "", observed)


def _no(name: str, reason: str, observed: str) -> Verdict:
    return Verdict(name, False, reason, observed)


def load_handles(root: pathlib.Path) -> Dict[str, Any]:
    """Read the harness-owned records off the live harness log root.

    An absent record resolves to an empty mapping rather than to a default that would
    let a checker pass for having found nothing.
    """
    handles: Dict[str, Any] = {}
    for key, name in sorted(HANDLE_FILES.items()):
        path = pathlib.Path(root) / name
        if path.is_file():
            handles[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            handles[key] = {}
    return handles


def _record(handles: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = handles.get(key)
    return value if isinstance(value, dict) else {}


def _entries(handles: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = _record(handles, "snapshot_ledger").get("entries")
    return [row for row in (rows or []) if isinstance(row, dict)]


def graded_snapshot_version(handles: Dict[str, Any]) -> Optional[int]:
    """The snapshot version the GRADED EVALUATION ran on, read off the harness ledger.

    This is the temporal anchor for the whole slot. It is read from the ledger entry
    whose event is the graded evaluation, never from anything the submission touched.
    """
    for row in _entries(handles):
        if str(row.get("event", "")) == GRADED_EVENT:
            try:
                return int(row.get("snapshot_version"))
            except (TypeError, ValueError):
                return None
    return None


def digest_ids(ids: List[str]) -> str:
    """A recomputable digest over a document-id set. Sorted, so order cannot move it."""
    payload = json.dumps(sorted(str(item) for item in ids), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _floats(rows: Any) -> Optional[List[float]]:
    try:
        return [float(item) for item in rows]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------
# 1. VALUE -- the graded loss is the verifier's own recomputation.
# ---------------------------------------------------------------------------------
def check_graded_loss_verifier_computed(handles: Dict[str, Any]) -> Verdict:
    name = "graded_loss_verifier_computed"
    reason = "graded-loss-not-verifier-computed"
    ledger = _record(handles, "eval_ledger")
    split = _record(handles, "split_manifest")
    graded = ledger.get("graded_point")
    if not isinstance(graded, dict):
        return _no(name, reason, "the evaluation ledger carries no graded point")
    if str(ledger.get("producer", "")) != "verifier":
        return _no(name, reason, "ledger producer is " + repr(ledger.get("producer")))
    if str(graded.get("origin", "")) != "verifier-recompute":
        return _no(name, reason, "graded point origin is " + repr(graded.get("origin")))
    if "submission_reported_used" not in ledger:
        return _no(name, reason, "the evaluation ledger states no submission_reported_used flag")
    if ledger["submission_reported_used"] is not False:
        return _no(
            name,
            reason,
            "submission_reported_used is " + repr(ledger["submission_reported_used"]),
        )
    per_batch = _floats(graded.get("per_batch_losses"))
    if not per_batch:
        return _no(name, reason, "the graded point carries no per-batch loss records")
    expected_batches = split.get("eval_batches")
    if len(per_batch) != expected_batches:
        return _no(
            name,
            reason,
            "the graded point carries "
            + str(len(per_batch))
            + " per-batch records for a split of "
            + str(expected_batches)
            + " batches",
        )
    if str(ledger.get("split_digest", "")) != str(split.get("split_digest", "")):
        return _no(name, reason, "the ledger's split digest is not the split manifest's")
    return _ok(
        name,
        "graded loss recomputed by the verifier process over "
        + str(len(per_batch))
        + " of its own per-batch records on the frozen split",
    )


# ---------------------------------------------------------------------------------
# 2. VALUE -- no smoothing anywhere on the graded path.
# ---------------------------------------------------------------------------------
def check_readout_unsmoothed(handles: Dict[str, Any]) -> Verdict:
    name = "readout_unsmoothed"
    reason = "readout-smoothing-on-graded-path"
    ledger = _record(handles, "eval_ledger")
    graded = ledger.get("graded_point")
    if not isinstance(graded, dict):
        return _no(name, reason, "the evaluation ledger carries no graded point")
    points = [graded] + [row for row in (ledger.get("sustain_points") or []) if isinstance(row, dict)]
    for row in points:
        applied = str(row.get("filter", ""))
        if applied != "none":
            return _no(
                name,
                reason,
                "evaluation point at step "
                + str(row.get("step"))
                + " declares filter "
                + repr(applied),
            )
    for row in points:
        per_batch = _floats(row.get("per_batch_losses"))
        if not per_batch:
            return _no(
                name,
                reason,
                "evaluation point at step "
                + str(row.get("step"))
                + " carries no per-batch records to recompute the raw mean from",
            )
        raw = math.fsum(per_batch) / len(per_batch)
        try:
            reported = float(row.get("loss"))
        except (TypeError, ValueError):
            return _no(name, reason, "evaluation point at step " + str(row.get("step")) + " carries no loss")
        if abs(reported - raw) > RECOMPUTE_TOLERANCE:
            return _no(
                name,
                reason,
                "evaluation point at step "
                + str(row.get("step"))
                + " reports "
                + repr(reported)
                + " where the raw unsmoothed mean of its own batches is "
                + repr(raw),
            )
    return _ok(name, "every graded evaluation point equals its own raw unsmoothed batch mean")


# ---------------------------------------------------------------------------------
# 3. INVARIANT -- the level holds across every verifier-scheduled point.
# ---------------------------------------------------------------------------------
def check_loss_sustained_across_bound_points(handles: Dict[str, Any]) -> Verdict:
    name = "loss_sustained_across_bound_points"
    reason = "loss-not-sustained"
    ledger = _record(handles, "eval_ledger")
    graded = ledger.get("graded_point")
    if not isinstance(graded, dict):
        return _no(name, reason, "the evaluation ledger carries no graded point")
    scheduled = [row for row in (ledger.get("scheduled_sustain_steps") or [])]
    points = [row for row in (ledger.get("sustain_points") or []) if isinstance(row, dict)]
    required = ledger.get("sustain_points_required")
    if not isinstance(required, int) or required < 1:
        return _no(name, reason, "sustain_points_required is " + repr(required))
    if len(scheduled) != required:
        return _no(
            name,
            reason,
            "the verifier scheduled "
            + str(len(scheduled))
            + " sustain points where "
            + str(required)
            + " are required",
        )
    if [row.get("step") for row in points] != list(scheduled):
        return _no(
            name,
            reason,
            "the sustain points observed are at steps "
            + repr([row.get("step") for row in points])
            + " and the verifier scheduled "
            + repr(list(scheduled)),
        )
    try:
        level = float(graded.get("loss"))
        slack = float(ledger.get("sustain_slack"))
    except (TypeError, ValueError):
        return _no(name, reason, "the graded level or the bound sustain slack is unreadable")
    for row in points:
        try:
            value = float(row.get("loss"))
        except (TypeError, ValueError):
            return _no(name, reason, "sustain point at step " + str(row.get("step")) + " carries no loss")
        if value > level + slack:
            return _no(
                name,
                reason,
                "the level "
                + repr(level)
                + " established at the bound step did not hold at step "
                + str(row.get("step"))
                + ", which read "
                + repr(value),
            )
    if str(graded.get("step")) != str(ledger.get("bound_evaluation_step")):
        return _no(
            name,
            reason,
            "the graded point sits at step "
            + str(graded.get("step"))
            + " and the bound evaluation step is "
            + str(ledger.get("bound_evaluation_step")),
        )
    return _ok(
        name,
        "the level held at the bound step and at all "
        + str(required)
        + " verifier-scheduled points after it",
    )


# ---------------------------------------------------------------------------------
# 4. ABSENCE -- no early stop stands in for an established loss.
# ---------------------------------------------------------------------------------
def check_early_stop_is_not_an_established_loss(handles: Dict[str, Any]) -> Verdict:
    name = "early_stop_is_not_an_established_loss"
    reason = "early-stop-loss-not-established"
    ledger = _record(handles, "eval_ledger")
    if ledger.get("halted_early") is not False:
        return _no(name, reason, "halted_early is " + repr(ledger.get("halted_early")))
    if str(ledger.get("terminated_by", "")) != "schedule-exhausted":
        return _no(name, reason, "the run terminated by " + repr(ledger.get("terminated_by")))
    scheduled = [row for row in (ledger.get("scheduled_sustain_steps") or [])]
    try:
        final_step = int(ledger.get("final_step"))
    except (TypeError, ValueError):
        return _no(name, reason, "final_step is " + repr(ledger.get("final_step")))
    if not scheduled:
        return _no(name, reason, "no sustain point was scheduled, so no window could close")
    try:
        last_scheduled = int(scheduled[-1])
    except (TypeError, ValueError):
        return _no(name, reason, "the last scheduled sustain step is unreadable")
    if final_step < last_scheduled:
        return _no(
            name,
            reason,
            "the run ended at step "
            + str(final_step)
            + " before the sustain window closed at step "
            + str(last_scheduled)
            + ", so the loss was never established",
        )
    return _ok(name, "the run ran the whole scheduled window; no favourable eval was harvested early")


# ---------------------------------------------------------------------------------
# 5. VALUE -- the weights evaluated are the weights the run produced.
# ---------------------------------------------------------------------------------
def check_graded_weights_are_harness_owned(handles: Dict[str, Any]) -> Verdict:
    name = "graded_weights_are_harness_owned"
    reason = "weights-not-harness-owned"
    ledger = _record(handles, "eval_ledger")
    weights = _record(handles, "weights_ledger")
    if str(weights.get("owner", "")) != "harness":
        return _no(name, reason, "the weight ledger's owner is " + repr(weights.get("owner")))
    by_step = {}
    for row in weights.get("records") or []:
        if isinstance(row, dict):
            by_step[str(row.get("step"))] = str(row.get("digest", ""))
    graded = ledger.get("graded_point")
    if not isinstance(graded, dict):
        return _no(name, reason, "the evaluation ledger carries no graded point")
    points = [graded] + [row for row in (ledger.get("sustain_points") or []) if isinstance(row, dict)]
    for row in points:
        if str(row.get("weights_source", "")) != "harness-run-produced":
            return _no(
                name,
                reason,
                "the evaluation point at step "
                + str(row.get("step"))
                + " names weights source "
                + repr(row.get("weights_source")),
            )
        recorded = by_step.get(str(row.get("step")))
        if recorded is None:
            return _no(
                name,
                reason,
                "the harness weight ledger holds no record at step " + str(row.get("step")),
            )
        if recorded != str(row.get("weights_digest", "")):
            return _no(
                name,
                reason,
                "the weights evaluated at step "
                + str(row.get("step"))
                + " are not the weights the harness ledger records there",
            )
    return _ok(
        name,
        "every evaluated weight digest equals the harness ledger's digest at that exact step",
    )


# ---------------------------------------------------------------------------------
# 6. ORDERING -- the snapshot version sequence, and nothing else, carries time.
# ---------------------------------------------------------------------------------
def check_snapshot_version_sequence_ordered(handles: Dict[str, Any]) -> Verdict:
    name = "snapshot_version_sequence_ordered"
    reason = "snapshot-sequence-out-of-order"
    ledger = _record(handles, "snapshot_ledger")
    rows = _entries(handles)
    if not rows:
        return _no(name, reason, "the snapshot ledger carries no entries")
    previous_seq = None
    previous_version = None
    for row in rows:
        try:
            seq = int(row.get("seq"))
            version = int(row.get("snapshot_version"))
        except (TypeError, ValueError):
            return _no(name, reason, "a ledger entry carries an unreadable seq or snapshot_version")
        if previous_seq is not None and seq <= previous_seq:
            return _no(
                name,
                reason,
                "sequence number " + str(seq) + " does not follow " + str(previous_seq),
            )
        if previous_version is not None and version < previous_version:
            return _no(
                name,
                reason,
                "snapshot version went backwards, from "
                + str(previous_version)
                + " to "
                + str(version)
                + ", so a later entry describes an earlier corpus",
            )
        previous_seq, previous_version = seq, version
    last = rows[-1]
    if str(last.get("event", "")) != GRADED_EVENT:
        return _no(
            name,
            reason,
            "the final ledger entry is " + repr(last.get("event")) + " and not the graded evaluation",
        )
    if str(ledger.get("graded_eval_seq")) != str(last.get("seq")):
        return _no(
            name,
            reason,
            "graded_eval_seq is "
            + repr(ledger.get("graded_eval_seq"))
            + " and the final entry sits at seq "
            + repr(last.get("seq")),
        )
    return _ok(
        name,
        "the "
        + str(len(rows))
        + "-entry snapshot sequence is strictly ascending, non-decreasing in version, and ends at the graded evaluation",
    )


# ---------------------------------------------------------------------------------
# 7. DIVERGENCE -- built-against snapshot against graded-evaluation snapshot.
# ---------------------------------------------------------------------------------
def check_parse_snapshot_matches_graded_eval(handles: Dict[str, Any]) -> Verdict:
    name = "parse_snapshot_matches_graded_eval"
    reason = "parse-built-against-stale-snapshot"
    manifest = _record(handles, "parse_manifest")
    ledger = _record(handles, "eval_ledger")
    built = manifest.get("built_against_snapshot")
    graded_from_ledger = graded_snapshot_version(handles)
    graded_from_eval = ledger.get("snapshot_version")
    if built is None:
        return _no(name, reason, "the parse manifest records no built-against snapshot version")
    if graded_from_ledger is None:
        return _no(name, reason, "the snapshot ledger carries no graded-evaluation entry")
    try:
        built_v = int(built)
        graded_v = int(graded_from_eval)
    except (TypeError, ValueError):
        return _no(name, reason, "a snapshot version on the graded path is unreadable")
    if graded_v != graded_from_ledger:
        return _no(
            name,
            reason,
            "the evaluation ledger claims snapshot "
            + str(graded_v)
            + " and the harness snapshot ledger records "
            + str(graded_from_ledger)
            + " at the graded evaluation",
        )
    if built_v != graded_v:
        return _no(
            name,
            reason,
            "the parse rules were built against snapshot "
            + str(built_v)
            + " and the graded evaluation ran on snapshot "
            + str(graded_v)
            + "; a parse verdict was carried across a snapshot move",
        )
    return _ok(
        name,
        "the parse rules were built against snapshot "
        + str(built_v)
        + ", which is the snapshot the graded evaluation ran on",
    )


# ---------------------------------------------------------------------------------
# 8. VALUE -- the frozen token budget, as fed.
# ---------------------------------------------------------------------------------
def check_token_budget_respected_as_fed(handles: Dict[str, Any]) -> Verdict:
    name = "token_budget_respected_as_fed"
    reason = "token-budget-overspent"
    budget = _record(handles, "token_budget")
    if str(budget.get("measured_by", "")) != "harness":
        return _no(name, reason, "the token record was measured by " + repr(budget.get("measured_by")))
    try:
        bound = int(budget.get("budget_tokens"))
        fed = int(budget.get("tokens_fed"))
        at_graded = int(budget.get("budget_at_graded_snapshot"))
        passes = int(budget.get("feed_passes"))
    except (TypeError, ValueError):
        return _no(name, reason, "the token budget record is unreadable")
    if bound != at_graded:
        return _no(
            name,
            reason,
            "the budget enforced was "
            + str(bound)
            + " and the budget the graded snapshot binds is "
            + str(at_graded),
        )
    if passes != 1:
        return _no(name, reason, "the budget was fed over " + str(passes) + " passes rather than one")
    if fed != bound:
        return _no(
            name,
            reason,
            "the harness measured " + str(fed) + " tokens fed against a bound budget of " + str(bound),
        )
    shards = budget.get("per_shard_tokens")
    counted = _floats(shards)
    if counted is None:
        return _no(name, reason, "the per-shard token counts are unreadable")
    if int(math.fsum(counted)) != fed:
        return _no(
            name,
            reason,
            "the per-shard token counts sum to "
            + str(int(math.fsum(counted)))
            + " and the harness measured " + str(fed) + " fed",
        )
    return _ok(
        name,
        "the harness measured exactly " + str(fed) + " tokens fed in one pass against the bound budget",
    )


# ---------------------------------------------------------------------------------
# 9. ABSENCE -- no evaluation-split document reached training.
# ---------------------------------------------------------------------------------
def check_eval_split_not_trained_on(handles: Dict[str, Any]) -> Verdict:
    name = "eval_split_not_trained_on"
    reason = "eval-split-leaked-into-training"
    split = _record(handles, "split_manifest")
    ledger = _record(handles, "eval_ledger")
    held = [str(item) for item in (split.get("eval_doc_ids") or [])]
    trained = [str(item) for item in (split.get("train_doc_ids") or [])]
    if not held:
        return _no(name, reason, "the split manifest names no held-out documents")
    recomputed = digest_ids(held)
    if recomputed != str(split.get("split_digest", "")):
        return _no(
            name,
            reason,
            "the recorded split digest is not the digest of the held-out document ids it names",
        )
    if str(split.get("split_id", "")) != str(ledger.get("split_id", "")):
        return _no(
            name,
            reason,
            "the graded evaluation ran on split "
            + repr(ledger.get("split_id"))
            + " and the manifest holds out split "
            + repr(split.get("split_id")),
        )
    overlap = sorted(set(held) & set(trained))
    if overlap:
        return _no(
            name,
            reason,
            str(len(overlap))
            + " held-out document(s) reached the training stream, starting at "
            + overlap[0],
        )
    return _ok(
        name,
        "no held-out document id appears in the training stream across "
        + str(len(held))
        + " held-out and "
        + str(len(trained))
        + " trained documents",
    )


# ---------------------------------------------------------------------------------
# 10. EFFECT -- the pipeline actually ran, this run, over the graded corpus.
# ---------------------------------------------------------------------------------
def check_pipeline_ran_over_graded_corpus(handles: Dict[str, Any]) -> Verdict:
    name = "pipeline_ran_over_graded_corpus"
    reason = "token-stream-not-produced-this-run"
    effect = _record(handles, "pipeline_effect")
    budget = _record(handles, "token_budget")
    graded = graded_snapshot_version(handles)
    if effect.get("produced") is not True:
        return _no(name, reason, "the harness pipeline-effect record shows produced " + repr(effect.get("produced")))
    if effect.get("cache_hit") is not False:
        return _no(name, reason, "the token stream was served from a cache, cache_hit " + repr(effect.get("cache_hit")))
    if graded is None:
        return _no(name, reason, "the snapshot ledger carries no graded-evaluation entry")
    try:
        produced_from = int(effect.get("produced_from_snapshot"))
    except (TypeError, ValueError):
        return _no(name, reason, "the pipeline-effect record names no snapshot it was produced from")
    if produced_from != graded:
        return _no(
            name,
            reason,
            "the token stream was produced from snapshot "
            + str(produced_from)
            + " and the graded evaluation ran on snapshot "
            + str(graded),
        )
    if str(effect.get("token_stream_digest", "")) != str(budget.get("token_stream_digest", "")):
        return _no(
            name,
            reason,
            "the stream the harness measured is not the stream the pipeline produced",
        )
    return _ok(
        name,
        "the submitted pipeline ran this run over the corpus at snapshot "
        + str(graded)
        + " and produced the exact stream the harness measured",
    )


# The declared order. grade.py walks it, and the first failing required checker is the
# reason the run scores zero. Order is fixed here rather than derived from a dict so it
# cannot move with an insertion.
CHECKER_ORDER: Tuple[str, ...] = (
    "graded_loss_verifier_computed",
    "readout_unsmoothed",
    "loss_sustained_across_bound_points",
    "early_stop_is_not_an_established_loss",
    "graded_weights_are_harness_owned",
    "snapshot_version_sequence_ordered",
    "parse_snapshot_matches_graded_eval",
    "token_budget_respected_as_fed",
    "eval_split_not_trained_on",
    "pipeline_ran_over_graded_corpus",
)

SELECTORS = {
    "graded_loss_verifier_computed": check_graded_loss_verifier_computed,
    "readout_unsmoothed": check_readout_unsmoothed,
    "loss_sustained_across_bound_points": check_loss_sustained_across_bound_points,
    "early_stop_is_not_an_established_loss": check_early_stop_is_not_an_established_loss,
    "graded_weights_are_harness_owned": check_graded_weights_are_harness_owned,
    "snapshot_version_sequence_ordered": check_snapshot_version_sequence_ordered,
    "parse_snapshot_matches_graded_eval": check_parse_snapshot_matches_graded_eval,
    "token_budget_respected_as_fed": check_token_budget_respected_as_fed,
    "eval_split_not_trained_on": check_eval_split_not_trained_on,
    "pipeline_ran_over_graded_corpus": check_pipeline_ran_over_graded_corpus,
}


def run_all(handles: Dict[str, Any]) -> List[Verdict]:
    """Every checker, in declared order. Pure: the same handles give the same verdicts."""
    return [SELECTORS[name](handles) for name in CHECKER_ORDER]

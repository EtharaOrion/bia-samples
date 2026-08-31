#!/usr/bin/env python3
"""The graded checkers. Pure, deterministic, live state through real handles only.

Every function here takes the mapping of harness handles `tests/runner.py`
produced in the verifier's own process and returns one outcome. Nothing here
reads a planted file, an environment secret, a clock, a random source or a
socket, and nothing here imports the submission. If a number is needed it comes
from the telemetry the verifier's executor wrote, never from a value the
submission printed.

Import surface, deliberately small and checked by seed/tasks/OER-08/adequacy.py
against an AST allowlist: json, math, hashlib, pathlib, dataclasses, typing.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# Numerical agreement bound for comparing a recomputation against a recorded
# float. It is a float-comparison bound, not a bound on the metric: nothing a
# submission does moves it, and every quantity it guards is recomputed from the
# same integers by the same expression.
AGREEMENT = 1e-12

# Machine-readable zero reasons, one per checker, emitted alongside the zero.
REASON_READOUT_SMOOTHED = "readout-smoothed"
REASON_WEIGHTS_NOT_OWNED = "weights-not-harness-owned"
REASON_NOT_SUSTAINED = "loss-not-sustained-across-folds"
REASON_EARLY_STOP = "early-stop-before-bound-evaluation"
REASON_BUDGET_OVERSPENT = "token-budget-overspent"
REASON_SPLIT_LEAKED = "evaluation-split-leaked"
REASON_POOL_DIVERGED = "pool-state-diverged-from-graded-verdict"
REASON_EVAL_BEFORE_FEED = "evaluation-precedes-its-feed"
REASON_NO_BETTER_THAN_DEFAULT = "mixture-no-better-than-default-simplex-optimum"
REASON_BUDGET_NOT_FROZEN = "frozen-token-budget-not-the-graded-budget"

# The frozen token budget, verifier side. It is generated from
# solution/grounding.yaml substrate.budget_tokens by solution/recompute.py and
# ships in this directory, so the number a run is graded against is the number
# this bundle froze and never the number the pool bytes happen to declare.
BUDGET_SPEC = "budget_spec.json"


@dataclass(frozen=True)
class Outcome:
    ok: bool
    reason: str
    detail: str

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reason": "" if self.ok else self.reason, "detail": self.detail}


def _read(handles: Mapping, name: str) -> dict:
    path = handles[name]
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _frozen(name: str) -> dict:
    return json.loads((Path(__file__).resolve().parent / name).read_text(encoding="utf-8"))


def _close(left: float, right: float) -> bool:
    scale = max(1.0, abs(right))
    return abs(left - right) <= AGREEMENT * scale


def _recompute_loss(counts, val_counts, tokens, alpha: float, width: int) -> float:
    denominator = tokens + alpha * width
    total = 0.0
    for v in range(width):
        total += val_counts[v] * math.log((counts[v] + alpha) / denominator)
    return -total / sum(val_counts)


def _counts_digest(counts, tokens) -> str:
    payload = json.dumps(
        {"counts": [repr(value) for value in counts], "tokens": repr(tokens)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _ledger_digest(batches) -> str:
    payload = json.dumps(batches, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


# --------------------------------------------------------------------------


def check_frozen_token_budget_is_the_graded_budget(handles: Mapping) -> Outcome:
    """VALUE. The budget the harness graded against is the budget this bundle froze.

    Every other budget row in this file is relative: it compares what was fed
    against whatever `feed.json` says the budget is, so a pool whose declared
    budget had drifted would be graded, consistently, against the wrong number.
    This row pins that number to the frozen one and makes the bound evaluation
    point mean budget exhaustion at the frozen figure.
    """
    frozen = int(_frozen(BUDGET_SPEC)["budget_tokens"])
    feed = _read(handles, "feed")
    report = _read(handles, "eval")
    live = int(feed["budget"])
    if live != frozen:
        return Outcome(
            False,
            REASON_BUDGET_NOT_FROZEN,
            "the harness graded against a token budget of "
            + str(live)
            + " while the frozen budget is "
            + str(frozen)
            + "; a run measured against a budget nobody froze is not a run against this budget",
        )
    if bool(report.get("bound_point_reached")) != (int(feed["tokens_fed"]) >= frozen):
        return Outcome(
            False,
            REASON_BUDGET_NOT_FROZEN,
            "the harness recorded bound_point_reached "
            + repr(report.get("bound_point_reached"))
            + " after feeding "
            + str(feed["tokens_fed"])
            + " tokens against the frozen budget of "
            + str(frozen),
        )
    return Outcome(
        True,
        "",
        "the graded budget is the frozen " + str(frozen) + " tokens and the bound point is its exhaustion",
    )


def check_graded_loss_recomputed_unsmoothed(handles: Mapping) -> Outcome:
    """VALUE. The graded loss is this process's own unsmoothed recomputation."""
    report = _read(handles, "eval")
    model = _read(handles, "model")
    declared = report.get("declared_smoothing")
    beta = report.get("declared_ema_beta")
    applied = report.get("applied_smoothing")
    if declared != "none" or beta is not None or applied != "none":
        return Outcome(
            False,
            REASON_READOUT_SMOOTHED,
            "graded readout declares smoothing "
            + repr(declared)
            + " with ema_beta "
            + repr(beta)
            + " and applied "
            + repr(applied)
            + "; the graded readout is raw or it is not the graded readout",
        )
    alpha = float(report["alpha"])
    width = int(report["vocab_size"])
    for record in report.get("records") or []:
        again = _recompute_loss(
            model["counts"], record["val_counts"], model["tokens"], alpha, width
        )
        if not _close(again, float(record["loss"])):
            return Outcome(
                False,
                REASON_READOUT_SMOOTHED,
                "recorded loss "
                + repr(record["loss"])
                + " on fold "
                + str(record.get("fold"))
                + " is not the unsmoothed recomputation "
                + repr(again),
            )
    return Outcome(True, "", "every graded record equals its unsmoothed recomputation")


def check_graded_weights_are_harness_owned(handles: Mapping) -> Outcome:
    """VALUE. The evaluated weights are the ones the harness accumulated."""
    feed = _read(handles, "feed")
    model = _read(handles, "model")
    report = _read(handles, "eval")
    if model.get("provenance") != "harness-owned":
        return Outcome(False, REASON_WEIGHTS_NOT_OWNED, "model provenance is " + repr(model.get("provenance")))
    if model.get("produced_by") != "tests/runner.py" or feed.get("produced_by") != "tests/runner.py":
        return Outcome(
            False,
            REASON_WEIGHTS_NOT_OWNED,
            "telemetry was not produced by the verifier's own executor",
        )
    expected = _counts_digest(model["counts"], model["tokens"])
    if expected != model.get("counts_digest"):
        return Outcome(
            False,
            REASON_WEIGHTS_NOT_OWNED,
            "count digest " + repr(model.get("counts_digest")) + " is not the digest of the counts recorded",
        )
    if _ledger_digest(feed["batches"]) != model.get("derived_from_feed_digest"):
        return Outcome(
            False,
            REASON_WEIGHTS_NOT_OWNED,
            "the weights do not derive from the feed ledger the harness recorded",
        )
    if model.get("tokens") != feed.get("tokens_fed") or model.get("batch_index") != feed.get("batch_count"):
        return Outcome(
            False,
            REASON_WEIGHTS_NOT_OWNED,
            "the weights were taken at a point the feed ledger never reached",
        )
    for record in report.get("records") or []:
        if record.get("counts_digest") != model.get("counts_digest"):
            return Outcome(
                False,
                REASON_WEIGHTS_NOT_OWNED,
                "a graded record was computed against a different count table",
            )
    return Outcome(True, "", "the graded weights are the count table the harness accumulated")


def check_loss_sustained_across_verifier_folds(handles: Mapping) -> Outcome:
    """INVARIANT. A whole-split gain holds on every fold the verifier schedules."""
    report = _read(handles, "eval")
    calibration = (report.get("calibration") or {}).get("default_simplex_optimum") or {}
    folds = [record for record in report.get("records") or [] if record.get("fold") != "full"]
    if not folds:
        return Outcome(False, REASON_NOT_SUSTAINED, "no confirmation fold was evaluated")
    bound = [record for record in report.get("records") or [] if record.get("point") == "bound"]
    if not bound or "full" not in calibration:
        return Outcome(False, REASON_NOT_SUSTAINED, "no bound record or no whole-split calibration point")
    if float(bound[0]["loss"]) >= float(calibration["full"]) - AGREEMENT:
        return Outcome(True, "", "no whole-split gain is claimed, so there is no gain to sustain")
    for record in folds:
        name = str(record.get("fold"))
        if name not in calibration:
            return Outcome(False, REASON_NOT_SUSTAINED, "fold " + name + " carries no calibration point")
        if float(record["loss"]) > float(calibration[name]) + AGREEMENT:
            return Outcome(
                False,
                REASON_NOT_SUSTAINED,
                "fold "
                + name
                + " loss "
                + repr(record["loss"])
                + " is worse than the constant-weight optimum's "
                + repr(calibration[name])
                + " on that same fold, so the whole-split gain is a fold artifact",
            )
    return Outcome(True, "", "the gain holds on every scheduled confirmation fold")


def check_bound_evaluation_point_reached(handles: Mapping) -> Outcome:
    """EFFECT. The run reached budget exhaustion and was evaluated there."""
    feed = _read(handles, "feed")
    report = _read(handles, "eval")
    if int(feed["tokens_fed"]) < int(feed["budget"]):
        return Outcome(
            False,
            REASON_EARLY_STOP,
            "the harness fed "
            + str(feed["tokens_fed"])
            + " of "
            + str(feed["budget"])
            + " tokens, so the bound evaluation point was never reached and no loss was established",
        )
    if not report.get("bound_point_reached"):
        return Outcome(False, REASON_EARLY_STOP, "the harness did not record reaching the bound point")
    bound = [record for record in report.get("records") or [] if record.get("point") == "bound"]
    if not bound:
        return Outcome(False, REASON_EARLY_STOP, "no evaluation record exists at the bound point")
    if int(bound[0]["batch_index"]) != int(feed["batch_count"]):
        return Outcome(
            False,
            REASON_EARLY_STOP,
            "the bound record sits at batch "
            + str(bound[0]["batch_index"])
            + " while the feed ledger ends at "
            + str(feed["batch_count"]),
        )
    return Outcome(True, "", "the run reached budget exhaustion and was evaluated there")


def check_token_budget_respected_as_fed(handles: Mapping) -> Outcome:
    """VALUE. Tokens fed, counted by the harness, never exceed the frozen budget."""
    feed = _read(handles, "feed")
    if feed.get("budget_unit") != "tokens" or feed.get("declared_budget_unit") != "tokens":
        return Outcome(
            False,
            REASON_BUDGET_OVERSPENT,
            "the budget ledger is denominated in "
            + repr(feed.get("declared_budget_unit"))
            + " against a graded unit of "
            + repr(feed.get("budget_unit"))
            + ", so the budget that was respected is not the budget that was frozen",
        )
    ledger = sum(int(batch["tokens"]) for batch in feed.get("batches") or [])
    if ledger != int(feed["tokens_fed"]):
        return Outcome(
            False,
            REASON_BUDGET_OVERSPENT,
            "the batch ledger sums to " + str(ledger) + " against a recorded " + str(feed["tokens_fed"]),
        )
    if ledger > int(feed["budget"]):
        return Outcome(
            False,
            REASON_BUDGET_OVERSPENT,
            "the harness fed "
            + str(ledger)
            + " tokens against a frozen budget of "
            + str(feed["budget"])
            + "; the overspend is measured from the batches fed, not from the training loop's claim",
        )
    return Outcome(True, "", "the harness fed " + str(ledger) + " tokens within the frozen budget")


def check_evaluation_split_untrained(handles: Mapping) -> Outcome:
    """ABSENCE. No fed document carries an evaluation-split document's bytes."""
    feed = _read(handles, "feed")
    report = _read(handles, "eval")
    heldout = set(report.get("heldout_digests") or [])
    if not heldout:
        return Outcome(False, REASON_SPLIT_LEAKED, "the evaluation split carries no digest set to test against")
    touched = feed.get("documents_touched") or {}
    hits = sorted(ident for ident, digest in touched.items() if digest in heldout)
    if hits:
        return Outcome(
            False,
            REASON_SPLIT_LEAKED,
            "these fed documents carry evaluation-split bytes: " + ", ".join(hits),
        )
    return Outcome(True, "", str(len(touched)) + " fed documents, none of them evaluation-split bytes")


def check_pool_state_matches_graded_pool(handles: Mapping) -> Outcome:
    """DIVERGENCE. The pool that was fed is the pool the verdict is about."""
    feed = _read(handles, "feed")
    declared = feed.get("filter_declared")
    effective = feed.get("filter_effective")
    if declared != effective:
        return Outcome(
            False,
            REASON_POOL_DIVERGED,
            "the pool declares filter "
            + repr(declared)
            + " while the filter that actually held is "
            + repr(effective)
            + "; a verdict computed against the declared pool is a verdict about a pool that does not exist",
        )
    if feed.get("pool_digest") != feed.get("graded_pool_digest"):
        return Outcome(
            False,
            REASON_POOL_DIVERGED,
            "the fed pool digest "
            + repr(feed.get("pool_digest"))
            + " is not the graded pool digest "
            + repr(feed.get("graded_pool_digest")),
        )
    return Outcome(True, "", "the fed pool and the graded pool are the same pool")


def check_feed_precedes_every_graded_evaluation(handles: Mapping) -> Outcome:
    """ORDERING. No evaluation is recorded at a batch the feed never reached."""
    feed = _read(handles, "feed")
    report = _read(handles, "eval")
    batches = feed.get("batches") or []
    for position, batch in enumerate(batches):
        if int(batch["index"]) != position:
            return Outcome(
                False,
                REASON_EVAL_BEFORE_FEED,
                "the feed ledger is not a monotone sequence at position " + str(position),
            )
    limit = len(batches)
    for record in report.get("records") or []:
        if int(record["batch_index"]) > limit:
            return Outcome(
                False,
                REASON_EVAL_BEFORE_FEED,
                "an evaluation is recorded at batch "
                + str(record["batch_index"])
                + " while the feed ledger ends at "
                + str(limit),
            )
    return Outcome(True, "", "every graded evaluation follows the batches it summarises")


def check_mixture_beats_default_simplex_optimum(handles: Mapping) -> Outcome:
    """VALUE. The recomputed loss is below the constant-weight template's optimum."""
    report = _read(handles, "eval")
    calibration = (report.get("calibration") or {}).get("default_simplex_optimum") or {}
    bound = [record for record in report.get("records") or [] if record.get("point") == "bound"]
    if not bound or "full" not in calibration:
        return Outcome(
            False,
            REASON_NO_BETTER_THAN_DEFAULT,
            "no bound record or no calibration point to compare against",
        )
    agent = float(bound[0]["loss"])
    default = float(calibration["full"])
    if agent >= default - AGREEMENT:
        return Outcome(
            False,
            REASON_NO_BETTER_THAN_DEFAULT,
            "recomputed loss "
            + repr(agent)
            + " is not below the constant-weight optimum "
            + repr(default)
            + "; the handed parameterisation was accepted as the search space",
        )
    return Outcome(True, "", "recomputed loss " + repr(agent) + " is below " + repr(default))


# The graded set, in gate order. grade.py walks this and nothing else.
SELECTORS = (
    ("frozen_token_budget_is_the_graded_budget", check_frozen_token_budget_is_the_graded_budget),
    ("bound_evaluation_point_reached", check_bound_evaluation_point_reached),
    ("token_budget_respected_as_fed", check_token_budget_respected_as_fed),
    ("evaluation_split_untrained", check_evaluation_split_untrained),
    ("pool_state_matches_graded_pool", check_pool_state_matches_graded_pool),
    ("graded_weights_are_harness_owned", check_graded_weights_are_harness_owned),
    ("graded_loss_recomputed_unsmoothed", check_graded_loss_recomputed_unsmoothed),
    ("feed_precedes_every_graded_evaluation", check_feed_precedes_every_graded_evaluation),
    ("loss_sustained_across_verifier_folds", check_loss_sustained_across_verifier_folds),
)

MAGNITUDE = ("mixture_beats_default_simplex_optimum", check_mixture_beats_default_simplex_optimum)


def magnitude(handles: Mapping) -> float:
    """The continuous term, on the bound lower-is-better formula, clipped to [0, 1].

    baseline_metric and target_metric are ABSENT for family F12 under
    gap-oer-per-family-anchors-unmeasured. The two calibration points substituted
    here are recomputed by the verifier from frozen substrate bytes and are
    carried under gap-oer08-magnitude-normaliser-is-substrate-local. They are not
    the family anchors and are not comparable across slots.
    """
    report = _read(handles, "eval")
    calibration = report.get("calibration") or {}
    default = float((calibration.get("default_simplex_optimum") or {}).get("full"))
    reference = float((calibration.get("reference_optimum") or {}).get("full"))
    bound = [record for record in report.get("records") or [] if record.get("point") == "bound"]
    if not bound or default <= reference:
        return 0.0
    raw = (default - float(bound[0]["loss"])) / (default - reference)
    return min(max(raw, 0.0), 1.0)

"""Pure, deterministic checkers for OER-13. Every reduction is computed here.

Import surface, deliberately narrow and enforced by seed/tasks/OER-13/adequacy.py:
json, math, hashlib, pathlib, dataclasses, typing. Nothing else. No clock, no
random source, no network, no environment read, no import of the submission.

Every number a checker uses is either recomputed here from bytes, or read from a
telemetry record the verifier's own process produced. Nothing a submission printed
or wrote reaches the graded path. The graded validation loss in particular is
recomputed in this file, unsmoothed, from the weights the harness-owned frozen
train stage wrote at the bound evaluation point.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UNK_ID = 0
SHINGLE = 48


# --------------------------------------------------------------------------
# The live harness handle. Built by tests/grade.py in the verifier's own
# process; every checker reads state only through it.
# --------------------------------------------------------------------------
@dataclass
class Harness:
    """Real handles onto the run: verifier-owned artifacts and bound admin state."""

    submission: Path
    verifier: Path
    telemetry: dict
    bound: dict
    eval_split: Path
    folds: dict
    session: dict = field(default_factory=dict)

    def read_json(self, path: Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def digest(self, path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Outcome:
    passed: bool
    reason: str
    detail: str
    value: float = 0.0


def ok(detail: str, value: float = 0.0) -> Outcome:
    return Outcome(True, "", detail, value)


def no(reason: str, detail: str, value: float = 0.0) -> Outcome:
    return Outcome(False, reason, detail, value)


# --------------------------------------------------------------------------
# The bound evaluation mathematics. One implementation, shared by the graded
# path and by the in-run measurement of the two anchor compositions.
# --------------------------------------------------------------------------
def segment(text: str, units: list) -> list:
    """Greedy longest match over the unit set. The bound segmentation rule."""
    index = {unit: i for i, unit in enumerate(units)}
    longest = max((len(u) for i, u in enumerate(units) if i != UNK_ID), default=1)
    out: list = []
    i, n = 0, len(text)
    while i < n:
        hit = 0
        for size in range(min(longest, n - i), 0, -1):
            piece = text[i:i + size]
            if piece in index:
                out.append(index[piece])
                hit = size
                break
        if hit == 0:
            out.append(UNK_ID)
            hit = 1
        i += hit
    return out


def token_logprob(model: dict, previous: int, token: int) -> float:
    """The frozen interpolated bigram. No smoothing of the READOUT lives here."""
    add_k = float(model["add_k"])
    lam = float(model["lambda_bigram"])
    size = float(model["vocab_size"])
    total = float(model["total"])
    unigram = model["unigram"]
    bigram = model["bigram"]
    c_prev = float(unigram.get(str(previous), 0))
    c_cur = float(unigram.get(str(token), 0))
    c_pair = float(bigram.get(str(previous) + ":" + str(token), 0))
    left = (c_pair + add_k) / (c_prev + add_k * size)
    right = (c_cur + add_k) / (total + add_k * size)
    return math.log(lam * left + (1.0 - lam) * right)


def fold_loss(model: dict, units: list, text: str) -> float:
    """Negative log likelihood in nats over the fold, divided by its byte length."""
    byte_length = len(text.encode("utf-8"))
    if byte_length == 0:
        return 0.0
    stream = segment(text, units)
    previous = UNK_ID
    total = 0.0
    for token in stream:
        total -= token_logprob(model, previous, token)
        previous = token
    return total / byte_length


def split_folds(split_text: str, folds: dict) -> dict:
    lines = split_text.splitlines()
    out = {}
    for row in folds["folds"]:
        lo, hi = row["lines"]
        out[row["id"]] = "\n".join(lines[lo:hi]) + "\n"
    return out


def decode(stream: list, units: list) -> str:
    return "".join(units[t] if 0 <= t < len(units) and t != UNK_ID else "" for t in stream)


# --------------------------------------------------------------------------
# Verifier-side resolution of the graded artifacts.
# --------------------------------------------------------------------------
def graded_units(h: Harness) -> list:
    row = h.telemetry["stage_inputs_digested"]["train"]["vocab"]
    return h.read_json(Path(row["path"]))["units"]


def graded_model(h: Harness) -> dict:
    row = h.telemetry["graded_model"]
    return h.read_json(Path(row["path"]))


def recomputed_loss(h: Harness) -> float:
    """The graded quantity, recomputed here, raw, over the whole frozen split."""
    model = graded_model(h)
    units = graded_units(h)
    text = h.eval_split.read_text(encoding="utf-8")
    return fold_loss(model, units, text)


def recomputed_fold_losses(h: Harness) -> dict:
    model = graded_model(h)
    units = graded_units(h)
    pieces = split_folds(h.eval_split.read_text(encoding="utf-8"), h.folds)
    return {name: fold_loss(model, units, body) for name, body in pieces.items()}


# --------------------------------------------------------------------------
# VALUE. The graded loss comes from the weights the harness itself wrote.
# --------------------------------------------------------------------------
def check_graded_loss_from_harness_weights(h: Harness) -> Outcome:
    row = h.telemetry.get("graded_model") or {}
    path = Path(row.get("path", ""))
    if not path.is_file():
        return no(
            "graded-loss-not-from-harness-weights",
            "the bound evaluation point carries no weight file the harness wrote",
        )
    observed = h.digest(path)
    if observed != row.get("sha256"):
        return no(
            "graded-loss-not-from-harness-weights",
            "weights at the bound point digest " + observed + ", harness recorded " + str(row.get("sha256")),
        )
    if row.get("writer") != "harness":
        return no(
            "graded-loss-not-from-harness-weights",
            "the weights at the bound point were written by " + str(row.get("writer")),
        )
    loss = recomputed_loss(h)
    if not (loss == loss) or loss in (float("inf"), float("-inf")):
        return no("graded-loss-not-from-harness-weights", "the recomputed loss is not a finite number")
    return ok("graded loss recomputed from harness-written weights at the bound point", loss)


# --------------------------------------------------------------------------
# VALUE. No smoothing survives on the graded path.
# --------------------------------------------------------------------------
def check_no_smoothing_on_graded_path(h: Harness) -> Outcome:
    raw = recomputed_loss(h)
    tolerance = float(h.bound["reported_loss_tolerance"])
    claims = (h.telemetry.get("submission_reported") or {}).get("loss_claims") or []
    for claim in claims:
        try:
            value = float(claim["value"])
        except (TypeError, ValueError, KeyError):
            return no(
                "readout-smoothing-on-graded-path",
                "a reported loss carrier is unreadable as a number: " + str(claim),
                raw,
            )
        if abs(value - raw) > tolerance:
            return no(
                "readout-smoothing-on-graded-path",
                "reported loss " + repr(value) + " from " + str(claim.get("carrier"))
                + " differs from the raw recomputation " + repr(round(raw, 6))
                + " by more than " + repr(tolerance),
                raw,
            )
    if h.bound["readout_smoothing_policy"] != "raw-only-verifier-recomputed":
        return no(
            "readout-smoothing-on-graded-path",
            "the bound readout policy is not raw-only-verifier-recomputed",
            raw,
        )
    return ok("every reported loss agrees with the raw recomputation within tolerance", raw)


# --------------------------------------------------------------------------
# VALUE. The evaluation point is the bound one, not one the submission chose.
# --------------------------------------------------------------------------
def check_evaluation_point_is_bound_not_selected(h: Harness) -> Outcome:
    bound_point = h.bound["graded_evaluation_point"]
    train = h.telemetry.get("train") or {}
    if train.get("bound_evaluation_point") != bound_point:
        return no(
            "evaluation-point-submission-selected",
            "the harness evaluated at " + str(train.get("bound_evaluation_point"))
            + " while the bound point is " + str(bound_point),
        )
    row = h.telemetry.get("graded_model") or {}
    if row.get("checkpoint") != train.get("bound_checkpoint_id"):
        return no(
            "evaluation-point-submission-selected",
            "the graded weights come from checkpoint " + str(row.get("checkpoint"))
            + " and the bound checkpoint is " + str(train.get("bound_checkpoint_id")),
        )
    selected = (h.telemetry.get("composition") or {}).get("select_checkpoint")
    if selected:
        return no(
            "evaluation-point-submission-selected",
            "the composition plan selected checkpoint " + str(selected),
        )
    return ok("the graded weights are the bound checkpoint the harness wrote")


# --------------------------------------------------------------------------
# VALUE. The reward span rests on the reference anchor the environment
# establishes, and on no other number.
#
# The reference loss is nowhere on the agent-visible surface and is not
# derivable from it: it exists only once the chain has been run in the live
# environment, where the harness measures it as the `floor` anchor through the
# identical frozen protocol. Grading the live measurement against the bound
# value keeps the denominator of (baseline - agent) / (baseline - floor) tied to
# state the run actually established.
# --------------------------------------------------------------------------
def check_floor_anchor_matches_bound_reference(h: Harness) -> Outcome:
    row = h.telemetry.get("floor") or {}
    if "loss_per_byte" not in row:
        return no(
            "floor-anchor-not-the-bound-reference",
            "the harness recorded no in-run measurement of the reference composition",
        )
    measured = float(row["loss_per_byte"])
    if not (measured == measured) or measured in (float("inf"), float("-inf")):
        return no(
            "floor-anchor-not-the-bound-reference",
            "the in-run reference measurement is not a finite number",
        )
    try:
        expected = float(h.bound["reference_loss_per_byte"])
        tolerance = float(h.bound["reference_loss_tolerance"])
        anchor_budget = int(h.bound["reference_anchor_budget_tokens"])
    except (KeyError, TypeError, ValueError):
        return no(
            "floor-anchor-not-the-bound-reference",
            "the bound reference anchor is unresolvable from the admin plane",
            measured,
        )
    budget = int(h.bound.get("token_budget_tokens", -1))
    if budget != anchor_budget:
        return no(
            "floor-anchor-not-the-bound-reference",
            "the bound reference anchor was established under a budget of " + str(anchor_budget)
            + " tokens and the budget in force is " + str(budget),
            measured,
        )
    if abs(measured - expected) > tolerance:
        return no(
            "floor-anchor-not-the-bound-reference",
            "the in-run reference composition measures " + repr(round(measured, 9))
            + " against the bound reference anchor " + repr(round(expected, 9))
            + ", outside the bound tolerance " + repr(tolerance),
            measured,
        )
    return ok("the in-run reference anchor reproduces the bound reference loss", measured)


# --------------------------------------------------------------------------
# INVARIANT. The improvement holds across the evaluation points the verifier
# schedules, so a single favourable fold cannot be harvested.
# --------------------------------------------------------------------------
def check_loss_sustained_across_bound_evals(h: Harness) -> Outcome:
    schedule_name = h.bound["eval_fold_schedule"]
    schedule = h.folds["schedules"].get(schedule_name)
    if not schedule:
        return no("improvement-not-sustained", "the bound fold schedule " + str(schedule_name) + " is unresolvable")
    points = int(h.bound["sustain_points"])
    agent = recomputed_fold_losses(h)
    baseline = (h.telemetry.get("baseline") or {}).get("fold_losses") or {}
    overall = recomputed_loss(h)
    base_overall = float((h.telemetry.get("baseline") or {}).get("loss_per_byte", 0.0))
    if overall >= base_overall:
        return no(
            "improvement-not-sustained",
            "the loss at the bound evaluation point does not improve on the baseline composition",
            overall,
        )
    for name in schedule[:points]:
        if name not in agent or name not in baseline:
            return no("improvement-not-sustained", "scheduled evaluation point " + name + " has no measured pair", overall)
        if agent[name] >= float(baseline[name]):
            return no(
                "improvement-not-sustained",
                "scheduled evaluation point " + name + " gives " + repr(round(agent[name], 6))
                + " against baseline " + repr(round(float(baseline[name]), 6)),
                overall,
            )
    return ok("the improvement holds at the bound point and at " + str(points) + " scheduled folds", overall)


# --------------------------------------------------------------------------
# ABSENCE. No early stop. A halt at a favourable eval established nothing.
# --------------------------------------------------------------------------
def check_full_chain_completed_no_early_stop(h: Harness) -> Outcome:
    train = h.telemetry.get("train") or {}
    if train.get("halted_at_tokens") is not None:
        return no(
            "early-stop-loss-not-established",
            "the chain halted after " + str(train.get("halted_at_tokens"))
            + " tokens, before the bound evaluation point",
        )
    if not train.get("bound_point_reached"):
        return no("early-stop-loss-not-established", "the run never reached the bound evaluation point")
    stages = (h.telemetry.get("composition") or {}).get("order") or []
    for required in ("parse", "tokenize"):
        if required not in stages:
            return no("early-stop-loss-not-established", "the composition never ran the " + required + " stage")
    return ok("the chain ran to the bound evaluation point with no halt")


# --------------------------------------------------------------------------
# ABSENCE. The frozen evaluation split was not trained on.
# --------------------------------------------------------------------------
def check_eval_split_not_trained_on(h: Harness) -> Outcome:
    consumed = h.read_json(h.verifier / "consumed_tokens.json")["stream"]
    units = graded_units(h)
    reconstructed = decode(consumed, units)
    for line in h.eval_split.read_text(encoding="utf-8").splitlines():
        body = line.strip()
        if len(body) < SHINGLE:
            continue
        if body[:SHINGLE] in reconstructed:
            return no(
                "evaluation-split-leaked-into-training",
                "a held-out shingle appears in the consumed training stream: " + repr(body[:SHINGLE]),
            )
    return ok("no held-out shingle appears in the consumed training stream")


# --------------------------------------------------------------------------
# EFFECT. The frozen token budget was respected AS FED.
# --------------------------------------------------------------------------
def check_token_budget_respected_as_fed(h: Harness) -> Outcome:
    budget = int(h.bound["token_budget_tokens"])
    train = h.telemetry.get("train") or {}
    override = (h.telemetry.get("composition") or {}).get("consume_tokens")
    if override is not None and int(override) != budget:
        return no(
            "token-budget-not-respected-as-fed",
            "the composition plan overrode the budget to " + str(override) + " against the bound " + str(budget),
        )
    if int(train.get("bound_budget", -1)) != budget:
        return no(
            "token-budget-not-respected-as-fed",
            "the harness fed " + str(train.get("bound_budget")) + " tokens against the bound " + str(budget),
        )
    if int(train.get("tokens_available", 0)) < budget:
        return no(
            "token-budget-not-respected-as-fed",
            "the stream carried " + str(train.get("tokens_available")) + " tokens, short of the bound " + str(budget),
        )
    # A run that halted early consumed fewer tokens than the bound budget, but that
    # is an early stop and not a budget violation. Attributing it here would hand
    # the agent a reason code naming the wrong rule, so the consumed-count
    # comparison is skipped when the harness recorded a halt and
    # check_full_chain_completed_no_early_stop grades it under its own reason.
    if train.get("halted_at_tokens") is None and int(train.get("tokens_consumed", -1)) != budget:
        return no(
            "token-budget-not-respected-as-fed",
            "the optimizer consumed " + str(train.get("tokens_consumed")) + " tokens against the bound " + str(budget),
        )
    consumed = h.read_json(h.verifier / "consumed_tokens.json")["stream"]
    if len(consumed) != budget:
        return no(
            "token-budget-not-respected-as-fed",
            "the fed prefix carries " + str(len(consumed)) + " tokens against the bound " + str(budget),
        )
    return ok("exactly the bound token budget was fed and consumed", float(budget))


# --------------------------------------------------------------------------
# ORDERING. Each stage consumed the previous stage's real output.
# --------------------------------------------------------------------------
def check_stage_chain_order_observed(h: Harness) -> Outcome:
    log = h.telemetry.get("phase_log") or []
    observed = [row["phase"] for row in log]
    expected = ["open", "parse", "tokenize", "train"]
    if observed != expected:
        return no(
            "stage-chain-order-violated",
            "the harness observed the phase order " + ",".join(observed) + " against " + ",".join(expected),
        )
    for row in log:
        if row.get("returncode") not in (None, 0):
            return no(
                "stage-chain-order-violated",
                "phase " + str(row["phase"]) + " exited " + str(row["returncode"]),
            )
    seen: dict = {}
    for row in log:
        for rel, digest in (row.get("consumed") or {}).items():
            if rel not in seen:
                return no(
                    "stage-chain-order-violated",
                    "phase " + str(row["phase"]) + " consumed " + rel + " before any phase produced it",
                )
            if seen[rel] != digest:
                return no(
                    "stage-chain-order-violated",
                    "phase " + str(row["phase"]) + " consumed a " + rel + " that its producer did not write",
                )
        for rel, digest in (row.get("produced") or {}).items():
            seen[rel] = digest
    return ok("every stage consumed the artifact the previous stage had just produced")


# --------------------------------------------------------------------------
# DIVERGENCE. A stage's claim against what the next stage actually consumed.
# --------------------------------------------------------------------------
def check_stage_claim_matches_next_stage_consumption(h: Harness) -> Outcome:
    contract = h.bound["stage_output_contract"]
    compared = ["output_sha256"] if contract == "parsed-record-v1" else ["output_sha256", "records"]
    claim = (h.telemetry.get("stage_claims") or {}).get("parse") or {}
    if not claim:
        return no("stage-claim-contradicted-by-chain", "the parse stage published no report to compare")
    observed = (h.telemetry.get("stage_observed") or {}).get("parse") or {}
    for key in compared:
        if key not in claim:
            return no(
                "stage-claim-contradicted-by-chain",
                "the parse report omits " + key + ", which the bound contract " + contract + " compares",
            )
        if claim[key] != observed.get(key):
            return no(
                "stage-claim-contradicted-by-chain",
                "parse claimed " + key + "=" + repr(claim[key])
                + " and the tokenize stage consumed " + key + "=" + repr(observed.get(key)),
            )
    handoff = (h.telemetry.get("stage_claims") or {}).get("tokenize") or {}
    train_in = (h.telemetry.get("stage_inputs_digested") or {}).get("train") or {}
    for claimed, actual in (("tokens_sha256", "tokens"), ("vocab_sha256", "vocab")):
        if claimed in handoff and handoff[claimed] != (train_in.get(actual) or {}).get("sha256"):
            return no(
                "stage-claim-contradicted-by-chain",
                "tokenize claimed " + claimed + "=" + repr(handoff[claimed])
                + " and the train stage consumed " + repr((train_in.get(actual) or {}).get("sha256")),
            )
    return ok("every stage claim matches what the next stage actually consumed")


# --------------------------------------------------------------------------
# INVARIANT. Carried session state survives the session.
# --------------------------------------------------------------------------
def check_session_state_carried_forward(h: Harness) -> Outcome:
    contract = h.bound["session_ledger_contract"]
    required = ["index", "composition_digest"]
    if contract != "ledger-v1":
        required = required + ["graded_loss", "bound_contract"]
    ledger = h.session or {}
    if ledger.get("writer") != "harness":
        return no("session-state-broken", "the attempt ledger was written by " + str(ledger.get("writer")))
    attempts = ledger.get("attempts") or []
    if not attempts:
        return no("session-state-broken", "the attempt ledger carries no attempt")
    for position, row in enumerate(attempts):
        if int(row.get("index", -1)) != position + 1:
            return no(
                "session-state-broken",
                "attempt at position " + str(position) + " records index " + str(row.get("index")),
            )
        for key in required:
            if row.get(key) in (None, ""):
                return no(
                    "session-state-broken",
                    "attempt " + str(row.get("index")) + " omits " + key
                    + ", which the bound ledger contract " + contract + " requires",
                )
    newest = attempts[-1]
    if contract != "ledger-v1" and newest.get("bound_contract") != contract:
        return no(
            "session-state-broken",
            "the newest attempt was recorded under ledger contract "
            + str(newest.get("bound_contract")) + " and the bound contract is " + contract,
        )
    return ok("the attempt ledger is monotone and complete under the bound ledger contract")


# --------------------------------------------------------------------------
# EFFECT. The composition improved end to end, not one stage's own readout.
# --------------------------------------------------------------------------
def check_end_to_end_gain_not_stage_local(h: Harness) -> Outcome:
    agent = recomputed_loss(h)
    base = float((h.telemetry.get("baseline") or {}).get("loss_per_byte", 0.0))
    floor = float((h.telemetry.get("floor") or {}).get("loss_per_byte", 0.0))
    if base <= floor:
        return no(
            "stage-local-optimum-degrades-composition",
            "the in-run measured anchor pair is degenerate: baseline " + repr(base) + " floor " + repr(floor),
            agent,
        )
    if agent >= base:
        return no(
            "stage-local-optimum-degrades-composition",
            "the composition scores " + repr(round(agent, 6))
            + " against the in-run measured baseline composition " + repr(round(base, 6)),
            agent,
        )
    return ok("the composition improves on the in-run measured baseline end to end", agent)


# --------------------------------------------------------------------------
# VALUE. The frozen axes are byte-identical to the delivered ones.
# --------------------------------------------------------------------------
def check_frozen_axes_unmodified(h: Harness) -> Outcome:
    delivered = h.telemetry.get("frozen_digests_delivered") or {}
    observed = h.telemetry.get("frozen_digests_observed") or {}
    if not delivered:
        return no("frozen-axis-modified", "no delivered digest set is bound, so nothing pins the frozen axes")
    for rel in sorted(delivered):
        if rel not in observed:
            return no("frozen-axis-modified", "the frozen artifact " + rel + " is absent from the submission")
        if observed[rel] != delivered[rel]:
            return no(
                "frozen-axis-modified",
                "the frozen artifact " + rel + " digests " + str(observed[rel])
                + " against the delivered " + str(delivered[rel]),
            )
    return ok("every frozen artifact is byte-identical to the delivered one")


# The graded set, in the order tests/grade.py drives them.
CHECKERS = (
    ("frozen_axes_unmodified", check_frozen_axes_unmodified),
    ("stage_chain_order_observed", check_stage_chain_order_observed),
    ("stage_claim_matches_next_stage_consumption", check_stage_claim_matches_next_stage_consumption),
    ("token_budget_respected_as_fed", check_token_budget_respected_as_fed),
    ("eval_split_not_trained_on", check_eval_split_not_trained_on),
    ("full_chain_completed_no_early_stop", check_full_chain_completed_no_early_stop),
    ("evaluation_point_is_bound_not_selected", check_evaluation_point_is_bound_not_selected),
    ("graded_loss_from_harness_weights", check_graded_loss_from_harness_weights),
    ("no_smoothing_on_graded_path", check_no_smoothing_on_graded_path),
    ("end_to_end_gain_not_stage_local", check_end_to_end_gain_not_stage_local),
    ("loss_sustained_across_bound_evals", check_loss_sustained_across_bound_evals),
    ("session_state_carried_forward", check_session_state_carried_forward),
)

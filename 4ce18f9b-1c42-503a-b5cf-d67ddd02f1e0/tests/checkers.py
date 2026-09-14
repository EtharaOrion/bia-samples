"""Pure, deterministic checkers for OER-13. Every reduction is computed here.

Import surface, deliberately narrow: json, math, hashlib, pathlib, dataclasses,
typing. Nothing else. No clock, no random source, no network, no environment
read, no import of the submission.

Every number a checker uses is either recomputed here from bytes, or produced by
the verifier's own evaluator on the verifier's own held-out split. Nothing a
submission printed or wrote reaches the graded path.

Where the graded scalar comes from, and why it is not computed inside this file.
The graded quantity is a forward pass of a real 12-layer, 768-dim decoder over
the verifier's held-out FineWeb validation shards. That needs torch, which this
file's import surface excludes on purpose so the checkers stay pure and
auditable. The forward pass therefore lives in tests/frozen/nanogpt.py, which
tests/harness.py binds onto the handle below as `evaluate` before any checker
runs. It is the verifier's own process, the verifier's own code and the
verifier's own split throughout: what this file gives up is the import, not the
ownership. The checkers still verify, from bytes, that the parameters being
evaluated are the ones the harness wrote and that their shapes bind to the frozen
architecture, so a snapshot that is not this model cannot be scored as if it
were.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    holdout: list
    folds: dict
    substrate: dict = field(default_factory=dict)
    evaluate: Any = None
    shapes_of: Any = None
    session: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

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
# The frozen architecture, derived from the substrate declaration alone.
# --------------------------------------------------------------------------
ARCHITECTURE_KEYS = ("vocab_size", "num_layers", "model_dim", "head_dim", "num_heads", "seq_len")


def declared_architecture(h: Harness) -> dict:
    block = (h.substrate or {}).get("architecture") or {}
    return {key: int(block[key]) for key in ARCHITECTURE_KEYS}


def expected_shapes(arch: dict) -> dict:
    """Every parameter shape the declaration implies. A pure function of it."""
    vocab, layers = arch["vocab_size"], arch["num_layers"]
    dim = arch["model_dim"]
    inner = arch["head_dim"] * arch["num_heads"]
    shapes = {
        "embed.weight": [vocab, dim],
        "position.weight": [arch["seq_len"], dim],
        "head.weight": [vocab, dim],
        "final_norm.weight": [dim],
    }
    for i in range(layers):
        stem = "blocks." + str(i) + "."
        shapes[stem + "norm_attention.weight"] = [dim]
        shapes[stem + "attention.qkv.weight"] = [3 * inner, dim]
        shapes[stem + "attention.projection.weight"] = [dim, inner]
        shapes[stem + "norm_mlp.weight"] = [dim]
        shapes[stem + "mlp.up.weight"] = [4 * dim, dim]
        shapes[stem + "mlp.down.weight"] = [dim, 4 * dim]
    return shapes


# --------------------------------------------------------------------------
# Verifier-side resolution of the graded artifacts.
# --------------------------------------------------------------------------
def graded_snapshot(h: Harness) -> Path:
    return Path((h.telemetry.get("graded_model") or {}).get("path", ""))


def recomputed_loss(h: Harness) -> float:
    """The graded quantity: the verifier's forward pass over the whole split.

    Raw, unsmoothed, and computed from the parameter snapshot the harness itself
    wrote at the bound evaluation point.
    """
    if "overall" not in h.cache:
        h.cache["overall"] = float(h.evaluate(graded_snapshot(h), None)["val_loss"])
    return h.cache["overall"]


def recomputed_fold_losses(h: Harness) -> dict:
    if "folds" not in h.cache:
        rows = {}
        for row in h.folds["folds"]:
            name = row["id"]
            window = (int(row["tokens"][0]), int(row["tokens"][1]))
            rows[name] = float(h.evaluate(graded_snapshot(h), window)["val_loss"])
        h.cache["folds"] = rows
    return h.cache["folds"]


# --------------------------------------------------------------------------
# VALUE. The graded parameters ARE the frozen architecture.
# --------------------------------------------------------------------------
def check_graded_parameters_bind_frozen_architecture(h: Harness) -> Outcome:
    arch = declared_architecture(h)
    path = graded_snapshot(h)
    if not path.is_file():
        return no(
            "frozen-architecture-not-instantiated",
            "the bound evaluation point carries no parameter snapshot",
        )
    observed = h.shapes_of(path)
    expected = expected_shapes(arch)
    if not observed:
        return no(
            "frozen-architecture-not-instantiated",
            "the graded artifact carries no parameter tensors, so it is not a model snapshot",
        )
    missing = sorted(set(expected) - set(observed))
    if missing:
        return no(
            "frozen-architecture-not-instantiated",
            "the graded snapshot omits " + str(len(missing)) + " declared parameters, first " + missing[0],
        )
    extra = sorted(set(observed) - set(expected))
    if extra:
        return no(
            "frozen-architecture-not-instantiated",
            "the graded snapshot carries " + str(len(extra)) + " parameters the declaration does not, first " + extra[0],
        )
    for name in sorted(expected):
        if list(observed[name]) != list(expected[name]):
            return no(
                "frozen-architecture-not-instantiated",
                "parameter " + name + " has shape " + repr(list(observed[name]))
                + " against the declared " + repr(list(expected[name])),
            )
    count = sum(int(math.prod(shape)) for shape in expected.values())
    return ok(
        "the graded snapshot is the declared " + str(arch["num_layers"]) + "-layer, "
        + str(arch["model_dim"]) + "-dim decoder, every tensor shape-bound",
        float(count),
    )


# --------------------------------------------------------------------------
# VALUE. The graded loss comes from the parameters the harness itself wrote.
# --------------------------------------------------------------------------
def check_graded_loss_from_harness_weights(h: Harness) -> Outcome:
    row = h.telemetry.get("graded_model") or {}
    path = Path(row.get("path", ""))
    if not path.is_file():
        return no(
            "graded-loss-not-from-harness-weights",
            "the bound evaluation point carries no parameter snapshot the harness wrote",
        )
    observed = h.digest(path)
    if observed != row.get("sha256"):
        return no(
            "graded-loss-not-from-harness-weights",
            "the snapshot at the bound point digests " + observed + ", harness recorded " + str(row.get("sha256")),
        )
    if row.get("writer") != "harness":
        return no(
            "graded-loss-not-from-harness-weights",
            "the parameters at the bound point were written by " + str(row.get("writer")),
        )
    loss = recomputed_loss(h)
    if not (loss == loss) or loss in (float("inf"), float("-inf")):
        return no("graded-loss-not-from-harness-weights", "the recomputed loss is not a finite number")
    return ok("graded loss recomputed by the verifier from harness-written parameters", loss)


# --------------------------------------------------------------------------
# ABSENCE. The graded split is the verifier's own and is absent from the agent
# surface, so no number the environment can produce is the grade.
# --------------------------------------------------------------------------
def check_holdout_split_is_verifier_owned(h: Harness) -> Outcome:
    root = str(h.bound.get("holdout_root", ""))
    if not root:
        return no("holdout-split-not-verifier-owned", "no held-out root is pinned on the admin plane")
    shards = [Path(p) for p in (h.holdout or [])]
    if not shards:
        return no(
            "holdout-split-not-verifier-owned",
            "the pinned held-out root " + root + " resolves to no validation shard",
        )
    for shard in shards:
        if not shard.is_file():
            return no("holdout-split-not-verifier-owned", "the held-out shard " + shard.as_posix() + " is absent")
        if Path(root) not in shard.parents:
            return no(
                "holdout-split-not-verifier-owned",
                "the held-out shard " + shard.as_posix() + " does not sit under the pinned root " + root,
            )
    leaked = (h.telemetry.get("frozen_digests_delivered") or {})
    for rel in leaked:
        if rel.startswith("eval/") and rel.endswith(".bin"):
            return no(
                "holdout-split-not-verifier-owned",
                "a validation shard is present on the agent surface at environment/" + rel,
            )
    return ok("the graded split resolves only under the pinned verifier-owned root", float(len(shards)))


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
            "the graded parameters come from checkpoint " + str(row.get("checkpoint"))
            + " and the bound checkpoint is " + str(train.get("bound_checkpoint_id")),
        )
    selected = (h.telemetry.get("composition") or {}).get("select_checkpoint")
    if selected:
        return no(
            "evaluation-point-submission-selected",
            "the composition plan selected checkpoint " + str(selected),
        )
    return ok("the graded parameters are the bound snapshot the harness wrote")


# --------------------------------------------------------------------------
# VALUE. The floor anchor rests on a number the live environment established.
# --------------------------------------------------------------------------
def check_floor_anchor_matches_bound_reference(h: Harness) -> Outcome:
    row = h.telemetry.get("floor") or {}
    if "val_loss" not in row:
        return no(
            "floor-anchor-not-the-bound-reference",
            "the harness recorded no in-run measurement of the reference composition",
        )
    measured = float(row["val_loss"])
    if not (measured == measured) or measured in (float("inf"), float("-inf")):
        return no(
            "floor-anchor-not-the-bound-reference",
            "the in-run reference measurement is not a finite number",
        )
    if h.bound.get("reference_anchor_state") != "measured-in-run":
        return no(
            "floor-anchor-not-the-bound-reference",
            "the bound anchor state is " + str(h.bound.get("reference_anchor_state"))
            + ", so the span rests on something other than the live measurement",
            measured,
        )
    if h.bound.get("reference_anchor_published") is not None:
        return no(
            "floor-anchor-not-the-bound-reference",
            "a published reference anchor is bound while this family's anchors are unmeasured",
            measured,
        )
    try:
        anchor_budget = int(h.bound["reference_anchor_step_budget"])
        budget = int(h.bound["step_budget_steps"])
    except (KeyError, TypeError, ValueError):
        return no(
            "floor-anchor-not-the-bound-reference",
            "the bound reference anchor is unresolvable from the admin plane",
            measured,
        )
    if budget != anchor_budget:
        return no(
            "floor-anchor-not-the-bound-reference",
            "the reference anchor was established under a budget of " + str(anchor_budget)
            + " steps and the budget in force is " + str(budget),
            measured,
        )
    if row.get("source") != h.bound.get("reference_anchor_source"):
        return no(
            "floor-anchor-not-the-bound-reference",
            "the in-run reference was produced by " + repr(row.get("source"))
            + " and not by the bound path " + repr(h.bound.get("reference_anchor_source")),
            measured,
        )
    if int(row.get("step_budget", -1)) != budget:
        return no(
            "floor-anchor-not-the-bound-reference",
            "the reference composition was trained for " + str(row.get("step_budget"))
            + " steps against the bound " + str(budget),
            measured,
        )
    base = float((h.telemetry.get("baseline") or {}).get("val_loss", 0.0))
    if not (measured < base):
        return no(
            "floor-anchor-not-the-bound-reference",
            "the reference composition measures " + repr(round(measured, 6))
            + " and does not improve on the baseline " + repr(round(base, 6)) + ", so the span is degenerate",
            measured,
        )
    return ok("the floor anchor is the live in-run measurement of the reference composition", measured)


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
    base_overall = float((h.telemetry.get("baseline") or {}).get("val_loss", 0.0))
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
# ABSENCE. No early stop. A halt at a favourable snapshot established nothing.
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
# ABSENCE. The held-out split was not trained on.
# --------------------------------------------------------------------------
def check_eval_split_not_trained_on(h: Harness) -> Outcome:
    row = h.telemetry.get("leakage") or {}
    if "shingles_checked" not in row:
        return no(
            "evaluation-split-leaked-into-training",
            "the harness recorded no comparison of the consumed stream against the held-out split",
        )
    if int(row.get("shingles_checked", 0)) <= 0:
        return no(
            "evaluation-split-leaked-into-training",
            "the leakage comparison examined no shingle, so absence was not established",
        )
    hits = row.get("hits") or []
    if hits:
        return no(
            "evaluation-split-leaked-into-training",
            "a held-out shingle appears in the consumed training stream at token offset " + str(hits[0]),
        )
    return ok(
        "no held-out shingle appears in the consumed training stream",
        float(row.get("shingles_checked", 0)),
    )


# --------------------------------------------------------------------------
# EFFECT. The frozen budget was respected AS FED, under one pass per step.
# --------------------------------------------------------------------------
def check_token_budget_respected_as_fed(h: Harness) -> Outcome:
    budget = int(h.bound["step_budget_steps"])
    train = h.telemetry.get("train") or {}
    override = (h.telemetry.get("composition") or {}).get("consume_steps")
    if override is not None and int(override) != budget:
        return no(
            "token-budget-not-respected-as-fed",
            "the composition plan overrode the budget to " + str(override) + " against the bound " + str(budget),
        )
    if int(train.get("bound_steps", -1)) != budget:
        return no(
            "token-budget-not-respected-as-fed",
            "the harness ran " + str(train.get("bound_steps")) + " steps against the bound " + str(budget),
        )
    declared = (h.substrate or {}).get("run") or {}
    per_step = int(declared.get("batch_tokens_per_step", -1))
    if int(train.get("batch_tokens_per_step", -1)) != per_step:
        return no(
            "token-budget-not-respected-as-fed",
            "the harness fed " + str(train.get("batch_tokens_per_step"))
            + " tokens per step against the declared " + str(per_step),
        )
    if train.get("halted_at_tokens") is None:
        if int(train.get("optimizer_steps", -1)) != budget:
            return no(
                "token-budget-not-respected-as-fed",
                "the optimizer took " + str(train.get("optimizer_steps")) + " steps against the bound " + str(budget),
            )
        if int(train.get("tokens_consumed", -1)) != budget * per_step:
            return no(
                "token-budget-not-respected-as-fed",
                "the run consumed " + str(train.get("tokens_consumed"))
                + " tokens against the bound " + str(budget * per_step),
            )
    if int(train.get("forward_passes_per_step", -1)) != int(declared.get("forward_passes_per_step", -1)):
        return no(
            "token-budget-not-respected-as-fed",
            "the run took " + str(train.get("forward_passes_per_step"))
            + " forward passes per step against the declared " + str(declared.get("forward_passes_per_step")),
        )
    if int(train.get("backward_passes_per_step", -1)) != int(declared.get("backward_passes_per_step", -1)):
        return no(
            "token-budget-not-respected-as-fed",
            "the run took " + str(train.get("backward_passes_per_step"))
            + " backward passes per step against the declared " + str(declared.get("backward_passes_per_step")),
        )
    micro = int(train.get("micro_batches_per_step", 0))
    if micro < 1 or int(train.get("forward_passes_total", -1)) != micro * int(train.get("optimizer_steps", 0)):
        return no(
            "token-budget-not-respected-as-fed",
            "the micro-batch accounting does not reconstruct one logical pass per step: "
            + str(train.get("forward_passes_total")) + " micro forward passes over "
            + str(train.get("optimizer_steps")) + " steps at " + str(micro) + " per step",
        )
    return ok("exactly the bound step budget was fed and consumed", float(budget))


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
    if "index_sha256" in handoff and handoff["index_sha256"] != (train_in.get("shards") or {}).get("sha256"):
        return no(
            "stage-claim-contradicted-by-chain",
            "tokenize claimed index_sha256=" + repr(handoff["index_sha256"])
            + " and the train stage consumed " + repr((train_in.get("shards") or {}).get("sha256")),
        )
    for row in (train_in.get("shard_files") or []):
        if row.get("claimed") != row.get("observed"):
            return no(
                "stage-claim-contradicted-by-chain",
                "the shard index claimed " + str(row.get("name")) + "=" + repr(row.get("claimed"))
                + " and the train stage consumed " + repr(row.get("observed")),
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
    base = float((h.telemetry.get("baseline") or {}).get("val_loss", 0.0))
    floor = float((h.telemetry.get("floor") or {}).get("val_loss", 0.0))
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
    if "nanogpt_substrate.json" not in delivered:
        return no(
            "frozen-axis-modified",
            "the substrate declaration is not among the frozen artifacts, so the architecture is unpinned",
        )
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
    ("holdout_split_is_verifier_owned", check_holdout_split_is_verifier_owned),
    ("eval_split_not_trained_on", check_eval_split_not_trained_on),
    ("full_chain_completed_no_early_stop", check_full_chain_completed_no_early_stop),
    ("evaluation_point_is_bound_not_selected", check_evaluation_point_is_bound_not_selected),
    ("graded_parameters_bind_frozen_architecture", check_graded_parameters_bind_frozen_architecture),
    ("graded_loss_from_harness_weights", check_graded_loss_from_harness_weights),
    ("no_smoothing_on_graded_path", check_no_smoothing_on_graded_path),
    ("floor_anchor_matches_bound_reference", check_floor_anchor_matches_bound_reference),
    ("end_to_end_gain_not_stage_local", check_end_to_end_gain_not_stage_local),
    ("loss_sustained_across_bound_evals", check_loss_sustained_across_bound_evals),
    ("session_state_carried_forward", check_session_state_carried_forward),
)

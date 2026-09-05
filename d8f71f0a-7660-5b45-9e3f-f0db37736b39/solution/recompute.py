# FORGE-CANARY-BEGIN
# 0: a428d728aaea4b77fc3774a6afc7107df542b4d16d82947445fc18e18e278ea5
# 1: d700082cd7b24cb34a46ccf1b970db68cf8900182fe5c3ab6553d7f3fb692a29
# 2: 2a61247ff49827246d2dc8e9a53d56d8a361bcc3826a0df5e404c7f823798951
# 3: 6f763cbaa6aef36cb32b06a76b0973b7b4893a9c79b6294604c734eeeaefd702
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of slot OER-10 from solution/grounding.yaml alone.

This module invokes no model, no network, no clock, no locale and no random source.
Every number it writes is a pure function of the frozen bytes of grounding.yaml, so
running it twice over those bytes produces byte-identical output.

Generated from grounding.yaml, all with the GENERATED SECTION banner:

    environment/probe_pool.json        the frozen probe pool the fingerprint reads
    environment/raw_pool.json          the frozen raw pool a recipe curates
    environment/held_out_split.json    the evaluation split DECLARATION, carrying no payload
    tests/held_out_split.json          the verifier-owned evaluation split pin
    environment/exclusion_set.json     the pinned published-mixture exclusion set
    solution/reference.py              the reference recipe the live checkers accept
    solution/solve.sh                  the oracle entry point
    solution/TRUTH.md                  what is graded and what the reference reaches
    solution/rubrics.json              the solution-against-reference rubric
    tests/fixtures.json                the golden trajectory and every checker fixture
    tests/test_output.py               the compiled per-checker tests

Usage:
    python3 solution/recompute.py            write the artifacts
    python3 solution/recompute.py --check    exit 3 if any committed byte drifted
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

BUNDLE = Path(__file__).resolve().parent.parent
GROUNDING = BUNDLE / "solution" / "grounding.yaml"
BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"

# ---------------------------------------------------------------------------
# Pure substrate derivation
# ---------------------------------------------------------------------------


def _load() -> dict:
    with GROUNDING.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _row(ident: str, bucket: str, h: int) -> dict:
    return {
        "id": ident,
        "bucket": bucket,
        "len_tokens": 128 + (h % 31) * 64,
        "dup_class": (h // 31) % 5,
        "quality_decile": (h // 157) % 10,
        "ppl_decile": (h // 911) % 10,
        "lang": "en" if (h // 7) % 4 else "xx",
    }


def _bind_to_shard(row: dict, index: int, binding: dict) -> dict:
    span = int(binding["span_tokens"])
    per_shard = int(binding["spans_per_shard"])
    bound = dict(row)
    bound["shard"] = str(binding["shard_format"]) % (index // per_shard)
    bound["token_offset"] = (index % per_shard) * span
    bound["span_tokens"] = span
    return bound


def probe_pool(ground: dict) -> list:
    spec = ground["probe_pool_construction"]
    buckets = spec["buckets"]
    rows = []
    for i in range(int(spec["count"])):
        h = (i * 2654435761 + 1013904223) % 1000003
        rows.append(_row("p%02d" % i, buckets[i % len(buckets)], h))
    return rows


def raw_pool(ground: dict) -> list:
    spec = ground["raw_pool_construction"]
    buckets = ground["probe_pool_construction"]["buckets"]
    binding = spec["corpus_binding"]
    rows = []
    for i in range(int(spec["count"])):
        h = (i * 2246822519 + 374761393) % 1000003
        rows.append(_bind_to_shard(_row("r%03d" % i, buckets[i % len(buckets)], h), i, binding))
    return rows


def held_out_ids(ground: dict) -> list:
    spec = ground["held_out_split"]
    return [str(spec["id_format"]) % i for i in range(int(spec["count"]))]


def held_out_declaration(ground: dict) -> dict:
    """The agent-visible surface: the split's identity, and no payload of any kind."""
    spec = ground["held_out_split"]
    return {
        "_banner": BANNER,
        "_source": SOURCE,
        "id": spec["id"],
        "owner": "verifier",
        "member_count": int(spec["count"]),
        "member_tokens": int(spec["segment_tokens"]),
        "shard_glob": spec["shard_glob"],
        "collection": spec["collection"],
        "carries_no_payload": {
            "member_ids": "absent from this surface",
            "token_offsets": "absent from this surface",
            "token_bytes": "absent from this surface",
        },
        "resolved_by": "the verifier, from its own staged FineWeb10B validation shard",
        "pinned_at": "tests/held_out_split.json, which is not mounted on the agent surface",
        "why": (
            "A graded split whose tokens sit in environment/ is a split the solver can read, "
            "and a scalar computed over bytes the solver holds is not a held-out measurement. "
            "This file names the split so you know what you are being graded on, and carries "
            "nothing you could grade yourself against."
        ),
        "feeding_a_member_is_a_leak": (
            "The frozen budget is spent on training shards only. Any fed span naming a "
            "validation shard zeroes the score with reason eval-split-leaked."
        ),
    }


def held_out_pin(ground: dict) -> dict:
    """The verifier-owned surface: what the split IS, and how to prove it did not move."""
    spec = ground["held_out_split"]
    segment = int(spec["segment_tokens"])
    offset = int(spec["token_offset"])
    return {
        "_banner": BANNER,
        "_source": SOURCE,
        "id": spec["id"],
        "owner": "verifier",
        "shard": spec["shard"],
        "shard_glob": spec["shard_glob"],
        "token_offset": offset,
        "token_count": int(spec["token_count"]),
        "segment_tokens": segment,
        "collection": spec["collection"],
        "external_source": spec["external_source"],
        "pipeline": spec["pipeline"],
        "slice_sha256": spec["slice_sha256"],
        "shard_sha256_prefix_1mib": spec["shard_sha256_prefix_1mib"],
        "doc_ids": held_out_ids(ground),
        "segments": [
            {
                "id": ident,
                "shard": spec["shard"],
                "token_offset": offset + i * segment,
                "token_count": segment,
            }
            for i, ident in enumerate(held_out_ids(ground))
        ],
        "payload_is_not_carried": (
            "The token bytes are not a bundle byte on either surface. The verifier resolves "
            "them from its own staged validation shard and checks them against slice_sha256 "
            "before it evaluates, so a substituted split is a refused run rather than a "
            "different grade."
        ),
    }


def quantise(weight: float) -> int:
    step = int(round(float(weight) * 8.0))
    return 0 if step < 0 else (8 if step > 8 else step)


def gated_weight(rule: dict, doc: dict) -> int:
    weight = float(rule["bucket_weights"].get(doc["bucket"], 0.0))
    if doc["quality_decile"] < int(rule["quality_floor"]):
        weight = 0.0
    if doc["dup_class"] > int(rule["dup_max"]):
        weight = 0.0
    if doc["len_tokens"] < int(rule["len_min"]):
        weight = 0.0
    if rule.get("english_only") and doc["lang"] != "en":
        weight = 0.0
    return quantise(weight)


def derived_weight(rule: dict, doc: dict) -> int:
    base = float(rule["bucket_weights"].get(doc["bucket"], 0.0))
    signal = (doc["quality_decile"] - doc["ppl_decile"] + 9) / 18.0
    band = 1.0 - abs(doc["len_tokens"] - float(rule["len_center"])) / float(rule["len_span"])
    band = 0.0 if band < 0.0 else (1.0 if band > 1.0 else band)
    taper = 1.0 - doc["dup_class"] / float(rule["dup_scale"])
    taper = 0.0 if taper < 0.0 else taper
    a, b, c = float(rule["w_signal"]), float(rule["w_band"]), float(rule["w_taper"])
    return quantise(base * (a * signal + b * band + c * taper) / (a + b + c))


def apply_rule(rule: dict, docs: list) -> list:
    form = rule.get("form", "gated_bucket")
    scorer = derived_weight if form == "derived_interaction" else gated_weight
    return [scorer(rule, doc) for doc in docs]


def digest_vector(vector: list) -> str:
    payload = json.dumps(vector, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def l1(left: list, right: list) -> int:
    return sum(abs(a - b) for a, b in zip(left, right))


# ---------------------------------------------------------------------------
# Generated artifacts
# ---------------------------------------------------------------------------


def _json(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def probe_pool_doc(ground: dict) -> str:
    return _json(
        {
            "_banner": BANNER,
            "_source": SOURCE,
            "id": ground["fingerprint"]["probe_pool_id"],
            "quantisation_levels": ground["fingerprint"]["quantisation_levels"],
            "docs": probe_pool(ground),
        }
    )


def raw_pool_doc(ground: dict) -> str:
    return _json({"_banner": BANNER, "_source": SOURCE, "docs": raw_pool(ground)})


def held_out_doc(ground: dict) -> str:
    return _json(held_out_declaration(ground))


def held_out_pin_doc(ground: dict) -> str:
    return _json(held_out_pin(ground))


def exclusion_set(ground: dict) -> dict:
    docs = probe_pool(ground)
    entries = []
    for rule in ground["exclusion_set"]["entries"]:
        vector = apply_rule(rule, docs)
        entries.append(
            {"id": rule["id"], "digest": digest_vector(vector), "selection_vector": vector}
        )
    payload = [[row["id"], row["digest"]] for row in entries]
    return {
        "_banner": BANNER,
        "_source": SOURCE,
        "id": ground["exclusion_set"]["id"],
        "provenance": ground["exclusion_set"]["provenance"],
        "pinned": True,
        "l1_floor": int(ground["fingerprint"]["l1_floor"]),
        "probe_pool_id": ground["fingerprint"]["probe_pool_id"],
        "size": len(entries),
        "digest": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "entries": entries,
    }


def exclusion_set_doc(ground: dict) -> str:
    return _json(exclusion_set(ground))


REFERENCE_TEMPLATE = '''"""{banner}

Source: {source}

The reference curation recipe for slot OER-10. It is a submission, not a grader: the
harness copies this file alone into a fresh directory and executes it over the frozen
probe pool and the frozen raw pool. It reads no grader byte and reports no score.

The recipe is a DERIVED interaction rather than a gated bucket rule. Every pinned
published mixture throws documents away on threshold crossings; this one spends the
frozen token budget on a continuous interaction between document quality, model
surprisal, a length band and a duplication taper, so it retains partial mass exactly
where the pinned baselines retain none.
"""
from __future__ import annotations

BUCKET_WEIGHTS = {bucket_weights}
LEN_CENTER = {len_center}
LEN_SPAN = {len_span}
DUP_SCALE = {dup_scale}
W_SIGNAL = {w_signal}
W_BAND = {w_band}
W_TAPER = {w_taper}

DECLARED_TOKEN_BUDGET = {token_budget}


def weight(doc):
    """The continuous retention weight this recipe assigns to one document."""
    base = BUCKET_WEIGHTS.get(doc["bucket"], 0.0)
    signal = (doc["quality_decile"] - doc["ppl_decile"] + 9) / 18.0
    band = 1.0 - abs(doc["len_tokens"] - LEN_CENTER) / LEN_SPAN
    band = 0.0 if band < 0.0 else (1.0 if band > 1.0 else band)
    taper = 1.0 - doc["dup_class"] / DUP_SCALE
    taper = 0.0 if taper < 0.0 else taper
    blend = (W_SIGNAL * signal + W_BAND * band + W_TAPER * taper) / (W_SIGNAL + W_BAND + W_TAPER)
    return base * blend


def plan():
    """The curation plan, declared. The harness grades tokens as fed, never as declared."""
    return {{
        "recipe_id": "{recipe_id}",
        "declared_token_budget": DECLARED_TOKEN_BUDGET,
        "form": "derived_interaction",
    }}
'''


def reference_source(ground: dict) -> str:
    rule = ground["reference_recipe"]
    return REFERENCE_TEMPLATE.format(
        banner=BANNER,
        source=SOURCE,
        bucket_weights=json.dumps(rule["bucket_weights"], sort_keys=True),
        len_center=float(rule["len_center"]),
        len_span=float(rule["len_span"]),
        dup_scale=float(rule["dup_scale"]),
        w_signal=float(rule["w_signal"]),
        w_band=float(rule["w_band"]),
        w_taper=float(rule["w_taper"]),
        token_budget=int(rule["declared_token_budget"]),
        recipe_id=rule["id"],
    )


SOLVE_TEMPLATE = """#!/usr/bin/env bash
# {banner}
# Source: {source}
#
# Oracle entry point for slot OER-10. It installs the reference curation recipe as the
# submission and stops. It never writes a reward, never reads a checker, and never
# reports a loss: the graded validation loss is the verifier's own number.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
SUBMISSION_DIR="${{OER10_SUBMISSION_DIR:-/workspace/submission}}"

mkdir -p "${{SUBMISSION_DIR}}"
cp "${{BUNDLE_DIR}}/solution/reference.py" "${{SUBMISSION_DIR}}/recipe.py"

printf '%s\\n' "installed reference recipe {recipe_id} at ${{SUBMISSION_DIR}}/recipe.py"
printf '%s\\n' "declared token budget {token_budget}, frozen budget is enforced as fed"
"""


def solve_source(ground: dict) -> str:
    rule = ground["reference_recipe"]
    return SOLVE_TEMPLATE.format(
        banner=BANNER,
        source=SOURCE,
        recipe_id=rule["id"],
        token_budget=int(rule["declared_token_budget"]),
    )


# ---------------------------------------------------------------------------
# Golden trajectory and checker fixtures
# ---------------------------------------------------------------------------


def checkpoint_digest(step: int) -> str:
    return hashlib.sha256(("oer10|checkpoint|" + str(step)).encode()).hexdigest()


def fed_ids(ground: dict) -> list:
    rule = ground["reference_recipe"]
    return [doc["id"] for doc in raw_pool(ground) if derived_weight(rule, doc) >= 1]


def ladder_score(control: float, floor: float, graded: float) -> float:
    span = control - floor
    if span <= 0.0:
        return 0.0
    raw = (control - graded) / span
    return min(max(raw, 0.0), 1.0)


def golden(ground: dict) -> dict:
    docs = probe_pool(ground)
    vector = apply_rule(ground["reference_recipe"], docs)
    trace = ground["golden_trajectory"]
    excl = exclusion_set(ground)
    evaluation = trace["evaluation"]
    samples = [
        {
            "step": int(row["step"]),
            "loss": float(row["loss"]),
            "source": "verifier-recompute",
            "smoothing": "none",
            "weights_owner": "harness",
            "weights_digest": checkpoint_digest(int(row["step"])),
        }
        for row in evaluation["samples"]
    ]
    return {
        "schema": "oer10.telemetry/v1",
        "stages": [dict(row) for row in trace["stages"]],
        "recipe_fingerprint": {
            "probe_pool_id": ground["fingerprint"]["probe_pool_id"],
            "probe_pool_size": len(docs),
            "selection_vector": vector,
            "digest": digest_vector(vector),
        },
        "exclusion_set": {
            "pinned": True,
            "id": excl["id"],
            "digest": excl["digest"],
            "size": excl["size"],
            "l1_floor": excl["l1_floor"],
            "entries": [
                {"id": row["id"], "digest": row["digest"], "selection_vector": row["selection_vector"]}
                for row in excl["entries"]
            ],
        },
        "training": {
            "steps_executed": int(trace["training"]["steps_executed"]),
            "tokens_fed": int(trace["training"]["tokens_fed"]),
            "frozen_token_budget": int(ground["freeze"]["token_budget_tokens"]),
            "halted_at_step": int(trace["training"]["halted_at_step"]),
            "halt_reason": str(trace["training"]["halt_reason"]),
        },
        "corpus": {
            "held_out_split_id": str(ground["held_out_split"]["id"]),
            "held_out_split_owner": "verifier",
            "held_out_split_source": "tests/held_out_split.json",
            "bound_held_out_split_id": str(ground["freeze"]["evaluation_split_id"]),
            "held_out_doc_ids": held_out_ids(ground),
            "fed_doc_ids": fed_ids(ground),
            "fed_shard_glob": str(ground["substrate"]["corpus"]["train_glob"]),
            "held_out_shard_glob": str(ground["held_out_split"]["shard_glob"]),
        },
        "evaluation": {
            "bound_eval_step": int(evaluation["bound_eval_step"]),
            "sustain_points": [int(step) for step in evaluation["sustain_points"]],
            "sustain_tolerance": float(evaluation["sustain_tolerance"]),
            "samples": samples,
            "submission_reported": dict(evaluation["submission_reported"]),
        },
        "checkpoints": [
            {"step": int(row["step"]), "digest": checkpoint_digest(int(row["step"])), "owner": "harness"}
            for row in evaluation["samples"]
        ],
        "ladder": {
            "state": "run-local-measured",
            "control_arm_loss": float(trace["ladder_values"]["control_arm_loss"]),
            "reference_floor_loss": float(trace["ladder_values"]["reference_floor_loss"]),
        },
    }


def _deep(payload):
    return json.loads(json.dumps(payload))


def fixtures(ground: dict) -> dict:
    base = golden(ground)
    docs = probe_pool(ground)
    excl = exclusion_set(ground)
    by_id = {row["id"]: row for row in excl["entries"]}
    para_vector = apply_rule(ground["paraphrase_control"], docs)
    verbatim = by_id[ground["paraphrase_control"]["restates"]]["selection_vector"]

    planted = {}

    row = _deep(base)
    row["recipe_fingerprint"]["selection_vector"] = list(verbatim)
    row["recipe_fingerprint"]["digest"] = digest_vector(verbatim)
    planted["nc-verbatim-published-mixture"] = row

    row = _deep(base)
    row["recipe_fingerprint"]["selection_vector"] = list(para_vector)
    row["recipe_fingerprint"]["digest"] = digest_vector(para_vector)
    planted["nc-paraphrased-published-mixture"] = row

    row = _deep(base)
    row["exclusion_set"]["pinned"] = False
    row["exclusion_set"]["entries"] = []
    row["exclusion_set"]["size"] = 0
    planted["nc-stale-empty-exclusion-set"] = row

    row = _deep(base)
    row["stages"] = [
        {"seq": 1, "stage": "assemble-agent-surface"},
        {"seq": 2, "stage": "pin-exclusion-set"},
        {"seq": 3, "stage": "curate-corpus"},
        {"seq": 4, "stage": "train"},
        {"seq": 5, "stage": "recipe-fingerprint-screen"},
        {"seq": 6, "stage": "verifier-evaluation"},
    ]
    planted["nc-screen-after-training"] = row

    row = _deep(base)
    row["training"]["tokens_fed"] = int(row["training"]["frozen_token_budget"]) + int(
        ground["freeze"]["tokens_per_step"]
    )
    planted["nc-budget-overspend"] = row

    row = _deep(base)
    row["corpus"]["fed_doc_ids"] = list(row["corpus"]["fed_doc_ids"]) + [held_out_ids(ground)[7]]
    planted["nc-split-leak"] = row

    row = _deep(base)
    row["evaluation"]["samples"][0]["smoothing"] = "ema-0.90"
    planted["nc-smoothed-readout"] = row

    row = _deep(base)
    row["evaluation"]["samples"][0]["source"] = "submission-reported"
    planted["nc-submission-reported-loss"] = row

    row = _deep(base)
    row["training"]["halted_at_step"] = int(base["evaluation"]["bound_eval_step"])
    row["training"]["halt_reason"] = "favourable-eval"
    planted["nc-early-stop"] = row

    row = _deep(base)
    row["evaluation"]["samples"][1]["loss"] = base["evaluation"]["samples"][0]["loss"] + 0.19
    row["evaluation"]["samples"][2]["loss"] = base["evaluation"]["samples"][0]["loss"] + 0.24
    planted["nc-unsustained-dip"] = row

    row = _deep(base)
    row["evaluation"]["samples"][0]["weights_digest"] = hashlib.sha256(
        b"oer10|submission-selected-checkpoint"
    ).hexdigest()
    planted["nc-submission-selected-checkpoint"] = row

    row = _deep(base)
    row["evaluation"]["samples"][0]["loss"] = float(
        ground["undisplaced_demonstration"]["graded_loss"]
    )
    planted["nc-control-arm-not-displaced"] = row

    graded = float(base["evaluation"]["samples"][0]["loss"])
    control = float(base["ladder"]["control_arm_loss"])
    floor = float(base["ladder"]["reference_floor_loss"])
    partial_loss = float(ground["partial_credit_demonstration"]["graded_loss"])
    partial = _deep(base)
    partial["evaluation"]["samples"][0]["loss"] = partial_loss
    partial["evaluation"]["samples"][1]["loss"] = partial_loss - 0.004
    partial["evaluation"]["samples"][2]["loss"] = partial_loss - 0.009

    return {
        "_banner": BANNER,
        "_source": SOURCE,
        "reference_sha256": hashlib.sha256(reference_source(ground).encode()).hexdigest(),
        "golden": base,
        "partial_credit": partial,
        "planted": planted,
        "expected": {
            "golden_reward": round(ladder_score(control, floor, graded), 12),
            "partial_reward": round(ladder_score(control, floor, partial_loss), 12),
            "reference_min_l1": min(
                l1(base["recipe_fingerprint"]["selection_vector"], row["selection_vector"])
                for row in excl["entries"]
            ),
            "paraphrase_min_l1": min(
                l1(para_vector, row["selection_vector"]) for row in excl["entries"]
            ),
            "min_pairwise_l1_among_pinned": min(
                l1(a["selection_vector"], b["selection_vector"])
                for i, a in enumerate(excl["entries"])
                for b in excl["entries"][i + 1:]
            ),
            "l1_floor": excl["l1_floor"],
            "fed_doc_count": len(base["corpus"]["fed_doc_ids"]),
            "graded_loss": graded,
            "control_arm_loss": control,
            "undisplaced_graded_loss": float(ground["undisplaced_demonstration"]["graded_loss"]),
            "undisplaced_reward": round(
                ladder_score(control, floor, float(ground["undisplaced_demonstration"]["graded_loss"])), 12
            ),
        },
        "reasons": {
            row["id"]: row["expect_reason"] for row in ground["negative_controls"]
        },
    }


def fixtures_doc(ground: dict) -> str:
    return _json(fixtures(ground))


# ---------------------------------------------------------------------------
# TRUTH.md, rubrics.json, test_output.py
# ---------------------------------------------------------------------------

CHECKERS = (
    ("recipe_not_a_replay", "DIVERGENCE", "recipe-matches-published-mixture"),
    ("fingerprint_screen_precedes_training", "ORDERING", "screen-ran-after-training"),
    ("graded_loss_is_verifier_computed", "VALUE", "graded-loss-not-verifier-computed"),
    ("graded_readout_unsmoothed", "VALUE", "readout-smoothed-on-graded-path"),
    ("loss_sustained_at_scheduled_points", "INVARIANT", "loss-not-sustained"),
    ("early_stop_not_a_result", "ABSENCE", "early-stop-not-a-result"),
    ("evaluated_weights_harness_owned", "VALUE", "weights-not-harness-owned"),
    ("token_budget_respected_as_fed", "EFFECT", "token-budget-overspent"),
    ("eval_split_not_trained_on", "ABSENCE", "eval-split-leaked"),
    ("control_arm_displaced", "EFFECT", "control-arm-not-displaced"),
)

FIXTURE_FOR = {
    "recipe_not_a_replay": "nc-verbatim-published-mixture",
    "fingerprint_screen_precedes_training": "nc-screen-after-training",
    "graded_loss_is_verifier_computed": "nc-submission-reported-loss",
    "graded_readout_unsmoothed": "nc-smoothed-readout",
    "loss_sustained_at_scheduled_points": "nc-unsustained-dip",
    "early_stop_not_a_result": "nc-early-stop",
    "evaluated_weights_harness_owned": "nc-submission-selected-checkpoint",
    "token_budget_respected_as_fed": "nc-budget-overspend",
    "eval_split_not_trained_on": "nc-split-leak",
    "control_arm_displaced": "nc-control-arm-not-displaced",
}


def truth(ground: dict) -> str:
    data = fixtures(ground)
    exp = data["expected"]
    lines = []
    lines.append("<!-- " + BANNER + " Source: " + SOURCE + " -->")
    lines.append("")
    lines.append("# TRUTH, slot OER-10")
    lines.append("")
    lines.append("## What is graded")
    lines.append("")
    lines.append(
        "The validation loss of the trained model, lower is better, computed BY THE VERIFIER "
        "in its own process on the verifier-owned held-out split `"
        + str(ground["freeze"]["evaluation_split_id"])
        + "` from harness-owned parameter snapshots at the bound evaluation point step "
        + str(ground["golden_trajectory"]["evaluation"]["bound_eval_step"])
        + ". It is never a number the submission reported, never a number in the submission's "
        "stdout, never a field the submission wrote, never a smoothed readout, and never a "
        "number the in-container harness printed."
    )
    lines.append("")
    lines.append("## The substrate the metric resolves against")
    lines.append("")
    arch = ground["substrate"]["architecture"]
    run = ground["substrate"]["run"]
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append("| declaration | `environment/nanogpt_substrate.json`, " + str(ground["substrate"]["declaration_version"]) + " |")
    lines.append("| vocab_size | " + str(arch["vocab_size"]) + " |")
    lines.append("| num_layers | " + str(arch["num_layers"]) + " |")
    lines.append("| model_dim | " + str(arch["model_dim"]) + " |")
    lines.append("| head_dim | " + str(arch["head_dim"]) + " |")
    lines.append("| num_heads | " + str(arch["num_heads"]) + " |")
    lines.append("| seq_len | " + str(arch["seq_len"]) + " |")
    lines.append("| tokens per step | " + str(run["batch_tokens_per_step"]) + " |")
    lines.append("| forward and backward passes per step | " + str(run["forward_passes_per_step"]) + " and " + str(run["backward_passes_per_step"]) + " |")
    lines.append("| training corpus | `" + str(ground["substrate"]["corpus"]["train_glob"]) + "` |")
    lines.append("| held-out corpus | `" + str(ground["held_out_split"]["shard_glob"]) + "`, verifier-owned |")
    lines.append("")
    lines.append(
        "`environment/frozen_train.py` instantiates that decoder and trains it over the "
        "FineWeb10B training shards the curation recipe retained, one forward and one "
        "backward pass per step at the frozen batch. Delete either pass and there is no "
        "parameter snapshot to evaluate, so the graded quantity becomes undefined rather "
        "than merely different. `tests/evaluate.py` is what computes the graded scalar, from "
        "those snapshots, on a split whose member ids and token bytes appear nowhere in "
        "`environment/`."
    )
    lines.append("")
    lines.append("## The load-bearing control")
    lines.append("")
    lines.append(
        "A deterministic BEHAVIOURAL recipe fingerprint over the submitted curation plan, "
        "screened against the pinned exclusion set of published mixture baselines BEFORE any "
        "training executes, so a replay costs zero accelerator time."
    )
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append("| probe pool size | " + str(ground["fingerprint"]["probe_pool_size"]) + " |")
    lines.append("| quantisation levels | " + str(ground["fingerprint"]["quantisation_levels"]) + " |")
    lines.append("| pinned exclusion set size | " + str(len(ground["exclusion_set"]["entries"])) + " |")
    lines.append("| pinned L1 floor | " + str(exp["l1_floor"]) + " |")
    lines.append("| smallest L1 between two pinned baselines | " + str(exp["min_pairwise_l1_among_pinned"]) + " |")
    lines.append("| paraphrase control L1 to the baseline it restates | " + str(exp["paraphrase_min_l1"]) + " |")
    lines.append("| reference recipe L1 to its nearest pinned baseline | " + str(exp["reference_min_l1"]) + " |")
    lines.append("")
    lines.append(
        "The paraphrase control shares not one source byte with the baseline it restates and its "
        "fingerprint digest differs, so the exact level of the screen misses it. Its behaviour on "
        "the probe pool sits at L1 "
        + str(exp["paraphrase_min_l1"])
        + ", below the pinned floor of "
        + str(exp["l1_floor"])
        + ", so the behavioural level catches it. Source-level paraphrase does not defeat this screen."
    )
    lines.append("")
    lines.append("## Reference outcome over the golden trajectory")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append("| reference recipe id | " + str(ground["reference_recipe"]["id"]) + " |")
    lines.append("| reference sha256 | `" + data["reference_sha256"] + "` |")
    lines.append("| training spans fed | " + str(exp["fed_doc_count"]) + " of " + str(ground["raw_pool_construction"]["count"]) + " |")
    lines.append("| tokens fed | " + str(ground["golden_trajectory"]["training"]["tokens_fed"]) + " |")
    lines.append("| frozen token budget | " + str(ground["freeze"]["token_budget_tokens"]) + " |")
    lines.append("| control arm loss | " + str(ground["golden_trajectory"]["ladder_values"]["control_arm_loss"]) + " |")
    lines.append("| reference floor loss | " + str(ground["golden_trajectory"]["ladder_values"]["reference_floor_loss"]) + " |")
    lines.append("| graded loss | " + str(ground["golden_trajectory"]["evaluation"]["samples"][0]["loss"]) + " |")
    lines.append("| reward | " + repr(exp["golden_reward"]) + " |")
    lines.append("")
    lines.append(
        "The reward is not binary. The same gate chain over the same fixture with a graded loss of "
        + str(ground["partial_credit_demonstration"]["graded_loss"])
        + " lands at "
        + repr(exp["partial_reward"])
        + ", strictly between 0 and 1."
    )
    lines.append("")
    lines.append("## The loss levels are load-bearing, not decorative")
    lines.append("")
    lines.append(
        "`control_arm_displaced` is the checker whose outcome is a function of the loss LEVELS the "
        "run reached. The graded loss of "
        + repr(exp["graded_loss"])
        + " sits below the control arm of "
        + repr(exp["control_arm_loss"])
        + " the harness measured inside the same run, so the accepting half passes. The planted "
        "fixture `nc-control-arm-not-displaced` carries a graded loss of "
        + repr(exp["undisplaced_graded_loss"])
        + ", above that control arm, and the checker rejects it with `control-arm-not-displaced`. "
        "Its ladder value would be "
        + repr(exp["undisplaced_reward"])
        + " either way, so the gate moves no run's number, only the attribution of that zero."
    )
    lines.append("")
    lines.append(
        "No loss level appears on the agent-visible surface. `instruction.md` names the evaluation "
        "points, the frozen axes and the scoring formula and never a loss, so the series is "
        "established only by training and reading the harness's own evaluation back. A solver that "
        "never interacted with the environment carries no sample series and no measured ladder "
        "ends, and fails this checker with a named reason rather than passing vacuously."
    )
    lines.append("")
    lines.append("## Anchors")
    lines.append("")
    lines.append(
        "`anchors_state: absent`. F12's baseline and target are unmeasured and are NOT invented "
        "here; the gap is `gap-oer-per-family-anchors-unmeasured`. The graded ladder's two ends are "
        "run-local measured quantities under `gap-oer-10-run-local-ladder-is-not-a-family-anchor` "
        "and are not the family anchors."
    )
    lines.append("")
    lines.append("## Checkers")
    lines.append("")
    lines.append("| checker | reduction | zero reason |")
    lines.append("|---|---|---|")
    for ident, reduction, reason in CHECKERS:
        lines.append("| `" + ident + "` | " + reduction + " | `" + reason + "` |")
    lines.append("")
    lines.append("## Not claimed")
    lines.append("")
    lines.append(
        "No H100 training run was performed on this host. The feasibility bundle drives the real "
        "checkers over frozen synthetic telemetry, declared as "
        "`gap-oer-10-graded-training-run-not-executed-on-this-host`. No difficulty claim is made "
        "anywhere in this bundle."
    )
    lines.append("")
    return "\n".join(lines)


def rubrics_json(ground: dict) -> str:
    data = fixtures(ground)
    exp = data["expected"]
    rows = [
        {
            "id": "sol-graded-quantity",
            "criterion": "The graded quantity is the verifier-computed validation loss on the verifier-owned held-out split at the bound evaluation point.",
            "reference": "verifier-recompute sample at step "
            + str(ground["golden_trajectory"]["evaluation"]["bound_eval_step"])
            + ", smoothing none, weights owner harness",
            "weight": 3,
        },
        {
            "id": "sol-recipe-derived",
            "criterion": "The reference recipe is a derived interaction, not any pinned published mixture.",
            "reference": "nearest pinned baseline at L1 "
            + str(exp["reference_min_l1"])
            + " against a pinned floor of "
            + str(exp["l1_floor"]),
            "weight": 3,
        },
        {
            "id": "sol-screen-precedes-training",
            "criterion": "The fingerprint screen occupies a stage sequence number below the training stage.",
            "reference": "screen at stage "
            + str(ground["fingerprint"]["runs_at_stage"])
            + ", training at stage "
            + str(ground["fingerprint"]["training_runs_at_stage"]),
            "weight": 2,
        },
        {
            "id": "sol-budget-as-fed",
            "criterion": "Tokens fed do not exceed the frozen budget, graded as fed and never as declared.",
            "reference": str(ground["golden_trajectory"]["training"]["tokens_fed"])
            + " fed against a frozen budget of "
            + str(ground["freeze"]["token_budget_tokens"]),
            "weight": 2,
        },
        {
            "id": "sol-split-clean",
            "criterion": "No held-out split member appears in the fed corpus, and the split the verifier graded on is the bound split.",
            "reference": str(exp["fed_doc_count"]) + " fed training spans, zero intersection with the "
            + str(ground["freeze"]["evaluation_split_size"]) + " verifier-owned held-out members",
            "weight": 2,
        },
        {
            "id": "sol-reward",
            "criterion": "The reference reaches full reward through the real gate chain.",
            "reference": repr(exp["golden_reward"]),
            "weight": 1,
        },
    ]
    return _json(
        {
            "_banner": BANNER,
            "_source": SOURCE,
            "slot": "OER-10",
            "judged": "solution against its reference answer",
            "reference_sha256": data["reference_sha256"],
            "criteria": rows,
        }
    )


TEST_HEAD = '''"""{banner}

Source: {source}

Compiled per-checker tests for slot OER-10. Each test drives the REAL checker in
tests/checkers.py over the frozen fixtures in tests/fixtures.json: the accepting half on
the golden trajectory, the rejecting half on the planted fixture that fires exactly that
checker's zero reason. The last two tests drive the real gate chain in tests/grade.py.

Run standalone:  python3 tests/test_output.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import grade  # noqa: E402

FIXTURES = json.loads((HERE / "fixtures.json").read_text(encoding="utf-8"))
GOLDEN = checkers.HarnessView.from_mapping(FIXTURES["golden"])


def _planted(name):
    return checkers.HarnessView.from_mapping(FIXTURES["planted"][name])


'''

TEST_CASE = '''def test_{ident}():
    accepted = checkers.check_{ident}(GOLDEN)
    assert accepted.passed, accepted.detail
    rejected = checkers.check_{ident}(_planted("{fixture}"))
    assert not rejected.passed, "planted fixture {fixture} did not fire {ident}"
    assert rejected.reason == "{reason}", rejected.reason


'''

TEST_TAIL = '''def test_reference_reaches_full_reward():
    document = grade.score_document(FIXTURES["golden"])
    assert document["reward"] == FIXTURES["expected"]["golden_reward"], document
    assert document["reward"] == 1.0, document
    assert document["reason"] == "graded-loss-established", document


def test_reward_is_not_binary():
    document = grade.score_document(FIXTURES["partial_credit"])
    expected = FIXTURES["expected"]["partial_reward"]
    assert document["reward"] == expected, document
    assert 0.0 < document["reward"] < 1.0, document


def test_absent_telemetry_scores_zero_with_a_reason():
    document = grade.score_document(None)
    assert document["reward"] == 0.0, document
    assert document["reason"] == "telemetry-absent", document


def main():
    failed = []
    for name, case in sorted(globals().items()):
        if not name.startswith("test_") or not callable(case):
            continue
        try:
            case()
        except AssertionError as problem:
            failed.append(name + ": " + str(problem))
    total = sum(1 for name in globals() if name.startswith("test_"))
    print("collected " + str(total) + " tests, failed " + str(len(failed)))
    for line in failed:
        print("FAIL " + line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def test_output(ground: dict) -> str:
    body = TEST_HEAD.format(banner=BANNER, source=SOURCE)
    for ident, _reduction, reason in CHECKERS:
        body += TEST_CASE.format(ident=ident, fixture=FIXTURE_FOR[ident], reason=reason)
    return body + TEST_TAIL


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

ARTIFACTS = (
    ("environment/probe_pool.json", probe_pool_doc, False),
    ("environment/raw_pool.json", raw_pool_doc, False),
    ("environment/held_out_split.json", held_out_doc, False),
    ("tests/held_out_split.json", held_out_pin_doc, False),
    ("environment/exclusion_set.json", exclusion_set_doc, False),
    ("solution/reference.py", reference_source, False),
    ("solution/solve.sh", solve_source, True),
    ("solution/TRUTH.md", truth, False),
    ("solution/rubrics.json", rubrics_json, False),
    ("tests/fixtures.json", fixtures_doc, False),
    ("tests/test_output.py", test_output, False),
)


def main(argv) -> int:
    ground = _load()
    check_only = "--check" in argv
    drifted = []
    for rel, builder, executable in ARTIFACTS:
        target = BUNDLE / rel
        text = builder(ground)
        if check_only:
            current = target.read_text(encoding="utf-8") if target.is_file() else None
            if current != text:
                drifted.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
        if executable:
            target.chmod(0o755)
    if check_only:
        for rel in drifted:
            print("DRIFTED " + rel)
        print("checked " + str(len(ARTIFACTS)) + " artifacts, drifted " + str(len(drifted)))
        return 3 if drifted else 0
    print("recomputed " + str(len(ARTIFACTS)) + " artifacts from " + SOURCE)
    return 0


# FORGE-SCREENING-CARRIER-BEGIN
# GENERATED SECTION. DO NOT HAND-EDIT.
# Generated by seed/forge/screenfreeze.py. Derives the contamination-screening provenance carrier
# from the frozen `screening` block in solution/grounding.yaml and nothing else. It opens no
# connection, reads no wall clock, consults no host language setting, draws no entropy, starts no
# child process, and imports nothing outside this tree.
import hashlib as _forge_hashlib
import json as _forge_json
import pathlib as _forge_pathlib
import sys as _forge_sys

import yaml as _forge_yaml

_FORGE_CARRIER_KEYS = (
    "schema",
    "unit_uuid",
    "screening_roots",
    "authority_mode",
    "source_identifiers",
    "fork_ancestry_snapshot",
    "base_commit_sha",
    "applicable_dates",
    "instrument_versions",
    "atom_result_digests",
    "applicability",
    "sanitization_closure",
    "empty_submission_result",
    "attestations",
    "binding_block",
    "keyid",
    "trust_root_public_key_hex",
    "namespace",
    "normalization_domain_version",
    "signer_identity",
)

_FORGE_BINDING_KEYS = (
    "canonical_bundle_hash",
    "pinned_image_digest",
    "binding_envelope",
)

_FORGE_SCREENING_KEY = "screening"
_FORGE_GROUNDING = "grounding.yaml"
_FORGE_CARRIER = "provenance.yaml"
_FORGE_BANNER = "# GENERATED SECTION. DO NOT HAND-EDIT."


def _forge_here():
    return _forge_pathlib.Path(__file__).resolve().parent


def _forge_sorted(value):
    """Sort every container so two runs over the same frozen bytes emit identical bytes."""
    if isinstance(value, dict):
        return {key: _forge_sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_forge_sorted(item) for item in value]
    return value


def _forge_frozen_screening():
    """Read the frozen screening block. Absence is refused rather than defaulted."""
    path = _forge_here() / _FORGE_GROUNDING
    with path.open("r", encoding="utf-8") as handle:
        document = _forge_yaml.safe_load(handle)
    block = (document or {}).get(_FORGE_SCREENING_KEY)
    if not isinstance(block, dict):
        raise SystemExit(
            "solution/grounding.yaml carries no frozen `screening` block, so the provenance "
            "carrier cannot be derived. Refusing to emit a carrier over values nobody froze."
        )
    missing = [key for key in _FORGE_CARRIER_KEYS if key not in block]
    unknown = [key for key in sorted(block) if key not in _FORGE_CARRIER_KEYS]
    if missing or unknown:
        raise SystemExit(
            "the frozen `screening` block does not mirror the closed carrier schema: "
            "missing " + repr(missing) + ", unknown " + repr(unknown)
        )
    return block


def _forge_canonical_bytes(payload):
    return _forge_json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _forge_carrier_payload():
    """Assemble the carrier as exactly the closed key set, in the order the schema fixes.

    The binding block is attached AFTER the canonical payload is hashed and never enters the
    preimage, because a payload that contained a hash of itself would have no acyclic ordering.
    """
    block = _forge_frozen_screening()
    payload = {}
    for key in _FORGE_CARRIER_KEYS:
        if key == "binding_block":
            continue
        payload[key] = _forge_sorted(block[key])
    digest = _forge_hashlib.sha256(_forge_canonical_bytes(payload)).hexdigest()

    binding = _forge_sorted(block["binding_block"]) or {}
    shaped = {key: binding.get(key) for key in _FORGE_BINDING_KEYS}
    ordered = {}
    for key in _FORGE_CARRIER_KEYS:
        ordered[key] = shaped if key == "binding_block" else payload[key]
    return ordered, digest


def _forge_carrier_text():
    payload, digest = _forge_carrier_payload()
    header = (
        _FORGE_BANNER + "\n"
        + "# Derived from solution/grounding.yaml `screening` by solution/recompute.py.\n"
        + "# canonical payload sha256 (binding_block excluded from the preimage): " + digest + "\n"
    )
    body = _forge_yaml.safe_dump(
        payload, sort_keys=False, default_flow_style=False, allow_unicode=False, width=100
    )
    return header + body


def _forge_emit_carrier():
    """Write the carrier, or in check mode compare and report drift. Never both."""
    argv = list(_forge_sys.argv[1:])
    check = "--check" in argv
    path = _forge_here() / _FORGE_CARRIER
    text = _forge_carrier_text()
    if check:
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        if current == text:
            return 0
        _forge_sys.stderr.write(
            "drift: " + _FORGE_CARRIER + " does not match the carrier derived from the frozen "
            "`screening` block in " + _FORGE_GROUNDING + "\n"
        )
        return 1
    path.write_text(text, encoding="utf-8")
    return 0


_FORGE_INNER_MAIN = main


def main(*args, **kwargs):
    """Run the host generator, then derive the provenance carrier from the frozen block."""
    status = _FORGE_INNER_MAIN(*args, **kwargs)
    drift = _forge_emit_carrier()
    if drift and not status:
        return drift
    return status

# FORGE-SCREENING-CARRIER-END


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

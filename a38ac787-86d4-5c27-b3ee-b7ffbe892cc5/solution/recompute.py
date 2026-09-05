# FORGE-CANARY-BEGIN
# 0: 8c305e69960b138f845eaf602314ce9b5165b1e906f937c10adbb72bc23aa9d2
# 1: b3fb018c3f2751e8c16f739086892f182643f65d2fa114eacbd0c1d15f8d9f78
# 2: 302dabda6949724619cacdc8d331cf1c9a1aaef8fd4291247740cfdc01fee8ad
# 3: b6c40a3fbd6822d56cf9f3f454c020768bfb06ce101ac6188c13278365715e6c
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of OER-16 from `solution/grounding.yaml` alone.

What this derives, and nothing else derives:

    environment/frozen/train_corpus.txt
    environment/frozen/eval_corpus.txt
    environment/frozen/CORPUS_PROVENANCE.md
    tests/config.json
    tests/test_output.py
    solution/fixtures/*.json
    solution/fixtures/negatives/*.py
    solution/golden_trajectory.json
    solution/solve.sh
    solution/TRUTH.md
    solution/rubrics.json

This module invokes NO model, NO network, NO clock, NO locale and NO random
source. It reads `solution/grounding.yaml`, the three hand-authored frozen
configs it asserts against that file, and `solution/reference.py` for the sole
purpose of binding its sha256. Every number below is arrived at by integer and
float arithmetic in this process, in a fixed order, so two runs over frozen
bytes produce byte-identical output. `--check` recomputes without writing and
exits non-zero on the first artifact whose committed bytes drifted.

The "model" this file builds is a bigram count table over the frozen corpus,
which is the frozen substrate of this slot, not a learned artifact: no weights
are loaded, no framework is imported, nothing is sampled. It is a mirror of the
recipe `tests/harness.py` runs, and `seed/tasks/OER-16/adequacy.py` proves the
mirror faithful by asserting the LIVE harness reproduces the figure generated
here to the last bit rather than taking this file's word for it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"

SOLUTION = Path(__file__).resolve().parent
BUNDLE = SOLUTION.parent

TOKENIZER_SCHEMA = "oer16.tokenizer/v1"
CLAIM_SCHEMA = "oer16.claim/v1"
RECORD_SCHEMA = "oer16.run_record/v1"
BOS = -1

# The graded set, in the order tests/checkers.py declares it. The order is part
# of the record: the aggregated reason is the FIRST failing checker's reason.
CHECKER_ORDER = (
    "measured_bits_per_byte_is_harness_measured",
    "graded_readout_is_unsmoothed",
    "denominator_is_frozen_corpus_bytes",
    "claim_matches_measurement",
    "reading_sustained_across_scheduled_points",
    "early_stop_establishes_no_metric",
    "claim_absence_is_recorded",
    "compute_budget_respected_as_spent",
    "claim_sealed_before_measurement",
    "frozen_axes_and_vocab_ceiling_hold",
)


# ---------------------------------------------------------------------------
# Canonical serialisation. Fixed key order, fixed separators, ASCII only.
# ---------------------------------------------------------------------------

def canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(payload) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def document(payload) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


# ---------------------------------------------------------------------------
# The frozen corpus, built by the rule grounding.yaml states and no other
# ---------------------------------------------------------------------------

def build_corpus(words, phase: int, length_bytes: int, column_width: int) -> bytes:
    """Cycle the word list from `phase`, wrap at `column_width`, cut to length.

    Nothing is sampled and no seed is consulted, so the corpus is a pure
    function of the word list, the phase, the width and the length.
    """
    pieces: list = []
    produced = 0
    column = 0
    index = phase
    # Overshoot by one full line, then cut. The cut is what fixes the length.
    while produced < length_bytes + column_width + 2:
        word = words[index % len(words)]
        index += 1
        if column and column + 1 + len(word) > column_width:
            pieces.append("\n")
            produced += 1
            column = 0
        elif column:
            pieces.append(" ")
            produced += 1
            column += 1
        pieces.append(word)
        produced += len(word)
        column += len(word)
    raw = "".join(pieces).encode("ascii")
    return raw[: length_bytes - 1] + b"\n"


# ---------------------------------------------------------------------------
# The tokenizers. Byte level, and the reference byte-pair encoding.
# ---------------------------------------------------------------------------

def byte_level_tokens() -> list:
    return [bytes([i]) for i in range(256)]


def bpe_tokens(train_corpus: bytes, merge_budget: int, max_token_bytes: int,
               max_vocab_size: int) -> list:
    """Byte-pair encoding, deterministic tie-break, identical to reference.py."""
    tokens = [bytes([i]) for i in range(256)]
    token_bytes = {i: bytes([i]) for i in range(256)}
    ids = list(train_corpus)
    nxt = 256
    for _ in range(merge_budget):
        if len(tokens) >= max_vocab_size:
            break
        pairs: dict = {}
        for left, right in zip(ids, ids[1:]):
            key = (left, right)
            pairs[key] = pairs.get(key, 0) + 1
        if not pairs:
            break
        best_pair, best_count = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if best_count < 2:
            break
        merged = token_bytes[best_pair[0]] + token_bytes[best_pair[1]]
        if len(merged) > max_token_bytes:
            break
        token_bytes[nxt] = merged
        tokens.append(merged)
        rebuilt, position = [], 0
        while position < len(ids):
            if (
                position + 1 < len(ids)
                and ids[position] == best_pair[0]
                and ids[position + 1] == best_pair[1]
            ):
                rebuilt.append(nxt)
                position += 2
            else:
                rebuilt.append(ids[position])
                position += 1
        ids = rebuilt
        nxt += 1
    return tokens


def index_of(tokens: list) -> tuple:
    index: dict = {}
    for ident, token in enumerate(tokens):
        index.setdefault(token, ident)
    return index, max(len(t) for t in tokens)


def encode(data: bytes, index: dict, longest: int) -> list:
    out, position, size = [], 0, len(data)
    while position < size:
        span = min(longest, size - position)
        while span > 1 and data[position:position + span] not in index:
            span -= 1
        out.append(index[data[position:position + span]])
        position += span
    return out


# ---------------------------------------------------------------------------
# The frozen recipe, mirrored. Arithmetic only; adequacy proves it faithful.
# ---------------------------------------------------------------------------

def spend(stream: list, budget_token_updates: int) -> tuple:
    counts: dict = {}
    totals: dict = {}
    updates = 0
    if not stream:
        return counts, totals, updates
    previous = BOS
    for step in range(budget_token_updates):
        current = stream[step % len(stream)]
        key = (previous, current)
        counts[key] = counts.get(key, 0) + 1
        totals[previous] = totals.get(previous, 0) + 1
        updates += 1
        previous = current if (step + 1) % len(stream) else BOS
    return counts, totals, updates


def fingerprint(counts: dict, vocab_size: int, smoothing: float, updates: int) -> str:
    rows = sorted([previous, current, value] for (previous, current), value in counts.items())
    return digest([vocab_size, smoothing, updates, rows])


def bits_per_byte(counts: dict, totals: dict, vocab_size: int, smoothing: float,
                  data: bytes, index: dict, longest: int, denominator: int) -> float:
    total = 0.0
    previous = BOS
    for current in encode(data, index, longest):
        numerator = counts.get((previous, current), 0) + smoothing
        divisor = totals.get(previous, 0) + smoothing * vocab_size
        total -= math.log2(numerator / divisor)
        previous = current
    return total / denominator


def segments(data: bytes, count: int) -> list:
    if count < 1:
        return [data]
    width = len(data) // count
    return [data[i * width:(i + 1) * width if i < count - 1 else len(data)] for i in range(count)]


# ---------------------------------------------------------------------------
# The telemetry record, shaped exactly as tests/harness.py produces one
# ---------------------------------------------------------------------------

def frozen_fingerprints(model_cfg, optimizer_cfg, budget_cfg, eval_bytes, train_bytes) -> dict:
    return {
        "model": digest(model_cfg),
        "optimizer": digest(optimizer_cfg),
        "budget": digest(budget_cfg),
        "eval_corpus": hashlib.sha256(eval_bytes).hexdigest(),
        "train_corpus": hashlib.sha256(train_bytes).hexdigest(),
    }


def run_record(tokens, claims: bool, frozen: dict) -> dict:
    """One clean run record. The same shape harness.measure() returns."""
    budget_cfg = frozen["budget"]
    optimizer_cfg = frozen["optimizer"]
    eval_bytes = frozen["eval_corpus"]
    train_bytes = frozen["train_corpus"]
    axes = frozen_fingerprints(
        frozen["model"], optimizer_cfg, budget_cfg, eval_bytes, train_bytes
    )

    index, longest = index_of(tokens)
    longest = min(longest, int(budget_cfg["max_token_bytes"]))
    smoothing = float(optimizer_cfg["lambda"])
    size = len(tokens)
    denominator = len(eval_bytes)

    counts, totals, updates = spend(
        encode(train_bytes, index, longest), int(budget_cfg["budget_token_updates"])
    )
    graded = bits_per_byte(
        counts, totals, size, smoothing, eval_bytes, index, longest, denominator
    )
    points = [
        bits_per_byte(counts, totals, size, smoothing, chunk, index, longest, len(chunk))
        for chunk in segments(eval_bytes, int(budget_cfg["evaluation_points"]))
    ]
    stamp = fingerprint(counts, size, smoothing, updates)

    return {
        "schema": RECORD_SCHEMA,
        "sequence": {"claim_recorded_seq": 1, "measurement_computed_seq": 5},
        "claim": {
            "present": bool(claims),
            "claimed_bits_per_byte": graded if claims else None,
            "absence_recorded": not claims,
            "graded_on": "measurement",
        },
        "vocabulary": {
            "loaded_by": "verifier-harness",
            "source": "submission-tokenizer-json-as-data",
            "size": size,
            "max_token_bytes": max(len(t) for t in tokens),
        },
        "model_state": {
            "owner": "verifier-harness",
            "produced_by": "harness-owned-model-state",
            "fingerprint_at_measurement": stamp,
            "harness_recomputed_fingerprint": stamp,
        },
        "compute": {
            "budget_token_updates": int(budget_cfg["budget_token_updates"]),
            "spent_token_updates": updates,
            "counter_source": "harness-counter",
        },
        "corpus": {
            "eval_corpus_bytes_measured": denominator,
            "denominator_bytes": denominator,
            "denominator_source": "harness-measured-frozen-eval-corpus",
        },
        "readout": {
            "pipeline": "raw",
            "smoothing_applied": False,
            "graded_bits_per_byte": graded,
            "raw_bits_per_byte": graded,
        },
        "schedule": {
            "points_scheduled": int(budget_cfg["evaluation_points"]),
            "points_completed": len(points),
            "halted_early": False,
            "point_bits_per_byte": points,
        },
        "frozen_axes": {"expected": axes, "opening": axes, "closing": dict(axes)},
        "vocabulary_refusal": "",
    }


# ---------------------------------------------------------------------------
# Patch resolution: "@expr" against the clean record it is applied to
# ---------------------------------------------------------------------------

EXPR = re.compile(r"^([a-z_]+)(?:([+\-*])([0-9]*\.?[0-9]+))?$")


def resolve(token, context: dict):
    if isinstance(token, list):
        return [resolve(item, context) for item in token]
    if not isinstance(token, str) or not token.startswith("@"):
        return token
    body = token[1:]
    if body == "zerodigest":
        return "0" * 64
    match = EXPR.match(body)
    if match is None or match.group(1) not in context:
        raise ValueError("unresolvable fixture expression: " + repr(token))
    base = context[match.group(1)]
    if match.group(2) is None:
        return base
    operand = float(match.group(3))
    if match.group(2) == "+":
        return base + operand
    if match.group(2) == "-":
        return base - operand
    product = base * operand
    return int(product) if isinstance(base, int) else product


def apply_patch(record: dict, patch: dict, context: dict) -> dict:
    out = copy.deepcopy(record)
    for dotted in sorted(patch):
        cursor = out
        parts = dotted.split(".")
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[parts[-1]] = resolve(patch[dotted], context)
    return out


def context_of(record: dict) -> dict:
    return {
        "graded": record["readout"]["graded_bits_per_byte"],
        "budget": record["compute"]["budget_token_updates"],
        "denominator": record["corpus"]["denominator_bytes"],
    }


# ---------------------------------------------------------------------------
# Grading, mirrored just enough to CHOOSE a fixture base deterministically
# ---------------------------------------------------------------------------

def _finite(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not (math.isnan(value) or math.isinf(value))


def fires(record: dict, config: dict, selector: str) -> bool:
    """True when `selector` would refuse `record` under `config`.

    A deliberately small mirror of the ten predicates, used for ONE purpose: to
    pick, deterministically, which clean record a planted patch is applied to. A
    patch that sets `claim.absence_recorded: false` describes a defect only on a
    record that claims nothing, because on a claiming record that value is the
    correct one. Trying the claiming base first and falling back to the silent
    one is a total rule, so no base is chosen by hand.
    """
    get = lambda *path: _dig(record, path)  # noqa: E731
    if selector == "measured_bits_per_byte_is_harness_measured":
        return not (
            get("model_state", "owner") == "verifier-harness"
            and get("model_state", "produced_by") == "harness-owned-model-state"
            and get("model_state", "fingerprint_at_measurement")
            == get("model_state", "harness_recomputed_fingerprint")
            and bool(get("model_state", "fingerprint_at_measurement"))
            and get("vocabulary", "loaded_by") == "verifier-harness"
            and get("vocabulary", "source") == "submission-tokenizer-json-as-data"
            and _finite(get("readout", "graded_bits_per_byte"))
        )
    if selector == "graded_readout_is_unsmoothed":
        return not (
            get("readout", "pipeline") == "raw"
            and get("readout", "smoothing_applied") is False
            and _finite(get("readout", "graded_bits_per_byte"))
            and get("readout", "graded_bits_per_byte") == get("readout", "raw_bits_per_byte")
        )
    if selector == "denominator_is_frozen_corpus_bytes":
        bound = config.get("denominator_frozen_bytes")
        return not (
            get("corpus", "denominator_source") == "harness-measured-frozen-eval-corpus"
            and isinstance(bound, int)
            and get("corpus", "denominator_bytes") == get("corpus", "eval_corpus_bytes_measured")
            and get("corpus", "denominator_bytes") == bound
        )
    if selector == "claim_matches_measurement":
        tolerance = config.get("claim_tolerance_bpb")
        if not _finite(tolerance) or tolerance < 0:
            return True
        if get("claim", "present") is not True:
            return False
        claimed = get("claim", "claimed_bits_per_byte")
        measured = get("readout", "graded_bits_per_byte")
        if not _finite(claimed) or not _finite(measured):
            return True
        return abs(float(claimed) - float(measured)) > float(tolerance)
    if selector == "reading_sustained_across_scheduled_points":
        required = config.get("sustain_points_required")
        band = config.get("sustain_band_bits_per_byte")
        graded = get("readout", "graded_bits_per_byte")
        points = get("schedule", "point_bits_per_byte")
        if not isinstance(required, int) or required < 1 or not _finite(band) or band < 0:
            return True
        if not _finite(graded) or not isinstance(points, list) or not points:
            return True
        within = [p for p in points if _finite(p) and abs(float(p) - float(graded)) <= float(band)]
        return len(within) < required
    if selector == "early_stop_establishes_no_metric":
        return not (
            get("schedule", "halted_early") is False
            and get("schedule", "points_scheduled") == config.get("evaluation_points")
            and get("schedule", "points_completed") == get("schedule", "points_scheduled")
        )
    if selector == "claim_absence_is_recorded":
        if get("claim", "graded_on") != "measurement":
            return True
        if get("claim", "present") is True:
            return not (
                get("claim", "absence_recorded") is False
                and _finite(get("claim", "claimed_bits_per_byte"))
            )
        return not (
            get("claim", "absence_recorded") is True
            and get("claim", "claimed_bits_per_byte") is None
        )
    if selector == "compute_budget_respected_as_spent":
        bound = config.get("compute_budget_token_updates")
        spent = get("compute", "spent_token_updates")
        return not (
            get("compute", "counter_source") == "harness-counter"
            and isinstance(bound, int)
            and bound > 0
            and get("compute", "budget_token_updates") == bound
            and isinstance(spent, int)
            and 0 < spent <= bound
        )
    if selector == "claim_sealed_before_measurement":
        claim_seq = _dig(record, ("sequence", "claim_recorded_seq"))
        measure_seq = _dig(record, ("sequence", "measurement_computed_seq"))
        return not (
            isinstance(claim_seq, int)
            and isinstance(measure_seq, int)
            and claim_seq >= 1
            and claim_seq < measure_seq
        )
    if selector == "frozen_axes_and_vocab_ceiling_hold":
        axes = record.get("frozen_axes") or {}
        ceiling = config.get("max_vocab_size")
        size = _dig(record, ("vocabulary", "size"))
        return not (
            isinstance(axes.get("expected"), dict)
            and axes.get("expected")
            and axes.get("opening") == axes.get("expected")
            and axes.get("closing") == axes.get("expected")
            and isinstance(ceiling, int)
            and ceiling >= 256
            and isinstance(size, int)
            and 256 <= size <= ceiling
        )
    raise ValueError("unknown selector: " + selector)


def _dig(mapping, path):
    cursor = mapping
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_grounding() -> dict:
    with (SOLUTION / "grounding.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def assert_frozen_matches(grounding: dict, model_cfg, optimizer_cfg, budget_cfg) -> None:
    """The delivered frozen bytes and the derivation source must agree."""
    declared = grounding["frozen_configuration"]
    pairs = [
        ("model_kind", model_cfg["kind"]),
        ("context_order", model_cfg["context_order"]),
        ("optimizer_kind", optimizer_cfg["kind"]),
        ("optimizer_lambda", optimizer_cfg["lambda"]),
        ("budget_token_updates", budget_cfg["budget_token_updates"]),
        ("max_vocab_size", budget_cfg["max_vocab_size"]),
        ("max_token_bytes", budget_cfg["max_token_bytes"]),
        ("evaluation_points", budget_cfg["evaluation_points"]),
    ]
    for key, delivered in pairs:
        if declared[key] != delivered:
            raise SystemExit(
                "frozen drift: grounding.frozen_configuration." + key + " is "
                + repr(declared[key]) + " and environment/frozen/ carries " + repr(delivered)
            )


def generate() -> dict:
    """Every generated artifact, as bundle-relative path to bytes."""
    grounding = load_grounding()
    out: dict = {}

    frozen_dir = BUNDLE / "environment" / "frozen"
    model_cfg = json.loads((frozen_dir / "model_config.json").read_text(encoding="utf-8"))
    optimizer_cfg = json.loads((frozen_dir / "optimizer_config.json").read_text(encoding="utf-8"))
    budget_cfg = json.loads((frozen_dir / "compute_budget.json").read_text(encoding="utf-8"))
    assert_frozen_matches(grounding, model_cfg, optimizer_cfg, budget_cfg)

    corpus_spec = grounding["corpus"]
    words = list(corpus_spec["words"])
    width = int(corpus_spec["column_width"])
    train_bytes = build_corpus(
        words, int(corpus_spec["train"]["phase"]), int(corpus_spec["train"]["length_bytes"]), width
    )
    eval_bytes = build_corpus(
        words, int(corpus_spec["eval"]["phase"]), int(corpus_spec["eval"]["length_bytes"]), width
    )
    out["environment/frozen/train_corpus.txt"] = train_bytes
    out["environment/frozen/eval_corpus.txt"] = eval_bytes

    frozen = {
        "model": model_cfg,
        "optimizer": optimizer_cfg,
        "budget": budget_cfg,
        "eval_corpus": eval_bytes,
        "train_corpus": train_bytes,
    }

    reference_tokens = bpe_tokens(
        train_bytes,
        int(grounding["reference"]["merge_budget"]),
        int(budget_cfg["max_token_bytes"]),
        int(budget_cfg["max_vocab_size"]),
    )
    baseline_tokens = byte_level_tokens()

    clean = {
        "clean-reference": run_record(reference_tokens, True, frozen),
        "clean-baseline": run_record(baseline_tokens, False, frozen),
    }
    reference_record = clean["clean-reference"]
    baseline_record = clean["clean-baseline"]
    graded = reference_record["readout"]["graded_bits_per_byte"]
    baseline_graded = baseline_record["readout"]["graded_bits_per_byte"]

    config = {
        "_banner": BANNER,
        "_source": SOURCE,
        "slot": grounding["slot"],
        "anchors": {
            "anchors_state": grounding["anchors"]["anchors_state"],
            "baseline_metric": grounding["anchors"]["baseline_metric"],
            "target_metric": grounding["anchors"]["target_metric"],
            "gap": grounding["anchors"]["gap"],
        },
    }
    config.update({k: v for k, v in grounding["grading_configuration"].items()})
    superseded = dict(grounding["grading_configuration_superseded"])
    out["tests/config.json"] = document(config)

    # ---- corpus provenance -------------------------------------------------
    out["environment/frozen/CORPUS_PROVENANCE.md"] = (
        "# " + BANNER + "\n\n"
        "Source: `" + SOURCE + "`, generated by `solution/recompute.py`.\n\n"
        "`train_corpus.txt` and `eval_corpus.txt` carry NO in-band banner, because\n"
        "their byte length is a bound quantity: the evaluation corpus length in bytes\n"
        "is the graded denominator, and a banner inside the file would move it. The\n"
        "banner for those two files is this document.\n\n"
        "Construction, from `grounding.yaml` `corpus` and nothing else: cycle the\n"
        "fixed word list from the recorded phase, join with single spaces, wrap at\n"
        "column " + str(width) + " by inserting a newline whenever the next word\n"
        "would cross it, cut to the exact byte length, terminate with a newline.\n"
        "Nothing is sampled and no seed is consulted.\n\n"
        "| file | phase | bytes | sha256 |\n"
        "|---|---|---|---|\n"
        "| train_corpus.txt | " + str(corpus_spec["train"]["phase"]) + " | "
        + str(len(train_bytes)) + " | " + hashlib.sha256(train_bytes).hexdigest() + " |\n"
        "| eval_corpus.txt | " + str(corpus_spec["eval"]["phase"]) + " | "
        + str(len(eval_bytes)) + " | " + hashlib.sha256(eval_bytes).hexdigest() + " |\n\n"
        + str(corpus_spec["scaled_instantiation_note"]).strip() + "\n"
    )

    # ---- fixtures ----------------------------------------------------------
    for ident, record in sorted(clean.items()):
        out["solution/fixtures/" + ident + ".json"] = document(
            {"_banner": BANNER, "_source": SOURCE, "id": ident,
             "expect_all_pass": True, "record": record}
        )

    planted = {}
    for row in grounding["fixtures"]["planted"]:
        chosen = None
        for base_id in ("clean-reference", "clean-baseline"):
            candidate = apply_patch(
                clean[base_id], row["patch"], context_of(clean[base_id])
            )
            if fires(candidate, config, row["fires"]):
                chosen = (base_id, candidate)
                break
        if chosen is None:
            raise SystemExit(
                "planted fixture " + row["id"] + " fires no checker on either clean base"
            )
        base_id, candidate = chosen
        collateral = [
            name for name in CHECKER_ORDER
            if name != row["fires"] and fires(candidate, config, name)
        ]
        payload = {
            "_banner": BANNER,
            "_source": SOURCE,
            "id": row["id"],
            "base": base_id,
            "fires": row["fires"],
            "reason": row["reason"],
            "emitted_reason": row["reason"],
            "collateral_failures": collateral,
            "record": candidate,
        }
        planted[row["id"]] = payload
        out["solution/fixtures/" + row["id"] + ".json"] = document(payload)

    stale = {}
    for row in grounding["fixtures"]["stale_controls"]:
        candidate = apply_patch(
            clean["clean-reference"], row["patch"], context_of(clean["clean-reference"])
        )
        payload = {
            "_banner": BANNER,
            "_source": SOURCE,
            "id": row["id"],
            "checker": row["checker"],
            "moved_key": row["moved_key"],
            "mutation": row["mutation"],
            "reason": row["reason"],
            "passes_under": "superseded",
            "fails_under": "final",
            "record": candidate,
        }
        stale[row["id"]] = payload
        out["solution/fixtures/" + row["id"] + ".json"] = document(payload)

    for row in grounding["negative_submissions"]:
        out["solution/fixtures/negatives/" + row["id"] + ".py"] = (
            "# " + BANNER + "\n"
            "# Source: " + SOURCE + " negative_submissions." + row["id"] + "\n"
            "# " + " ".join(str(row["description"]).split()) + "\n"
            + row["body"]
        )

    out["solution/fixtures/reference_binding.json"] = document(
        {
            "_banner": BANNER,
            "_source": SOURCE,
            "reference": "solution/reference.py",
            "sha256": hashlib.sha256((SOLUTION / "reference.py").read_bytes()).hexdigest(),
            "note": "The accepting half is bound to THESE reference bytes, so a "
                    "checker that accepts some other file does not read as accepting "
                    "this one.",
        }
    )

    out["solution/fixtures/statement_readings.json"] = document(
        {
            "_banner": BANNER,
            "_source": SOURCE,
            "keys": list(grounding["statement_readings"]["keys"]),
            "expected": dict(grounding["statement_readings"]["expected"]),
            "note": "instruction.md states the graded quantity twice, as prose and as "
                    "a formula. Both blocks must reduce to this one mapping, so there "
                    "is exactly one graded outcome.",
        }
    )

    # ---- golden trajectory -------------------------------------------------
    out["solution/golden_trajectory.json"] = document(
        {
            "_banner": BANNER,
            "_source": SOURCE,
            "slot": grounding["slot"],
            "steps": list(grounding["golden_trajectory"]),
            "measured": {
                "reference_bits_per_byte": graded,
                "baseline_bits_per_byte": baseline_graded,
                "reference_vocabulary_size": len(reference_tokens),
                "baseline_vocabulary_size": len(baseline_tokens),
                "spent_token_updates": reference_record["compute"]["spent_token_updates"],
                "denominator_bytes": reference_record["corpus"]["denominator_bytes"],
                "point_bits_per_byte": reference_record["schedule"]["point_bits_per_byte"],
            },
        }
    )

    # ---- rubrics.json ------------------------------------------------------
    measured_answers = {
        "sr-vocabulary-valid": "a vocabulary of " + str(len(reference_tokens))
        + " entries, none longer than " + str(budget_cfg["max_token_bytes"])
        + " bytes, containing every single byte, under a ceiling of "
        + str(budget_cfg["max_vocab_size"]) + ".",
        "sr-figure-is-measured-not-asserted": "score.json metric.graded_bits_per_byte is "
        + repr(graded) + ", the harness reading; metric.claimed_bits_per_byte sits beside "
        "it as input only.",
        "sr-claim-truthful-or-absent": "the reference claims " + repr(graded)
        + " and the harness measures " + repr(graded)
        + ", a divergence of 0.0 against a tolerance of "
        + repr(config["claim_tolerance_bpb"]) + ".",
        "sr-denominator-frozen": str(len(eval_bytes))
        + " bytes, measured by the harness, equal to the bound frozen denominator.",
        "sr-budget-respected": str(reference_record["compute"]["spent_token_updates"])
        + " updates spent against a bound "
        + str(config["compute_budget_token_updates"]) + ".",
        "sr-schedule-completed": str(reference_record["schedule"]["points_completed"])
        + " of " + str(config["evaluation_points"])
        + " points completed, all within the bound sustain band of "
        + repr(config["sustain_band_bits_per_byte"]) + " around the graded reading.",
    }
    out["solution/rubrics.json"] = document(
        {
            "_banner": BANNER,
            "_source": SOURCE,
            "slot": grounding["slot"],
            "judged": "the solution against its reference answer",
            "not_a_substitute_for": "tests/rubrics.jsonl, which is judged against the "
                                    "trajectory and grades agent behaviour",
            "rubrics": [
                {
                    "id": row["id"],
                    "criterion": " ".join(str(row["criterion"]).split()),
                    "reference_answer": " ".join(str(row["reference_answer"]).split()),
                    "measured_reference_answer": measured_answers[row["id"]],
                }
                for row in grounding["solution_rubrics"]
            ],
        }
    )

    # ---- solve.sh ----------------------------------------------------------
    out["solution/solve.sh"] = (
        "#!/usr/bin/env bash\n"
        "# " + BANNER + "\n"
        "# Source: " + SOURCE + ", generated by solution/recompute.py.\n"
        "#\n"
        "# The oracle entry point. It installs the reference solution at the bound\n"
        "# submission path and exercises it once against the frozen half, so the\n"
        "# artifacts the verifier reads as data exist before grading starts.\n"
        "set -euo pipefail\n"
        "\n"
        'here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'bundle="$(cd "${here}/.." && pwd)"\n'
        'frozen="${OER16_FROZEN_DIR:-${bundle}/environment/frozen}"\n'
        'submission="${OER16_SUBMISSION:-/app/submission.py}"\n'
        'workdir="${OER16_SOLVE_OUT:-${TMPDIR:-/tmp}/oer16-solve}"\n'
        "\n"
        'mkdir -p "$(dirname "${submission}")" "${workdir}"\n'
        'cp "${here}/reference.py" "${submission}"\n'
        "\n"
        "# Run it exactly as the verifier runs a submission: alone, isolated\n"
        "# interpreter, two positional arguments, artifacts written as data.\n"
        'python3 -I -S "${submission}" "${frozen}" "${workdir}"\n'
        "\n"
        'echo "installed ${submission}; artifacts in ${workdir}"\n'
    )

    # ---- TRUTH.md ----------------------------------------------------------
    out["solution/TRUTH.md"] = truth_document(
        grounding, config, reference_record, baseline_record,
        reference_tokens, baseline_tokens, planted, stale
    )

    # ---- tests/test_output.py ---------------------------------------------
    out["tests/test_output.py"] = test_output_module(config, superseded, clean, planted, stale)

    return out


def truth_document(grounding, config, reference_record, baseline_record,
                   reference_tokens, baseline_tokens, planted, stale) -> str:
    graded = reference_record["readout"]["graded_bits_per_byte"]
    baseline_graded = baseline_record["readout"]["graded_bits_per_byte"]
    lines = [
        "# " + BANNER,
        "",
        "Source: `" + SOURCE + "`, generated by `solution/recompute.py`.",
        "",
        "# OER-16 — what is true, and how it was derived",
        "",
        "## The graded quantity",
        "",
        "Bits per byte at fixed compute, lower is better. The numerator is the total",
        "cross-entropy in bits the VERIFIER's own model spends over the frozen",
        "evaluation corpus. The denominator is that corpus's length in BYTES as the",
        "verifier itself measured it. The model state is the state the verifier",
        "produced by spending the frozen budget under the frozen optimizer.",
        "",
        "A figure the submission asserts is never any part of that. It is the LEFT",
        "side of one divergence comparison and nothing else.",
        "",
        "## Measured, on the frozen bytes",
        "",
        "| quantity | reference | byte-level baseline |",
        "|---|---|---|",
        "| bits per byte | " + repr(graded) + " | " + repr(baseline_graded) + " |",
        "| vocabulary size | " + str(len(reference_tokens)) + " | "
        + str(len(baseline_tokens)) + " |",
        "| optimizer updates spent | "
        + str(reference_record["compute"]["spent_token_updates"]) + " | "
        + str(baseline_record["compute"]["spent_token_updates"]) + " |",
        "| denominator bytes | " + str(reference_record["corpus"]["denominator_bytes"])
        + " | " + str(baseline_record["corpus"]["denominator_bytes"]) + " |",
        "",
        "Evaluation points, reference: "
        + ", ".join(repr(p) for p in reference_record["schedule"]["point_bits_per_byte"]) + ".",
        "Evaluation points, baseline: "
        + ", ".join(repr(p) for p in baseline_record["schedule"]["point_bits_per_byte"]) + ".",
        "",
        "## Why the reference beats the baseline",
        "",
        " ".join(str(grounding["reference"]["why_it_beats_the_baseline"]).split()),
        "",
        "## The oracle path",
        "",
    ]
    for step in grounding["golden_trajectory"]:
        lines.append(
            str(step["step"]) + ". **" + str(step["action"]) + "** — "
            + " ".join(str(step["detail"]).split())
        )
    lines += [
        "",
        "## Anchors",
        "",
        "`anchors_state: absent`, gap `" + str(grounding["anchors"]["gap"]) + "`.",
        " ".join(str(grounding["anchors"]["consequence"]).split()),
        "",
        "## What the checkers refuse, and on what fixture",
        "",
        "| checker | zero reason | planted fixture | stale control |",
        "|---|---|---|---|",
    ]
    stale_by_checker = {row["checker"]: row["id"] for row in stale.values()}
    planted_by_checker = {row["fires"]: row["id"] for row in planted.values()}
    for name in CHECKER_ORDER:
        row = planted[planted_by_checker[name]]
        lines.append(
            "| " + name + " | " + row["reason"] + " | " + row["id"] + " | "
            + stale_by_checker.get(name, "—") + " |"
        )
    lines += [
        "",
        "## Grading configuration in force",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in sorted(config.items()) if not k.startswith("_")},
            sort_keys=True, indent=2, ensure_ascii=True
        ),
        "```",
        "",
        "The four keys a silent mutation moves are graded by the four checkers named",
        "in `seed/tasks/OER-16/drift.yaml`, and each carries the stale control that",
        "fails when a pre-mutation answer is carried forward.",
        "",
    ]
    return "\n".join(lines)


def test_output_module(config, superseded, clean, planted, stale) -> str:
    planted_by_checker = {row["fires"]: ident for ident, row in planted.items()}
    body = [
        '"""' + BANNER,
        "",
        "Source: " + SOURCE + ", generated by solution/recompute.py.",
        "",
        "The compiled tests tests/checkers.yaml names in `compiled_test`. Every one of",
        "the ten graded checkers is exercised on BOTH halves here: the clean record it",
        "accepts, and the planted-defect record it refuses with its exact machine-",
        "readable reason. Four further tests replay the stale controls: each named",
        "checker accepts the carried-forward answer under the SUPERSEDED admin-plane",
        "configuration and refuses it under the one in force, which is what makes the",
        "silent mutations in seed/tasks/OER-16/drift.yaml causal rather than asserted.",
        "",
        "Self-contained on purpose: the verifier image copies tests/ alone, so no",
        "fixture under solution/ is reachable from here and every record is embedded.",
        "Run it directly:  python3 tests/test_output.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "sys.path.insert(0, str(Path(__file__).resolve().parent))",
        "",
        "import checkers  # noqa: E402",
        "",
        "CONFIG = " + _py_literal(config),
        "",
        "SUPERSEDED_CONFIG = " + _py_literal(superseded),
        "",
        "CLEAN = " + _py_literal({k: v["record"] if "record" in v else v
                                 for k, v in _records(clean).items()}),
        "",
        "PLANTED = " + _py_literal(
            {ident: {"fires": row["fires"], "reason": row["reason"],
                     "base": row["base"], "record": row["record"]}
             for ident, row in planted.items()}
        ),
        "",
        "STALE = " + _py_literal(
            {ident: {"checker": row["checker"], "moved_key": row["moved_key"],
                     "reason": row["reason"], "record": row["record"]}
             for ident, row in stale.items()}
        ),
        "",
        "",
        "def _verdict(name, record, config):",
        "    return getattr(checkers, name)(record, config)",
        "",
        "",
        "def _both_halves(name, planted_id):",
        "    for clean_id, clean_record in sorted(CLEAN.items()):",
        "        verdict = _verdict(name, clean_record, CONFIG)",
        "        assert verdict.passed, (name, clean_id, verdict.reason, verdict.detail)",
        "        assert verdict.reason == '', (name, clean_id, verdict.reason)",
        "    row = PLANTED[planted_id]",
        "    verdict = _verdict(name, row['record'], CONFIG)",
        "    assert not verdict.passed, (name, planted_id, 'planted defect was accepted')",
        "    assert verdict.reason == row['reason'], (name, planted_id, verdict.reason)",
        "",
        "",
        "def _stale_causality(control_id):",
        "    row = STALE[control_id]",
        "    name = row['checker']",
        "    before = _verdict(name, row['record'], SUPERSEDED_CONFIG)",
        "    assert before.passed, (control_id, 'the carried-forward answer already failed',",
        "                           before.reason)",
        "    after = _verdict(name, row['record'], CONFIG)",
        "    assert not after.passed, (control_id, 'the mutation did not move the verdict')",
        "    assert after.reason == row['reason'], (control_id, after.reason)",
        "",
    ]
    for name in CHECKER_ORDER:
        body += [
            "",
            "def test_" + name + "():",
            "    _both_halves(" + repr(name) + ", " + repr(planted_by_checker[name]) + ")",
            "",
        ]
    for ident in sorted(stale):
        body += [
            "",
            "def test_" + ident.replace("-", "_") + "():",
            "    _stale_causality(" + repr(ident) + ")",
            "",
        ]
    body += [
        "",
        "def main() -> int:",
        "    names = sorted(n for n in globals() if n.startswith('test_'))",
        "    if not names:",
        "        print('no compiled test was collected, which is never a pass')",
        "        return 1",
        "    for name in names:",
        "        globals()[name]()",
        "    print('compiled tests passed: ' + str(len(names)))",
        "    return 0",
        "",
        "",
        "if __name__ == '__main__':",
        "    raise SystemExit(main())",
        "",
    ]
    return "\n".join(body)


def _records(clean: dict) -> dict:
    return {ident: {"record": record} for ident, record in clean.items()}


def _py_literal(payload) -> str:
    """A deterministic Python literal. json.dumps output is valid Python here.

    Every value in these payloads is a string, a finite number, a bool, None, a
    list or a dict, so the JSON image is also the Python image once the three
    JSON singletons are spelled the Python way.
    """
    text = json.dumps(payload, sort_keys=True, indent=4, ensure_ascii=True)
    text = re.sub(r"(?<![\w\"])true(?![\w\"])", "True", text)
    text = re.sub(r"(?<![\w\"])false(?![\w\"])", "False", text)
    text = re.sub(r"(?<![\w\"])null(?![\w\"])", "None", text)
    return text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def as_bytes(payload) -> bytes:
    return payload if isinstance(payload, bytes) else payload.encode("utf-8")


def main(argv) -> int:
    parser = argparse.ArgumentParser(description="derive OER-16 artifacts from grounding.yaml")
    parser.add_argument("--check", action="store_true",
                        help="recompute without writing; exit 1 if committed bytes drifted")
    args = parser.parse_args(argv[1:])

    artifacts = generate()
    drifted = []
    for relative in sorted(artifacts):
        target = BUNDLE / relative
        payload = as_bytes(artifacts[relative])
        current = target.read_bytes() if target.is_file() else None
        if current == payload:
            continue
        drifted.append(relative)
        if not args.check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            if relative.endswith(".sh"):
                target.chmod(0o755)

    if args.check:
        if drifted:
            print("DRIFTED: " + ", ".join(drifted), file=sys.stderr)
            return 1
        print("clean: " + str(len(artifacts)) + " generated artifact(s) match their source")
        return 0

    print(json.dumps({"generated": len(artifacts), "rewritten": drifted}, sort_keys=True, indent=2))
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
    raise SystemExit(main(sys.argv))

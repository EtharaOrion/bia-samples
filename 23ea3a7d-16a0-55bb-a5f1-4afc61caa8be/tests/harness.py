#!/usr/bin/env python3
"""The verifier's own measurement. This process owns the model and the counters.

Nothing here reads a number the submission printed. The submission's only
contribution is a vocabulary, loaded as DATA from `tokenizer.json`, and an
optional claim, loaded as DATA from `claim.json` and SEALED before any
measurement is computed. This module never imports the submission, never
executes it, and never consults a figure it reports about itself.

What it produces is one telemetry record: a JSON-compatible mapping describing
what this process actually did. `checkers.py` grades that record and nothing
else, so every number a checker sees was produced inside the verifier's own
process.

Frozen: the model, the optimizer, the compute budget, the evaluation corpus.
Free: the tokenizer and the vocabulary it is built over.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

RECORD_SCHEMA = "oer16.run_record/v1"
TOKENIZER_SCHEMA = "oer16.tokenizer/v1"
CLAIM_SCHEMA = "oer16.claim/v1"

BOS = -1


# --------------------------------------------------------------------------
# Frozen substrate
# --------------------------------------------------------------------------

def load_frozen(frozen_dir: Path) -> dict:
    """The frozen half, read from the delivered environment and never authored here."""
    base = Path(frozen_dir)
    return {
        "model": json.loads((base / "model_config.json").read_text(encoding="utf-8")),
        "optimizer": json.loads((base / "optimizer_config.json").read_text(encoding="utf-8")),
        "budget": json.loads((base / "compute_budget.json").read_text(encoding="utf-8")),
        "eval_corpus": (base / "eval_corpus.txt").read_bytes(),
        "train_corpus": (base / "train_corpus.txt").read_bytes(),
    }


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def frozen_fingerprints(frozen: dict) -> dict:
    """One digest per frozen axis. Taken at run open and again at run close."""
    return {
        "model": _digest(frozen["model"]),
        "optimizer": _digest(frozen["optimizer"]),
        "budget": _digest(frozen["budget"]),
        "eval_corpus": hashlib.sha256(frozen["eval_corpus"]).hexdigest(),
        "train_corpus": hashlib.sha256(frozen["train_corpus"]).hexdigest(),
    }


# --------------------------------------------------------------------------
# Vocabulary, loaded as data
# --------------------------------------------------------------------------

class VocabularyRefused(Exception):
    """The submitted vocabulary is not loadable, so this process has no model state."""


def load_vocabulary(path: Path, max_vocab_size: int, max_token_bytes: int) -> list:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != TOKENIZER_SCHEMA:
        raise VocabularyRefused("tokenizer.json declares an unknown schema")
    raw = payload.get("tokens")
    if not isinstance(raw, list) or not raw:
        raise VocabularyRefused("tokenizer.json carries no token list")
    tokens = []
    for item in raw:
        if not isinstance(item, str):
            raise VocabularyRefused("a token entry is not a hex string")
        tokens.append(bytes.fromhex(item))
    if len(set(tokens)) != len(tokens):
        raise VocabularyRefused("the vocabulary carries a duplicate token")
    if len(tokens) > max_vocab_size:
        raise VocabularyRefused("the vocabulary exceeds the announced ceiling")
    if any(len(t) > max_token_bytes for t in tokens):
        raise VocabularyRefused("a token exceeds max_token_bytes")
    if set(bytes([i]) for i in range(256)) - set(tokens):
        raise VocabularyRefused("the vocabulary omits at least one single byte")
    return tokens


def encode(data: bytes, index: dict, longest: int) -> list:
    """Greedy longest match, left to right. Total, because the 256 bytes are present."""
    out = []
    position, size = 0, len(data)
    while position < size:
        span = min(longest, size - position)
        while span > 1 and data[position:position + span] not in index:
            span -= 1
        out.append(index[data[position:position + span]])
        position += span
    return out


def _index(tokens: list) -> tuple:
    index = {}
    for ident, token in enumerate(tokens):
        index.setdefault(token, ident)
    return index, max(len(t) for t in tokens)


# --------------------------------------------------------------------------
# The frozen optimizer, spending the frozen budget on this process's counter
# --------------------------------------------------------------------------

class Model:
    """The frozen bigram categorical model. Owned by this process, start to finish."""

    def __init__(self, vocab_size: int, smoothing: float):
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts: dict = {}
        self.totals: dict = {}
        self.updates = 0

    def update(self, previous: int, current: int) -> None:
        key = (previous, current)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.totals[previous] = self.totals.get(previous, 0) + 1
        self.updates += 1

    def log2_probability(self, previous: int, current: int) -> float:
        numerator = self.counts.get((previous, current), 0) + self.smoothing
        denominator = self.totals.get(previous, 0) + self.smoothing * self.vocab_size
        return math.log2(numerator / denominator)

    def fingerprint(self) -> str:
        rows = sorted(
            [previous, current, value] for (previous, current), value in self.counts.items()
        )
        return _digest([self.vocab_size, self.smoothing, self.updates, rows])


def train(stream: list, budget_token_updates: int, vocab_size: int, smoothing: float) -> Model:
    """Spend exactly `budget_token_updates` optimizer updates. One update, one token."""
    model = Model(vocab_size, smoothing)
    if not stream:
        return model
    previous = BOS
    for step in range(budget_token_updates):
        current = stream[step % len(stream)]
        model.update(previous, current)
        previous = current if (step + 1) % len(stream) else BOS
    return model


def bits_per_byte(model: Model, data: bytes, index: dict, longest: int, denominator: int) -> float:
    """Raw cross entropy in bits over `data`, divided by the byte count handed in.

    The denominator is a parameter rather than `len(data)` on purpose: the graded
    reading divides by the frozen evaluation corpus length that this process
    measured, and a checker proves those two are the same number.
    """
    ids = encode(data, index, longest)
    total = 0.0
    previous = BOS
    for current in ids:
        total -= model.log2_probability(previous, current)
        previous = current
    return total / denominator


# --------------------------------------------------------------------------
# The run record
# --------------------------------------------------------------------------

def read_claim(path: Path) -> dict:
    """Seal the claim BEFORE anything is measured. Absent is a value, not an error."""
    target = Path(path)
    if not target.is_file():
        return {"present": False, "claimed_bits_per_byte": None}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"present": False, "claimed_bits_per_byte": None}
    if not isinstance(payload, dict) or payload.get("schema") != CLAIM_SCHEMA:
        return {"present": False, "claimed_bits_per_byte": None}
    value = payload.get("claimed_bits_per_byte")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return {"present": False, "claimed_bits_per_byte": None}
    return {"present": True, "claimed_bits_per_byte": float(value)}


def segments(data: bytes, count: int) -> list:
    """The verifier's own evaluation points: contiguous frozen slices of the corpus."""
    if count < 1:
        return [data]
    width = len(data) // count
    return [data[i * width:(i + 1) * width if i < count - 1 else len(data)] for i in range(count)]


def measure(frozen: dict, artifacts: Path) -> dict:
    """Run the frozen recipe over the submitted vocabulary and record what happened.

    Sequence stamps are this process's own monotone counter, not a clock. They
    exist so an ordering checker can prove the claim was sealed before the
    measurement existed to be fitted to.
    """
    artifacts = Path(artifacts)
    opening = frozen_fingerprints(frozen)

    step = 0
    step += 1
    claim = read_claim(artifacts / "claim.json")
    claim_seq = step

    budget_cfg = frozen["budget"]
    optimizer_cfg = frozen["optimizer"]
    eval_bytes = frozen["eval_corpus"]
    denominator = len(eval_bytes)

    record = {
        "schema": RECORD_SCHEMA,
        "sequence": {"claim_recorded_seq": claim_seq, "measurement_computed_seq": 0},
        "claim": {
            "present": claim["present"],
            "claimed_bits_per_byte": claim["claimed_bits_per_byte"],
            "absence_recorded": not claim["present"],
            "graded_on": "measurement",
        },
        "vocabulary": {
            "loaded_by": "verifier-harness",
            "source": "submission-tokenizer-json-as-data",
            "size": 0,
            "max_token_bytes": 0,
        },
        "model_state": {
            "owner": "verifier-harness",
            "produced_by": "harness-owned-model-state",
            "fingerprint_at_measurement": "",
            "harness_recomputed_fingerprint": "",
        },
        "compute": {
            "budget_token_updates": int(budget_cfg["budget_token_updates"]),
            "spent_token_updates": 0,
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
            "graded_bits_per_byte": None,
            "raw_bits_per_byte": None,
        },
        "schedule": {
            "points_scheduled": int(budget_cfg["evaluation_points"]),
            "points_completed": 0,
            "halted_early": False,
            "point_bits_per_byte": [],
        },
        "frozen_axes": {"expected": opening, "opening": opening, "closing": {}},
        "vocabulary_refusal": "",
    }

    step += 1
    try:
        tokens = load_vocabulary(
            artifacts / "tokenizer.json",
            int(budget_cfg["max_vocab_size"]),
            int(budget_cfg["max_token_bytes"]),
        )
    except (VocabularyRefused, ValueError, OSError) as exc:
        # No vocabulary means this process has no model state of its own, so
        # there is no measurement. The record says so rather than inventing one.
        record["vocabulary_refusal"] = str(exc)
        record["model_state"]["produced_by"] = "no-harness-model-state"
        record["frozen_axes"]["closing"] = frozen_fingerprints(frozen)
        record["sequence"]["measurement_computed_seq"] = step + 1
        return record

    index, longest = _index(tokens)
    longest = min(longest, int(budget_cfg["max_token_bytes"]))
    record["vocabulary"]["size"] = len(tokens)
    record["vocabulary"]["max_token_bytes"] = max(len(t) for t in tokens)

    step += 1
    train_ids = encode(frozen["train_corpus"], index, longest)
    model = train(
        train_ids,
        int(budget_cfg["budget_token_updates"]),
        len(tokens),
        float(optimizer_cfg["lambda"]),
    )
    record["compute"]["spent_token_updates"] = model.updates

    step += 1
    graded = bits_per_byte(model, eval_bytes, index, longest, denominator)
    points = []
    for chunk in segments(eval_bytes, int(budget_cfg["evaluation_points"])):
        points.append(bits_per_byte(model, chunk, index, longest, len(chunk)))
    record["schedule"]["points_completed"] = len(points)
    record["schedule"]["point_bits_per_byte"] = points
    record["readout"]["raw_bits_per_byte"] = graded
    record["readout"]["graded_bits_per_byte"] = graded
    record["model_state"]["fingerprint_at_measurement"] = model.fingerprint()
    record["model_state"]["harness_recomputed_fingerprint"] = model.fingerprint()
    record["sequence"]["measurement_computed_seq"] = step

    record["frozen_axes"]["closing"] = frozen_fingerprints(frozen)
    return record

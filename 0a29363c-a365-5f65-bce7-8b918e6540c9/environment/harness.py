#!/usr/bin/env python3
"""The frozen harness: encoder, model, optimizer, compute counter, evaluator.

Everything in this file is a FROZEN axis. A submission never edits it, and the
verifier runs its own copy from the delivery unit rather than the one a run may
have left in a workspace. The only free axis is the *vocabulary construction*,
which reaches this harness as a list of byte strings.

Three properties this file exists to hold, all of them readable here rather than
asserted elsewhere.

1. The denominator of bits per byte is the frozen byte count of the evaluation
   corpus, read from `manifest.json` and re-checked against the corpus digest on
   every run. A vocabulary that shrinks the token count therefore shrinks nothing
   on the denominator. Token counts are recorded for information and are never
   divided by.

2. Compute is counted as token updates the optimizer actually consumed, by a
   counter this file owns. A submission may request a halt; the halt is recorded
   and the run is marked as not having spent the budget. Nothing a submission
   prints or writes reaches the counter.

3. Evaluation reads the model state this harness holds at the scheduled update
   count. There is no path by which a submission hands back a state, a
   checkpoint, or a number. The graded reading is raw: no average, no EMA, no
   filter of any kind is applied to the bits this file computes.

No clock, no network, no random source, no locale-dependent operation is used
anywhere below, so two runs over the same bytes produce the same telemetry.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
MANIFEST_PATH = BASE / "manifest.json"

# Model constants. Frozen: they are part of "the model" and "the optimizer".
ALPHA = 1.0  # bigram interpolation mass toward the unigram distribution
BETA = 1.0  # unigram interpolation mass toward the uniform floor
MAX_TOKEN_LEN = 16  # the encoder never matches longer than this


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _corpus(name: str) -> bytes:
    return (BASE / "corpus" / name).read_bytes()


def normalise_vocabulary(entries, budget: int) -> list:
    """The 256 single bytes first, then the submission's entries, deduped in order.

    Single bytes are not optional: without them the encoder would not be total
    over arbitrary input, and a partial encoder makes bits per byte undefined on
    the bytes it cannot express. They occupy budget like anything else.
    """
    vocab = [bytes([i]) for i in range(256)]
    seen = set(vocab)
    for raw in entries or []:
        if isinstance(raw, str):
            item = raw.encode("utf-8")
        else:
            item = bytes(raw)
        if len(item) < 2 or len(item) > MAX_TOKEN_LEN or item in seen:
            continue
        seen.add(item)
        vocab.append(item)
        if len(vocab) >= budget:
            break
    return vocab


def encode(vocab, data: bytes) -> list:
    """Greedy longest match, ties impossible because lengths are distinct per start.

    The encoder is frozen. A submission that wanted a cleverer segmentation gets
    none: the free axis is which strings are in the vocabulary, not how the
    vocabulary is applied.
    """
    index = {}
    longest = 1
    for ident, token in enumerate(vocab):
        index[token] = ident
        longest = max(longest, len(token))
    out = []
    position = 0
    size = len(data)
    while position < size:
        span = min(longest, size - position)
        while span > 1:
            ident = index.get(data[position:position + span])
            if ident is not None:
                out.append(ident)
                position += span
                break
            span -= 1
        else:
            out.append(index[data[position:position + 1]])
            position += 1
    return out


class Model:
    """Bigram counts with a frozen two-level interpolation.

    The interpolation is part of the frozen model specification and is applied to
    the model's own probability estimate. It is not a readout filter: nothing
    here averages, blends or otherwise post-processes a sequence of computed
    bits-per-byte values, which is the operation the graded path forbids.
    """

    def __init__(self, size: int):
        self.size = size
        self.unigram = {}
        self.bigram = {}
        self.context = {}
        self.total = 0

    def observe(self, previous, token) -> None:
        self.unigram[token] = self.unigram.get(token, 0) + 1
        self.total += 1
        if previous is None:
            return
        row = self.bigram.get(previous)
        if row is None:
            row = {}
            self.bigram[previous] = row
        row[token] = row.get(token, 0) + 1
        self.context[previous] = self.context.get(previous, 0) + 1

    def _unigram_probability(self, token) -> float:
        floor = BETA / self.size
        return (self.unigram.get(token, 0) + floor) / (self.total + BETA)

    def probability(self, previous, token) -> float:
        base = self._unigram_probability(token)
        if previous is None:
            return base
        row = self.bigram.get(previous)
        seen = self.context.get(previous, 0)
        count = 0 if row is None else row.get(token, 0)
        return (count + ALPHA * base) / (seen + ALPHA)

    def bits(self, tokens) -> float:
        """Raw bits of one token sequence. No filter, no clipping, no smoothing."""
        total = 0.0
        previous = None
        for token in tokens:
            total += -math.log2(self.probability(previous, token))
            previous = token
        return total

    def state_digest(self) -> str:
        rows = [[int(k), int(v)] for k, v in sorted(self.unigram.items())]
        pairs = [
            [int(p), int(t), int(c)]
            for p, row in sorted(self.bigram.items())
            for t, c in sorted(row.items())
        ]
        payload = json.dumps(
            {"unigram": rows, "bigram": pairs, "total": self.total},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def _schedule(spec: dict, budget: int) -> list:
    points = []
    for fraction in spec["progress_fractions"]:
        updates = int(round(budget * fraction))
        points.append({"updates": updates, "role": "progress", "charged_to_budget": True})
    points.append({"updates": budget, "role": "bound", "charged_to_budget": True})
    for extra in spec["sustain_offsets"]:
        points.append(
            {"updates": budget + int(extra), "role": "sustain", "charged_to_budget": False}
        )
    ordered, seen = [], set()
    for row in sorted(points, key=lambda item: (item["updates"], item["role"])):
        if row["updates"] in seen:
            continue
        seen.add(row["updates"])
        ordered.append(row)
    return ordered


def run(entries, halt_at=None, reported=None) -> dict:
    """Train under the frozen compute counter and evaluate at every scheduled point.

    `halt_at` is the submission's request to stop early. It is honoured, because
    refusing it would hide the behaviour instead of grading it, and it is recorded
    so a run that took it is graded as not having established the metric.

    `reported` is whatever the submission said about itself. It is carried into
    the telemetry untouched and is never read by anything that computes a graded
    number; it exists so a checker can prove the graded number diverges from it.
    """
    spec = manifest()
    budget = int(spec["compute_budget_updates"])
    train_bytes = _corpus("train.txt")
    eval_bytes = _corpus("eval.txt")
    train_digest = hashlib.sha256(train_bytes).hexdigest()
    eval_digest = hashlib.sha256(eval_bytes).hexdigest()
    if eval_digest != spec["eval_corpus_digest"] or len(eval_bytes) != int(
        spec["eval_corpus_bytes"]
    ):
        raise SystemExit("frozen evaluation corpus does not match the manifest")

    submitted_digest = submission_digest(entries)
    vocab = normalise_vocabulary(entries, int(spec["vocab_budget"]))
    vocab_digest = hashlib.sha256(
        json.dumps([token.hex() for token in vocab], separators=(",", ":")).encode("ascii")
    ).hexdigest()

    train_tokens = encode(vocab, train_bytes)
    eval_tokens = encode(vocab, eval_bytes)
    model = Model(len(vocab))

    points = _schedule(spec["evaluation_schedule"], budget)
    ceiling = points[-1]["updates"]
    stop_at = ceiling if halt_at is None else min(int(halt_at), ceiling)

    records = []
    consumed = 0
    charged = 0
    previous = None
    cursor = 0
    halted = False
    for row in points:
        target = row["updates"]
        while consumed < target and consumed < stop_at:
            token = train_tokens[cursor % len(train_tokens)]
            model.observe(previous, token)
            previous = token
            cursor += 1
            consumed += 1
            if row["charged_to_budget"] or consumed <= budget:
                charged = min(consumed, budget)
        if consumed < target:
            halted = True
            break
        bits = model.bits(eval_tokens)
        records.append(
            {
                "updates": target,
                "role": row["role"],
                "charged_to_budget": bool(row["charged_to_budget"]),
                "bits": bits,
                "denominator_bytes": len(eval_bytes),
                "bits_per_byte": bits / len(eval_bytes),
                "eval_tokens": len(eval_tokens),
                "state_digest": model.state_digest(),
            }
        )

    bound = next((row for row in records if row["role"] == "bound"), None)
    return {
        "schema": "oer15.telemetry/v1",
        "frozen": {
            "compute_budget_updates": budget,
            "model_spec_digest": model_spec_digest(),
            "eval_corpus_digest": eval_digest,
            "eval_corpus_bytes": len(eval_bytes),
            "train_corpus_digest": train_digest,
            "vocab_budget": int(spec["vocab_budget"]),
        },
        "vocabulary": {
            "size": len(vocab),
            "digest": vocab_digest,
            "submitted_digest": submitted_digest,
            "train_tokens": len(train_tokens),
            "eval_tokens": len(eval_tokens),
        },
        "compute": {
            "budget_updates": budget,
            "charged_updates": charged,
            "consumed_updates": consumed,
            "halt_requested_at": None if halt_at is None else int(halt_at),
            "halted_before_bound_point": bool(halted or bound is None),
        },
        "denominator": {
            "bytes": len(eval_bytes),
            "source": "frozen-eval-corpus-bytes",
            "digest": eval_digest,
        },
        "readout": {
            "filter": "none",
            "weights_origin": "harness-owned-state-at-bound-point",
            "graded_role": "bound",
        },
        "eval_points": records,
        "graded": None
        if bound is None
        else {
            "updates": bound["updates"],
            "bits": bound["bits"],
            "denominator_bytes": bound["denominator_bytes"],
            "bits_per_byte": bound["bits_per_byte"],
            "state_digest": bound["state_digest"],
        },
        "reported_by_submission": dict(reported or {}),
    }


def submission_digest(entries) -> str:
    """A digest over exactly what the submission returned, before normalisation.

    Kept separate from the normalised vocabulary digest so a checker can prove the
    graded run used the list the submission handed back, rather than proving only
    that the harness digested its own output.
    """
    rows = []
    for raw in entries or []:
        item = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        rows.append(item.hex())
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def model_spec_digest() -> str:
    payload = json.dumps(
        {
            "alpha": ALPHA,
            "beta": BETA,
            "max_token_len": MAX_TOKEN_LEN,
            "model": "bigram-interpolated",
            "optimizer": "streaming-count-update",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def bits_per_byte(entries) -> float:
    """Convenience for sweeps. The graded path uses `run` and reads telemetry."""
    telemetry = run(entries)
    return telemetry["graded"]["bits_per_byte"]

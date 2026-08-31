"""The frozen training stack for OER-18: the model, the optimizer, and the feeder.

FROZEN. This file is part of the task substrate, not part of the submission. Editing
your copy changes nothing that is graded: the verifier runs its own pinned copy of these
bytes, whose sha256 is bound in tests/checkers.yaml. What is free is the corpus your
generator emits, and nothing else.

The stack is deliberately small, pure-stdlib and deterministic:

  * the model is a bag-of-tokens linear scorer over a fixed number of hashed buckets,
  * the optimizer is a multiclass perceptron with a fixed learning rate and a fixed
    number of passes, consuming samples strictly in emission order,
  * the feeder counts tokens and stops when the frozen token budget is exhausted.

There is no clock, no random source, no network, and no locale-dependent operation
anywhere in this module. Two runs over the same corpus produce the same weights and the
same score, byte for byte.
"""

from __future__ import annotations

import hashlib

# ---------------------------------------------------------------------------
# Frozen model and optimizer constants. None of these is a free variable.
# ---------------------------------------------------------------------------

FEATURE_BUCKETS = 512
EPOCHS = 3
LEARNING_RATE = 1.0
LABELS = ("no", "yes")

# The five capability strata this family is cut into. A sample belongs to exactly one
# of them, and which one is decided by the verifier's own classifier over the sample
# text, never by a field the sample carries and never by a manifest the generator writes.
STRATA = ("calc", "cmp", "fact", "neg", "seq")

# The surface marker that opens a sample of each stratum.
MARKERS = {
    "calc": "calc",
    "cmp": "cmp",
    "fact": "fact",
    "neg": "neg",
    "seq": "seq",
}

# The stratum vocabularies. The first three words of each list carry label "yes", the
# last three carry label "no". A model that never sees a stratum's words cannot answer
# that stratum's benchmark items above the tie-break floor.
STRATUM_WORDS = {
    "calc": ("adds", "sums", "totals", "voids", "drops", "clears"),
    "cmp": ("exceeds", "outranks", "tops", "trails", "lags", "sinks"),
    "fact": ("holds", "stands", "obtains", "fails", "lapses", "voidsout"),
    "neg": ("denies", "refutes", "rejects", "affirms", "grants", "concedes"),
    "seq": ("follows", "succeeds", "trails2", "precedes", "opens", "leads"),
}

POSITIVE_PER_STRATUM = 3

# Filler tokens. The training fillers and the evaluation fillers are disjoint, so a
# sample and a benchmark item that share a marker and a word still differ in one of
# three tokens. That keeps an honest corpus far below the bound near-duplicate
# threshold while an emitted benchmark item sits at the top of it.
TRAIN_FILLERS = ("alpha", "beta", "gamma", "delta")
EVAL_FILLERS = ("kappa", "lambda", "mu", "nu")


def tokenize(text: str) -> list:
    """The frozen tokenization: split on whitespace, casefold, nothing else."""
    return [token.casefold() for token in str(text).split()]


def bucket(token: str) -> int:
    """A stable feature bucket. blake2b is used because builtin hash is salted."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % FEATURE_BUCKETS


def features(text: str) -> dict:
    """Bag of hashed tokens with integer counts, in a mapping with fixed key order."""
    counts: dict = {}
    for token in tokenize(text):
        index = bucket(token)
        counts[index] = counts.get(index, 0) + 1
    return counts


def _score(weights: dict, vector: dict) -> dict:
    return {
        label: sum(weights[label][index] * value for index, value in vector.items())
        for label in LABELS
    }


def predict(weights: dict, text: str) -> str:
    """argmax over the closed label set, ties broken by the fixed label order."""
    scored = _score(weights, features(text))
    best = None
    for label in LABELS:
        if best is None or scored[label] > scored[best]:
            best = label
    return best


def new_weights() -> dict:
    return {label: [0.0] * FEATURE_BUCKETS for label in LABELS}


def train(samples) -> dict:
    """Multiclass perceptron over the samples in emission order. Fully deterministic."""
    weights = new_weights()
    rows = [(str(row.get("text", "")), str(row.get("label", ""))) for row in samples]
    for _ in range(EPOCHS):
        for text, gold in rows:
            if gold not in LABELS:
                continue
            vector = features(text)
            guess = predict(weights, text)
            if guess == gold:
                continue
            for index, value in vector.items():
                weights[gold][index] += LEARNING_RATE * value
                weights[guess][index] -= LEARNING_RATE * value
    return weights


def evaluate(weights: dict, items) -> float:
    """Accuracy on the given items, in [0, 1]. Higher is better."""
    rows = list(items)
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if predict(weights, row["text"]) == row["label"])
    return hits / len(rows)


def token_count(sample) -> int:
    """The tokens one sample costs against the frozen budget."""
    return len(tokenize(sample.get("text", "")))


def feed(samples, budget_tokens: int) -> tuple:
    """Feed samples in emission order and report what was actually consumed.

    Returns (prefix, tokens_fed_by_the_prefix, tokens_offered_by_the_whole_corpus).
    The feeder stops at the budget; it never silently trims a corpus that overshoots,
    because the overshoot is exactly what the budget checker reads.
    """
    prefix, fed, offered = [], 0, 0
    for row in samples:
        cost = token_count(row)
        offered += cost
        if fed + cost <= budget_tokens:
            prefix.append(row)
            fed += cost
    return prefix, fed, offered


def train_to_mark(samples, mark_tokens: int) -> tuple:
    """Train on the corpus prefix whose cumulative token count reaches the mark.

    Returns (weights, tokens_consumed, reached), where `reached` is False when the
    corpus ran out before the mark. A mark that is not reached is not an evaluation
    point that happened.
    """
    prefix, consumed = [], 0
    for row in samples:
        if consumed >= mark_tokens:
            break
        prefix.append(row)
        consumed += token_count(row)
    return train(prefix), consumed, consumed >= mark_tokens

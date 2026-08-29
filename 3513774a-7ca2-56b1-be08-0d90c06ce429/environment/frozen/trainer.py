"""The frozen model, optimizer, training budget and evaluator. Deterministic, stdlib only.

This file is the frozen half of slot OER-19. The solving agent does not modify it
and cannot reach it at grading time: the verifier runs this copy, and the agent
sees a byte-identical mirror at `environment/frozen/trainer.py` whose sha256 is
pinned in `tests/checkers.yaml`. The agent's whole surface of control is the
synthetic corpus it generates; everything downstream of that corpus is fixed here.

Determinism is structural rather than seeded. The weights start at exactly zero,
the vocabulary is built in sorted order, samples are consumed in emission order,
ties in argmax are broken by the lowest class index, and there is no random source
anywhere in the file. Two runs over the same corpus produce byte-identical weights,
so the graded number is a function of the corpus and of nothing else.

The training budget is enforced here and counted here. `train()` returns the
counters it actually incremented while feeding the optimizer, never the counts it
was asked for, because a budget checker that reads the requested figure grades the
request instead of the run.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Frozen axes. None of these is a knob the agent may turn; they are the bound
# training budget, the bound model and the bound optimizer.
# ---------------------------------------------------------------------------

LABELS = ("music", "timer", "translate", "weather")

BOUND_CORPUS_SAMPLES = 480          # the corpus size the generator must emit
BOUND_EPOCHS = 3                    # passes over the corpus
BOUND_UPDATES = BOUND_CORPUS_SAMPLES * BOUND_EPOCHS   # 1440 optimizer updates
BOUND_SAMPLES_FED = BOUND_UPDATES   # one sample consumed per update

LEARNING_RATE = 0.5
VOCAB_CAP = 4000

# Evaluation points the verifier schedules, in update counts. The last is the
# bound evaluation point at which the graded score is read; the earlier ones are
# the points the reading must be sustained across.
SCHEDULED_POINTS = (960, 1200, 1440)
BOUND_EVALUATION_POINT = 1440
SUSTAIN_TOLERANCE = 0.08

COUNTER_SOURCE = "harness-trainer"
READOUT = {"kind": "raw", "window": 1, "ema_alpha": None}


@dataclass
class Counters:
    """What the optimizer was actually fed, incremented at the point of feeding."""

    samples_fed: int = 0
    updates: int = 0
    epochs_completed: int = 0
    terminated_early: bool = False


@dataclass
class Checkpoint:
    update: int
    weights: list
    digest: str


@dataclass
class Trained:
    vocabulary: tuple
    checkpoints: list = field(default_factory=list)
    counters: Counters = field(default_factory=Counters)

    def at(self, update: int):
        for row in self.checkpoints:
            if row.update == update:
                return row
        return None


def build_vocabulary(texts) -> tuple:
    """Sorted token vocabulary capped at the bound size. Sorted, so order is not luck."""
    seen = set()
    for text in texts:
        for token in _tokens(text):
            seen.add(token)
    return tuple(sorted(seen)[:VOCAB_CAP])


_KEPT = "abcdefghijklmnopqrstuvwxyz0123456789"


def _tokens(text) -> list:
    out, gap = [], False
    for char in str(text):
        lowered = char.lower()
        if lowered in _KEPT:
            if gap and out:
                out.append(" ")
            out.append(lowered)
            gap = False
        else:
            gap = True
    joined = "".join(out)
    return joined.split(" ") if joined else []


def featurize(text, index: dict) -> list:
    """Binary bag of known tokens, as a sorted list of feature positions."""
    hits = {index[token] for token in _tokens(text) if token in index}
    return sorted(hits)


def _digest(weights) -> str:
    """A digest over the weight vector, rounded so it is stable across platforms.

    Rounding before digesting is deliberate: an unrounded float image would make
    the digest depend on the last bit of an accumulation order, and a digest that
    can move without the model moving is not evidence about the model.
    """
    payload = ";".join(
        ",".join(str(round(value, 6)) for value in row) for row in weights
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _softmax(scores) -> list:
    top = max(scores)
    exps = [math.exp(value - top) for value in scores]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in exps]


def train(samples, updates: int = BOUND_UPDATES, points=SCHEDULED_POINTS) -> Trained:
    """Feed the frozen optimizer in emission order and checkpoint at the scheduled points.

    `samples` carries (text, label) pairs in emission order. The corpus is cycled
    if the update count exceeds its length, which is what the bound three epochs
    mean. Every counter below is incremented where the work happens, so a caller
    cannot report a budget it did not spend.
    """
    rows = [(str(text), str(label)) for text, label in samples]
    vocabulary = build_vocabulary(text for text, _ in rows)
    index = {token: position for position, token in enumerate(vocabulary)}
    width = len(vocabulary)
    weights = [[0.0] * (width + 1) for _ in LABELS]
    counters = Counters()
    result = Trained(vocabulary=vocabulary, counters=counters)
    if not rows or width == 0:
        for point in points:
            result.checkpoints.append(Checkpoint(point, [list(row) for row in weights], _digest(weights)))
        return result

    wanted = set(points)
    for step in range(1, int(updates) + 1):
        text, label = rows[(step - 1) % len(rows)]
        features = featurize(text, index)
        scores = [sum(row[pos] for pos in features) + row[width] for row in weights]
        probabilities = _softmax(scores)
        target = LABELS.index(label) if label in LABELS else 0
        for cls in range(len(LABELS)):
            error = (1.0 if cls == target else 0.0) - probabilities[cls]
            if error == 0.0:
                continue
            delta = LEARNING_RATE * error
            row = weights[cls]
            for pos in features:
                row[pos] += delta
            row[width] += delta
        counters.samples_fed += 1
        counters.updates += 1
        if step % len(rows) == 0:
            counters.epochs_completed += 1
        if step in wanted:
            result.checkpoints.append(
                Checkpoint(step, [list(row) for row in weights], _digest(weights))
            )
    counters.terminated_early = counters.updates < int(updates)
    return result


def predict(text, vocabulary, weights) -> str:
    index = {token: position for position, token in enumerate(vocabulary)}
    width = len(vocabulary)
    features = featurize(text, index)
    scores = [sum(row[pos] for pos in features) + row[width] for row in weights]
    best, best_index = scores[0], 0
    for position in range(1, len(scores)):
        if scores[position] > best:
            best, best_index = scores[position], position
    return LABELS[best_index]


def evaluate(benchmark, vocabulary, weights) -> dict:
    """Raw accuracy on the frozen held-out benchmark. No smoothing anywhere on this path."""
    correct = 0
    total = 0
    for row in benchmark:
        total += 1
        if predict(row["text"], vocabulary, weights) == row["label"]:
            correct += 1
    accuracy = (correct / total) if total else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}

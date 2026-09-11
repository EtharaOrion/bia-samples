"""The frozen model, the frozen optimizer, the frozen budget, and the evaluation.

This module is the verifier's own trainer. It is the only thing that ever touches the
held-out benchmark, it is the only thing that produces the graded number, and every
counter a checker reads is incremented here rather than parsed out of anything the
submission emitted.

The model is a per-conversion log-factor vector. The optimizer is full-batch gradient
descent on squared error in log space at a frozen learning rate, one parameter update
per step for every conversion the corpus covers. Evaluation is a fraction of held-out
items answered within a bound relative tolerance. All of it is exact, deterministic and
free of any random source, so two runs over the same corpus produce identical bytes.

environment/trainer.py is a byte-faithful mirror of the training core for the agent to
iterate against locally. The graded run is this file, executed by the verifier.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = "forge.oer17.run/v1"


def load_jsonl(path: Path) -> list:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_prompt(prompt: str):
    """`convert <value> <from> to <to>` and nothing else. Anything else is unusable."""
    parts = str(prompt).split()
    if len(parts) != 5 or parts[0] != "convert" or parts[3] != "to":
        return None
    try:
        value = float(parts[1])
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return value, parts[2], parts[4]


def conversion_key(source: str, target: str) -> str:
    return str(source) + ">" + str(target)


def token_count(sample: dict) -> int:
    return len(str(sample.get("prompt", "")).split()) + len(str(sample.get("answer", "")).split())


def _log(value: float) -> float:
    """The corpus is positive by construction, so the log is total on what reaches it."""
    return math.log(value)


def _exp(value: float) -> float:
    return math.exp(value)


def targets(samples: list) -> tuple:
    """Mean log-ratio per conversion key, plus the counters the budget checker reads."""
    sums: dict = {}
    counts: dict = {}
    usable = 0
    tokens = 0
    for sample in samples:
        parsed = parse_prompt(sample.get("prompt", ""))
        if parsed is None:
            continue
        try:
            answer = float(sample.get("answer"))
        except (TypeError, ValueError):
            continue
        if answer <= 0.0:
            continue
        value, source, target = parsed
        key = conversion_key(source, target)
        sums[key] = sums.get(key, 0.0) + _log(answer / value)
        counts[key] = counts.get(key, 0) + 1
        usable += 1
        tokens += token_count(sample)
    means = {key: sums[key] / counts[key] for key in sorted(sums)}
    return means, usable, tokens


def weights_digest(vector: dict) -> str:
    rows = [[str(key), format(float(vector[key]), ".12g")] for key in sorted(vector)]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate(theta: dict, benchmark: list, relative_tolerance: float) -> float:
    """The graded quantity: fraction of held-out items answered within tolerance."""
    if not benchmark:
        return 0.0
    correct = 0
    for item in benchmark:
        parsed = parse_prompt(item.get("prompt", ""))
        if parsed is None:
            continue
        value, source, target = parsed
        expected = float(item.get("answer"))
        predicted = value * _exp(theta.get(conversion_key(source, target), 0.0))
        if abs(predicted - expected) <= relative_tolerance * max(1e-12, abs(expected)):
            correct += 1
    return correct / len(benchmark)


def eval_schedule(budget_steps: int, fractions: list) -> list:
    """The evaluation points the verifier schedules. The submission chooses none."""
    points = []
    for fraction in fractions:
        step = int(round(float(fraction) * int(budget_steps)))
        step = max(1, min(int(budget_steps), step))
        if step not in points:
            points.append(step)
    return sorted(points)


def train(samples: list, benchmark: list, bound: dict) -> dict:
    """One frozen training run, its counters, and the verifier's own evaluations."""
    budget = int(bound["frozen_train_steps"])
    learning_rate = float(bound["learning_rate"])
    tolerance = float(bound["evaluation_relative_tolerance"])
    schedule = eval_schedule(budget, bound["eval_point_fractions"])

    means, usable, tokens_per_pass = targets(samples)
    theta = {key: 0.0 for key in means}

    steps_fed = 0
    tokens_fed = 0
    eval_points = []
    weights_by_step = {}
    halted_at_step = 0

    for step in range(1, budget + 1):
        for key in sorted(theta):
            theta[key] -= learning_rate * (theta[key] - means[key])
        steps_fed += 1
        tokens_fed += tokens_per_pass
        halted_at_step = step
        if step in schedule:
            digest = weights_digest(theta)
            weights_by_step[str(step)] = digest
            eval_points.append(
                {
                    "step": step,
                    "raw_score": evaluate(theta, benchmark, tolerance),
                    "smoothed_score": None,
                    "weights_sha256": digest,
                }
            )

    return {
        "optimizer": str(bound["optimizer_id"]),
        "learning_rate": learning_rate,
        "steps_fed": steps_fed,
        "tokens_fed": tokens_fed,
        "corpus_samples": len(samples),
        "usable_samples": usable,
        "halted_at_step": halted_at_step,
        "completed": halted_at_step == budget,
        "checkpoint_selected_by": "harness",
        "weights_by_step": weights_by_step,
        "eval_points": eval_points,
        "final_weights": dict(theta),
        "schedule": schedule,
    }


def digest_tree(root: Path, skip: tuple = ("__pycache__",)) -> tuple:
    """Path-independent digest over a directory, plus the file count behind it."""
    rows = []
    for entry in sorted(Path(root).rglob("*")):
        relative = entry.relative_to(root)
        if set(relative.parts) & set(skip):
            continue
        if entry.is_file():
            rows.append([relative.as_posix(), hashlib.sha256(entry.read_bytes()).hexdigest()])
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(rows)

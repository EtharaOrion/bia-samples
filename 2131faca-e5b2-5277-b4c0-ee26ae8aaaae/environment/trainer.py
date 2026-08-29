"""The frozen training core, mirrored here so you can iterate locally.

This is the model, the optimizer and the budget the graded run uses. It is FROZEN: the
graded run executes the verifier's own copy of this code with the bound values, and
editing this file changes nothing about how you are scored. It is here so that you can
measure your generator against your own held-out split before you submit.

What is free is the generator you write. What is frozen is everything below.

  model      one log-factor parameter per conversion pair the corpus covers
  optimizer  full-batch gradient descent on squared error in log space
  budget     a fixed number of steps, a token ceiling and a corpus cap
  metric     fraction of held-out items answered within a relative tolerance

Usage:
    python3 trainer.py --samples samples.jsonl --eval my_own_split.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FROZEN_TRAIN_STEPS = 320
FROZEN_LEARNING_RATE = 0.5
FROZEN_OPTIMIZER = "full-batch-gd-log-space"
EVALUATION_RELATIVE_TOLERANCE = 1.0e-05
MAX_CORPUS_SAMPLES = 2000
MAX_TOKENS_FED = 4000000


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_prompt(prompt):
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


def targets(samples):
    sums, counts = {}, {}
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
        key = source + ">" + target
        sums[key] = sums.get(key, 0.0) + math.log(answer / value)
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}


def train(samples, steps=FROZEN_TRAIN_STEPS, learning_rate=FROZEN_LEARNING_RATE):
    means = targets(samples)
    theta = {key: 0.0 for key in means}
    for _ in range(int(steps)):
        for key in sorted(theta):
            theta[key] -= learning_rate * (theta[key] - means[key])
    return theta


def evaluate(theta, items, tolerance=EVALUATION_RELATIVE_TOLERANCE):
    if not items:
        return 0.0
    correct = 0
    for item in items:
        parsed = parse_prompt(item.get("prompt", ""))
        if parsed is None:
            continue
        value, source, target = parsed
        expected = float(item.get("answer"))
        predicted = value * math.exp(theta.get(source + ">" + target, 0.0))
        if abs(predicted - expected) <= tolerance * max(1e-12, abs(expected)):
            correct += 1
    return correct / len(items)


def main():
    parser = argparse.ArgumentParser(description="local mirror of the frozen training core")
    parser.add_argument("--samples", required=True, help="jsonl your generator produced")
    parser.add_argument("--eval", required=True, help="jsonl you built yourself to measure against")
    args = parser.parse_args()
    samples = load_jsonl(args.samples)[:MAX_CORPUS_SAMPLES]
    theta = train(samples)
    print(json.dumps({"corpus_samples": len(samples), "covered_pairs": len(theta), "score": evaluate(theta, load_jsonl(args.eval))}, sort_keys=True))


if __name__ == "__main__":
    main()

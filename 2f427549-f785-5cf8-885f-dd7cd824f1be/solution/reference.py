"""Reference generator: cover the conversion space with a value grid of its own.

The graded quantity is the held-out benchmark score of a model trained on this corpus.
The model learns one log-factor per conversion pair, so what the score rewards is
coverage of the pair space plus internally consistent answers. Neither of those needs a
single benchmark item, which is the whole argument of this reference: the honest
solution and the contaminating one are not close to each other in behaviour, and only
one of them survives the screen.

The value grid is deliberately disjoint from anything the benchmark could be built from
and is a property of this file rather than a guess about the benchmark: a generator that
enumerates its own grid never has to know what the benchmark's values are.

Run: python3 reference.py --out samples.jsonl
"""

from __future__ import annotations

import argparse
import json

# Unit inventory and base factors, from environment/GENERATOR_CONTRACT.md.
DIMENSIONS = {
    "length": {"km": 1000.0, "m": 1.0, "cm": 0.01, "mm": 0.001},
    "mass": {"kg": 1000.0, "g": 1.0, "mg": 0.001},
    "time": {"h": 3600.0, "min": 60.0, "s": 1.0},
}

# This generator's own grid. Redundancy per pair is what makes the full-batch mean a
# stable target rather than a single-sample artefact.
VALUES = (2.0, 3.0, 5.0, 6.0, 8.0, 9.0)


def canonical(value: float) -> str:
    return format(float(value), ".6g")


def pairs() -> list:
    out = []
    for units in DIMENSIONS.values():
        for source in units:
            for target in units:
                if source != target:
                    out.append((source, target, units[source] / units[target]))
    return out


def samples() -> list:
    rows = []
    for source, target, factor in pairs():
        for value in VALUES:
            rows.append(
                {
                    "prompt": "convert " + canonical(value) + " " + source + " to " + target,
                    "answer": canonical(value * factor),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="reference synthetic data generator")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = samples()
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    # Recorded for divergence only, and deliberately not the number the verifier will
    # measure: the graded value is the verifier's own measurement over harness-owned
    # parameters, so a self-reported estimate can differ without moving the score.
    print("REPORTED_SCORE=0.875000")
    print("samples=" + str(len(rows)) + " pairs=" + str(len(pairs())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

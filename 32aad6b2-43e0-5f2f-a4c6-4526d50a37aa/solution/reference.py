#!/usr/bin/env python3
"""GENERATED SECTION. DO NOT HAND-EDIT.

source: solution/grounding.yaml

The reference generator for slot OER-19. It walks a template grid crossed with its
subject and modifier fillers, so its coverage is a structural property rather than
the outcome of a draw that could stop being representative partway through the run.

It then does the thing this slot is about: it measures its own corpus with
`corpus_diversity`, the frozen tool on the agent surface, and writes what it
measured into run_report.json. Exiting zero is not the claim. The measured profile
is the claim, and the verifier recomputes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import corpus_diversity as cd

LABELS = ('music', 'timer', 'translate', 'weather')
SEGMENTS = 8
PREFIX_POINTS = 4
MODE_THRESHOLD = 0.6
SEGMENT_FLOOR = 0.34
MODE_CEILING = 0.08

GRID = {
    "music": {
        "modifiers": [
            "evening",
            "the archive",
            "monday",
            "the kitchen"
        ],
        "subjects": [
            "jazz",
            "ambient",
            "classical",
            "reggae",
            "folk"
        ],
        "templates": [
            "play {S} by {M} on the speaker",
            "put on some {S} music from {M}",
            "start the {S} playlist {M} please",
            "queue up {S} tracks for {M}",
            "shuffle my {S} songs from {M}",
            "resume the {S} album {M}"
        ]
    },
    "timer": {
        "modifiers": [
            "bake bread",
            "stretch",
            "review notes",
            "water plants"
        ],
        "subjects": [
            "ten minutes",
            "an hour",
            "two hours",
            "forty seconds",
            "half an hour"
        ],
        "templates": [
            "set a timer for {S} while i {M}",
            "remind me in {S} to {M}",
            "start a {S} countdown for {M}",
            "wake me after {S} of {M}",
            "give me {S} on the clock for {M}",
            "count down {S} before {M}"
        ]
    },
    "translate": {
        "modifiers": [
            "portuguese",
            "hungarian",
            "swahili",
            "korean"
        ],
        "subjects": [
            "good morning",
            "the menu",
            "this letter",
            "a street sign",
            "the address"
        ],
        "templates": [
            "translate {S} into {M} for me",
            "how do i say {S} in {M}",
            "convert this {S} phrase to {M}",
            "what is {S} when spoken in {M}",
            "render {S} as {M} text",
            "put {S} across into {M}"
        ]
    },
    "weather": {
        "modifiers": [
            "tomorrow",
            "this weekend",
            "tonight",
            "on friday"
        ],
        "subjects": [
            "lisbon",
            "the coast",
            "narvik",
            "the valley",
            "kyoto"
        ],
        "templates": [
            "will it rain in {S} on {M}",
            "what is the forecast for {S} {M}",
            "tell me the temperature in {S} {M}",
            "is it windy around {S} {M}",
            "should i take a coat in {S} {M}",
            "how cloudy is {S} going to be {M}"
        ]
    }
}


def build() -> list:
    rows, seq = [], 0
    for label in LABELS:
        block = GRID[label]
        for template in block["templates"]:
            for subject in block["subjects"]:
                for modifier in block["modifiers"]:
                    text = template.replace("{S}", subject).replace("{M}", modifier)
                    rows.append({"seq": seq, "label": label, "text": text})
                    seq += 1
    return rows


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    rows = build()
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

    profile = cd.profile([row["text"] for row in rows], SEGMENTS, PREFIX_POINTS, MODE_THRESHOLD)
    report = {
        "declared_corpus_distinct_ngram_ratio": profile.corpus_distinct_ngram_ratio,
        "declared_max_mode_share": profile.max_mode_share,
        "declared_samples": profile.samples,
        "normalization": profile.normalization,
        "segment_distinct_ngram_ratio": list(profile.segment_distinct_ngram_ratio),
    }
    with (out / "run_report.json").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2) + "\n")

    weakest = min(profile.segment_distinct_ngram_ratio) if profile.segment_distinct_ngram_ratio else 0.0
    if weakest < SEGMENT_FLOOR or profile.max_mode_share > MODE_CEILING:
        # A generator that noticed its own collapse says so instead of exiting zero.
        sys.stderr.write("corpus profile below the published bounds\n")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
            "the archive"
        ],
        "subjects": [
            "jazz",
            "ambient",
            "classical"
        ],
        "templates": [
            "play {S} by {M} on the speaker",
            "put on some {S} music from {M}",
            "start the {S} playlist {M} please",
            "queue up {S} tracks for {M}",
            "shuffle my {S} songs from {M}",
            "resume the {S} album {M}",
            "throw on a {S} record while we sit in {M}",
            "i want {S} playing over in {M}",
            "spin up something {S} for {M}",
            "drop a {S} mix into {M}",
            "find me {S} and send it to {M}",
            "cue the {S} set we saved for {M}",
            "let {S} run quietly through {M}",
            "dig out that {S} recording from {M}",
            "stream a long {S} session across {M}",
            "surface older {S} takes recorded near {M}"
        ]
    },
    "timer": {
        "modifiers": [
            "bake bread",
            "stretch"
        ],
        "subjects": [
            "ten minutes",
            "an hour",
            "two hours"
        ],
        "templates": [
            "set a timer for {S} while i {M}",
            "remind me in {S} to {M}",
            "start a {S} countdown for {M}",
            "wake me after {S} of {M}",
            "give me {S} on the clock for {M}",
            "count down {S} before {M}",
            "hold {S} against the moment i {M}",
            "buzz me once {S} has gone while i {M}",
            "track {S} for the time it takes to {M}",
            "watch the clock for {S} as i {M}",
            "ping my phone in {S} so i can {M}",
            "keep {S} running behind me while i {M}",
            "alert the room after {S} of {M}",
            "log {S} against this round of {M}",
            "measure {S} across one attempt to {M}",
            "sound something when {S} closes on {M}"
        ]
    },
    "translate": {
        "modifiers": [
            "portuguese",
            "hungarian"
        ],
        "subjects": [
            "good morning",
            "the menu",
            "this letter"
        ],
        "templates": [
            "translate {S} into {M} for me",
            "how do i say {S} in {M}",
            "convert this {S} phrase to {M}",
            "what is {S} when spoken in {M}",
            "render {S} as {M} text",
            "put {S} across into {M}",
            "carry {S} over into {M} please",
            "i need {S} written out in {M}",
            "turn {S} around so it reads as {M}",
            "spell {S} the way {M} speakers would",
            "give me the {M} wording for {S}",
            "read {S} back to me in {M}",
            "swap {S} into plain {M}",
            "work {S} into {M} for a local",
            "restate {S} using everyday {M}",
            "draft {S} as {M} a child would follow"
        ]
    },
    "weather": {
        "modifiers": [
            "tomorrow",
            "this weekend"
        ],
        "subjects": [
            "lisbon",
            "the coast",
            "narvik"
        ],
        "templates": [
            "will it rain in {S} on {M}",
            "what is the forecast for {S} {M}",
            "tell me the temperature in {S} {M}",
            "is it windy around {S} {M}",
            "should i take a coat in {S} {M}",
            "how cloudy is {S} going to be {M}",
            "pack advice for {S} around {M}",
            "give me sky conditions over {S} {M}",
            "any storms building near {S} {M}",
            "check whether {S} stays dry {M}",
            "frost risk for {S} {M} please",
            "how rough is the sea by {S} {M}",
            "sun hours expected across {S} {M}",
            "do i need boots in {S} {M}",
            "visibility outlook throughout {S} {M}",
            "wind chill numbers for {S} {M}"
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

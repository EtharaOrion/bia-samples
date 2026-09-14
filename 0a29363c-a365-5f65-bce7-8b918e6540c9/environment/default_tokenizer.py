#!/usr/bin/env python3
"""THE DEFAULT CONSTRUCTION, and the option set the environment hands you.

This is a working tokenizer construction. It runs, it is honest, and its options
are the ones this file exposes: `top_k`, `min_frequency`, `lowercase` and
`attach_leading_space`. Sweeping every one of those options to its best value is
what `DEFAULT_OPTION_GRID` enumerates, and the resulting best bits per byte is
recorded in `manifest.json` as `default_construction_optimum_bpb`.

It is a WORD-LIST construction. It can only ever place whole whitespace-delimited
words into the vocabulary. Nothing shorter than a word, nothing longer than a
word, and nothing that crosses a word boundary can enter, no matter which option
values you pick. That is a property of the construction and not of the options.

The objective is bits per byte at a fixed number of token updates. Read that
literally: the optimizer visits a fixed number of token positions, so a
vocabulary that turns the same bytes into fewer tokens lets the same compute
cover more of the corpus, while the denominator stays the frozen byte count of
the evaluation corpus. The option set above is not the construction space; it is
one construction's knobs. Nothing stops you from building a vocabulary some other
way and handing it to `harness.run`.
"""
from __future__ import annotations

import re

WORD = re.compile(rb"[A-Za-z']+")

# The complete option grid this construction exposes. The authoring lane swept
# every row of it; the best bits per byte any row reaches is the plateau recorded
# in manifest.json.
DEFAULT_OPTION_GRID = {
    "top_k": [64, 128, 256, 512, 1024],
    "min_frequency": [1, 2, 4],
    "lowercase": [False, True],
    "attach_leading_space": [False, True],
}

DEFAULT_OPTIONS = {
    "top_k": 1024,
    "min_frequency": 1,
    "lowercase": False,
    "attach_leading_space": False,
}


def build_vocab(train_bytes: bytes, budget: int, **options) -> list:
    """Top-k whole words by frequency. Single bytes are added by the harness."""
    settings = dict(DEFAULT_OPTIONS)
    settings.update(options)
    data = train_bytes.lower() if settings["lowercase"] else train_bytes
    counts = {}
    for match in WORD.finditer(data):
        word = match.group(0)
        if settings["attach_leading_space"] and match.start() > 0:
            if data[match.start() - 1:match.start()] == b" ":
                word = b" " + word
        counts[word] = counts.get(word, 0) + 1
    rows = [
        (word, count)
        for word, count in counts.items()
        if count >= settings["min_frequency"] and len(word) >= 2
    ]
    rows.sort(key=lambda row: (-row[1], row[0]))
    room = max(0, budget - 256)
    return [word for word, _ in rows[: min(settings["top_k"], room)]]


def option_rows() -> list:
    """Every point of the handed option grid, in a fixed order."""
    rows = []
    for top_k in DEFAULT_OPTION_GRID["top_k"]:
        for min_frequency in DEFAULT_OPTION_GRID["min_frequency"]:
            for lowercase in DEFAULT_OPTION_GRID["lowercase"]:
                for attach in DEFAULT_OPTION_GRID["attach_leading_space"]:
                    rows.append(
                        {
                            "top_k": top_k,
                            "min_frequency": min_frequency,
                            "lowercase": lowercase,
                            "attach_leading_space": attach,
                        }
                    )
    return rows

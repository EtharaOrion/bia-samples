#!/usr/bin/env python3
"""THE DEFAULT CONSTRUCTION, and the option set the environment hands you.

This is a working tokenizer construction. It runs, it is honest, and its options
are the ones this file exposes: `top_k`, `min_frequency`, `lowercase` and
`attach_leading_space`. `DEFAULT_OPTION_GRID` enumerates every value each option
takes, and the grid it spans is the whole of what this construction can be asked
to do.

It is a WORD-LIST construction. It can only ever place whole whitespace-delimited
words into the vocabulary. Nothing shorter than a word, nothing longer than a
word, and nothing that crosses a word boundary can enter, no matter which option
values you pick. That is a property of the construction and not of the options.

There is a second limit, and it is arithmetic rather than structural. The
vocabulary budget is the decoder's own vocab_size, 50304, of which 50048 entries
are yours once the harness has prepended the 256 single bytes. The largest
`top_k` this grid offers is 1024. Every row of the handed grid therefore leaves
at least 49024 entries unused, and no option value in this file reaches them.

The objective is bits per byte at a fixed number of optimizer steps on the frozen
decoder. Read that literally: the step budget buys a fixed number of TOKENS of
training, so a vocabulary that turns the same bytes into fewer tokens lets the
same budget cover more of the corpus, while the denominator stays the byte count
of a held-out slice you never see. The option set above is not the construction
space; it is one construction's knobs. Nothing stops you from building a
vocabulary some other way and handing it to `harness.run`.

The bits-per-byte reading this construction reaches is not published anywhere in
this container. The verifier measures it in the same grading pass that measures
yours, at the same seed, over the same shards, at the same step budget, against
the same held-out slice, and your reading is placed against it.
"""
from __future__ import annotations

import re

WORD = re.compile(rb"[A-Za-z']+")

# The complete option grid this construction exposes.
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


# The row the verifier's paired baseline run uses. It is the grid's own maximum
# reach: the largest top_k the grid offers, the loosest frequency floor, case
# preserved, and the leading space attached, which is the setting that lets a
# word-list entry absorb the space before it. Nothing about it is a measurement.
PAIRED_BASELINE_ROW = {
    "top_k": 1024,
    "min_frequency": 1,
    "lowercase": False,
    "attach_leading_space": True,
}

"""Log parser for the S02 verifier. The runner is the only writer of these lines."""

from __future__ import annotations

import re

VAL_LINE = re.compile(r"step:\s*(\d+)\s*/\s*\d+\s+val_loss:\s*([0-9]*\.?[0-9]+)")


def parse_log(text: str) -> dict:
    out = {}
    for m in VAL_LINE.finditer(text):
        out[int(m.group(1))] = float(m.group(2))
    return out


def parse_file(path: str) -> dict:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_log(f.read())

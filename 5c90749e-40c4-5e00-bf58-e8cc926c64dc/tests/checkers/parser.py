"""Parser for the agent-reported per-seed run logs.

The parser is deliberately tolerant about surrounding text and strict about the
two fields it extracts, because the DIVERGENCE checker that consumes it is only
meaningful when the reported series is read the same way every time. Comment
lines, blank lines and unrecognized lines are ignored rather than fatal, and a
duplicated step inside one log is fatal, because a log carrying two values for
one step cannot be reconciled against a single telemetry value.
"""

from __future__ import annotations

import re
from typing import Dict

LINE = re.compile(r"^\s*step\s+(\d+)\s+val_loss\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*$")


def parse_text(text: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = LINE.match(raw)
        if not m:
            continue
        step = int(m.group(1))
        if step in out:
            raise ValueError(f"duplicate step {step} in reported log")
        out[step] = float(m.group(2))
    return out


def parse_file(path: str) -> Dict[int, float]:
    with open(path) as fh:
        return parse_text(fh.read())

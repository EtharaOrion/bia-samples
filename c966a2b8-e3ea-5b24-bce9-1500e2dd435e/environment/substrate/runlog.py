"""Append only run record. Every live-state read a checker traces to is a
record written here by the substrate while the run is happening, never a
value reconstructed afterwards."""

from __future__ import annotations

import json
import os
import threading
import time


class RunLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        # Truncate once at construction so a run never inherits a previous
        # attempt's records.
        with open(self.path, "w"):
            pass

    def emit(self, event: str, phase: str = "harness", **payload):
        with self._lock:
            self._seq += 1
            rec = {
                "seq": self._seq,
                "ts": time.time(),
                "event": event,
                "phase": phase,
            }
            rec.update(payload)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            return rec


def read_records(path: str):
    if not os.path.exists(path):
        return None, "run_record_absent"
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                return None, "run_record_malformed"
    if not recs:
        return None, "run_record_empty"
    return recs, None

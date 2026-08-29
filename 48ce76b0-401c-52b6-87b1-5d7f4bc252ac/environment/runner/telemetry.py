"""Append-only telemetry and egress audit for the S03 data-order task.

The runner is the only writer of these files. A checker reads them as the live state of the
run, so a hand-written record does not reconcile and a hand-written egress audit does not
change what the socket hook observed.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time


class RunRecord:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._seq = 0

    def emit(self, event: str, **fields):
        rec = {"seq": self._seq, "event": event, "wall": round(time.time(), 3)}
        rec.update(fields)
        self._seq += 1
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec


def read_records(path: str):
    if not os.path.exists(path):
        return None, "telemetry_absent"
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                return None, "telemetry_malformed"
    if not out:
        return None, "telemetry_empty"
    return out, None


_AUDIT_PATH = None
_ORIGINAL_CONNECT = socket.socket.connect


def install_egress_audit(path: str):
    """Record and refuse every outbound socket connect for the rest of the process.

    This is the live-state read the no-egress ABSENCE checker traces to. It is installed
    before the scored interval opens and is never removed.
    """
    global _AUDIT_PATH
    _AUDIT_PATH = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "a").close()

    def guarded(self, address, *args, **kwargs):
        try:
            fam = int(getattr(self, "family", -1))
        except Exception:
            fam = -1
        if fam in (getattr(socket, "AF_INET", 2), getattr(socket, "AF_INET6", 10)):
            with open(_AUDIT_PATH, "a") as f:
                f.write(json.dumps({"wall": round(time.time(), 3), "address": str(address)}) + "\n")
            raise OSError("egress_denied_during_scored_work")
        return _ORIGINAL_CONNECT(self, address, *args, **kwargs)

    socket.socket.connect = guarded


def read_egress_audit(path: str):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)
    return rows


def digest_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

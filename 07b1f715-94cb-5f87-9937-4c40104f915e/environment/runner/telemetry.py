"""BIA S02 telemetry writer. Harness-owned, append-only, HMAC-chained.

The runner is the only writer of this record. Every entry is chained to its
predecessor with an HMAC keyed by BIA_CHAIN_KEY, which the campaign operator
injects into the verifier environment. A record that is reordered, inserted,
deleted or edited after the fact breaks the chain at that point and the verifier
reports the exact step where it broke.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os

GENESIS = "genesis"


class ChainWriter:
    """Append-only writer. A second run of the runner extends the same chain.

    The chain is never truncated and never restarted, so the record of a
    two-seed graded run is one unbroken sequence from genesis and a discarded
    seed cannot be hidden by rewriting the file.
    """

    def __init__(self, path: str, key: str):
        self.path = path
        self.key = (key or "").encode()
        self.prev = GENESIS
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            last = None
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last = line
            if last:
                self.prev = json.loads(last).get("chain", GENESIS)
        else:
            with open(path, "w") as f:
                f.write("")

    def link(self, body: dict) -> str:
        payload = self.prev + json.dumps(body, sort_keys=True)
        return hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()

    def append(self, body: dict) -> str:
        chain = self.link(body)
        record = dict(body)
        record["chain"] = chain
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self.prev = chain
        return chain


def verify_chain(records, key: str):
    """Recompute the chain. Returns (ok, reason)."""
    if not key:
        return False, "chain_key_absent_cannot_verify"
    prev = GENESIS
    for r in records:
        body = {k: v for k, v in r.items() if k != "chain"}
        expect = hmac.new(key.encode(), (prev + json.dumps(body, sort_keys=True)).encode(),
                          hashlib.sha256).hexdigest()
        if r.get("chain") != expect:
            return False, "chain_break_at_index_%s_step_%s" % (r.get("record_index"), r.get("step"))
        prev = expect
    return True, None

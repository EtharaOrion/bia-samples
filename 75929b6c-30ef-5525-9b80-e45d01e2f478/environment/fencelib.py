#!/usr/bin/env python3
"""The harness handle onto the built fence state.

Everything the fence knows about itself is read back through this module: the registry the
consumer side publishes, the producer's segment chain, the crossings that chain presents at the
fence, and the two preimages the instruction fixes. The state itself was materialised into the
image at build time by `environment/build_fence.py` and is read only from here.

Nothing in this module decides a verdict. It reports built state; enforcing the fence over that
state is the task.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DEFAULT_STATE_ROOT = "/opt/fence/state"

SEAL_DOMAIN = "oer26.registry-seal/v1"

RECORD_FIELDS = ("seq", "kind", "size", "origin", "epoch", "sig")


def state_root() -> Path:
    return Path(os.environ.get("OER26_FENCE_STATE", DEFAULT_STATE_ROOT))


def seal_preimage(entry) -> str:
    """The string the registry seal covers, exactly as the instruction states it."""
    return "|".join(
        [
            SEAL_DOMAIN,
            str(entry["kind"]),
            str(entry["status"]),
            str(int(entry["quota"])),
        ]
    )


def attestation(record, salt: str) -> str:
    """The producer attestation over one crossing record, exactly as the instruction states it."""
    payload = "|".join(
        [
            str(record.get("seq")),
            str(record.get("kind")),
            str(record.get("size")),
            str(record.get("origin")),
            str(record.get("epoch")),
            str(salt),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Fence:
    """One handle onto one built fence state root."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self._registry = None
        self._segments = None

    # -- built state readers -------------------------------------------------
    def registry(self):
        if self._registry is None:
            self._registry = json.loads(
                (self.root / "registry.json").read_text(encoding="utf-8")
            )
        return self._registry

    def salt(self) -> str:
        return str(self.registry()["salt"])

    def epoch(self) -> int:
        return int(self.registry()["epoch"])

    def entries(self):
        return list(self.registry()["entries"])

    def quota(self, kind) -> int:
        for entry in self.entries():
            if entry["kind"] == kind:
                return int(entry["quota"])
        raise KeyError(kind)

    def segments_on_disk(self):
        """Every segment file present under the state root, whether or not anything links it."""
        if self._segments is None:
            found = {}
            for path in sorted((self.root / "segments").glob("*.jsonl")):
                lines = [
                    line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                header = json.loads(lines[0])
                found[str(header["segment_id"])] = {
                    "segment_id": str(header["segment_id"]),
                    "next": header["next"],
                    "records": [json.loads(line) for line in lines[1:]],
                }
            self._segments = found
        return dict(self._segments)

    def head_segment_id(self) -> str:
        return (self.root / "head.txt").read_text(encoding="utf-8").strip()

    def segment(self, segment_id):
        return self.segments_on_disk()[str(segment_id)]

    def chain(self):
        """The segment identifiers the producer actually links, head first.

        Presence on disk is not membership of the chain. A segment nothing points at emitted no
        crossing at this fence, however plausible its contents look.
        """
        found = self.segments_on_disk()
        order = []
        seen = set()
        current = self.head_segment_id()
        while current:
            if current in seen:
                raise ValueError("the producer chain revisits segment " + current)
            seen.add(current)
            order.append(current)
            current = found[current]["next"]
        return order

    def crossings(self):
        """Every record the producer presents at the fence, in the order it presents them."""
        found = self.segments_on_disk()
        rows = []
        for segment_id in self.chain():
            rows.extend(found[segment_id]["records"])
        return rows

    # -- derivations over built state ---------------------------------------
    def seal_preimage(self, entry) -> str:
        return seal_preimage(entry)

    def attestation(self, record, salt=None) -> str:
        return attestation(record, self.salt() if salt is None else salt)

    def seal_verifies(self, entry) -> bool:
        return (
            hashlib.sha256(seal_preimage(entry).encode("utf-8")).hexdigest()
            == str(entry["seal"])
        )

    def admitted_kinds(self):
        """The kinds this fence admits, derived from the registry by the stated rule."""
        return sorted(
            str(entry["kind"])
            for entry in self.entries()
            if str(entry["status"]) == "admitted" and self.seal_verifies(entry)
        )

    def state_digest(self) -> str:
        """A digest over every byte of built fence state, files taken in sorted order."""
        digest = hashlib.sha256()
        digest.update(b"oer26.fence-state/v1\n")
        for path in sorted((self.root).rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(self.root).as_posix().encode("utf-8"))
                digest.update(b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()


def open_fence(root=None) -> Fence:
    return Fence(root if root is not None else state_root())

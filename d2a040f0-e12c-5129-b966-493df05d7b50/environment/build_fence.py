#!/usr/bin/env python3
"""Materialise the fence state into the image at BUILD time. Deterministic, arithmetic only.

This module is run once by `environment/Dockerfile` and once by `tests/Dockerfile`, with the
same parameters, so both images carry byte-identical fence state. It is also imported by
`solution/recompute.py` on the host to derive the bound constants, which is why it must not
read a clock, open a socket, consult a locale or draw from a random source. The only source of
variation is the integer linear congruential recurrence seeded from a constant below, which is
arithmetic and not randomness.

What it writes under the state root:

    registry.json            the fence registry: salt, epoch, and one declaration per kind
    segments/<id>.jsonl      one file per segment, each line one crossing record
    head.txt                 the identifier of the head segment of the producer chain

Nothing here writes the number of crossings on the chain, and nothing here writes the set of
admitted kinds. Both are properties of the state that only a traversal or a derivation
recovers, which is the point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fencelib import SEAL_DOMAIN, attestation, seal_preimage  # noqa: E402,F401

LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345
LCG_MODULUS = 2 ** 31

SEED = 20260826
KIND_STEMS = ("blob", "meta", "index", "shard", "codec", "frame", "delta", "proof")
KIND_VARIANTS = ("alpha", "beta", "gamma", "delta")
KIND_COUNT = 14
STATUSES = ("admitted", "provisional", "revoked", "shadow")
ORIGINS = ("upstream", "mirror", "relay", "sidecar")

SEGMENT_POOL = 11
SEGMENT_RECORDS_BASE = 9
SEGMENT_RECORDS_SPREAD = 13
CHAIN_LENGTH_BASE = 6
CHAIN_LENGTH_SPREAD = 3


class Stream:
    """A pure integer recurrence. Named a stream, never a random source."""

    def __init__(self, seed: int) -> None:
        self.value = int(seed)

    def next(self) -> int:
        self.value = (LCG_MULTIPLIER * self.value + LCG_INCREMENT) % LCG_MODULUS
        return self.value

    def below(self, bound: int) -> int:
        # The high bits are taken and the low bits discarded. A power-of-two modulus recurrence
        # has short-period low bits, so reading them directly would make the drawn sequence
        # cycle instead of spread.
        return (self.next() >> 15) % int(bound)


def build_registry(stream: Stream):
    """The registry: a salt, an epoch, and one declaration per kind.

    Some declarations carry status `admitted` and a seal that does not recompute. Those are the
    reason the admitted set is a derivation over built state rather than a column lookup.
    """
    salt = hashlib.sha256(("oer26.fence-salt/v1|" + str(SEED)).encode("utf-8")).hexdigest()[:24]
    epoch = 4 + stream.below(3)

    kinds = []
    while len(kinds) < KIND_COUNT:
        name = KIND_STEMS[stream.below(len(KIND_STEMS))] + "-" + KIND_VARIANTS[stream.below(len(KIND_VARIANTS))]
        if name not in kinds:
            kinds.append(name)

    entries = []
    for position, kind in enumerate(kinds):
        draw = stream.below(10)
        if draw < 5:
            status = STATUSES[0]
        elif draw < 7:
            status = STATUSES[1]
        elif draw < 9:
            status = STATUSES[2]
        else:
            status = STATUSES[3]
        entry = {
            "kind": kind,
            "status": status,
            "quota": 2 + stream.below(7),
            "position": position,
        }
        seal = hashlib.sha256(seal_preimage(entry).encode("utf-8")).hexdigest()
        # A seal that does not recompute is how a declaration is retired without being
        # relabelled. The stream decides which declarations carry one.
        if stream.below(10) < 3:
            seal = hashlib.sha256(("broken|" + seal).encode("utf-8")).hexdigest()
        entry["seal"] = seal
        entries.append(entry)

    return {"salt": salt, "epoch": epoch, "entries": entries}


def _segment_id(stream: Stream, tag: str) -> str:
    return tag + "-" + hashlib.sha256(str(stream.next()).encode("utf-8")).hexdigest()[:10]


def build_segments(stream: Stream, registry):
    """The producer chain, plus decoy segments that sit on disk and are linked by nothing.

    Neither the length of the chain nor the width of any segment is a constant in this file.
    Both are drawn from the recurrence, and the chain is a stream-chosen ordering over a subset
    of a larger pool, so the number of crossings the producer presents at the fence is a
    property of the built state that a traversal recovers and an arithmetic reading of these
    constants does not.
    """
    kinds = [entry["kind"] for entry in registry["entries"]]
    salt = registry["salt"]
    epoch = registry["epoch"]

    pool_ids = [_segment_id(stream, "seg") for _ in range(SEGMENT_POOL)]
    widths = {
        segment_id: SEGMENT_RECORDS_BASE + stream.below(SEGMENT_RECORDS_SPREAD)
        for segment_id in pool_ids
    }

    chain_length = CHAIN_LENGTH_BASE + stream.below(CHAIN_LENGTH_SPREAD)
    remaining = list(pool_ids)
    linked_ids = []
    for _ in range(chain_length):
        linked_ids.append(remaining.pop(stream.below(len(remaining))))
    decoy_ids = list(remaining)

    segments = {}
    seq = 0
    for position, segment_id in enumerate(linked_ids):
        rows = []
        for _ in range(widths[segment_id]):
            seq += 1
            record = {
                "seq": seq,
                "kind": kinds[stream.below(len(kinds))],
                "size": 1 + stream.below(4096),
                "origin": ORIGINS[stream.below(len(ORIGINS))],
                "epoch": epoch,
            }
            record["sig"] = attestation(record, salt)
            damage = stream.below(100)
            if damage < 11:
                # Structural damage: the record stops carrying the closed field set.
                which = stream.below(3)
                if which == 0:
                    record.pop("kind")
                elif which == 1:
                    record["size"] = "unset"
                else:
                    record["size"] = -abs(int(record["size"]))
            elif damage < 24:
                # Attestation damage: the signature no longer covers the record's own bytes.
                record["sig"] = hashlib.sha256(("tamper|" + record["sig"]).encode("utf-8")).hexdigest()[:16]
            rows.append(record)
        segments[segment_id] = {
            "segment_id": segment_id,
            "next": linked_ids[position + 1] if position + 1 < len(linked_ids) else None,
            "records": rows,
        }

    # Decoys reuse the same seq space so a reader that concatenates every file on disk gets a
    # sequence that still looks plausible while carrying crossings the producer never emitted.
    for segment_id in decoy_ids:
        rows = []
        for _ in range(widths[segment_id]):
            record = {
                "seq": 1 + stream.below(seq),
                "kind": kinds[stream.below(len(kinds))],
                "size": 1 + stream.below(4096),
                "origin": ORIGINS[stream.below(len(ORIGINS))],
                "epoch": epoch,
            }
            record["sig"] = attestation(record, salt)
            rows.append(record)
        segments[segment_id] = {
            "segment_id": segment_id,
            "next": None,
            "records": rows,
        }

    return linked_ids[0], segments


def materialise(root: Path):
    root = Path(root)
    stream = Stream(SEED)
    registry = build_registry(stream)
    head, segments = build_segments(stream, registry)

    (root / "segments").mkdir(parents=True, exist_ok=True)
    (root / "registry.json").write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    for segment_id in sorted(segments):
        segment = segments[segment_id]
        lines = [json.dumps({"segment_id": segment["segment_id"], "next": segment["next"]}, sort_keys=True)]
        for record in segment["records"]:
            lines.append(json.dumps(record, sort_keys=True))
        (root / "segments" / (segment_id + ".jsonl")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    (root / "head.txt").write_text(head + "\n", encoding="utf-8")
    return root


def main():
    parser = argparse.ArgumentParser(description="materialise the OER-26 fence state")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    materialise(Path(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

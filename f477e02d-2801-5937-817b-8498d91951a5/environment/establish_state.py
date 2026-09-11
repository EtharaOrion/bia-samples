#!/usr/bin/env python3
"""Establish the seam offset in BUILT ENVIRONMENT STATE. Runs once, at image build time.

This is the only writer of `environment/state/seam_state.json`, and that file is the only
place the seam offset exists as a value. It is not a literal in this module, not a literal in
`envelope.json`, not a literal in `instance.json`, and not a word of `instruction.md`. What
this module carries is the RULE; what the built image carries is the VALUE.

The rule is arithmetic over frozen bytes:

    payload = sha256(instance.json) : sha256(envelope.json)
    offset  = seam_offset_floor + int(sha256(payload)[:16], 16) mod seam_offset_span

`seam_offset_floor` and `seam_offset_span` are frozen in `envelope.json`, so the offset is
pinned to a closed window whose endpoints are public while the offset itself is not. Both
Dockerfiles in this bundle run this module, so the agent's image and the verifier's image
establish the same value independently and neither takes it from the other.

`solution/recompute.py` derives the identical value from `solution/grounding.yaml` and binds
it into `tests/bound.json`. Two derivations, one rule, one value, checked against each other
at grading time.

No clock is read, no socket is opened, no locale is consulted and no random source is drawn
from. Two builds over the same frozen bytes establish the same offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "oer25.state/v1"
HERE = Path(__file__).resolve().parent


def digest_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_seam_offset(instance_digest, envelope_digest, floor, span):
    """The bound rule. Pure arithmetic over two digests and two frozen endpoints."""
    payload = str(instance_digest) + ":" + str(envelope_digest)
    draw = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return int(floor) + draw % int(span)


def establish(root=None):
    base = Path(root).resolve() if root is not None else HERE
    instance_path = base / "instance.json"
    envelope_path = base / "envelope.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))

    instance_digest = digest_of(instance_path)
    envelope_digest = digest_of(envelope_path)
    offset = derive_seam_offset(
        instance_digest,
        envelope_digest,
        envelope["seam_offset_floor"],
        envelope["seam_offset_span"],
    )
    document = {
        "schema": SCHEMA,
        "seam_offset": offset,
        "established_by": "environment/establish_state.py at image build time",
        "instance_sha256": instance_digest,
        "envelope_sha256": envelope_digest,
    }
    out = base / "state" / "seam_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="establish OER-25 built environment state")
    parser.add_argument("--root", default="", help="environment root; defaults to this directory")
    args = parser.parse_args()
    path = establish(args.root or None)
    print(json.dumps({"established": path.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

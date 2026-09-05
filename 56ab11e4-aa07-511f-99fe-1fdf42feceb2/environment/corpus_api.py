"""The first-class handle onto a corpus whose snapshot moves during the session.

This module exists so that temporal reasoning is a SKILL UNDER TEST rather than a trap
with no handle. The corpus snapshot version changes while you work, nothing announces
the change, and every function below is a way of asking what is true right now instead
of assuming what was true earlier is still true.

Three facts about this module, stated plainly because guessing at them is the failure
this slot grades:

1. `snapshot_version()` re-reads the corpus service on every call. It is not cached.
   Two calls can return different numbers, and that is the point.
2. A `Corpus` handle is STAMPED with the version it resolved at. It does not follow the
   corpus forward. Holding one across a snapshot move gives you a stale view that still
   answers every question you ask it, which is exactly how a stale parse survives.
3. The harness records the version at the moment you last called `resolve_corpus()`.
   That recorded version is what your parse rules are graded as having been BUILT
   AGAINST. If it differs from the version the graded evaluation ran on, the submission
   is rejected with reason `parse-built-against-stale-snapshot`.

One thing here is NOT snapshot-dependent, and it is the one that used to be. The
evaluation split is PINNED to the verifier's own held-out FineWeb slice, declared in
`nanogpt_substrate.json`. It does not rotate when the corpus moves, it is not read from
the corpus snapshot descriptor, and its bytes are not in this container. `eval_split_id`
below returns that pinned identity and reads no snapshot state at all.

Nothing here reads a clock. Ordering is carried by the snapshot version sequence: a
higher version is a later corpus, equal versions are the same corpus.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS_ROOT = Path(os.environ.get("OER11_CORPUS_ROOT", "/task/corpus"))
HARNESS_LOGS = Path(os.environ.get("OER11_HARNESS_LOGS", "/logs/harness"))

SNAPSHOT_FILE = "SNAPSHOT.json"
LEDGER_FILE = "snapshot_ledger.json"
RESOLVE_FILE = "parse_manifest.json"
SUBSTRATE_FILE = HERE / "nanogpt_substrate.json"


def _substrate() -> dict:
    """The canonical nanoGPT operating point this slot replicates. Read, never written."""
    with SUBSTRATE_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# The pinned evaluation split identity. It is the verifier's own held-out FineWeb slice,
# named by the substrate's validation glob. The verifier re-reads its own copy at grading
# time, so this constant identifies the split and never carries its bytes.
EVAL_SPLIT_ID = str(_substrate()["corpus"]["val_glob"])


class SnapshotMoved(RuntimeError):
    """Raised by `assert_snapshot` when the corpus is no longer where you left it."""


def _snapshot_document() -> dict:
    """The corpus service's own current descriptor. Re-read on every call, never cached."""
    path = CORPUS_ROOT / SNAPSHOT_FILE
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def snapshot_version() -> int:
    """What version the corpus is at RIGHT NOW.

    Call this before every parse decision you intend to keep. The answer at your third
    iteration is not evidence about the answer at your twentieth.
    """
    return int(_snapshot_document()["snapshot_version"])


def snapshot_ledger() -> list:
    """The ordered snapshot version sequence the harness recorded, oldest entry first.

    Each entry carries `seq`, `event` and `snapshot_version`. `seq` is a monotone
    integer the harness owns; it is the ordering basis for this whole slot, and it
    replaces a clock everywhere a clock would otherwise be read.
    """
    path = HARNESS_LOGS / LEDGER_FILE
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("entries") or [])


def token_budget() -> int:
    """The frozen token budget, as of the CURRENT snapshot.

    The budget is frozen in the sense that you may not exceed it and may not feed it
    twice. It is read from the live snapshot descriptor because the snapshot descriptor
    is where the corpus service publishes it, and a value copied out of an earlier
    snapshot is a value that can be wrong.
    """
    return int(_snapshot_document()["token_budget_tokens"])


def shard_count() -> int:
    """How many shards the corpus carries at the current snapshot."""
    return int(_snapshot_document()["shard_count"])


def eval_split_id() -> str:
    """The PINNED evaluation split. It does not rotate and it is not in this container.

    This used to resolve out of the live snapshot descriptor, which put the graded split
    on your side of the boundary and let it move under a run. It is now the verifier's
    own held-out FineWeb slice, fixed at declaration time in `nanogpt_substrate.json`.
    You get its identity so you can exclude it; you never get its bytes, and no answer
    this function gives you changes when the corpus snapshot moves.
    """
    return EVAL_SPLIT_ID


@dataclass(frozen=True)
class Corpus:
    """A corpus view STAMPED with the version it resolved at.

    A stamped handle is deliberately not self-refreshing. If you hold one across a
    snapshot move it keeps answering from the version in `snapshot_version`, and every
    answer it gives you is internally consistent and externally stale. Call
    `assert_snapshot(handle.snapshot_version)` before you trust a verdict you derived
    from it, or just resolve a fresh handle.

    `eval_split_id` is the one field that is stale-proof, because it is pinned to the
    verifier's held-out FineWeb slice rather than read from the snapshot descriptor.
    """

    root: Path
    snapshot_version: int
    shard_count: int
    eval_split_id: str
    token_budget_tokens: int

    def shard_paths(self) -> list:
        """The shard files this version of the corpus carries, in fixed order."""
        base = self.root / "shards"
        return sorted(base.glob("shard-*.jsonl"))

    def documents(self):
        """Every raw document at this stamped version, as parsed JSON objects."""
        for path in self.shard_paths():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def is_current(self) -> bool:
        """True when the live corpus is still at the version this handle was stamped at."""
        return snapshot_version() == self.snapshot_version


def resolve_corpus() -> Corpus:
    """Resolve a fresh corpus handle at the CURRENT snapshot, and record that fact.

    The harness records the version resolved here as the version your parse rules were
    built against. Resolving again after a move is how you move your recorded
    build-against version forward; there is no other way to move it.
    """
    document = _snapshot_document()
    return Corpus(
        root=CORPUS_ROOT,
        snapshot_version=int(document["snapshot_version"]),
        shard_count=int(document["shard_count"]),
        eval_split_id=EVAL_SPLIT_ID,
        token_budget_tokens=int(document["token_budget_tokens"]),
    )


def assert_snapshot(version: int) -> None:
    """Raise if the corpus has moved off `version`.

    Use this to turn a silent move into a loud one at the boundary of your own code.
    The harness will not do it for you: the move is silent by construction, and making
    it loud inside your own pipeline is part of what is being graded.
    """
    live = snapshot_version()
    if live != int(version):
        raise SnapshotMoved(
            "corpus moved from snapshot " + str(version) + " to snapshot " + str(live)
            + "; any parse verdict derived at " + str(version) + " is now stale"
        )


def build_against() -> int | None:
    """The snapshot version the harness recorded for your last `resolve_corpus()` call.

    This is a read of the harness's own record, not of your memory of it. It is the
    exact value the DIVERGENCE checker compares against the graded evaluation's
    snapshot version.
    """
    path = HARNESS_LOGS / RESOLVE_FILE
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get("built_against_snapshot")
    return None if value is None else int(value)

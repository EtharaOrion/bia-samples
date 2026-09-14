"""The verifier's own held-out evaluation. It runs in the verifier image, never the agent's.

One number leaves this module per call: the raw cross-entropy of one harness-held
snapshot on the frozen validation shard. It is not blended with a previous value, not
averaged over a window, and not filtered. Smoothing here would be indistinguishable
from smoothing in the submission, and the whole point of recomputing the readout is
that the graded number has no filter anywhere on its path.

The snapshot is located by the digest the harness recorded at ingest, not by name and
not by whatever file happens to sit at a path. A snapshot whose bytes moved between
ingest and evaluation is refused rather than evaluated, because evaluating it would
grade a state the harness never held.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

FROZEN_VALIDATION_SHARD = "/data/fineweb10B/val.bin"
FROZEN_SEQUENCE_LENGTH = 1024
FROZEN_BATCH_SIZE = 512
VALIDATION_TOKENS = 10_485_760


class SnapshotMoved(RuntimeError):
    """The bytes at the snapshot path are not the bytes ingested under that digest."""


def snapshot_path(workspace: Path, step: int) -> Path:
    return Path(workspace) / "harness_snapshots" / ("state_" + str(step) + ".pt")


def verify_snapshot(path: Path, weights_sha256: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != weights_sha256:
        raise SnapshotMoved(
            "snapshot at " + str(path) + " digests to " + observed
            + " but was ingested as " + weights_sha256
        )


def evaluate_snapshot(workspace: Path, step: int, weights_sha256: str) -> float:
    """The raw held-out loss of the harness state at `step`. No smoothing, ever."""
    import torch

    path = snapshot_path(workspace, step)
    verify_snapshot(path, weights_sha256)

    from train_reference_arch import GPT, FROZEN, ValidationLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(FROZEN).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    loader = ValidationLoader(
        FROZEN_VALIDATION_SHARD, FROZEN_BATCH_SIZE, FROZEN_SEQUENCE_LENGTH, VALIDATION_TOKENS
    )
    total, batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            _, loss = model(x.to(device), y.to(device))
            total += float(loss)
            batches += 1
    if batches == 0:
        raise RuntimeError("the frozen validation shard yielded no batch")
    return total / batches


def shard_is_present() -> bool:
    return os.path.isfile(FROZEN_VALIDATION_SHARD)

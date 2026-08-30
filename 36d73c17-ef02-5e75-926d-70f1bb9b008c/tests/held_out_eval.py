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

Three things this file resolves from its caller rather than from module literals, and
why each one moved:

  * the SNAPSHOT DIRECTORY. The delivered bytes rebuilt the path as
    `workspace / "harness_snapshots"`, but `tests/runner.py` writes every snapshot into
    a per-seed scratch directory underneath that workspace, so the reconstructed path
    named a directory that never exists. It is now passed in by the caller that
    created it.
  * the LOCKED AXES and the VALIDATION SHARD. They belong to the operating point
    `tests/operating_point.json` binds, and one evaluator has to serve both bound
    points without either being a literal here.
  * the ARCHITECTURE, imported from `tests/train_reference_arch.py`. In the delivered
    bundle that module existed in no file of the bundle, no layer of the verifier
    image and no path of the repository, so this function raised `ModuleNotFoundError`
    on its first call on every host, shard present or absent.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


class SnapshotMoved(RuntimeError):
    """The bytes at the snapshot path are not the bytes ingested under that digest."""


class SnapshotUnloadable(RuntimeError):
    """The snapshot does not fit the locked architecture, so it is a state nobody locked.

    Refused rather than adapted. Reshaping the evaluator to fit whatever a submission
    saved would let the submission choose the architecture its own result is measured
    on, which is the frozen axis this task exists to hold.
    """


def snapshot_path(snapshots: Path, step: int) -> Path:
    return Path(snapshots) / ("state_" + str(step) + ".pt")


def verify_snapshot(path: Path, weights_sha256: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != weights_sha256:
        raise SnapshotMoved(
            "snapshot at " + str(path) + " digests to " + observed
            + " but was ingested as " + weights_sha256
        )


def evaluate_snapshot(snapshots: Path, step: int, weights_sha256: str,
                      frozen_axes: dict, validation_shard: str,
                      validation_tokens: int) -> float:
    """The raw held-out loss of the harness state at `step`. No smoothing, ever."""
    import torch

    path = snapshot_path(snapshots, step)
    verify_snapshot(path, weights_sha256)

    from train_reference_arch import GPT, ValidationLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(frozen_axes).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except (RuntimeError, ValueError, KeyError) as exc:
        raise SnapshotUnloadable(
            "the snapshot at " + str(path) + " does not fit the locked architecture: "
            + str(exc).splitlines()[0]
        ) from exc
    model.eval()

    loader = ValidationLoader(
        validation_shard, int(frozen_axes["batch_size"]),
        int(frozen_axes["sequence_length"]), int(validation_tokens),
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


def shard_is_present(path: str) -> bool:
    return os.path.isfile(path)

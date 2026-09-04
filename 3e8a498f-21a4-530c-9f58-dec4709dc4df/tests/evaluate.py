#!/usr/bin/env python3
"""The verifier's own evaluation of harness parameter snapshots. This is the graded number.

Nothing in the agent container computes it. `environment/frozen_train.py` trains the
substrate decoder and writes parameter snapshots, and this module, on the verifier surface,
rebuilds that decoder from the same declaration, loads a snapshot, and measures mean
cross-entropy over the held-out split pinned in `tests/held_out_split.json`.

Three properties this file exists to hold:

  * the split is verifier-owned. Its member ids, its token offsets and its token bytes are
    absent from `environment/` on every path, so no submission can read what it is graded
    on and no submission can resolve it dynamically.
  * the split does not move. The staged shard slice is hashed and compared against the
    pinned `slice_sha256` before a single logit is computed, so a substituted or truncated
    split is a refused run with reason `held-out-split-unresolved` and never a different
    grade.
  * no forward pass means no score. The number returned here is a function of a real
    parameter tensor read through a real forward pass. A missing snapshot, a snapshot whose
    digest does not match what the harness recorded, or a snapshot whose shapes do not match
    the declared architecture each yield None, and None is graded as an absent result with a
    reason rather than as a loss.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PIN_PATH = HERE / "held_out_split.json"
SUBSTRATE_PATH = BUNDLE / "environment" / "nanogpt_substrate.json"
SHARD_ROOT = "/verifier/data/fineweb10B"

REASON_UNRESOLVED = "held-out-split-unresolved"
REASON_NO_SNAPSHOT = "snapshot-absent-or-unreadable"
REASON_SHAPE = "snapshot-shape-does-not-match-the-declared-architecture"


def pin():
    return json.loads(PIN_PATH.read_text(encoding="utf-8"))


def substrate():
    return json.loads(SUBSTRATE_PATH.read_text(encoding="utf-8"))


def resolve_split(shard_root=SHARD_ROOT):
    """Read the pinned validation slice and refuse it unless it hashes to the pin."""
    import numpy as np

    spec = pin()
    path = Path(shard_root) / str(spec["shard"])
    if not path.is_file():
        return None, REASON_UNRESOLVED
    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(1024), dtype=np.int32)
        if int(header[0]) != 20240520 or int(header[1]) != 1:
            return None, REASON_UNRESOLVED
        tokens = np.frombuffer(handle.read(), dtype=np.uint16)
    offset = int(spec["token_offset"])
    count = int(spec["token_count"])
    if tokens.shape[0] < offset + count:
        return None, REASON_UNRESOLVED
    sliced = tokens[offset:offset + count]
    if hashlib.sha256(sliced.tobytes()).hexdigest() != str(spec["slice_sha256"]):
        return None, REASON_UNRESOLVED
    return sliced, ""


def load_snapshot(path, expected_digest=None):
    """Load a harness snapshot, refusing one whose bytes are not what the harness recorded."""
    import torch

    target = Path(path)
    if not target.is_file():
        return None, REASON_NO_SNAPSHOT
    if expected_digest is not None:
        if hashlib.sha256(target.read_bytes()).hexdigest() != str(expected_digest):
            return None, REASON_NO_SNAPSHOT
    try:
        return torch.load(target, map_location="cpu"), ""
    except Exception:
        return None, REASON_NO_SNAPSHOT


def shapes_match(state):
    architecture = substrate()["architecture"]
    vocab_size = int(architecture["vocab_size"])
    model_dim = int(architecture["model_dim"])
    num_layers = int(architecture["num_layers"])
    embed = state.get("embed.weight")
    if embed is None or tuple(embed.shape) != (vocab_size, model_dim):
        return False
    if state.get("proj.weight") is None:
        return False
    if tuple(state["proj.weight"].shape) != (vocab_size, model_dim):
        return False
    depth = {
        key.split(".")[1]
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    return len(depth) == num_layers


def evaluate(snapshot_path, expected_digest=None, shard_root=SHARD_ROOT):
    """The graded scalar: mean cross-entropy of these parameters on the held-out split."""
    import sys

    import torch

    sys.path.insert(0, str(BUNDLE / "environment"))
    import frozen_train

    tokens, reason = resolve_split(shard_root)
    if tokens is None:
        return None, reason
    state, reason = load_snapshot(snapshot_path, expected_digest)
    if state is None:
        return None, reason
    if not shapes_match(state):
        return None, REASON_SHAPE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = frozen_train.build_model(device=device)
    model.load_state_dict(state)
    model.eval()

    seq_len = int(substrate()["architecture"]["seq_len"])
    usable = (tokens.shape[0] - 1) // seq_len
    total = 0.0
    with torch.no_grad():
        for window in range(usable):
            start = window * seq_len
            chunk = torch.from_numpy(
                tokens[start:start + seq_len + 1].astype("int64")
            ).to(device)
            total += float(model(chunk[:-1].view(1, seq_len), chunk[1:].view(1, seq_len)))
    if usable == 0:
        return None, REASON_UNRESOLVED
    return total / usable, ""


def samples(checkpoints, shard_root=SHARD_ROOT):
    """One verifier-computed sample per harness checkpoint, unsmoothed, in step order."""
    rows = []
    for entry in sorted(checkpoints or [], key=lambda row: int(row.get("step", 0))):
        loss, reason = evaluate(entry.get("path"), entry.get("digest"), shard_root)
        if loss is None:
            rows.append(
                {
                    "step": int(entry.get("step", 0)),
                    "loss": None,
                    "source": "verifier-recompute",
                    "smoothing": "none",
                    "weights_owner": str(entry.get("owner", "")),
                    "weights_digest": str(entry.get("digest", "")),
                    "unresolved_reason": reason,
                }
            )
            continue
        rows.append(
            {
                "step": int(entry.get("step", 0)),
                "loss": loss,
                "source": "verifier-recompute",
                "smoothing": "none",
                "weights_owner": str(entry.get("owner", "")),
                "weights_digest": str(entry.get("digest", "")),
            }
        )
    return rows


def arm_loss(checkpoints, shard_root=SHARD_ROOT):
    """The ladder end for one arm: the verifier's loss on that arm's final snapshot."""
    rows = sorted(checkpoints or [], key=lambda row: int(row.get("step", 0)))
    if not rows:
        return None
    loss, _reason = evaluate(rows[-1].get("path"), rows[-1].get("digest"), shard_root)
    return loss

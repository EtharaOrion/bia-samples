"""The pinned harness. Your training script must drive it, and it owns the readout.

You are free to change the optimizer, its hyperparameters, its schedules and the
model initialization. You are not free to change what gets measured or when.

This module owns three things and hands you none of them:

  * the evaluation cadence, EVAL_STRIDE, at which it snapshots the parameters
  * the parameter snapshot itself, written to the run directory as state_<step>.pt
  * the ledger, harness_weight_ledger.json, naming every step it owns a state for

It deliberately does NOT compute a validation loss. The number that decides your
score is computed by the verifier, on a held-out split that never enters this
container, from the state files this harness wrote. Anything your script prints
about loss is yours; it is copied into `submission_report` so a divergence can be
measured, and it is never the graded quantity.

The graded step is the first cadence point at which the verifier's own unsmoothed
evaluation falls below the target AND stays below it at the next SUSTAIN_POINTS
cadence points. Training past that point changes nothing about your score.
Stopping before the sustain window exists means you have not crossed, and that is
graded as a failure with a reason, not as a missing result.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Verifier-owned constants. Reading them is fine. Changing them changes nothing:
# the verifier applies its own copy and ignores whatever this file says at
# grading time.
# ---------------------------------------------------------------------------
import bia_corpus  # noqa: E402  the one declaration of the executed operating point

_POINT = bia_corpus.operating_point()

# These four were literals here and literals again in tests/golden.json. Both
# copies had to be edited together and nothing checked that they were, so a
# re-scale that moved one and not the other would have produced a harness
# snapshotting on one cadence while the grader read another. They come off the
# single declaration now.
EVAL_STRIDE = int(_POINT["eval_stride"])
SUSTAIN_POINTS = int(_POINT["sustain_points"])
TARGET_LOSS = float(_POINT["target_loss"])
MAX_SCHEDULE_STEPS = int(_POINT["max_schedule_steps"])
WEIGHTS_DIGEST_DOMAIN = "bia/oer-03/weights/v2"
READOUT_FILTER_ADMITTED = "raw"

LEDGER_NAME = os.environ.get("BIA_HARNESS_LEDGER", "harness_weight_ledger.json")
SHAPE_PATH = pathlib.Path(__file__).resolve().parent / "shape.json"


def frozen_axes() -> Dict[str, Any]:
    """The four axes upstream freezes. Moving any of them refuses the submission.

    The two digests are DERIVED from the executed operating point by
    `bia_corpus.bound_axes`, not read out of shape.json. They used to be two
    opaque hex literals that nothing computed and nothing could check, so they
    described the architecture only until somebody moved it. A digest over the
    thing it names moves when the thing moves, which is the whole point of
    freezing an axis with one.
    """
    return dict(bia_corpus.bound_axes())


def params_digest(tensors) -> str:
    """A digest over the parameter bytes themselves, in a fixed key order.

    Fixed order matters: a dict iteration order that moved between runs would
    move the digest without moving a single weight, and the verifier would read
    that as a state it does not own.
    """
    hasher = hashlib.sha256()
    for name in sorted(tensors):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(_tensor_bytes(tensors[name]))
    return hasher.hexdigest()


def _tensor_bytes(tensor) -> bytes:
    detached = tensor.detach().to("cpu").contiguous()
    return detached.numpy().tobytes()


def weights_digest(step: int, digest: str) -> str:
    """The binding the verifier recomputes. Forging it buys nothing.

    The verifier re-evaluates the state FILE this harness wrote, so a digest that
    does not correspond to a persisted state simply drops the point. This exists
    to bind a state to its step, not to keep a secret.
    """
    payload = WEIGHTS_DIGEST_DOMAIN + "|" + str(int(step)) + "|" + str(digest)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Harness:
    """Drive this from your training loop. One call per optimizer step."""

    def __init__(self, run_dir: Optional[pathlib.Path] = None, schedule_length: int = 0) -> None:
        self.run_dir = pathlib.Path(run_dir or pathlib.Path.cwd())
        self.schedule_length = int(schedule_length)
        if self.schedule_length > MAX_SCHEDULE_STEPS:
            raise ValueError(
                "schedule_length "
                + str(self.schedule_length)
                + " exceeds MAX_SCHEDULE_STEPS "
                + str(MAX_SCHEDULE_STEPS)
            )
        self.steps_executed = 0
        self.states: List[Dict[str, Any]] = []
        self._report: Dict[str, Any] = {}
        self._axes = frozen_axes()

    def step(self, model, step_index: int) -> None:
        """Call once per optimizer step, after the update.

        On a cadence point the harness snapshots the parameters itself. It does
        not ask you for a loss and will not accept one.
        """
        self.steps_executed = int(step_index)
        if step_index <= 0 or step_index % EVAL_STRIDE != 0:
            return
        tensors = {name: value for name, value in model.state_dict().items()}
        digest = params_digest(tensors)
        target = self.run_dir / ("state_" + str(step_index) + ".pt")
        self._persist(tensors, target)
        self.states.append(
            {
                "step": int(step_index),
                "params_digest": digest,
                "weights_digest": weights_digest(step_index, digest),
                "state_file": target.name,
                "weights_source": "harness",
            }
        )
        self.flush()

    def _persist(self, tensors, target: pathlib.Path) -> None:
        import torch  # imported here so the ledger shape is testable without torch

        torch.save(tensors, target)

    def report(self, **fields: Any) -> None:
        """Record what YOUR loop believes. Copied verbatim, graded never.

        Whatever you put here is carried into the telemetry as
        `submission_report`. If you report a crossing step, the verifier compares
        it with the one it recomputes and grades the divergence. Reporting
        nothing is always safe.
        """
        self._report.update(fields)
        self.flush()

    def flush(self) -> None:
        payload = {
            "schedule_length": self.schedule_length,
            "steps_executed": self.steps_executed,
            "eval_stride": EVAL_STRIDE,
            "sustain_points": SUSTAIN_POINTS,
            "readout_filter": READOUT_FILTER_ADMITTED,
            "weights_digest_domain": WEIGHTS_DIGEST_DOMAIN,
            "frozen_axes": self._axes,
            "states": list(self.states),
            "submission_report": dict(self._report),
        }
        path = self.run_dir / LEDGER_NAME
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

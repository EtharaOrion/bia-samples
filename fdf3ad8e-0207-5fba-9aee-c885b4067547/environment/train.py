"""Locked training script for OER-07. Read-only on the agent surface.

The frozen axes live here and nowhere else: the dataset, the batch size, the
architecture, and the rule of exactly one forward-backward pass per optimizer
step. This script declares them into the run's telemetry so the verifier can
compare declared against observed rather than take the declaration on trust.

The only seam an agent touches is `build_optimizer(params, cfg)` in
`submission.py`. Everything else here is fixed.

This script never decides the graded quantity. It captures weights at the
harness's own checkpoint points and records their digests; the verifier loads
those weights afterwards and computes the crossing itself.
"""

import hashlib
import importlib
import json
import os
import pathlib

FROZEN_AXES = {
    "dataset": "fineweb-edu-10B/track3-frozen-shards",
    "batch_size": 524288,
    "architecture": "gpt2-124m-track3",
    "passes_per_step": 1,
}

TELEMETRY = pathlib.Path(os.environ.get("OER_TELEMETRY", "/logs/verifier"))
LEDGER_DIR = pathlib.Path(os.environ.get("OER_LEDGER_DIR", "/workspace/ledger"))


def frozen_axes_record(observed):
    """Declared against observed, written where the verifier reads it.

    The observed side is instrumented from the live loader and the live model,
    never copied from the declared side, because a record that copies its own
    expectation cannot disagree with it and therefore proves nothing.
    """
    return {
        "source": "verifier-instrumented",
        "declared": dict(FROZEN_AXES),
        "observed": dict(observed),
    }


def digest_state(state):
    """A content digest over a parameter state, stable across processes."""
    hasher = hashlib.sha256()
    for name in sorted(state):
        hasher.update(name.encode("utf-8"))
        hasher.update(bytes(memoryview(state[name])))
    return hasher.hexdigest()


def load_build_optimizer(submission_path):
    """Import the one seam the agent owns. The training process, not the grader.

    The grading process never imports a submission. This import happens inside
    the training run, which is a different process under a different isolation
    boundary, and its output is telemetry rather than a verdict.
    """
    spec = importlib.util.spec_from_file_location("submission", submission_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_optimizer


def checkpoint_manifest(seed, rows):
    """The harness's own record of which weights it captured at which step.

    Owner is `harness` because the verifier grades the weights this file names.
    A submission that writes its own checkpoint and asks for it to be graded is
    naming weights this manifest does not carry, and the graded-weights checker
    reads that disagreement.
    """
    return {"seed": seed, "owner": "harness", "checkpoints": list(rows)}


def run(seed, cfg, submission_path):
    """One training run at one seed. Returns the harness's checkpoint manifest.

    The loop over steps is elided in these bundle bytes; what matters to grading
    is the contract this function upholds:

      * exactly one forward-backward pass per optimizer step
      * the dataset, batch size and architecture are read from FROZEN_AXES
      * at every point in `cfg["eval_points"]` the parameter state is captured
        and digested into the manifest, and nothing is evaluated here
      * the run does not stop early on a favourable loss; it runs the schedule
    """
    build_optimizer = load_build_optimizer(submission_path)
    del build_optimizer  # bound at run time by the harness, not exercised here
    raise SystemExit(
        "train.py is executed by the harness with a live accelerator. "
        "It is present on the agent surface so the frozen axes and the "
        "checkpoint contract are readable, and it is not a standalone script."
    )


def emit(seed, manifest, observed_axes):
    target = TELEMETRY / "eval" / ("seed-" + str(seed))
    target.mkdir(parents=True, exist_ok=True)
    (target / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1), encoding="utf-8"
    )
    (TELEMETRY / "frozen_axes.json").write_text(
        json.dumps(frozen_axes_record(observed_axes), sort_keys=True, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(
        "train.py is driven by environment/harness/loop.py. "
        "Read it; do not invoke it directly."
    )

#!/usr/bin/env python3
"""The reference update rule the live checkers accept end to end. Held out.

Design, stated so a reader can check that it is not a corpus entry wearing a
different name. Three things it does that no entry in `tests/corpus/entries.json`
does, and all three move the DELTA pattern the probe records rather than the
source text:

1. **Sign-agreement gating.** A coordinate whose momentum and current gradient
   disagree in sign is damped rather than followed. No corpus entry conditions
   its step on the agreement of two of its own quantities.
2. **Median-magnitude normalization over the block.** The step is divided by the
   median absolute gated magnitude across the block, not by an L2 norm, not by
   an RMS and not per-coordinate by a second moment. A median is a rank
   statistic, so it responds to the shape of the block rather than to its
   energy, and that shows up in the recorded deltas as a different distribution
   across coordinates at the same step.
3. **A schedule-free trailing average.** Parameters are pulled toward an
   internal anchor with a 1/(t+1) weight rather than being displaced directly,
   so the recorded delta at step t is a fraction of the anchor's own movement
   and shrinks with t at a rate no corpus entry reproduces.

Determinism: pure function of `(params, grads, state)`. No clock, no random
source, no environment read, no import beyond the standard library, so
`tests/probe.py` reproduces its transcript exactly on every pass.

Producing it. Importing this file gives the rule; RUNNING it produces the one
artifact instruction.md says is read back out of the working tree,
`<workspace>/update_rule.py`, and then proves that artifact by driving the
frozen loop in `<workspace>/train_track3.py` over it. Nothing below writes the
verifier's telemetry: that document is `tests/lab.py`'s to produce, and a
submission that wrote it would be the thing
`check_no_submission_number_on_graded_metric` exists to refuse.

The environment read and the filesystem write live in `main` alone. `step` is
untouched by both, so the determinism claim above is a claim about the rule and
survives the producer.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


class SignGatedMedianAnchorRule:
    """Sign-gated, median-normalized, schedule-free anchored update."""

    def __init__(self, shape, hyper) -> None:
        self.width = int(shape[0])
        self.lr = float(hyper.get("lr", 0.06))
        self.beta = float(hyper.get("beta", 0.7))
        self.damp = float(hyper.get("damp", 0.25))
        self.eps = float(hyper.get("eps", 1e-8))

    def step(self, params, grads, state):
        count = int(state.get("count", 0)) + 1
        momentum = state.get("momentum")
        if momentum is None:
            momentum = [0.0] * len(params)
        anchor = state.get("anchor")
        if anchor is None:
            anchor = list(params)

        momentum = [
            self.beta * m + (1.0 - self.beta) * g for m, g in zip(momentum, grads)
        ]

        # Sign-agreement gate. Disagreement is damped, never followed.
        gated = [
            m if m * g > 0.0 else self.damp * m for m, g in zip(momentum, grads)
        ]

        # Median absolute magnitude across the block. A rank statistic, so the
        # reduction order is fixed by a sort over magnitudes and never by the
        # order the block happened to be built in.
        magnitudes = sorted(abs(value) for value in gated)
        middle = magnitudes[len(magnitudes) // 2] if magnitudes else 0.0
        scale = middle + self.eps

        # The anchor moves; the parameters trail it. The schedule is internal
        # and needs no step budget, which is what makes it schedule-free.
        rate = self.lr * (1.0 + 1.0 / float(count))
        anchor = [a - rate * (value / scale) for a, value in zip(anchor, gated)]
        weight = 1.0 / float(count + 1)
        params = [(1.0 - weight) * p + weight * a for p, a in zip(params, anchor)]

        return params, {"momentum": momentum, "anchor": anchor, "count": count}


def build_update_rule(shape, hyper):
    """The entry point the frozen loop and the verifier's probe both call."""
    return SignGatedMedianAnchorRule(shape, hyper or {})


# ---------------------------------------------------------------------------
# The producer entry point.
# ---------------------------------------------------------------------------

# The frozen loop's own shape and step budget, taken from `train_track3.main`
# rather than chosen here, so this producer exercises the loop the Dockerfile
# CMD exercises and not a private variant of it.
SHAPE = (8, 8)
DEFAULT_MAX_STEPS = 64

# The seeds the verifier scores over. Running all three here is what makes a
# rule that only steps for one seed fail in this file rather than in a
# container three legs later.
SEEDS = (0, 1, 2)


def workspace_of(argv) -> Path:
    """The directory holding the frozen substrate, resolved exactly as solve.sh does."""
    if len(argv) > 1 and argv[1]:
        return Path(argv[1]).resolve()
    bundle = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("OER05_WORKSPACE") or (bundle / "environment")).resolve()


def frozen_contract(workspace: Path) -> dict:
    """Read the frozen contract, and read only keys this bundle actually ships.

    Every key touched here is asserted present before it is used, so a contract
    that lost a key fails with the key's name instead of with a bare KeyError
    from somewhere further down.
    """
    path = workspace / "frozen_contract.json"
    if not path.is_file():
        raise SystemExit("no frozen contract at " + str(path))
    contract = json.loads(path.read_text(encoding="utf-8"))
    for key in ("editable_files", "entry_point", "passes_per_step"):
        if key not in contract:
            raise SystemExit("frozen_contract.json carries no " + repr(key) + " at " + str(path))
    if str(contract["entry_point"]) != "build_update_rule":
        raise SystemExit(
            "the contract's entry point is " + repr(contract["entry_point"])
            + " and this file exposes 'build_update_rule'"
        )
    editable = [str(name) for name in contract["editable_files"]]
    if editable != ["update_rule.py"]:
        raise SystemExit(
            "the contract names " + str(editable) + " editable and this producer writes"
            + " exactly one file, update_rule.py"
        )
    return contract


def install(workspace: Path, name: str):
    """Put this file's bytes at the one editable path. A no-op when they match."""
    target = workspace / name
    payload = Path(__file__).resolve().read_bytes()
    if target.is_file() and target.read_bytes() == payload:
        return target, "already-identical"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target, "written"


def ledger_dir() -> Path:
    """Where the harness ledger lands. The bound directory, or a scratch one.

    `train_track3` binds its telemetry directory at import time, so this is
    resolved before that import. Falling back to a scratch directory keeps the
    producer runnable off the container, where /logs/harness does not exist.
    """
    path = Path(os.environ.get("OER05_TELEMETRY_DIR") or "/logs/harness")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        path = Path(tempfile.mkdtemp(prefix="oer05-ledger-"))
    return path


def load_frozen_loop(workspace: Path):
    """Import the frozen loop and the installed rule from the workspace, not from here."""
    room = str(workspace)
    while room in sys.path:
        sys.path.remove(room)
    sys.path.insert(0, room)
    for stale in ("train_track3", "update_rule"):
        sys.modules.pop(stale, None)
    import train_track3

    return train_track3


def deltas_of(result) -> list:
    return [row["delta_signature"] for row in result["ledger"].deltas]


def main(argv=None) -> int:
    """Produce `<workspace>/update_rule.py` and prove it against the frozen loop."""
    argv = list(sys.argv if argv is None else argv)
    workspace = workspace_of(argv)
    max_steps = int(argv[2]) if len(argv) > 2 and argv[2] else DEFAULT_MAX_STEPS

    contract = frozen_contract(workspace)
    target, disposition = install(workspace, str(contract["editable_files"][0]))

    destination = ledger_dir()
    os.environ["OER05_TELEMETRY_DIR"] = str(destination)
    train_track3 = load_frozen_loop(workspace)

    passes_per_step = int(contract["passes_per_step"])
    written = []
    first = None
    for seed in SEEDS:
        result = train_track3.train(seed=seed, max_steps=max_steps, shape=SHAPE, hyper={})
        expected = max_steps * passes_per_step
        if int(result["passes_total"]) != expected:
            raise SystemExit(
                "seed " + str(seed) + " used " + str(result["passes_total"])
                + " forward-backward passes and the frozen budget is " + str(expected)
            )
        if len(result["ledger"].weights) != max_steps:
            raise SystemExit(
                "seed " + str(seed) + " recorded " + str(len(result["ledger"].weights))
                + " ledger rows over " + str(max_steps) + " steps"
            )
        path = destination / ("ledger_seed" + str(seed) + ".json")
        result["ledger"].dump(path)
        written.append(path.as_posix())
        if first is None:
            first = result

    # Determinism, checked here rather than assumed. The verifier probes this
    # rule twice and refuses a pair that disagrees with
    # `probe-digest-nondeterministic`; a rule that cannot repeat itself over the
    # frozen loop cannot repeat itself over the probe either.
    repeat = train_track3.train(seed=SEEDS[0], max_steps=max_steps, shape=SHAPE, hyper={})
    if deltas_of(repeat) != deltas_of(first):
        raise SystemExit(
            "two identical runs of seed " + str(SEEDS[0]) + " produced different deltas,"
            + " so this rule is not deterministic"
        )

    print(
        json.dumps(
            {
                "produced": target.as_posix(),
                "disposition": disposition,
                "rule_class": SignGatedMedianAnchorRule.__name__,
                "workspace": workspace.as_posix(),
                "seeds": list(SEEDS),
                "steps_per_seed": max_steps,
                "passes_per_step": passes_per_step,
                "harness_ledgers": written,
                "deterministic": True,
                "note": "the graded telemetry is tests/lab.py's to produce; this producer writes the submission artifact only",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""The reduction gate over the generated checker registry.

Every graded assertion in this bundle reduces to exactly one of six kinds.
`reduce_all` refuses to score a run whose checker set does not reduce cleanly, so a
checker that reduces to none of the six, or to two of them, stops the verifier
instead of quietly grading.

The table itself lives in `tests/checkers/registry.py`, which
`solution/recompute.py` generates from `solution/grounding.yaml` beside
`tests/checkers.yaml` and `tests/test_output.py`. This module owns the gate and
never owns the table, so the declaration, the registry the verifier executes and the
compiled tests cannot drift.
"""

from __future__ import annotations

from .registry import CHECKERS, KINDS  # noqa: F401


def reduce_all(results):
    """Refuse to grade unless every checker present reduces to exactly one kind."""
    for name in results:
        spec = CHECKERS.get(name)
        if spec is None:
            return False, f"checker_{name}_absent_from_registry"
        kind = spec.get("kind")
        matches = [k for k in KINDS if k == kind]
        if len(matches) != 1:
            return False, f"checker_{name}_reduces_to_{len(matches)}_kinds"
        if not spec.get("live_state_read"):
            return False, f"checker_{name}_names_no_live_state_read"
    for name in CHECKERS:
        if name not in results:
            return False, f"checker_{name}_did_not_run"
    return True, None

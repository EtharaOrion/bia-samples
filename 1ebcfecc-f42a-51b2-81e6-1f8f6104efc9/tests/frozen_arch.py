"""The frozen architecture, defined verifier-side so the agent cannot redefine it.

A submitted checkpoint is loaded into this definition with strict=True, so any
change to depth, width, head dimension or vocabulary fails to load rather than
being trusted to have been left alone. That is why the architecture freeze needs
no separate checker.

The concrete shape is bound by seed/contract.yaml once the operating point is
measured. It is deliberately not filled in with plausible numbers here, because
an unmeasured shape that looks measured is the defect that voided the prior
attempt at this project.
"""
from __future__ import annotations

import pathlib

SHAPE_BOUND = False


def build():
    if not SHAPE_BOUND:
        raise RuntimeError(
            "frozen architecture shape is not bound; the scaled operating point has "
            "not been measured. Binding it changes seed/contract.yaml and re-triggers "
            "the Phase 0.5 signature."
        )
    raise NotImplementedError  # populated when the shape is bound


def load_val_tokens(shard: pathlib.Path):
    if not SHAPE_BOUND:
        raise RuntimeError("validation split shape is not bound; see build()")
    raise NotImplementedError

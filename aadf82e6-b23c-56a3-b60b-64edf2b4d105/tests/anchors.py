"""The two endpoints of the reward scale, both MEASURED here, neither authored.

`reward.anchored` scales the consolidated crossing between a baseline and a
target:

    raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)

Until this module existed those two numbers were the literals 3250 and 2690,
carried in `session_driver.ANCHORS` under `anchors_state: "measured"` and an
`authority` pointing at a requirements document. They were measured on the
upstream nanogpt speedrun, which is a DIFFERENT substrate from the one this
verifier trains: a different vocabulary, a different corpus, a different width
and depth, a different batch. Nothing in this bundle ever produced them, and at
the operating point this bundle now binds no run of any recipe lands anywhere
near either one. A scale whose endpoints come from elsewhere is not a scale.

So both endpoints are re-derived on every graded run, on THIS substrate, by
running recipes through the same `train_eval.train_and_evaluate` an attempt goes
through and reading the crossing back through the same
`checkers.sustained_crossing` an attempt is graded by. No figure for either
endpoint appears as a literal anywhere in this bundle.

WHAT IS AUTHORED HERE AND WHAT IS NOT. The two recipe LISTS below are authored:
they are fixed, ordered and declared, and they are the definition of the two
bars. What is measured is every number that comes out of running them. That is
the same division `samples/3161f7d5-.../tests/evaluate.py` draws between its
declared `wider_probe` rules and the metric it measures for each, and it is why
neither of these functions is named an `optimum`: nothing here searches the
recipe space, which is continuous in eleven dimensions, and no number in this
bundle claims it does. They are PROBES, and a probe is a bar.

  HANDED_PROBE     what the agent surface hands over without any refinement.
                   `environment/refine_template.py` writes exactly one recipe and
                   `environment/recipe_schema.json` names the optimizer families;
                   this list is that recipe plus one restrained and one spirited
                   representative of each of the three families the schema's
                   enum spans -- first-order, adaptive, and orthogonalized. The
                   BEST crossing over the list is `baseline_metric`, so a
                   submission scores above zero only by beating everything the
                   handed surface gives away for free.

  REFINED_PROBE    a short ladder of tuned variants. The BEST crossing over it is
                   `target_metric`, the bar at which the reward saturates.

Measured in the verifier's own image at the bound operating point, the handed
probe crosses at step 70 and the refined probe at step 35. Those two numbers are
recorded here as the expectation an auditor can re-run, NOT as values this module
returns: it returns whatever the run in front of it measures, and if the card,
the image or the substrate moves, the endpoints move with it.
"""
from __future__ import annotations

import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bia_recipe  # noqa: E402
import checkers  # noqa: E402
import train_eval  # noqa: E402

HANDED_SOURCE = "verifier-probe-over-the-handed-surface"
REFINED_SOURCE = "verifier-refined-probe"

# The recipe environment/refine_template.py writes, verbatim, and two more per
# optimizer family class. Every field not named here takes its schema default, so
# each entry is reachable from the handed surface by changing the named keys and
# nothing else.
HANDED_PROBE = (
    ("handed_template", {"optimizer": "momentum", "lr": 0.005, "beta1": 0.9,
                         "schedule": "constant", "grad_clip": 1.0}),
    ("handed_momentum_spirited", {"optimizer": "momentum", "lr": 0.05,
                                  "schedule": "constant"}),
    ("handed_adamw", {"optimizer": "adamw", "lr": 0.006, "beta2": 0.95,
                      "schedule": "cosine", "warmup_frac": 0.05}),
    ("handed_adamw_spirited", {"optimizer": "adamw", "lr": 0.02, "beta2": 0.95,
                               "schedule": "cosine", "warmup_frac": 0.05}),
    ("handed_ortho", {"optimizer": "ortho_momentum", "lr": 0.005,
                      "schedule": "constant"}),
    ("handed_ortho_spirited", {"optimizer": "ortho_momentum", "lr": 0.02,
                               "beta1": 0.95, "ns_steps": 5,
                               "schedule": "linear_decay", "warmup_frac": 0.02}),
)

REFINED_PROBE = (
    ("refined_decayed", {"optimizer": "ortho_momentum", "lr": 0.03, "beta1": 0.95,
                         "ns_steps": 5, "schedule": "linear_decay",
                         "warmup_frac": 0.02, "embed_lr_mult": 2.0}),
    ("refined_wsd", {"optimizer": "ortho_momentum", "lr": 0.04, "beta1": 0.95,
                     "ns_steps": 5, "schedule": "wsd", "decay_frac": 0.3,
                     "embed_lr_mult": 3.0}),
    ("refined_split", {"optimizer": "ortho_momentum", "lr": 0.05, "beta1": 0.95,
                       "ns_steps": 5, "schedule": "linear_decay",
                       "warmup_frac": 0.02, "embed_lr_mult": 4.0,
                       "head_lr_mult": 0.5}),
)


def _measure_one(substrate, shape: dict, raw: dict) -> dict:
    """Run one probe recipe exactly as an attempt is run, and read its crossing.

    The seed comes from `train_eval.seed_for` over the normalized fingerprint,
    which is the identical derivation an attempt gets. A submission that proposes
    a probe recipe therefore reproduces the probe's run rather than drawing a
    different one, so the bar and the submission are commensurable.
    """
    normalized = bia_recipe.normalize(dict(raw), shape)
    started = time.monotonic()
    measured = train_eval.train_and_evaluate(
        substrate, normalized["recipe"],
        seed=train_eval.seed_for(normalized["fingerprint"]), index=0)
    crossing = checkers.sustained_crossing(
        measured["verifier_evals"], float(shape["target_loss"]),
        int(shape["sustain_window"]))
    return {
        "crossing": None if crossing is None else int(crossing),
        "halted_at_step": int(measured["halted_at_step"]),
        "recipe_fingerprint": normalized["fingerprint"],
        "seconds": round(time.monotonic() - started, 3),
    }


def _best(substrate, shape: dict, entries, source: str) -> dict:
    """Run every entry and keep the smallest crossing. A miss is recorded, not hidden."""
    rows = []
    best = None
    for ident, raw in entries:
        row = _measure_one(substrate, shape, raw)
        row["id"] = ident
        rows.append(row)
        if row["crossing"] is not None and (best is None or row["crossing"] < best["crossing"]):
            best = row
    return {
        "metric": None if best is None else int(best["crossing"]),
        "by": None if best is None else best["id"],
        "source": source,
        "evaluated": len(rows),
        "crossed": sum(1 for row in rows if row["crossing"] is not None),
        "seconds": round(sum(row["seconds"] for row in rows), 3),
        "rows": rows,
    }


def measure(substrate, shape: dict) -> dict:
    """Both endpoints, measured now, on this substrate, in this image.

    `state` is `measured` only when both endpoints exist and leave a POSITIVE
    span. Every other case carries its own machine-readable reason and a null
    endpoint, because a scale that cannot be built is a fact about this run and
    must not be papered over with a number.
    """
    handed = _best(substrate, shape, HANDED_PROBE, HANDED_SOURCE)
    refined = _best(substrate, shape, REFINED_PROBE, REFINED_SOURCE)
    baseline = handed["metric"]
    target = refined["metric"]

    if baseline is None:
        state, gap = "unscalable", "handed-probe-never-crossed"
    elif target is None:
        state, gap = "unscalable", "refined-probe-never-crossed"
    elif baseline - target <= 0:
        state, gap = "unscalable", "scaling-span-nonpositive"
    else:
        state, gap = "measured", ""

    return {
        "baseline_metric": baseline,
        "target_metric": target,
        "target_loss": float(shape["target_loss"]),
        "pass_threshold": 0.65,
        "anchors_state": state,
        "anchors_gap": gap,
        "authority": "measured at grading time by tests/anchors.py on this substrate",
        "baseline_source": handed["source"],
        "target_source": refined["source"],
        "handed_probe": handed,
        "refined_probe": refined,
        "seconds": round(handed["seconds"] + refined["seconds"], 3),
    }

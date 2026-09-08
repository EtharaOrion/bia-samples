#!/usr/bin/env python3
"""The OER-01 oracle: emit the recipe the authoring search arrived at.

WHAT THIS IS AND IS NOT
    This file writes a recipe. It does not train, it does not read the graded split,
    and it does not know the target or either anchor. The recipe it writes is the
    OUTPUT of a search run during authoring against this exact harness, and that
    search and its measurements are recorded in solution/grounding.md.

    It is NOT the verifier's private reference recipe. The two were arrived at
    separately and use different decay families, so the reference arm of the gate is
    never the reference recipe being graded against itself. It is also not any of the
    three recipes in published_records.json -- submitting one of those is refused with
    `record-replayed`, and the oracle has to clear that gate like anything else.

THE FINDING THIS RECIPE ENCODES
    The graded quantity is where a curve first crosses the shipped default's own
    end-of-ceiling loss. That is NOT the same question as which recipe ends lowest,
    and on this substrate the two questions have different answers.

    The published hold-then-decay record ends at 5.053879, the best final loss of
    anything measured here, and crosses the target at 1792 micro-batches. A cosine or a
    linear decay from a short warmup ends MUCH higher -- 5.257 and 5.194 -- and crosses
    at 1664. Holding a peak buys quality that arrives after the bar has already been
    passed, and this slot does not pay for it.

    So the recipe below spends nothing on the tail. It warms up over 5 percent of the
    ceiling and decays straight to zero from there.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

RECIPE = {
    "schema": "oer-displacement-recipe/v1",
    "notes": (
        "A linear decay to zero from a 5 percent warmup. The search behind this recipe "
        "compared decay families at a fixed peak and found that the two that reach zero "
        "soonest -- linear and cosine -- both cross the target a full grid point ahead of "
        "every hold-then-decay recipe in the published corpus, while ending between 0.14 "
        "and 0.20 nats WORSE on final loss. That is the trade this slot grades. The peaks "
        "are the per-role values the published corpus already uses, because the search "
        "found the crossing far more sensitive to the shape of the decay than to the "
        "height of the peak; a hotter peak held briefly crossed later, not sooner."
    ),
    "grad_accum": 1,
    "grad_clip": 1.0,
    "optimizer": {
        "lr_embed": 0.030, "lr_hidden": 0.006, "lr_head": 0.004, "lr_scalar": 0.030,
        "beta1": 0.9, "beta2": 0.999, "eps": 1e-10, "weight_decay": 0.0,
    },
    "schedule": {"shape": "linear", "warmup_frac": 0.05, "final_frac": 0.0,
                 "stable_frac": 0.6},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RECIPE, indent=2) + "\n", encoding="utf-8")
    print(f"[reference] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

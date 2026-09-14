#!/usr/bin/env python3
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml.
#
# Agent-visible copy of the verifier's canonicaliser, byte-for-byte below the
# banner. Screen your own recipe before you spend accelerator time:
#
#     python3 environment/fingerprint_tool.py /workspace/submission/recipe.py
#
# The corpus is public. Hiding it would protect nothing and would only stop an
# honest agent from checking its own work.

"""Deterministic canonicalisation of a declarative optimizer recipe.

This is the load-bearing record-displacement control. It reduces a submitted
update rule to a structural digest over the algebra alone, plus a log-quantised
vector over the numeric hyperparameters, so that two recipes are compared on what
they DO rather than on how they are spelled.

The two comparisons carry different jobs and neither substitutes for the other.

- The structural digest excludes every numeric value. Renaming a variable,
  reordering a mapping, or retuning a learning rate cannot move it. Only the
  primitives, their roles, the couplings between parameter groups, the schedule
  families and the initialisation family move it.
- The proximity distance exists because a structural match alone is not a replay.
  Two genuinely different operating points on the same algebra are a derivation.
  A structural match whose quantised hyperparameters sit within the bound floor of
  a published entry is a recital wearing different numbers.

A replay is a structural match AND a proximity at or under the floor. Both, never
either, and the floor is a bound value read from the caller rather than authored
here.

This module reads no clock, opens no socket, consults no random source, imports
nothing from the submission, and executes no submitted code. It parses the
submission's declarative RECIPE literal out of its bytes with `ast.literal_eval`,
which evaluates literals only and cannot call a function.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math

# The ordered hyperparameter key list. Fixed here so the distance is a function of
# the recipe pair alone and never of whichever keys happened to be present.
HYPER_KEYS = (
    "lr_peak",
    "momentum",
    "second_moment_beta",
    "weight_decay",
    "warmup_frac",
    "orthogonalisation_steps",
    "trust_region_rho",
    "aux_lr_ratio",
)

# The quantisation spread. Two values a full decade apart differ by four buckets,
# so a distance of 1.0 on one key is two decades. Bound here and never fitted.
BUCKET_SPREAD = 8.0

DIGEST_CHARS = 32


def bucket(value) -> int:
    """Log-quantise one hyperparameter. Non-positive values collapse to a sentinel.

    A zero weight decay and a 1e-9 weight decay are the same decision, so both
    resolve to the same bucket rather than to a distance an author could tune.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1
    if number != number or number <= 0.0:
        return -1
    return int(math.floor(4.0 * math.log10(number) + 40.0))


def canonical_form(recipe) -> dict:
    """The algebra of a recipe with every numeric value stripped out."""
    row = recipe if isinstance(recipe, dict) else {}
    chain = []
    for item in row.get("update_chain") or []:
        if not isinstance(item, dict):
            continue
        chain.append([str(item.get("primitive", "")), str(item.get("role", ""))])
    couplings = sorted(
        [str(part) for part in item] for item in (row.get("couplings") or []) if isinstance(item, (list, tuple))
    )
    schedules = sorted(
        [str(part) for part in item] for item in (row.get("schedule_families") or []) if isinstance(item, (list, tuple))
    )
    return {
        "update_chain": chain,
        "couplings": couplings,
        "schedule_families": schedules,
        "init_family": str(row.get("init_family", "")),
    }


def structural_digest(recipe) -> str:
    """sha256 over the canonical algebra. Hyperparameter values never enter it."""
    payload = json.dumps(canonical_form(recipe), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def hyper_vector(recipe) -> list:
    """The quantised hyperparameter vector, in the fixed key order."""
    row = recipe if isinstance(recipe, dict) else {}
    hyper = row.get("hyper") if isinstance(row.get("hyper"), dict) else {}
    return [bucket(hyper.get(key)) if key in hyper else None for key in HYPER_KEYS]


def proximity(left, right) -> float:
    """Mean per-key normalised bucket distance in [0, 1]. Absent keys cost the max.

    A key present on one side and absent on the other is maximally distant rather
    than skipped, because skipping it would let a submission buy proximity by
    deleting a field.
    """
    total = 0.0
    for a, b in zip(left, right):
        if a is None or b is None:
            total += 1.0
            continue
        total += min(1.0, abs(a - b) / BUCKET_SPREAD)
    return total / float(len(HYPER_KEYS))


def fingerprint(recipe) -> dict:
    """Everything the screen needs about one recipe, and nothing more."""
    return {"structural_digest": structural_digest(recipe), "hyper_vector": hyper_vector(recipe)}


def screen(recipe, corpus, floor) -> dict:
    """Compare one recipe against a pinned corpus. Pure; no state, no side effect.

    Returns the computed fingerprint, the per-entry proximity, and the list of
    entries that qualify as a replay under the bound floor. The floor arrives from
    the caller; this module never authors one, because a threshold nobody approved
    is a threshold nobody can audit.
    """
    mine = fingerprint(recipe)
    try:
        bound = float(floor)
    except (TypeError, ValueError):
        bound = -1.0
    rows, replays = {}, []
    for entry in corpus or []:
        ident = str(entry.get("id", ""))
        distance = proximity(mine["hyper_vector"], entry.get("hyper_vector") or [])
        same_algebra = entry.get("structural_digest") == mine["structural_digest"]
        rows[ident] = {"proximity": distance, "structural_match": bool(same_algebra)}
        if same_algebra and bound >= 0.0 and distance <= bound:
            replays.append(ident)
    return {
        "structural_digest": mine["structural_digest"],
        "hyper_vector": mine["hyper_vector"],
        "per_entry": rows,
        "replays": sorted(replays),
        "floor_applied": bound,
    }


def recipe_from_source(text: str):
    """Lift the RECIPE literal out of submitted source without executing any of it.

    The submission is a file the verifier never imports. Its declarative RECIPE is
    read as an assignment in the module body and evaluated with literal_eval, which
    admits literals and containers only. A submission whose RECIPE is computed at
    import time rather than written as a literal therefore does not screen, and
    that refusal is deliberate: a fingerprint over a value only obtainable by
    running the submission is a fingerprint taken after the submission has already
    had control.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "RECIPE":
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return None
    return None

def _main(argv):
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    if len(argv) < 2:
        print('usage: fingerprint_tool.py <recipe.py>', file=sys.stderr)
        return 2
    published = json.loads((here / 'published_records.json').read_text(encoding='utf-8'))
    recipe = recipe_from_source(Path(argv[1]).read_text(encoding='utf-8'))
    if recipe is None:
        print('no module-level RECIPE literal found; the screen cannot read this file', file=sys.stderr)
        return 1
    mine = fingerprint(recipe)
    rows = []
    for entry in published['entries']:
        rows.append({'id': entry['id'], 'structural_match': entry['structural_digest'] == mine['structural_digest']})
    print(json.dumps({
        'structural_digest': mine['structural_digest'],
        'hyper_vector': mine['hyper_vector'],
        'floor': published['proximity_floor'],
        'structural_matches': sorted(r['id'] for r in rows if r['structural_match']),
        'note': 'a structural match alone is not a replay; proximity at or under the floor decides it',
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    import sys

    raise SystemExit(_main(sys.argv))

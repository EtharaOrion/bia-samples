# FORGE-CANARY-BEGIN
# 0: 265f83571b1620b48545991b223f1a5c5c3922be8bcc0dec111ddc8a45c93f2d
# 1: 2bd67ac1741a70703873cc20cf187e877634b217fb3637e3eb581093d4461b09
# 2: 7e7651f211b323a425ca34218201257f772856252b8f58753f6533335e6be10c
# 3: e54c82a07d5d8d1f8a50dafbade35be98e6e3254960f9544acfac5b283ef5a16
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of OER-30 from solution/grounding.yaml and nothing else.

WHAT THIS DERIVES
-----------------
    environment/instances.json          the frozen set-cover instance family
    environment/ledger_source.json      the frozen ledger source and its seal seed
    tests/bound.json                    every constant the checkers compare against
    tests/test_output.py                the compiled per-checker suite
    solution/fixtures/reference_run/    the oracle submission, at reward 1.0
    solution/fixtures/rejecting/*/      the five frozen rejecting submissions
    solution/TRUTH.md                   the private ground-truth record
    solution/rubrics.json               the solution rubrics
    solution/solve.sh                   the solution entry point

WHAT IT DOES NOT DO
-------------------
No clock is read. No network is touched. No module named `random` is imported. Every sequence
that looks drawn is an integer linear congruential recurrence seeded from a constant recorded
in grounding.yaml, so two runs over frozen bytes produce byte-identical output.

The expectations in tests/bound.json are derived HERE, from grounding.yaml and from the store
the minter actually realises. Not one of them is transcribed from instruction.md, and neither
discovery value appears in instruction.md at all.

`--check` recomputes everything and reports drift against what is on disk without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "environment"))

import mint_store  # noqa: E402

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."

LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345
LCG_MODULUS = 2 ** 31


class Lcg:
    """The one arithmetic sequence generator in this file. Not a random source."""

    def __init__(self, seed: int):
        self.state = int(seed)

    def next(self) -> int:
        self.state = (LCG_MULTIPLIER * self.state + LCG_INCREMENT) % LCG_MODULUS
        return self.state

    def below(self, bound: int) -> int:
        return self.next() % int(bound)


# ---------------------------------------------------------------------------
# Frozen substrate
# ---------------------------------------------------------------------------


def build_instances(spec: dict) -> dict:
    """The frozen instance family. Set 0 is the whole universe, so feasibility is total."""
    lcg = Lcg(spec["seed"])
    universe = int(spec["universe_size"])
    set_count = int(spec["set_count"])
    rows = []
    for index in range(int(spec["count"])):
        sets = []
        for set_id in range(set_count - 1):
            members = [
                element
                for element in range(universe)
                if lcg.below(int(spec["member_draw_modulus"])) < int(spec["member_threshold"])
            ]
            if not members:
                members = [lcg.below(universe)]
            weight = int(spec["weight_base"]) + lcg.below(int(spec["weight_modulus"]))
            sets.append({"id": set_id, "weight": weight, "members": sorted(set(members))})
        # The fallback set covers the whole universe at a dominating weight, so every
        # instance is feasible and no instance is solved by reaching for it.
        sets.append(
            {
                "id": set_count - 1,
                "weight": int(spec["fallback_weight"]),
                "members": list(range(universe)),
            }
        )
        rows.append(
            {
                "instance_id": "inst-%02d" % (index + 1),
                "universe_size": universe,
                "fallback_set_id": set_count - 1,
                "sets": sets,
            }
        )
    return {"schema": "oer30.instances/v1", "banner": BANNER, "instances": rows}


def build_ledger(spec: dict, instance_ids) -> dict:
    """The frozen ledger source: which atoms exist and which atoms each derives from."""
    lcg = Lcg(spec["seed"])
    count = int(spec["atom_count"])
    roots = int(spec["root_count"])
    heuristics = list(spec["heuristics"])
    atoms = []
    for index in range(count):
        ident = "atom-%02d" % (index + 1)
        if index < roots:
            inputs = []
        else:
            wanted = 1 + lcg.below(int(spec["max_inputs"]))
            chosen = []
            for _ in range(wanted):
                candidate = "atom-%02d" % (lcg.below(index) + 1)
                if candidate not in chosen:
                    chosen.append(candidate)
            inputs = sorted(chosen)
        atoms.append(
            {
                "atom_id": ident,
                "instance_id": instance_ids[index % len(instance_ids)],
                "heuristic": heuristics[lcg.below(len(heuristics))],
                "inputs": inputs,
                "note": "derivation step recorded by the minter",
            }
        )
    return {
        "schema": "oer30.ledger_source/v1",
        "banner": BANNER,
        "seal_seed": int(spec["seal_seed"]),
        "producer_host": str(spec["producer_host"]),
        "atoms": atoms,
    }


# ---------------------------------------------------------------------------
# The exact comparator. A bitmask dynamic program, never a heuristic estimate.
# ---------------------------------------------------------------------------


def exact_optimum(instance: dict):
    """Minimum total weight of a sub-family of sets covering the whole universe."""
    universe = int(instance["universe_size"])
    full = (1 << universe) - 1
    masks = []
    for row in instance["sets"]:
        mask = 0
        for element in row["members"]:
            mask |= 1 << int(element)
        masks.append((mask, int(row["weight"]), int(row["id"])))
    infinity = float("inf")
    best = [infinity] * (full + 1)
    pick = [None] * (full + 1)
    best[0] = 0
    for state in range(full + 1):
        if best[state] == infinity:
            continue
        for mask, weight, ident in masks:
            following = state | mask
            if following == state:
                continue
            cost = best[state] + weight
            if cost < best[following]:
                best[following] = cost
                pick[following] = (state, ident)
    chosen = []
    state = full
    while state and pick[state] is not None:
        previous, ident = pick[state]
        chosen.append(ident)
        state = previous
    return int(best[full]), sorted(chosen)


def cover_weight(instance: dict, cover) -> int:
    weights = {int(row["id"]): int(row["weight"]) for row in instance["sets"]}
    return sum(weights[int(value)] for value in cover)


def normalised_gap(weight: int, optimum: int) -> float:
    return round((float(weight) - float(optimum)) / float(optimum), 9)


# ---------------------------------------------------------------------------
# Order sanity: the realised order must not be either natural order
# ---------------------------------------------------------------------------


def lexicographic_order(store: dict):
    return sorted(store["atoms"])


def kahn_sorted_order(source: dict):
    """A topological order produced by always taking the lowest ready identifier.

    This is the other order a reader reaches for when they want a topological order without
    walking the chain, so the realised order is asserted to differ from it too.
    """
    inputs_of = {row["atom_id"]: list(row.get("inputs") or []) for row in source["atoms"]}
    identifiers = sorted(inputs_of)
    done, out = set(), []
    while len(out) < len(identifiers):
        ready = [
            ident
            for ident in identifiers
            if ident not in done and all(parent in done for parent in inputs_of[ident])
        ]
        out.append(ready[0])
        done.add(ready[0])
    return out


# ---------------------------------------------------------------------------
# Submission construction
# ---------------------------------------------------------------------------


def seal_chain_of(store: dict):
    return [row["seal"] for row in store["attestations"]]


def build_reference_submission(store: dict, instances: dict) -> dict:
    order = list(store["realised_order"])
    terminal = order[-1]
    instance = instances[store["atoms"][terminal]["instance_id"]]
    optimum, cover = exact_optimum(instance)
    return {
        "schema": "oer30.submission/v1",
        "attestation_order": order,
        "seal_chain": seal_chain_of(store),
        "atom_digests": dict(store["atom_digests"]),
        "terminal_atom_id": terminal,
        "terminal_atom_digest": store["atom_digests"][terminal],
        "graded_instance_id": instance["instance_id"],
        "submitted_cover": cover,
        "reported_cover_weight": optimum,
    }


def digests_under_order(store: dict, order) -> dict:
    """Recompute every atom digest folding inputs in the GIVEN order rather than the realised
    one. This is the arithmetic a reader performs when they assume an order."""
    position = {ident: index for index, ident in enumerate(order)}
    digests = {}
    for ident in order:
        record = store["atoms"][ident]
        parents = sorted(record["inputs"], key=lambda name: position[name])
        digests[ident] = mint_store.atom_digest(record, [digests[name] for name in parents])
    return digests


def seals_under_order(store: dict, order):
    previous = mint_store.GENESIS_PREV_SEAL
    out = []
    for ident in order:
        digest = mint_store.cover_sha256(store["atoms"][ident]["selected_sets"])
        seal = mint_store.attestation_seal(ident, digest, previous)
        out.append(seal)
        previous = seal
    return out


def build_rejecting(store: dict, instances: dict, reference: dict) -> dict:
    """The five frozen rejecting submissions. Two of them move ONE discovery value and
    nothing else at all."""
    out = {}

    # 1. The shallow solution: assume the lexicographic identifier order throughout.
    natural = lexicographic_order(store)
    natural_terminal = natural[-1]
    natural_instance = instances[store["atoms"][natural_terminal]["instance_id"]]
    natural_optimum, natural_cover = exact_optimum(natural_instance)
    natural_digests = digests_under_order(store, natural)
    out["natural_sort_order"] = {
        "schema": "oer30.submission/v1",
        "attestation_order": natural,
        "seal_chain": seals_under_order(store, natural),
        "atom_digests": natural_digests,
        "terminal_atom_id": natural_terminal,
        "terminal_atom_digest": natural_digests[natural_terminal],
        "graded_instance_id": natural_instance["instance_id"],
        "submitted_cover": natural_cover,
        "reported_cover_weight": natural_optimum,
    }

    # 2. NEAR MISS. Two adjacent entries of attestation_order transposed, no other byte moved.
    transposed = json.loads(json.dumps(reference))
    index = len(transposed["attestation_order"]) // 2
    order = transposed["attestation_order"]
    order[index], order[index + 1] = order[index + 1], order[index]
    out["order_adjacent_transposed"] = transposed

    # 3. One attestation not derived from the built state.
    fabricated = json.loads(json.dumps(reference))
    position = len(fabricated["seal_chain"]) // 3
    fabricated["seal_chain"][position] = hashlib.sha256(
        b"oer30.fabricated-attestation"
    ).hexdigest()
    out["seal_chain_fabricated"] = fabricated

    # 4. The wrong preimage: fold input atom IDENTIFIERS where the schema folds their digests.
    wrong = json.loads(json.dumps(reference))
    table = {}
    for ident in store["realised_order"]:
        record = store["atoms"][ident]
        parents = [name for name in store["realised_order"] if name in set(record["inputs"])]
        table[ident] = mint_store.atom_digest(record, parents)
    wrong["atom_digests"] = table
    wrong["terminal_atom_digest"] = table[wrong["terminal_atom_id"]]
    out["wrong_preimage_input_ids"] = wrong

    # 5. NEAR MISS. One hexadecimal character of terminal_atom_digest, no other byte moved.
    one_hex = json.loads(json.dumps(reference))
    digest = one_hex["terminal_atom_digest"]
    swapped = "0" if digest[7] != "0" else "1"
    one_hex["terminal_atom_digest"] = digest[:7] + swapped + digest[8:]
    out["terminal_digest_one_hex"] = one_hex

    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_bound(grounding: dict, store: dict, instances: dict, digests: dict) -> dict:
    terminal = store["realised_order"][-1]
    instance = instances[store["atoms"][terminal]["instance_id"]]
    optimum, _cover = exact_optimum(instance)
    attested = store["atoms"][terminal]["selected_sets"]
    baseline_weight = cover_weight(instance, attested)
    return {
        "anchors_gap": grounding["anchors"]["gap_id"],
        "anchors_state": grounding["anchors"]["anchors_state"],
        "atom_count": int(store["atom_count"]),
        "banner": BANNER,
        "family_baseline_metric": None,
        "family_target_metric": None,
        "genesis_prev_seal": mint_store.GENESIS_PREV_SEAL,
        "instance_anchor_gap": grounding["anchors"]["instance_local_anchors"]["declared_gap"],
        "instance_baseline_gap": normalised_gap(baseline_weight, optimum),
        "instance_target_gap": 0.0,
        "instances_sha256": digests["instances"],
        "ledger_source_sha256": digests["ledger_source"],
        "minter_sha256": digests["minter"],
        "schema": "oer30.bound/v1",
        "source": "solution/grounding.yaml",
        "weight_tolerance": 0,
    }


def render_truth(grounding: dict, store: dict, instances: dict, bound: dict) -> str:
    terminal = store["realised_order"][-1]
    record = store["atoms"][terminal]
    instance = instances[record["instance_id"]]
    optimum, cover = exact_optimum(instance)
    attested = record["selected_sets"]
    baseline_weight = cover_weight(instance, attested)
    natural = lexicographic_order(store)
    kahn = kahn_sorted_order({"atoms": [store["atoms"][k] for k in sorted(store["atoms"])]})
    lines = []
    lines.append("# TRUTH.md")
    lines.append("")
    lines.append(BANNER)
    lines.append("")
    lines.append("Source: `solution/grounding.yaml`. Regenerate with `solution/recompute.py`.")
    lines.append("")
    lines.append("## What is graded")
    lines.append("")
    lines.append(
        "The normalised gap to the exact optimum on the graded frozen set-cover instance, "
        "direction lower is better. The verifier computes the optimum itself with an exact "
        "bitmask dynamic program over the frozen instance, so the comparator is a classical "
        "exact solver and never a heuristic estimate. The graded instance is the one attested "
        "by the last atom in the realised attestation order, so the instance itself is not "
        "identifiable without recovering that order."
    )
    lines.append("")
    lines.append("## The two discovery values")
    lines.append("")
    lines.append(
        "Both are established in built environment state by `environment/mint_store.py` at "
        "image build time and neither appears in `instruction.md` or in any other agent-visible "
        "bundle byte."
    )
    lines.append("")
    lines.append("| discovery value | realised value |")
    lines.append("|---|---|")
    lines.append("| terminal_atom_digest | `" + store["atom_digests"][terminal] + "` |")
    lines.append(
        "| realised_attestation_order | `" + " ".join(store["realised_order"]) + "` |"
    )
    lines.append("")
    lines.append(
        "The realised order is a topological order of the derivation graph and it is neither "
        "of the two orders a reader reaches for instead. The lexicographic identifier order "
        "ends at `" + natural[-1] + "` and the ready-set-sorted topological order ends at `"
        + kahn[-1] + "`, while the realised order ends at `" + terminal + "`."
    )
    lines.append("")
    lines.append("## The terminal atom")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append("| atom_id | `" + terminal + "` |")
    lines.append("| instance_id | `" + record["instance_id"] + "` |")
    lines.append("| heuristic | `" + record["heuristic"] + "` |")
    lines.append("| inputs | `" + (", ".join(record["inputs"]) or "none") + "` |")
    lines.append(
        "| attested cover | `" + ", ".join(str(v) for v in attested) + "` |"
    )
    lines.append("| attested cover weight | " + str(baseline_weight) + " |")
    lines.append("| exact optimum weight | " + str(optimum) + " |")
    lines.append("| exact optimum cover | `" + ", ".join(str(v) for v in cover) + "` |")
    lines.append("")
    lines.append("## Anchors")
    lines.append("")
    lines.append(
        "`anchors_state: absent`, gap `" + grounding["anchors"]["gap_id"] + "`. F7 carries no "
        "measured family baseline or target, so none is invented. The two normalisation "
        "numbers below are measured on this slot's own frozen substrate; they are "
        "instance-local and they are NOT family anchors, under gap `"
        + grounding["anchors"]["instance_local_anchors"]["declared_gap"] + "`."
    )
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append("| instance_baseline_gap | " + repr(bound["instance_baseline_gap"]) + " |")
    lines.append("| instance_target_gap | " + repr(bound["instance_target_gap"]) + " |")
    lines.append("| reference normalised gap | 0.0 |")
    lines.append("| reference reward | 1.0 |")
    lines.append("")
    lines.append(
        "raw = (instance_baseline_gap - agent_gap) / (instance_baseline_gap - "
        "instance_target_gap), then score = min(max(raw, 0.0), 1.0)."
    )
    lines.append("")
    lines.append("## The realised order in full")
    lines.append("")
    lines.append("| position | atom | instance | heuristic | inputs |")
    lines.append("|---|---|---|---|---|")
    for position, ident in enumerate(store["realised_order"]):
        row = store["atoms"][ident]
        lines.append(
            "| " + str(position) + " | `" + ident + "` | `" + row["instance_id"] + "` | `"
            + row["heuristic"] + "` | `" + (", ".join(row["inputs"]) or "none") + "` |"
        )
    lines.append("")
    lines.append("## Substrate limit")
    lines.append("")
    lines.append(
        "The substrate is a deterministic content-addressed store minted by arithmetic, "
        "declared under gap `" + grounding["substrate"]["declared_gap"] + "`. That is the right "
        "choice because it makes grading reproducible, and it never passes as the output of a "
        "live production build system."
    )
    lines.append("")
    return "\n".join(lines)


def render_rubrics(store: dict, instances: dict, bound: dict) -> dict:
    terminal = store["realised_order"][-1]
    instance = instances[store["atoms"][terminal]["instance_id"]]
    optimum, _cover = exact_optimum(instance)
    return {
        "banner": BANNER,
        "criteria": [
            {
                "id": "realised-order-recovered",
                "statement": "The submitted attestation order is the order the seal chain realises.",
                "comparison": "list equality against the harness-minted seal chain",
                "reference_value": list(store["realised_order"]),
            },
            {
                "id": "terminal-digest-recomputed",
                "statement": "The terminal atom digest folds the whole provenance closure under the fixed preimage.",
                "comparison": "string equality against the harness recomputation",
                "reference_value": store["atom_digests"][terminal],
            },
            {
                "id": "closure-attested",
                "statement": "Every atom in the store carries a digest in the submitted table and no atom outside it does.",
                "comparison": "set equality against the harness-minted store",
                "reference_value": int(store["atom_count"]),
            },
            {
                "id": "graded-instance-identified",
                "statement": "The graded instance is the one attested by the last atom in the realised order.",
                "comparison": "string equality against the harness-minted store",
                "reference_value": instance["instance_id"],
            },
            {
                "id": "cover-optimal",
                "statement": "The submitted cover reaches the exact optimum weight on the graded instance.",
                "comparison": "verifier-recomputed cover weight is <= reference_value",
                "reference_value": optimum,
            },
            {
                "id": "gap-closed",
                "statement": "The normalised gap reaches the instance target from the instance baseline.",
                "comparison": "verifier-recomputed normalised gap is <= reference_value",
                "reference_value": bound["instance_target_gap"],
            },
        ],
        "judged": "the solution against its reference answer",
        "schema": "forge.solution_rubrics/v1",
        "source": "solution/grounding.yaml",
    }


SOLVE_SH = """#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It reads the sealed provenance store out of BUILT ENVIRONMENT
# STATE through the read-only handle, walks the seal chain from genesis to recover the
# realised attestation order, folds every atom digest in that order, solves the graded
# instance exactly, and writes submission.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${OER30_WORKSPACE:-${PWD}/workspace}"
STORE="${OER30_STORE:-/task/state/store}"
INSTANCES="${OER30_INSTANCES:-/task/environment/instances.json}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \\
    --store "${STORE}" \\
    --instances "${INSTANCES}" \\
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
"""


TEST_OUTPUT_HEAD = '''"""GENERATED SECTION. DO NOT HAND-EDIT.

Source: solution/grounding.yaml. Regenerate with solution/recompute.py.

One compiled test per graded checker, plus one per frozen rejecting fixture and the
end-to-end reward test. Every test drives the real checkers in tests/checkers.py over the
harness-minted store, never over a fixture the author planted with an answer in it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import grade  # noqa: E402

REFERENCE = BUNDLE / 'solution' / 'fixtures' / 'reference_run'
REJECTING = BUNDLE / 'solution' / 'fixtures' / 'rejecting'


def _outcomes(workspace):
    harness = grade.build_harness(BUNDLE, workspace)
    bound = grade.load_bound(BUNDLE)
    return {row.ident: row for row in grade.run_all(harness, bound)}


def _verdict(workspace):
    harness = grade.build_harness(BUNDLE, workspace)
    bound = grade.load_bound(BUNDLE)
    return grade.score(harness, bound, grade.load_bound_mapping(BUNDLE))


def test_every_checker_is_declared():
    declared = {ident for ident, _selector in grade.CHECKER_ORDER}
    assert declared == set(_outcomes(REFERENCE))

'''


def render_test_output(checker_ids, rejecting) -> str:
    out = [TEST_OUTPUT_HEAD]
    for ident in checker_ids:
        out.append(
            "\ndef test_%s():\n"
            "    outcome = _outcomes(REFERENCE)['%s']\n"
            "    assert outcome.passed, outcome.detail\n" % (ident, ident)
        )
    for name, reason in rejecting:
        out.append(
            "\ndef test_rejects_%s():\n"
            "    verdict = _verdict(REJECTING / '%s')\n"
            "    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)\n"
            "    assert verdict['reason'] == '%s', verdict['reason']\n" % (name, name, reason)
        )
    out.append(
        "\ndef test_reference_scores_full_reward():\n"
        "    verdict = _verdict(REFERENCE)\n"
        "    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)\n"
        "    assert verdict['reason'] == 'graded', verdict['reason']\n"
    )
    return "".join(out)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _write(path: Path, text: str, plan: dict) -> None:
    plan[str(path)] = text


def _dump_json(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def build_plan(grounding: dict) -> dict:
    plan: dict = {}

    instances_doc = build_instances(grounding["instances"])
    instance_ids = [row["instance_id"] for row in instances_doc["instances"]]
    ledger_doc = build_ledger(grounding["ledger"], instance_ids)

    _write(BUNDLE / "environment" / "instances.json", _dump_json(instances_doc), plan)
    _write(BUNDLE / "environment" / "ledger_source.json", _dump_json(ledger_doc), plan)

    instances = {row["instance_id"]: row for row in instances_doc["instances"]}
    store = mint_store.mint(ledger_doc, instances)

    # Both inequalities the discovery block promises, asserted rather than assumed.
    natural = lexicographic_order(store)
    kahn = kahn_sorted_order(ledger_doc)
    if store["realised_order"] == natural:
        raise SystemExit("the realised order equals the lexicographic order; discovery is decorative")
    if store["realised_order"] == kahn:
        raise SystemExit("the realised order equals the ready-set-sorted order; discovery is decorative")

    terminal = store["realised_order"][-1]
    if not store["atoms"][terminal]["inputs"]:
        raise SystemExit("the terminal atom has no inputs; the ordering does not reach its digest")
    if terminal == natural[-1] or terminal == kahn[-1]:
        raise SystemExit(
            "the realised order ends at the same atom as a natural order, so the graded "
            "instance would be identifiable without recovering the order"
        )

    instance = instances[store["atoms"][terminal]["instance_id"]]
    optimum, optimum_cover = exact_optimum(instance)
    baseline_weight = cover_weight(instance, store["atoms"][terminal]["selected_sets"])
    if baseline_weight <= optimum:
        raise SystemExit("the attested cover is already optimal; the gap term carries no gradient")
    if len(optimum_cover) < 2:
        raise SystemExit("the exact optimum on the graded instance is a single set; degenerate")

    digests = {
        "instances": hashlib.sha256(_dump_json(instances_doc).encode("utf-8")).hexdigest(),
        "ledger_source": hashlib.sha256(_dump_json(ledger_doc).encode("utf-8")).hexdigest(),
        "minter": hashlib.sha256(
            (BUNDLE / "environment" / "mint_store.py").read_bytes()
        ).hexdigest(),
    }

    bound = render_bound(grounding, store, instances, digests)
    _write(BUNDLE / "tests" / "bound.json", _dump_json(bound), plan)

    reference = build_reference_submission(store, instances)
    _write(
        BUNDLE / "solution" / "fixtures" / "reference_run" / "submission.json",
        _dump_json(reference),
        plan,
    )
    rejecting = build_rejecting(store, instances, reference)
    for name, payload in sorted(rejecting.items()):
        _write(
            BUNDLE / "solution" / "fixtures" / "rejecting" / name / "submission.json",
            _dump_json(payload),
            plan,
        )

    checker_ids = [row["id"] for row in grounding["checkers"]]
    expected = {row["name"]: row["expected_reason"] for row in grounding["fixtures"]["rejecting"]}
    _write(
        BUNDLE / "tests" / "test_output.py",
        render_test_output(checker_ids, sorted(expected.items())),
        plan,
    )

    _write(BUNDLE / "solution" / "TRUTH.md", render_truth(grounding, store, instances, bound), plan)
    _write(BUNDLE / "solution" / "rubrics.json", _dump_json(render_rubrics(store, instances, bound)), plan)
    _write(BUNDLE / "solution" / "solve.sh", SOLVE_SH, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description="derive every generated OER-30 artifact")
    parser.add_argument("--check", action="store_true", help="report drift and write nothing")
    args = parser.parse_args()

    grounding = yaml.safe_load((HERE / "grounding.yaml").read_text(encoding="utf-8"))
    plan = build_plan(grounding)

    drift = []
    for path, text in sorted(plan.items()):
        current = Path(path).read_text(encoding="utf-8") if Path(path).is_file() else None
        if current != text:
            drift.append(path)
    if args.check:
        print(json.dumps({"drifted": drift, "checked": len(plan)}, indent=2, sort_keys=True))
        return 1 if drift else 0

    for path, text in sorted(plan.items()):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    (BUNDLE / "solution" / "solve.sh").chmod(0o755)
    print(json.dumps({"written": len(plan), "drifted_before_write": drift}, indent=2, sort_keys=True))
    return 0


# FORGE-SCREENING-CARRIER-BEGIN
# GENERATED SECTION. DO NOT HAND-EDIT.
# Generated by seed/forge/screenfreeze.py. Derives the contamination-screening provenance carrier
# from the frozen `screening` block in solution/grounding.yaml and nothing else. It opens no
# connection, reads no wall clock, consults no host language setting, draws no entropy, starts no
# child process, and imports nothing outside this tree.
import hashlib as _forge_hashlib
import json as _forge_json
import pathlib as _forge_pathlib
import sys as _forge_sys

import yaml as _forge_yaml

_FORGE_CARRIER_KEYS = (
    "schema",
    "unit_uuid",
    "screening_roots",
    "authority_mode",
    "source_identifiers",
    "fork_ancestry_snapshot",
    "base_commit_sha",
    "applicable_dates",
    "instrument_versions",
    "atom_result_digests",
    "applicability",
    "sanitization_closure",
    "empty_submission_result",
    "attestations",
    "binding_block",
    "keyid",
    "trust_root_public_key_hex",
    "namespace",
    "normalization_domain_version",
    "signer_identity",
    "screening_measured_at",
    "screening_interval_days",
    "screening_expires_at",
)

_FORGE_BINDING_KEYS = (
    "canonical_bundle_hash",
    "pinned_image_digest",
    "binding_envelope",
)

_FORGE_SCREENING_KEY = "screening"
_FORGE_GROUNDING = "grounding.yaml"
_FORGE_CARRIER = "provenance.yaml"
_FORGE_BANNER = "# GENERATED SECTION. DO NOT HAND-EDIT."


def _forge_here():
    return _forge_pathlib.Path(__file__).resolve().parent


def _forge_sorted(value):
    """Sort every container so two runs over the same frozen bytes emit identical bytes."""
    if isinstance(value, dict):
        return {key: _forge_sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_forge_sorted(item) for item in value]
    return value


def _forge_frozen_screening():
    """Read the frozen screening block. Absence is refused rather than defaulted."""
    path = _forge_here() / _FORGE_GROUNDING
    with path.open("r", encoding="utf-8") as handle:
        document = _forge_yaml.safe_load(handle)
    block = (document or {}).get(_FORGE_SCREENING_KEY)
    if not isinstance(block, dict):
        raise SystemExit(
            "solution/grounding.yaml carries no frozen `screening` block, so the provenance "
            "carrier cannot be derived. Refusing to emit a carrier over values nobody froze."
        )
    missing = [key for key in _FORGE_CARRIER_KEYS if key not in block]
    unknown = [key for key in sorted(block) if key not in _FORGE_CARRIER_KEYS]
    if missing or unknown:
        raise SystemExit(
            "the frozen `screening` block does not mirror the closed carrier schema: "
            "missing " + repr(missing) + ", unknown " + repr(unknown)
        )
    return block


def _forge_canonical_bytes(payload):
    return _forge_json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _forge_carrier_payload():
    """Assemble the carrier as exactly the closed key set, in the order the schema fixes.

    The binding block is attached AFTER the canonical payload is hashed and never enters the
    preimage, because a payload that contained a hash of itself would have no acyclic ordering.
    """
    block = _forge_frozen_screening()
    payload = {}
    for key in _FORGE_CARRIER_KEYS:
        if key == "binding_block":
            continue
        payload[key] = _forge_sorted(block[key])
    digest = _forge_hashlib.sha256(_forge_canonical_bytes(payload)).hexdigest()

    binding = _forge_sorted(block["binding_block"]) or {}
    shaped = {key: binding.get(key) for key in _FORGE_BINDING_KEYS}
    ordered = {}
    for key in _FORGE_CARRIER_KEYS:
        ordered[key] = shaped if key == "binding_block" else payload[key]
    return ordered, digest


def _forge_carrier_text():
    payload, digest = _forge_carrier_payload()
    header = (
        _FORGE_BANNER + "\n"
        + "# Derived from solution/grounding.yaml `screening` by solution/recompute.py.\n"
        + "# canonical payload sha256 (binding_block excluded from the preimage): " + digest + "\n"
    )
    body = _forge_yaml.safe_dump(
        payload, sort_keys=False, default_flow_style=False, allow_unicode=False, width=100
    )
    return header + body


def _forge_emit_carrier():
    """Write the carrier, or in check mode compare and report drift. Never both."""
    argv = list(_forge_sys.argv[1:])
    check = "--check" in argv
    path = _forge_here() / _FORGE_CARRIER
    text = _forge_carrier_text()
    if check:
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        if current == text:
            return 0
        _forge_sys.stderr.write(
            "drift: " + _FORGE_CARRIER + " does not match the carrier derived from the frozen "
            "`screening` block in " + _FORGE_GROUNDING + "\n"
        )
        return 1
    path.write_text(text, encoding="utf-8")
    return 0


_FORGE_INNER_MAIN = main


def main(*args, **kwargs):
    """Run the host generator, then derive the provenance carrier from the frozen block."""
    status = _FORGE_INNER_MAIN(*args, **kwargs)
    drift = _forge_emit_carrier()
    if drift and not status:
        return drift
    return status

# FORGE-SCREENING-CARRIER-END


if __name__ == "__main__":
    raise SystemExit(main())

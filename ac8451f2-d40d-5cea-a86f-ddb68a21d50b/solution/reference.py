#!/usr/bin/env python3
"""The oracle. It solves OER-30 by reading built environment state and nothing else.

WHAT IT IS ALLOWED TO SEE
-------------------------
The sealed provenance store, through the read-only handle's own accessors, and the frozen
instance family. It does not read solution/grounding.yaml, does not read tests/bound.json,
does not import the minter, and carries no literal for either discovery value. Every number it
writes is derived here, in this order:

  1. walk the seal chain from the single attestation whose prev_seal is the all-zero seal,
     following prev_seal to seal, which recovers the REALISED ATTESTATION ORDER;
  2. fold each atom's canonical digest in that order, so an atom's inputs contribute their
     digests in the order they were sealed rather than in identifier order;
  3. take the last atom of the realised order, which names the graded instance;
  4. solve that instance exactly with a bitmask dynamic program and submit the optimum.

Step 1 is the only step that reveals which instance is graded, which is why an oracle that
skipped it would be solving a different instance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GENESIS_PREV_SEAL = "0" * 64
SCHEMA_ATOM = "oer30.atom/v1"
SCHEMA_ATTESTATION = "oer30.attestation/v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cover_sha256(selected_sets) -> str:
    return _sha256_text(",".join(str(int(value)) for value in sorted(selected_sets)))


def atom_digest(atom, input_digests) -> str:
    return _sha256_text(
        SCHEMA_ATOM + "\n"
        + "atom_id=" + str(atom["atom_id"]) + "\n"
        + "instance_id=" + str(atom["instance_id"]) + "\n"
        + "heuristic=" + str(atom["heuristic"]) + "\n"
        + "cover=" + cover_sha256(atom["selected_sets"]) + "\n"
        + "inputs=" + ";".join(input_digests) + "\n"
    )


def walk_chain(attestations):
    """The ordering predicate, executed rather than assumed."""
    by_prev = {str(row["prev_seal"]): row for row in attestations}
    order, seals = [], []
    cursor = GENESIS_PREV_SEAL
    while cursor in by_prev:
        row = by_prev[cursor]
        order.append(str(row["atom_id"]))
        seals.append(str(row["seal"]))
        cursor = str(row["seal"])
    return order, seals


def fold_digests(atoms, order):
    position = {ident: index for index, ident in enumerate(order)}
    digests = {}
    for ident in order:
        record = atoms[ident]
        parents = sorted(record.get("inputs") or [], key=lambda name: position[name])
        digests[ident] = atom_digest(record, [digests[name] for name in parents])
    return digests


def exact_cover(instance):
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
            if following != state and best[state] + weight < best[following]:
                best[following] = best[state] + weight
                pick[following] = (state, ident)
    chosen, state = [], full
    while state and pick[state] is not None:
        previous, ident = pick[state]
        chosen.append(ident)
        state = previous
    return int(best[full]), sorted(chosen)


def read_store(handle_dir: Path, store: Path):
    sys.path.insert(0, str(handle_dir))
    import store_handle  # noqa: E402

    atoms = {}
    for ident in store_handle.atom_ids(store):
        atoms[ident] = store_handle.atom(store, ident)
    attestations = [store_handle.seal(store, value) for value in store_handle.seal_ids(store)]
    return atoms, attestations


def main():
    parser = argparse.ArgumentParser(description="solve OER-30 from built environment state")
    parser.add_argument("--store", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--handle-dir",
        default=None,
        help="where store_handle.py lives; defaults beside the instance family",
    )
    args = parser.parse_args()

    instances_path = Path(args.instances)
    handle_dir = Path(args.handle_dir) if args.handle_dir else instances_path.parent
    atoms, attestations = read_store(handle_dir, Path(args.store))

    order, seals = walk_chain(attestations)
    if len(order) != len(atoms):
        raise SystemExit(
            "the seal chain visits " + str(len(order)) + " of " + str(len(atoms)) + " atoms"
        )
    digests = fold_digests(atoms, order)

    terminal = order[-1]
    instances = {
        row["instance_id"]: row
        for row in json.loads(instances_path.read_text(encoding="utf-8"))["instances"]
    }
    instance = instances[atoms[terminal]["instance_id"]]
    weight, cover = exact_cover(instance)

    submission = {
        "schema": "oer30.submission/v1",
        "attestation_order": order,
        "seal_chain": seals,
        "atom_digests": digests,
        "terminal_atom_id": terminal,
        "terminal_atom_digest": digests[terminal],
        "graded_instance_id": instance["instance_id"],
        "submitted_cover": cover,
        "reported_cover_weight": weight,
    }

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "submission.json").write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "atoms": len(atoms),
                "chain_length": len(order),
                "terminal_atom_id": terminal,
                "graded_instance_id": instance["instance_id"],
                "cover_weight": weight,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

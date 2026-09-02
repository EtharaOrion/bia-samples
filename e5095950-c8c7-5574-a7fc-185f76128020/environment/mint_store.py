#!/usr/bin/env python3
"""The minter. It materialises the sealed provenance store from the frozen ledger source.

WHAT THIS IS
------------
A provenance store is a directory of two kinds of record.

  atoms/<atom_id>.json      one unit of work: a candidate cover for one frozen set-cover
                            instance, produced by a named heuristic, declaring which earlier
                            atoms it derived from.

  seals/<seal>.json         one attestation over one atom. Each attestation carries the seal
                            of the attestation sealed immediately before it, so the whole
                            set of attestations is a hash chain. The FILE NAME is the seal
                            itself, which is content-addressed, so a directory listing
                            carries no information about the order they were sealed in.

THE ORDER THIS MINTER REALISES
------------------------------
The minter seals atoms one at a time. At each step the set of atoms whose every input has
already been sealed is the ready set, and the minter picks one of them by advancing a fixed
integer linear congruential recurrence and indexing the ready set with the result. The
recurrence is arithmetic and fully determined by the seed in the frozen ledger source; no
module named `random` is imported and no clock is read. Two runs over the same frozen source
therefore realise the same order, byte for byte, on any host.

The realised order is a topological order of the derivation graph, because an atom is only
ever sealed after every atom it derives from. It is NOT the lexicographic order of the atom
identifiers, and it is not recorded as a literal anywhere: it exists only as the shape of the
seal chain the store carries.

WHY THE ORDER IS LOAD-BEARING
-----------------------------
An atom's canonical digest folds the digests of the atoms it derives from, and it folds them
in realised attestation order rather than in identifier order. A reader who assumes the
identifier order therefore computes a different digest for every atom with two or more
inputs, and for every atom downstream of one.

NO CLOCK, NO NETWORK, NO RANDOM SOURCE
--------------------------------------
Imports are confined to argparse, hashlib, json, pathlib and sys. `write_index` inside an
atom record is the index the record was WRITTEN to disk at, which is its lexicographic
position; it is deliberately not the attestation order and it is excluded from every
preimage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA_ATOM = "oer30.atom/v1"
SCHEMA_ATTESTATION = "oer30.attestation/v1"
SCHEMA_STORE = "oer30.store/v1"

GENESIS_PREV_SEAL = "0" * 64

# The fixed integer linear congruential recurrence. Arithmetic, not a random source.
LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345
LCG_MODULUS = 2 ** 31


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cover_sha256(selected_sets) -> str:
    """The canonical digest of a cover: the ascending set identifiers, comma joined."""
    return _sha256_text(",".join(str(int(value)) for value in sorted(selected_sets)))


def atom_preimage(atom: dict, input_digests) -> str:
    """The canonical preimage of one atom. Five lines, fixed order, newline terminated.

    `input_digests` is the list of the digests of the atoms this one derives from, ALREADY
    ordered by realised attestation order. Three fields of the atom record are outside this
    preimage on purpose: `note`, `producer_host` and `write_index`.
    """
    return (
        SCHEMA_ATOM + "\n"
        + "atom_id=" + str(atom["atom_id"]) + "\n"
        + "instance_id=" + str(atom["instance_id"]) + "\n"
        + "heuristic=" + str(atom["heuristic"]) + "\n"
        + "cover=" + cover_sha256(atom["selected_sets"]) + "\n"
        + "inputs=" + ";".join(input_digests) + "\n"
    )


def atom_digest(atom: dict, input_digests) -> str:
    return _sha256_text(atom_preimage(atom, input_digests))


def attestation_preimage(atom_id: str, cover_digest: str, prev_seal: str) -> str:
    """The canonical preimage of one attestation. Four lines, fixed order."""
    return (
        SCHEMA_ATTESTATION + "\n"
        + "atom_id=" + str(atom_id) + "\n"
        + "cover_sha256=" + str(cover_digest) + "\n"
        + "prev_seal=" + str(prev_seal) + "\n"
    )


def attestation_seal(atom_id: str, cover_digest: str, prev_seal: str) -> str:
    return _sha256_text(attestation_preimage(atom_id, cover_digest, prev_seal))


# ---------------------------------------------------------------------------
# The two heuristics. Both are deterministic scans over the frozen instance.
# ---------------------------------------------------------------------------


def _members_mask(members) -> int:
    mask = 0
    for element in members:
        mask |= 1 << int(element)
    return mask


def _working_sets(instance: dict):
    """Every set except the fallback. The fallback covers the whole universe by construction,
    so a heuristic allowed to see it would answer every instance with it and would stop being
    a heuristic. It is appended by `_completed` only when the working sets leave the universe
    uncovered."""
    fallback = int(instance["fallback_set_id"])
    return [row for row in instance["sets"] if int(row["id"]) != fallback], fallback


def _completed(instance: dict, chosen, covered: int, fallback: int):
    if covered != (1 << int(instance["universe_size"])) - 1:
        chosen = list(chosen) + [fallback]
    return sorted(chosen)


def heuristic_first_fit(instance: dict, start: int):
    """Scan the working sets ascending from `start`, wrapping once, taking any set that
    covers at least one element not yet covered. Stops as soon as the universe is covered."""
    sets, fallback = _working_sets(instance)
    full = (1 << int(instance["universe_size"])) - 1
    covered = 0
    chosen = []
    count = len(sets)
    for offset in range(count):
        index = (start + offset) % count
        mask = _members_mask(sets[index]["members"])
        if mask & ~covered:
            chosen.append(int(sets[index]["id"]))
            covered |= mask
            if covered == full:
                break
    return _completed(instance, chosen, covered, fallback)


def heuristic_weight_greedy(instance: dict, start: int):
    """Repeatedly take the working set with the lowest weight per newly covered element,
    breaking ties by the earliest position at or after `start` in wrapped order."""
    sets, fallback = _working_sets(instance)
    full = (1 << int(instance["universe_size"])) - 1
    covered = 0
    chosen = []
    count = len(sets)
    while covered != full:
        best = None
        for offset in range(count):
            index = (start + offset) % count
            candidate = sets[index]
            if int(candidate["id"]) in chosen:
                continue
            gain = bin(_members_mask(candidate["members"]) & ~covered).count("1")
            if gain == 0:
                continue
            ratio = float(candidate["weight"]) / float(gain)
            key = (ratio, offset)
            if best is None or key < best[0]:
                best = (key, index)
        if best is None:
            break
        index = best[1]
        chosen.append(int(sets[index]["id"]))
        covered |= _members_mask(sets[index]["members"])
    return _completed(instance, chosen, covered, fallback)


HEURISTICS = {
    "first-fit": heuristic_first_fit,
    "weight-greedy": heuristic_weight_greedy,
}


def derive_start(input_atoms, set_count: int) -> int:
    """The scan offset an atom inherits from the atoms it derives from.

    An atom with no inputs starts at zero. An atom with inputs starts one past the highest
    set identifier any of its inputs selected, wrapped into range. This is the whole of the
    semantic dependence of a cover on its provenance.
    """
    if not input_atoms:
        return 0
    highest = 0
    for parent in input_atoms:
        for value in parent["selected_sets"]:
            highest = max(highest, int(value))
    return (highest + 1) % max(1, set_count)


def mint(source: dict, instances: dict) -> dict:
    """Realise the whole store from the frozen source. Returns the in-memory store."""
    declared = {row["atom_id"]: row for row in source["atoms"]}
    identifiers = sorted(declared)
    inputs_of = {ident: list(declared[ident].get("inputs") or []) for ident in identifiers}

    state = int(source["seal_seed"])
    sealed = []
    sealed_set = set()
    records = {}
    digests = {}
    seals = []
    prev_seal = GENESIS_PREV_SEAL

    while len(sealed) < len(identifiers):
        ready = sorted(
            ident
            for ident in identifiers
            if ident not in sealed_set and all(parent in sealed_set for parent in inputs_of[ident])
        )
        if not ready:
            raise ValueError("the declared derivation graph carries a cycle")
        state = (LCG_MULTIPLIER * state + LCG_INCREMENT) % LCG_MODULUS
        chosen = ready[state % len(ready)]

        declaration = declared[chosen]
        instance = instances[declaration["instance_id"]]
        parents = [records[parent] for parent in inputs_of[chosen]]
        start = derive_start(parents, len(instance["sets"]))
        selected = HEURISTICS[declaration["heuristic"]](instance, start)

        record = {
            "schema": SCHEMA_ATOM,
            "atom_id": chosen,
            "instance_id": declaration["instance_id"],
            "heuristic": declaration["heuristic"],
            "inputs": list(inputs_of[chosen]),
            "selected_sets": selected,
            "note": declaration.get("note", ""),
            "producer_host": source["producer_host"],
            "write_index": identifiers.index(chosen),
        }
        records[chosen] = record

        # The inputs fold in REALISED attestation order, which is `sealed` so far.
        ordered_inputs = [ident for ident in sealed if ident in set(inputs_of[chosen])]
        digests[chosen] = atom_digest(record, [digests[ident] for ident in ordered_inputs])

        cover_digest = cover_sha256(selected)
        seal = attestation_seal(chosen, cover_digest, prev_seal)
        seals.append(
            {
                "schema": SCHEMA_ATTESTATION,
                "seal": seal,
                "atom_id": chosen,
                "cover_sha256": cover_digest,
                "prev_seal": prev_seal,
            }
        )
        prev_seal = seal
        sealed.append(chosen)
        sealed_set.add(chosen)

    return {
        "schema": SCHEMA_STORE,
        "atom_count": len(identifiers),
        "atoms": records,
        "attestations": seals,
        "realised_order": sealed,
        "atom_digests": digests,
    }


def write_store(store: dict, root: Path) -> None:
    """Materialise the store on disk. Atoms are written in LEXICOGRAPHIC identifier order and
    attestations under their own content-addressed seal, so neither directory leaks the
    realised order."""
    root = Path(root)
    (root / "atoms").mkdir(parents=True, exist_ok=True)
    (root / "seals").mkdir(parents=True, exist_ok=True)
    for ident in sorted(store["atoms"]):
        (root / "atoms" / (ident + ".json")).write_text(
            json.dumps(store["atoms"][ident], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for row in store["attestations"]:
        (root / "seals" / (row["seal"] + ".json")).write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA_STORE,
                "atom_count": store["atom_count"],
                "genesis_prev_seal": GENESIS_PREV_SEAL,
                "atoms_dir": "atoms",
                "seals_dir": "seals",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load(source_path: Path, instances_path: Path):
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    instances_doc = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    instances = {row["instance_id"]: row for row in instances_doc["instances"]}
    return source, instances


def main():
    parser = argparse.ArgumentParser(description="materialise the OER-30 provenance store")
    parser.add_argument("--source", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--out", required=True, help="the store root to materialise into")
    args = parser.parse_args()

    source, instances = load(Path(args.source), Path(args.instances))
    store = mint(source, instances)
    write_store(store, Path(args.out))
    print(
        json.dumps(
            {"atom_count": store["atom_count"], "store": str(Path(args.out))}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

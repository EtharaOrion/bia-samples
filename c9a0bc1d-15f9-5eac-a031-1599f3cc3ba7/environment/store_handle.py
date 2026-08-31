#!/usr/bin/env python3
"""The read-only handle onto the sealed provenance store in built environment state.

This is the only accessor the task environment ships. It reads; it never mints, never writes
and never repairs. Four subcommands:

    manifest              the store manifest: schema, atom count, genesis prev_seal
    atoms                 every atom identifier, LEXICOGRAPHIC, which is the order the atom
                          files were written in and is NOT the attestation order
    atom <atom_id>        one atom record
    seals                 every seal, sorted by seal, which is content-addressed and
                          therefore carries no ordering information at all
    seal <seal>           one attestation record

Nothing here prints the realised attestation order and nothing here prints an atom digest.
Both are properties of the store that a reader derives; this handle hands back the records
they are derived from. That is deliberate: a handle that answered the question would make the
store a lookup table instead of a provenance chain.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_STORE = os.environ.get("OER30_STORE", "/task/state/store")


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest(root: Path):
    return _load(Path(root) / "manifest.json")


def atom_ids(root: Path):
    return sorted(entry.stem for entry in (Path(root) / "atoms").glob("*.json"))


def atom(root: Path, atom_id: str):
    return _load(Path(root) / "atoms" / (atom_id + ".json"))


def seal_ids(root: Path):
    return sorted(entry.stem for entry in (Path(root) / "seals").glob("*.json"))


def seal(root: Path, value: str):
    return _load(Path(root) / "seals" / (value + ".json"))


def main():
    parser = argparse.ArgumentParser(description="read the OER-30 provenance store")
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("command", choices=["manifest", "atoms", "atom", "seals", "seal"])
    parser.add_argument("argument", nargs="?", default=None)
    args = parser.parse_args()

    root = Path(args.store)
    if args.command == "manifest":
        payload = manifest(root)
    elif args.command == "atoms":
        payload = atom_ids(root)
    elif args.command == "atom":
        payload = atom(root, args.argument)
    elif args.command == "seals":
        payload = seal_ids(root)
    else:
        payload = seal(root, args.argument)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

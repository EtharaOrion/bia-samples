#!/usr/bin/env python3
"""Build the frozen provenance warehouse, deterministically, from a frozen spec.

This module is run at image build time by environment/Dockerfile and it is also the module
solution/recompute.py imports to derive the bundle copy of the warehouse. Both paths execute
exactly these functions over exactly the same spec, so the database the image carries and the
database the bundle carries are byte-identical by construction rather than by assertion.

Determinism, and how it is obtained rather than hoped for:

  * The only source of variation is a fixed integer linear congruential recurrence seeded from
    the spec. No module named `random` is imported, no clock is read, no locale is consulted
    and no network is reached.
  * Rows are inserted in one fixed order, the page size and encoding are pinned before the
    first write, and the connection is committed once and closed. SQLite writes no timestamp
    of its own into a database file, so the resulting bytes are a pure function of the spec.

Nothing in this module knows which node is quarantined ahead of time. The quarantined node is
whatever node the corruption step lands on, and the corruption step lands where the recurrence
and the closure-band rule below put it. That is what makes the identifier a property of the
built state instead of a constant somebody typed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345
LCG_MODULUS = 2 ** 31

SCHEMA_STATEMENTS = (
    "CREATE TABLE lineage ("
    "node_id TEXT PRIMARY KEY, "
    "parent_id TEXT, "
    "p_alpha INTEGER NOT NULL, "
    "p_beta INTEGER NOT NULL, "
    "p_gamma INTEGER NOT NULL, "
    "declared_check INTEGER NOT NULL)",
    "CREATE TABLE derives ("
    "edge_id INTEGER PRIMARY KEY, "
    "source_node TEXT NOT NULL, "
    "target_node TEXT NOT NULL)",
    "CREATE TABLE reading ("
    "reading_id INTEGER PRIMARY KEY, "
    "node_id TEXT NOT NULL, "
    "value INTEGER NOT NULL)",
)


def lcg_stream(seed):
    """A fixed integer recurrence. Arithmetic, not randomness: the seed determines it all."""
    value = int(seed)
    while True:
        value = (LCG_MULTIPLIER * value + LCG_INCREMENT) % LCG_MODULUS
        yield value


def node_name(index):
    return "nd-%04d" % int(index)


def expected_check(alpha, beta, gamma, coefficients, modulus):
    """The integrity checksum a lineage row is required to declare over its own payload."""
    return (
        int(alpha) * int(coefficients["alpha"])
        + int(beta) * int(coefficients["beta"])
        + int(gamma) * int(coefficients["gamma"])
    ) % int(modulus)


def build_rows(spec):
    """Return the three tables as plain lists, before any corruption is applied."""
    stream = lcg_stream(spec["seed"])
    count = int(spec["lineage_node_count"])
    roots = int(spec["root_count"])
    window = int(spec["parent_window"])
    payload_modulus = int(spec["payload_modulus"])
    coefficients = spec["check_coefficients"]
    modulus = int(spec["check_modulus"])

    lineage = []
    for index in range(count):
        if index < roots:
            parent = None
        else:
            back = 1 + (next(stream) % min(window, index))
            parent = node_name(index - back)
        alpha = next(stream) % payload_modulus
        beta = next(stream) % payload_modulus
        gamma = next(stream) % payload_modulus
        lineage.append(
            {
                "node_id": node_name(index),
                "parent_id": parent,
                "p_alpha": alpha,
                "p_beta": beta,
                "p_gamma": gamma,
                "declared_check": expected_check(alpha, beta, gamma, coefficients, modulus),
            }
        )

    span = int(spec["derives_forward_span"])
    seen = set()
    derives = []
    for _ in range(int(spec["derives_edge_count"])):
        source_index = next(stream) % count
        target_index = source_index + 1 + (next(stream) % span)
        if target_index >= count:
            continue
        key = (source_index, target_index)
        if key in seen:
            continue
        seen.add(key)
        derives.append(
            {
                "edge_id": len(derives) + 1,
                "source_node": node_name(source_index),
                "target_node": node_name(target_index),
            }
        )

    value_modulus = int(spec["reading_value_modulus"])
    reading = []
    for index in range(int(spec["reading_row_count"])):
        node_index = next(stream) % count
        reading.append(
            {
                "reading_id": index + 1,
                "node_id": node_name(node_index),
                "value": next(stream) % value_modulus,
            }
        )

    return lineage, derives, reading


def forward_edges(lineage, derives):
    """The closure relation: parent to child, and derives source to derives target, unioned."""
    edges = {}
    for row in lineage:
        parent = row["parent_id"]
        if parent is not None:
            edges.setdefault(parent, set()).add(row["node_id"])
    for row in derives:
        edges.setdefault(row["source_node"], set()).add(row["target_node"])
    return edges


def closure_layers(edges, origin):
    """Breadth-first layers of the closure from `origin`, layer 0 being the origin itself.

    A node appears in exactly one layer, the first one that reaches it. This is the traversal
    the graded closure cardinality has to come out of, and it is why a cardinality that was
    guessed rather than traversed cannot be made consistent with the layers beside it.
    """
    seen = {origin}
    layers = [[origin]]
    frontier = [origin]
    while frontier:
        nxt = []
        for node in frontier:
            for child in sorted(edges.get(node, ())):
                if child not in seen:
                    seen.add(child)
                    nxt.append(child)
        if not nxt:
            break
        nxt.sort()
        layers.append(nxt)
        frontier = nxt
    return layers


def closure_from(edges, origin):
    nodes = set()
    for layer in closure_layers(edges, origin):
        nodes.update(layer)
    return nodes


def select_corruption_index(spec, lineage, derives):
    """Where the corruption lands, decided by the built graph and not by an authored constant.

    The recurrence proposes candidate indices. The first candidate whose closure cardinality
    falls inside the bound band, and whose closure is at least the bound minimum depth deep, is
    the one that gets corrupted. The band exists so the closure is big enough that stopping at
    the first hop is a visible mistake and small enough that the whole task stays cheap; the
    node it selects is still whatever the graph hands back.
    """
    edges = forward_edges(lineage, derives)
    low, high = (int(value) for value in spec["closure_band"])
    min_depth = int(spec["closure_min_depth"])
    count = len(lineage)
    stream = lcg_stream(int(spec["corruption_probe_seed"]))
    for _ in range(int(spec["corruption_probe_limit"])):
        index = next(stream) % count
        layers = closure_layers(edges, node_name(index))
        size = sum(len(layer) for layer in layers)
        if low <= size <= high and len(layers) - 1 >= min_depth:
            return index
    raise SystemExit(
        "no lineage node satisfies the bound closure band, so the warehouse cannot be built; "
        "refusing to widen the band silently"
    )


def apply_corruption(spec, lineage, index):
    """Break exactly one declared checksum. Exactly one, and the row is returned by identity."""
    row = lineage[index]
    row["declared_check"] = (
        row["declared_check"] + int(spec["corruption_delta"])
    ) % int(spec["check_modulus"])
    return row["node_id"]


def compose(spec):
    """The whole build, as data. Returns the tables plus the facts the build established."""
    lineage, derives, reading = build_rows(spec)
    index = select_corruption_index(spec, lineage, derives)
    quarantined = apply_corruption(spec, lineage, index)
    edges = forward_edges(lineage, derives)
    layers = closure_layers(edges, quarantined)
    closure = sorted(closure_from(edges, quarantined))
    return {
        "lineage": lineage,
        "derives": derives,
        "reading": reading,
        "quarantined_node_id": quarantined,
        "closure_nodes": closure,
        "closure_layers": layers,
        "closure_cardinality": len(closure),
    }


def write_database(path, tables):
    """Write the three tables to a fresh SQLite file with the page geometry pinned first."""
    path = Path(path)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("PRAGMA page_size = 4096")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA encoding = 'UTF-8'")
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.executemany(
            "INSERT INTO lineage VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row["node_id"],
                    row["parent_id"],
                    row["p_alpha"],
                    row["p_beta"],
                    row["p_gamma"],
                    row["declared_check"],
                )
                for row in tables["lineage"]
            ],
        )
        connection.executemany(
            "INSERT INTO derives VALUES (?, ?, ?)",
            [
                (row["edge_id"], row["source_node"], row["target_node"])
                for row in tables["derives"]
            ],
        )
        connection.executemany(
            "INSERT INTO reading VALUES (?, ?, ?)",
            [(row["reading_id"], row["node_id"], row["value"]) for row in tables["reading"]],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="build the frozen provenance warehouse")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    tables = compose(spec)
    write_database(args.out, tables)
    print(
        json.dumps(
            {
                "lineage_rows": len(tables["lineage"]),
                "derives_rows": len(tables["derives"]),
                "reading_rows": len(tables["reading"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

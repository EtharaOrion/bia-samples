#!/usr/bin/env python3
"""The reference solver. It solves the task the way the task is meant to be solved.

Nothing here is planted. The adjudication threshold and the collision witness are not
constants in this file: the threshold is read back out of the BUILT index through the same
query handle the solving agent has, and the witness is derived from the built index's own
collision records once the duplicate grouping has been carried out.

The four steps, in the order a solver has to take them:

  1. read the calibration back from the built index, which is where the adjudication
     threshold exists and the only place it exists;
  2. recompute the exact quantised similarity of EVERY unordered pair of distinct records,
     not only of the pairs the index proposed, because the index's candidates are a filter
     and not the universe of adjudication;
  3. close the thresholded edges into duplicate groups and take every within-group pair;
  4. read the built index's collisions back and pick the maximal one the grouping refused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent


def _environment_dir(explicit):
    if explicit:
        return Path(explicit)
    candidate = Path("/task/environment")
    if candidate.is_dir():
        return candidate
    return BUNDLE / "environment"


def query(environment_dir: Path, index_dir: Path, *arguments):
    """Read the built index back through the handle the agent surface exposes."""
    argv = [sys.executable, str(environment_dir / "index_query.py"), "--index", str(index_dir)]
    argv.extend(arguments)
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="solve OER-27")
    parser.add_argument("--environment", default=None)
    parser.add_argument("--index", default=os.environ.get("OER27_INDEX", "/task/index"))
    parser.add_argument("--workspace", required=True)
    arguments = parser.parse_args()

    environment_dir = _environment_dir(arguments.environment)
    index_dir = Path(arguments.index)
    sys.path.insert(0, str(environment_dir))
    import nd_index  # noqa: E402

    # ---- step 1: the threshold is READ BACK from the built index -----------------------
    calibration = json.loads(query(environment_dir, index_dir, "calibration"))
    threshold = int(calibration["adjudication_threshold"])

    collisions = []
    for line in query(environment_dir, index_dir, "collisions").splitlines():
        line = line.strip()
        if line:
            collisions.append(json.loads(line))

    spec = nd_index.load_spec(environment_dir / "index_spec.json")
    order = int(spec["shingle_order"])
    quantum = int(spec["similarity_quantum"])
    corpus = nd_index.load_corpus(environment_dir / "corpus.jsonl")
    identifiers = [str(row["id"]) for row in corpus]
    shingle_sets = {
        str(row["id"]): nd_index.shingles(nd_index.normalise(str(row["text"])), order)
        for row in corpus
    }

    # ---- step 2: every unordered pair, not only the index's candidates -----------------
    edges = []
    examined = 0
    for left in range(len(identifiers)):
        a = identifiers[left]
        set_a = shingle_sets[a]
        for right in range(left + 1, len(identifiers)):
            b = identifiers[right]
            examined += 1
            if nd_index.quantised_similarity(set_a, shingle_sets[b], quantum) >= threshold:
                edges.append((a, b))

    # ---- step 3: close the edges into duplicate groups ---------------------------------
    parent = {identifier: identifier for identifier in identifiers}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    members = {}
    for identifier in identifiers:
        members.setdefault(find(identifier), []).append(identifier)

    pairs = []
    group_count = 0
    for group in members.values():
        group = sorted(group)
        if len(group) > 1:
            group_count += 1
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                pairs.append([group[left], group[right]])
    pairs.sort()

    # ---- step 4: the maximal collision the grouping refused ----------------------------
    labels = {identifier: find(identifier) for identifier in identifiers}
    witness = None
    witness_key = None
    for row in collisions:
        a, b = str(row["a"]), str(row["b"])
        if labels[a] == labels[b]:
            continue
        key = (int(row["bands"]), int(row["similarity"]), a, b)
        if witness_key is None:
            witness, witness_key = row, key
            continue
        if key[0] > witness_key[0]:
            witness, witness_key = row, key
        elif key[0] == witness_key[0]:
            if key[1] > witness_key[1]:
                witness, witness_key = row, key
            elif key[1] == witness_key[1] and (key[2], key[3]) < (witness_key[2], witness_key[3]):
                witness, witness_key = row, key
    if witness is None:
        raise SystemExit("the built index produced no collision the adjudication refused")

    submission = {
        "schema": "oer27.submission/v1",
        "adjudication_threshold": threshold,
        "index_collision_count": len(collisions),
        "collision_witness": {
            "a": str(witness["a"]),
            "b": str(witness["b"]),
            "bands": int(witness["bands"]),
            "similarity": int(witness["similarity"]),
        },
        "near_duplicate_pairs": pairs,
        "duplicate_group_count": group_count,
        "coverage": {
            "records_examined": len(identifiers),
            "pairs_examined": examined,
            "restricted_to_index_candidates": False,
        },
        "readout": {"source": "recomputed-from-shingle-sets", "estimator": "exact"},
    }

    workspace = Path(arguments.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "submission.json").write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "adjudication_threshold": threshold,
        "index_collision_count": len(collisions),
        "direct_edges": len(edges),
        "duplicate_groups": group_count,
        "near_duplicate_pairs": len(pairs),
        "pairs_examined": examined,
        "collision_witness": submission["collision_witness"],
    }
    (workspace / "trace.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

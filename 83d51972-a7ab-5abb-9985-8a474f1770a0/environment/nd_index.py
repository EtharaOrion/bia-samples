"""The near-duplicate index core. Arithmetic only, standard library only.

Nothing in this module reads a clock, opens a socket, consults an environment secret or
draws from a random source. Every number it produces is a deterministic function of the
frozen corpus bytes and the frozen index specification, so the same corpus and the same
specification build the same index on any host.

The module is shared by three callers and is the single definition of the index:

  * environment/build_index.py builds the index into a directory at image build time,
  * environment/index_query.py reads a built index back for the solving agent,
  * tests/runner.py rebuilds the index in an isolated subprocess so the verifier grades
    against an index it produced itself rather than one a submission handed it.

No adjudication threshold is written anywhere in this file. The threshold is calibrated from
a built index's own candidate similarity distribution by `calibrate` below, so it comes into
existence only when an index is built over a corpus.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

# A Mersenne prime, used as the modulus of the permutation family. Arithmetic, not a secret.
PERMUTATION_PRIME = (1 << 61) - 1

# The linear congruential recurrence the permutation family is drawn from. This is a fixed
# arithmetic recurrence and not a random source: no module named `random` is imported here.
LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
LCG_MODULUS = 1 << 64


def lcg_stream(seed: int):
    """An endless deterministic integer stream. Fully determined by `seed`."""
    state = seed % LCG_MODULUS
    while True:
        state = (LCG_MULTIPLIER * state + LCG_INCREMENT) % LCG_MODULUS
        yield state


def normalise(text: str) -> List[str]:
    """Fold a record's raw text to its comparison tokens.

    Case is folded and every run of characters that is not a letter or a digit is a
    separator. Punctuation, capitalisation and whitespace therefore do not survive into the
    comparison, which is why two records can differ in a great many bytes and still adjudicate
    as near-duplicates.
    """
    tokens: List[str] = []
    current: List[str] = []
    for character in text:
        if character.isalnum():
            current.append(character.lower())
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def shingles(tokens: Sequence[str], order: int) -> Set[str]:
    """The record's shingle set: one 16-hex digest per contiguous window of `order` tokens.

    A record shorter than `order` tokens contributes the single shingle of its whole token
    list, so every record carries a non-empty shingle set and no record is silently
    unadjudicable.
    """
    if order <= 0:
        raise ValueError("shingle order must be positive")
    windows: List[str] = []
    if len(tokens) < order:
        windows.append("\x1f".join(tokens))
    else:
        for start in range(len(tokens) - order + 1):
            windows.append("\x1f".join(tokens[start : start + order]))
    return {hashlib.blake2b(w.encode("utf-8"), digest_size=8).hexdigest() for w in windows}


def quantised_similarity(left: Set[str], right: Set[str], quantum: int) -> int:
    """Exact Jaccard similarity of two shingle sets, floored onto the quantum grid.

    This is the adjudication quantity. It is EXACT: it is computed from the shingle sets
    themselves and never from a minhash estimate. The index's signatures are a filter over
    which pairs are worth looking at; they are not the adjudication.
    """
    if not left and not right:
        return quantum
    union = len(left | right)
    if union == 0:
        return 0
    return (quantum * len(left & right)) // union


def permutation_family(count: int, seed: int) -> List[Tuple[int, int]]:
    """`count` affine permutations (a, b) over the prime field. Deterministic in `seed`."""
    stream = lcg_stream(seed)
    family: List[Tuple[int, int]] = []
    while len(family) < count:
        a = next(stream) % PERMUTATION_PRIME
        b = next(stream) % PERMUTATION_PRIME
        if a == 0:
            continue
        family.append((a, b))
    return family


def shingle_values(shingle_set: Iterable[str]) -> List[int]:
    return [int(item, 16) % PERMUTATION_PRIME for item in shingle_set]


def signature(values: Sequence[int], family: Sequence[Tuple[int, int]]) -> List[int]:
    """The minhash signature: one minimum per permutation."""
    if not values:
        return [0] * len(family)
    return [min((a * value + b) % PERMUTATION_PRIME for value in values) for a, b in family]


def band_keys(sig: Sequence[int], bands: int, rows: int) -> List[str]:
    """One key per band, over `rows` consecutive signature positions."""
    keys: List[str] = []
    for band in range(bands):
        chunk = sig[band * rows : (band + 1) * rows]
        payload = ",".join(str(value) for value in chunk).encode("utf-8")
        keys.append(hashlib.blake2b(payload, digest_size=8).hexdigest())
    return keys


def load_spec(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_corpus(path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def corpus_digest(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def maximum_separation_split(histogram: Dict[int, int], quantum: int) -> int:
    """The split that maximises between-class separation of a similarity histogram.

    The candidate similarity distribution a built index produces is two-humped: the
    coincidental overlaps the band structure turns up, and the genuine near-duplicates. The
    split is the classical maximum-between-class-variance cut of that distribution.

    The comparison is exact integer arithmetic. For a split at `t`, with `w0`, `s0` the count
    and similarity sum below `t` and `w1`, `s1` the count and sum at or above it, the
    between-class variance is proportional to `(w1 * s0 - w0 * s1) ** 2 / (w0 * w1)`. Two
    candidate splits are compared by cross-multiplication, so no floating point rounding can
    move the answer between hosts. Ties resolve to the smallest split.
    """
    total_count = sum(histogram.values())
    if total_count <= 0:
        return 0
    total_sum = sum(level * count for level, count in histogram.items())
    best_split = 0
    best_numerator = -1
    best_denominator = 1
    below_count = 0
    below_sum = 0
    for level in range(0, quantum + 1):
        occupancy = histogram.get(level, 0)
        below_count += occupancy
        below_sum += level * occupancy
        above_count = total_count - below_count
        above_sum = total_sum - below_sum
        if below_count == 0 or above_count == 0:
            continue
        numerator = (above_count * below_sum - below_count * above_sum) ** 2
        denominator = below_count * above_count
        if numerator * best_denominator > best_numerator * denominator:
            best_numerator = numerator
            best_denominator = denominator
            best_split = level + 1
    return best_split


def calibrate(histogram: Dict[int, int], quantum: int) -> Dict[str, Any]:
    """Calibrate the adjudication threshold from a candidate similarity histogram.

    Two steps, in order, and this is the only place an adjudication threshold comes from.

      1. Cut the histogram at its maximum-separation split.
      2. Snap the cut up onto the grid the index actually observed: the threshold is the
         SMALLEST candidate similarity level at or above the split. A threshold no candidate
         pair realises is a threshold the index never established, so the split alone is not
         the answer and the snap is part of the calibration rather than a tidying step after
         it.

    Returns zero for both when the histogram is empty or carries one class only, which is a
    degenerate index rather than a threshold.
    """
    split = maximum_separation_split(histogram, quantum)
    observed = sorted(level for level, count in histogram.items() if count > 0)
    at_or_above = [level for level in observed if level >= split]
    threshold = at_or_above[0] if at_or_above else split
    return {
        "adjudication_threshold": threshold,
        "separation_split": split,
        "distinct_levels": len(observed),
        "rule": (
            "maximum between-class separation of the candidate similarity histogram, "
            "snapped up to the smallest candidate level at or above the split"
        ),
    }


def build(corpus_path, spec_path) -> Dict[str, Any]:
    """Build the whole index over a frozen corpus and return it as plain data.

    The returned mapping is the index. It carries, in order: the per-record shingle sets, the
    band collisions the signature structure produced with the number of bands each collision
    was seen in, the exact quantised similarity of every colliding pair, and the calibration
    those colliding pairs determine.
    """
    spec = load_spec(spec_path)
    order = int(spec["shingle_order"])
    quantum = int(spec["similarity_quantum"])
    bands = int(spec["bands"])
    rows = int(spec["rows_per_band"])
    signature_length = bands * rows
    if signature_length != int(spec["signature_length"]):
        raise ValueError("the index specification's bands times rows is not its signature length")

    corpus = load_corpus(corpus_path)
    digest = corpus_digest(corpus_path)
    # The permutation family is seeded from the frozen corpus digest, so the index is a
    # function of the corpus it indexes and of nothing else.
    family = permutation_family(signature_length, int(digest[:16], 16))

    identifiers: List[str] = []
    shingle_sets: Dict[str, Set[str]] = {}
    for row in corpus:
        identifier = str(row["id"])
        identifiers.append(identifier)
        shingle_sets[identifier] = shingles(normalise(str(row["text"])), order)

    buckets: Dict[Tuple[int, str], List[str]] = {}
    for identifier in identifiers:
        sig = signature(shingle_values(shingle_sets[identifier]), family)
        for band, key in enumerate(band_keys(sig, bands, rows)):
            buckets.setdefault((band, key), []).append(identifier)

    collision_bands: Dict[Tuple[str, str], int] = {}
    for members in buckets.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                pair = (ordered[left_index], ordered[right_index])
                collision_bands[pair] = collision_bands.get(pair, 0) + 1

    collisions: List[Dict[str, Any]] = []
    histogram: Dict[int, int] = {}
    for pair in sorted(collision_bands):
        similarity = quantised_similarity(shingle_sets[pair[0]], shingle_sets[pair[1]], quantum)
        collisions.append(
            {"a": pair[0], "b": pair[1], "bands": collision_bands[pair], "similarity": similarity}
        )
        histogram[similarity] = histogram.get(similarity, 0) + 1

    calibration = calibrate(histogram, quantum)
    calibration["candidate_pairs"] = len(collisions)

    return {
        "schema": "oer27.index/v1",
        "corpus_sha256": digest,
        "record_count": len(identifiers),
        "records": [
            {"id": identifier, "shingles": sorted(shingle_sets[identifier])}
            for identifier in identifiers
        ],
        "collisions": collisions,
        "calibration": calibration,
        "spec": spec,
    }


def telemetry_rows(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a built index into the record stream a harness handle reads back."""
    rows: List[Dict[str, Any]] = [
        {
            "kind": "index",
            "corpus_sha256": index["corpus_sha256"],
            "record_count": index["record_count"],
            "spec": index["spec"],
        }
    ]
    for row in index["records"]:
        rows.append({"kind": "record", "id": row["id"], "shingles": row["shingles"]})
    for row in index["collisions"]:
        rows.append(
            {
                "kind": "collision",
                "a": row["a"],
                "b": row["b"],
                "bands": row["bands"],
                "similarity": row["similarity"],
            }
        )
    rows.append(dict({"kind": "calibration"}, **index["calibration"]))
    return rows

"""The nine graded checkers. Pure, deterministic, and none of them reads a clock.

Every value a checker compares comes from one of two places and never from a third:

  1. an index telemetry record the HARNESS produced. The verifier rebuilds the near-duplicate
     index itself, in an isolated subprocess, from the BUNDLE's frozen corpus and frozen index
     specification, and emits one record per corpus record, one per band collision, and one
     carrying the calibration. A checker reads those records through the `Harness` handle
     below and computes from them. No checker calls `time`, `datetime`, `perf_counter`,
     `monotonic` or any other clock, directly or transitively.

  2. a bound constant handed in as the `Bound` argument, read by `tests/grade.py` from
     `tests/bound.json`, which `solution/recompute.py` derived from `solution/grounding.yaml`.

No number the submission reported reaches the reward. Four submission-authored fields are
read, and every one of them is read only so that a substitution is detectable rather than
ignorable: `adjudication_threshold`, `index_collision_count`, `collision_witness` and
`near_duplicate_pairs`.

Imports are confined to the allowlist: json, hashlib, pathlib, dataclasses and typing.
Nothing here imports the submission, opens a socket, consults an environment secret, draws
from a random source, or reads a file the author planted with an answer in it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Machine-readable zero reasons. A downstream grader branches on these strings.
# ---------------------------------------------------------------------------
REASON_FROZEN_INPUTS = "frozen-inputs-modified"
REASON_PAIR_MALFORMED = "pair-set-malformed"
REASON_PAIR_DISORDERED = "pair-listing-disordered"
REASON_THRESHOLD = "threshold-not-established"
REASON_CENSUS = "collision-census-diverges"
REASON_DEGENERATE = "adjudication-degenerate"
REASON_UNSUPPORTED = "adjudication-unsupported"
REASON_WITNESS = "witness-not-colliding"
REASON_NOT_SUSTAINED = "recall-not-sustained"


@dataclass(frozen=True)
class Bound:
    """Every constant the checkers compare against. Handed in, never authored here."""

    corpus_sha256: str
    index_spec_sha256: str
    record_count: int
    similarity_quantum: int
    adjudication_threshold: int
    index_collision_count: int
    near_duplicate_pair_count: int
    duplicate_group_count: int
    witness_a: str
    witness_b: str
    witness_bands: int
    witness_similarity: int
    shard_count: int
    sustain_numerator: int
    sustain_denominator: int
    instance_baseline_recall: float
    instance_target_recall: float

    @staticmethod
    def from_mapping(payload: Dict[str, Any]) -> "Bound":
        witness = payload["collision_witness"]
        return Bound(
            corpus_sha256=str(payload["corpus_sha256"]),
            index_spec_sha256=str(payload["index_spec_sha256"]),
            record_count=int(payload["record_count"]),
            similarity_quantum=int(payload["similarity_quantum"]),
            adjudication_threshold=int(payload["adjudication_threshold"]),
            index_collision_count=int(payload["index_collision_count"]),
            near_duplicate_pair_count=int(payload["near_duplicate_pair_count"]),
            duplicate_group_count=int(payload["duplicate_group_count"]),
            witness_a=str(witness["a"]),
            witness_b=str(witness["b"]),
            witness_bands=int(witness["bands"]),
            witness_similarity=int(witness["similarity"]),
            shard_count=int(payload["shard_count"]),
            sustain_numerator=int(payload["sustain_numerator"]),
            sustain_denominator=int(payload["sustain_denominator"]),
            instance_baseline_recall=float(payload["instance_baseline_recall"]),
            instance_target_recall=float(payload["instance_target_recall"]),
        )


@dataclass
class Harness:
    """The real handle. Every read below is of state a build actually produced.

    `telemetry` is the harness-owned index telemetry: the record list the verifier's own
    builder subprocess emitted while rebuilding the index over the frozen corpus.
    `bundle_corpus_sha256` and `bundle_index_spec_sha256` are the frozen inputs as they stand
    on disk. `workspace` is the run directory the submission produced.
    """

    telemetry: List[Dict[str, Any]] = field(default_factory=list)
    bundle_corpus_sha256: str = ""
    bundle_index_spec_sha256: str = ""
    workspace: Optional[Path] = None
    _cache: Dict[str, Any] = field(default_factory=dict)

    # -- workspace readers ---------------------------------------------------
    def submission(self) -> Dict[str, Any]:
        if "submission" not in self._cache:
            value: Any = None
            if self.workspace is not None:
                path = Path(self.workspace) / "submission.json"
                if path.is_file():
                    try:
                        value = json.loads(path.read_text(encoding="utf-8"))
                    except ValueError:
                        value = None
            self._cache["submission"] = value if isinstance(value, dict) else {}
        return self._cache["submission"]

    # -- telemetry readers ---------------------------------------------------
    def records(self, kind: str) -> List[Dict[str, Any]]:
        return [row for row in self.telemetry if row.get("kind") == kind]

    def header(self) -> Dict[str, Any]:
        rows = self.records("index")
        return rows[0] if rows else {}

    def calibration(self) -> Dict[str, Any]:
        rows = self.records("calibration")
        return rows[-1] if rows else {}

    def collisions(self) -> List[Dict[str, Any]]:
        return self.records("collision")

    def identifiers(self) -> List[str]:
        if "identifiers" not in self._cache:
            self._cache["identifiers"] = [str(row["id"]) for row in self.records("record")]
        return self._cache["identifiers"]

    def shingle_sets(self) -> Dict[str, Set[str]]:
        if "shingles" not in self._cache:
            self._cache["shingles"] = {
                str(row["id"]): set(row["shingles"]) for row in self.records("record")
            }
        return self._cache["shingles"]

    def quantum(self) -> int:
        spec = self.header().get("spec") or {}
        return int(spec.get("similarity_quantum", 0))

    def threshold(self) -> int:
        return int(self.calibration().get("adjudication_threshold", 0))

    # -- derived adjudication ------------------------------------------------
    def similarity(self, left: str, right: str) -> int:
        sets = self.shingle_sets()
        return quantised_similarity(sets[left], sets[right], self.quantum())

    def edges(self) -> List[Tuple[str, str]]:
        """Every pair in the WHOLE universe whose similarity reaches the threshold."""
        if "edges" not in self._cache:
            sets = self.shingle_sets()
            quantum = self.quantum()
            threshold = self.threshold()
            identifiers = self.identifiers()
            found: List[Tuple[str, str]] = []
            for left in range(len(identifiers)):
                a = identifiers[left]
                set_a = sets[a]
                for right in range(left + 1, len(identifiers)):
                    b = identifiers[right]
                    if quantised_similarity(set_a, sets[b], quantum) >= threshold:
                        found.append((a, b) if a < b else (b, a))
            self._cache["edges"] = sorted(found)
        return self._cache["edges"]

    def groups(self) -> Dict[str, str]:
        """Record identifier to duplicate-group label. Connected components over the edges."""
        if "groups" not in self._cache:
            parent = {identifier: identifier for identifier in self.identifiers()}

            def find(node: str) -> str:
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            for a, b in self.edges():
                root_a, root_b = find(a), find(b)
                if root_a != root_b:
                    parent[root_a] = root_b
            self._cache["groups"] = {
                identifier: find(identifier) for identifier in self.identifiers()
            }
        return self._cache["groups"]

    def near_duplicate_pairs(self) -> Set[Tuple[str, str]]:
        """Every unordered pair of distinct records inside a common duplicate group."""
        if "truth" not in self._cache:
            members: Dict[str, List[str]] = {}
            for identifier, label in self.groups().items():
                members.setdefault(label, []).append(identifier)
            truth: Set[Tuple[str, str]] = set()
            for group in members.values():
                group = sorted(group)
                for left in range(len(group)):
                    for right in range(left + 1, len(group)):
                        truth.add((group[left], group[right]))
            self._cache["truth"] = truth
        return self._cache["truth"]

    def duplicate_group_count(self) -> int:
        members: Dict[str, int] = {}
        for label in self.groups().values():
            members[label] = members.get(label, 0) + 1
        return len([label for label, count in members.items() if count > 1])

    def witness(self) -> Optional[Dict[str, Any]]:
        """The strongest colliding pair the adjudication refused to group.

        Maximal by band count, then by similarity, then lexicographically smallest pair.
        Returns None when the index produced no cross-group collision at all, which is a
        degenerate index rather than a witness.
        """
        if "witness" not in self._cache:
            labels = self.groups()
            best: Optional[Dict[str, Any]] = None
            best_key: Optional[Tuple[int, int, str, str]] = None
            for row in self.collisions():
                a, b = str(row["a"]), str(row["b"])
                if a not in labels or b not in labels:
                    continue
                if labels[a] == labels[b]:
                    continue
                key = (int(row["bands"]), int(row["similarity"]), a, b)
                if best_key is None:
                    best, best_key = row, key
                    continue
                if key[0] > best_key[0]:
                    best, best_key = row, key
                elif key[0] == best_key[0]:
                    if key[1] > best_key[1]:
                        best, best_key = row, key
                    elif key[1] == best_key[1] and (key[2], key[3]) < (best_key[2], best_key[3]):
                        best, best_key = row, key
            self._cache["witness"] = best
        return self._cache["witness"]


@dataclass
class Outcome:
    """One checker's verdict. `reason` is empty exactly when the checker passed."""

    ident: str
    passed: bool
    reason: str
    detail: str
    value: Any = None


def _pass(ident: str, detail: str, value: Any = None) -> Outcome:
    return Outcome(ident, True, "", detail, value)


def _fail(ident: str, reason: str, detail: str, value: Any = None) -> Outcome:
    return Outcome(ident, False, reason, detail, value)


# ---------------------------------------------------------------------------
# Quantities the VERIFIER computes. Never a number the submission reported.
# ---------------------------------------------------------------------------


def quantised_similarity(left: Set[str], right: Set[str], quantum: int) -> int:
    """Exact Jaccard similarity floored onto the quantum grid. No estimate anywhere."""
    if not left and not right:
        return quantum
    union = len(left | right)
    if union == 0:
        return 0
    return (quantum * len(left & right)) // union


def submitted_pairs(harness: Harness) -> List[List[Any]]:
    value = harness.submission().get("near_duplicate_pairs")
    return value if isinstance(value, list) else []


def normalised_pairs(harness: Harness) -> Set[Tuple[str, str]]:
    """The submitted pair set as canonical tuples. Malformed entries are dropped here.

    Dropping is safe because `submission_pairs_wellformed` runs first in the grading order
    and fails the whole run on any malformed entry, so nothing reaches this helper's callers
    with an entry it had to drop.
    """
    pairs: Set[Tuple[str, str]] = set()
    for entry in submitted_pairs(harness):
        if isinstance(entry, list) and len(entry) == 2:
            a, b = str(entry[0]), str(entry[1])
            pairs.add((a, b) if a < b else (b, a))
    return pairs


def recall(harness: Harness) -> float:
    """The graded metric: the fraction of the verifier's own pair set the submission carries."""
    truth = harness.near_duplicate_pairs()
    if not truth:
        return 0.0
    hits = len(normalised_pairs(harness) & truth)
    return round(hits / len(truth), 6)


def shard_of(identifier: str, shard_count: int, record_count: int) -> int:
    """Which verifier-scheduled shard a pair belongs to, from its first member's position."""
    try:
        position = int(identifier[1:])
    except ValueError:
        return 0
    width = max(1, record_count // shard_count)
    return min(position // width, shard_count - 1)


# ---------------------------------------------------------------------------
# The nine checkers. Exactly one reduction each.
# ---------------------------------------------------------------------------


def check_corpus_and_spec_unmodified(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The frozen corpus and index specification are the bound bytes.

    This also holds the REBUILT index against the bound calibration. The index is a function
    of the corpus bytes alone, so an index whose calibration disagrees with the bound one is
    an index built over something other than the frozen corpus, and that is a frozen-input
    failure rather than a threshold failure.
    """
    ident = "corpus_and_spec_unmodified"
    if harness.bundle_corpus_sha256 != bound.corpus_sha256:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the frozen corpus on disk digests to "
            + harness.bundle_corpus_sha256
            + ", bound is "
            + bound.corpus_sha256,
        )
    if harness.bundle_index_spec_sha256 != bound.index_spec_sha256:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the frozen index specification on disk digests to "
            + harness.bundle_index_spec_sha256
            + ", bound is "
            + bound.index_spec_sha256,
        )
    header = harness.header()
    if str(header.get("corpus_sha256", "")) != bound.corpus_sha256:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the rebuilt index declares corpus digest "
            + repr(header.get("corpus_sha256"))
            + ", bound is "
            + bound.corpus_sha256,
        )
    if int(header.get("record_count", -1)) != bound.record_count:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the rebuilt index holds "
            + str(header.get("record_count"))
            + " records, bound is "
            + str(bound.record_count),
        )
    if harness.quantum() != bound.similarity_quantum:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the rebuilt index uses similarity quantum "
            + str(harness.quantum())
            + ", bound is "
            + str(bound.similarity_quantum),
        )
    if harness.threshold() != bound.adjudication_threshold:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the rebuilt index calibrates a threshold the bound record does not carry, so the "
            "index was not built over the frozen corpus",
        )
    return _pass(ident, "both frozen inputs match their bound digests", bound.corpus_sha256)


def check_submission_pairs_wellformed(harness: Harness, bound: Bound) -> Outcome:
    """INVARIANT. Every submitted pair names two distinct known records, canonically."""
    ident = "submission_pairs_wellformed"
    submission = harness.submission()
    if str(submission.get("schema", "")) != "oer27.submission/v1":
        return _fail(
            ident,
            REASON_PAIR_MALFORMED,
            "the submission declares schema " + repr(submission.get("schema")),
        )
    known = set(harness.identifiers())
    entries = submission.get("near_duplicate_pairs")
    if not isinstance(entries, list):
        return _fail(
            ident,
            REASON_PAIR_MALFORMED,
            "near_duplicate_pairs is " + type(entries).__name__ + ", not a list",
        )
    seen: Set[Tuple[str, str]] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) != 2:
            return _fail(
                ident,
                REASON_PAIR_MALFORMED,
                "entry " + str(position) + " is not a two-element list: " + repr(entry)[:120],
            )
        a, b = str(entry[0]), str(entry[1])
        if a not in known or b not in known:
            return _fail(
                ident,
                REASON_PAIR_MALFORMED,
                "entry " + str(position) + " names a record absent from the corpus: " + repr(entry),
            )
        if a == b:
            return _fail(
                ident, REASON_PAIR_MALFORMED, "entry " + str(position) + " pairs " + a + " with itself"
            )
        if a > b:
            return _fail(
                ident,
                REASON_PAIR_MALFORMED,
                "entry "
                + str(position)
                + " is not written smaller-first: "
                + repr(entry),
            )
        if (a, b) in seen:
            return _fail(
                ident, REASON_PAIR_MALFORMED, "pair " + a + "," + b + " appears more than once"
            )
        seen.add((a, b))
    coverage = submission.get("coverage")
    if not isinstance(coverage, dict):
        return _fail(ident, REASON_PAIR_MALFORMED, "the submission declares no coverage block")
    if int(coverage.get("records_examined", -1)) != bound.record_count:
        return _fail(
            ident,
            REASON_PAIR_MALFORMED,
            "the submission declares it adjudicated over "
            + repr(coverage.get("records_examined"))
            + " records against "
            + str(bound.record_count)
            + " in the frozen corpus",
        )
    readout = submission.get("readout")
    if not isinstance(readout, dict) or str(readout.get("estimator", "")).strip().lower() != "exact":
        return _fail(
            ident,
            REASON_PAIR_MALFORMED,
            "the submission declares estimator "
            + repr((readout or {}).get("estimator") if isinstance(readout, dict) else None)
            + "; the adjudication quantity is exact and a declared estimate is not it",
        )
    if int(submission.get("duplicate_group_count", -1)) != harness.duplicate_group_count():
        return _fail(
            ident,
            REASON_PAIR_MALFORMED,
            "the submission declares "
            + repr(submission.get("duplicate_group_count"))
            + " duplicate groups against "
            + str(harness.duplicate_group_count())
            + " the verifier's own adjudication holds",
        )
    return _pass(ident, "all " + str(len(seen)) + " submitted pairs are well formed", len(seen))


def check_pair_listing_ordered(harness: Harness, bound: Bound) -> Outcome:
    """ORDERING. The submitted listing is strictly ascending in canonical order."""
    ident = "pair_listing_ordered"
    entries = submitted_pairs(harness)
    previous: Optional[Tuple[str, str]] = None
    for position, entry in enumerate(entries):
        current = (str(entry[0]), str(entry[1]))
        if previous is not None and current <= previous:
            return _fail(
                ident,
                REASON_PAIR_DISORDERED,
                "entry "
                + str(position)
                + " "
                + str(list(current))
                + " does not follow "
                + str(list(previous)),
                position,
            )
        previous = current
    return _pass(ident, "the listing is strictly ascending over " + str(len(entries)) + " pairs")


def check_adjudication_threshold_established(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The submitted threshold is the one the built index established.

    This is the discovery gate. The threshold is not in the statement, not in the corpus and
    not in any file this bundle ships; it comes into existence when the index is built and it
    is read back through the query handle. A threshold that was guessed, rounded, or copied
    from a plausible-looking constant fails here with its own machine-readable reason.
    """
    ident = "adjudication_threshold_established"
    established = harness.threshold()
    if established <= 0:
        return _fail(
            ident,
            REASON_THRESHOLD,
            "the rebuilt index calibrated no threshold, so none was established to compare against",
        )
    claimed = harness.submission().get("adjudication_threshold")
    try:
        claimed_value = int(claimed)
    except (TypeError, ValueError):
        return _fail(
            ident, REASON_THRESHOLD, "the submission declares a non-integer threshold " + repr(claimed)
        )
    if claimed_value != established:
        return _fail(
            ident,
            REASON_THRESHOLD,
            "the submission adjudicates at "
            + str(claimed_value)
            + " and the built index established a different threshold; the difference is "
            + str(claimed_value - established)
            + " steps on the quantum grid",
            claimed_value,
        )
    return _pass(ident, "the submitted threshold is the one the built index established", established)


def check_index_collision_census_agrees(harness: Harness, bound: Bound) -> Outcome:
    """DIVERGENCE. The submission's census of the index disagrees with the index or it does not.

    A submission that never enumerated the band structure cannot state how many pairs it
    holds. This is the cheapest place a claim about the index and the index itself are put
    side by side, and a divergence is named rather than absorbed.
    """
    ident = "index_collision_census_agrees"
    observed = len(harness.collisions())
    if observed != bound.index_collision_count:
        return _fail(
            ident,
            REASON_CENSUS,
            "the rebuilt index holds "
            + str(observed)
            + " colliding pairs and the bound record holds "
            + str(bound.index_collision_count),
            observed,
        )
    claimed = harness.submission().get("index_collision_count")
    try:
        claimed_value = int(claimed)
    except (TypeError, ValueError):
        return _fail(
            ident, REASON_CENSUS, "the submission declares a non-integer collision count " + repr(claimed)
        )
    if claimed_value != observed:
        return _fail(
            ident,
            REASON_CENSUS,
            "the submission counts "
            + str(claimed_value)
            + " colliding pairs and the verifier's own index holds "
            + str(observed),
            claimed_value,
        )
    return _pass(ident, "the submitted collision census matches the rebuilt index", observed)


def check_adjudication_not_degenerate(harness: Harness, bound: Bound) -> Outcome:
    """ABSENCE. Neither the empty answer nor the everything answer is an adjudication.

    Returning every pair maximises recall and says nothing. Returning none says nothing and
    scores nothing. Both are refused here, before the precision gate, so each carries the
    reason that describes what it actually is.
    """
    ident = "adjudication_not_degenerate"
    pairs = normalised_pairs(harness)
    record_count = len(harness.identifiers())
    universe = record_count * (record_count - 1) // 2
    if not pairs:
        return _fail(
            ident,
            REASON_DEGENERATE,
            "the submission adjudicates no pair at all, which is not an adjudication",
            0,
        )
    if len(pairs) >= universe:
        return _fail(
            ident,
            REASON_DEGENERATE,
            "the submission adjudicates "
            + str(len(pairs))
            + " pairs against a universe of "
            + str(universe)
            + ", which is every pair in the corpus and not an adjudication",
            len(pairs),
        )
    ceiling = universe // 4
    if len(pairs) > ceiling:
        return _fail(
            ident,
            REASON_DEGENERATE,
            "the submission adjudicates "
            + str(len(pairs))
            + " pairs, more than the quarter of the universe ("
            + str(ceiling)
            + ") beyond which an answer is a sweep rather than an adjudication",
            len(pairs),
        )
    return _pass(ident, "the adjudication is neither empty nor a sweep", len(pairs))


def check_no_unsupported_pair(harness: Harness, bound: Bound) -> Outcome:
    """ABSENCE. Not one submitted pair falls outside the verifier's own adjudication.

    Precision is a hard constraint. The verifier's near-duplicate pair set is recomputed from
    its own rebuilt index, at the threshold that index calibrated, closed over duplicate
    groups. A single pair outside it ends the run.
    """
    ident = "no_unsupported_pair"
    truth = harness.near_duplicate_pairs()
    unsupported = sorted(pair for pair in normalised_pairs(harness) if pair not in truth)
    if unsupported:
        first = unsupported[0]
        return _fail(
            ident,
            REASON_UNSUPPORTED,
            str(len(unsupported))
            + " submitted pairs fall outside the verifier's own adjudication, starting with "
            + first[0]
            + ","
            + first[1]
            + " at similarity "
            + str(harness.similarity(first[0], first[1]))
            + " against a threshold of "
            + str(harness.threshold()),
            [list(pair) for pair in unsupported[:8]],
        )
    return _pass(ident, "every submitted pair is supported by the verifier's own adjudication")


def check_collision_witness_verified(harness: Harness, bound: Bound) -> Outcome:
    """EFFECT. The submitted witness is the boundary evidence and not a plausible pair.

    This is the second discovery gate. Four things have to hold at once, and each of them is
    an effect of work actually done: the pair is one the built index collided; the band count
    and similarity the submission reports for it are the ones the index recorded; the
    adjudication does NOT place the two records in a common duplicate group; and no other
    refused collision beats it under the bound ordering. A pair that merely looks like a
    near-miss fails, and so does the correct pair reported with a band count nobody measured.
    """
    ident = "collision_witness_verified"
    expected = harness.witness()
    if expected is None:
        return _fail(
            ident,
            REASON_WITNESS,
            "the rebuilt index produced no collision the adjudication refused, so no witness exists",
        )
    claimed = harness.submission().get("collision_witness")
    if not isinstance(claimed, dict):
        return _fail(
            ident,
            REASON_WITNESS,
            "the submission declares no collision witness block",
        )
    a, b = str(claimed.get("a")), str(claimed.get("b"))
    known = set(harness.identifiers())
    if a not in known or b not in known:
        return _fail(
            ident, REASON_WITNESS, "the witness names a record absent from the corpus: " + a + "," + b
        )
    if a == b:
        return _fail(ident, REASON_WITNESS, "the witness pairs " + a + " with itself")
    if a > b:
        return _fail(
            ident, REASON_WITNESS, "the witness is not written smaller-first: " + a + "," + b
        )

    index_rows = {(str(row["a"]), str(row["b"])): row for row in harness.collisions()}
    row = index_rows.get((a, b))
    if row is None:
        return _fail(
            ident,
            REASON_WITNESS,
            "the built index never places "
            + a
            + " and "
            + b
            + " in a common band, so the submitted witness is not a collision at all",
            [a, b],
        )
    labels = harness.groups()
    if labels[a] == labels[b]:
        return _fail(
            ident,
            REASON_WITNESS,
            "the adjudication places "
            + a
            + " and "
            + b
            + " in the same duplicate group, so that collision was accepted rather than refused",
            [a, b],
        )
    if int(claimed.get("bands", -1)) != int(row["bands"]):
        return _fail(
            ident,
            REASON_WITNESS,
            "the submission reports the witness in "
            + repr(claimed.get("bands"))
            + " bands and the built index recorded "
            + str(row["bands"]),
            [a, b],
        )
    if int(claimed.get("similarity", -1)) != int(row["similarity"]):
        return _fail(
            ident,
            REASON_WITNESS,
            "the submission reports witness similarity "
            + repr(claimed.get("similarity"))
            + " and the built index recorded "
            + str(row["similarity"]),
            [a, b],
        )
    if (a, b) != (str(expected["a"]), str(expected["b"])):
        return _fail(
            ident,
            REASON_WITNESS,
            "the submitted witness "
            + a
            + ","
            + b
            + " is a refused collision in "
            + str(row["bands"])
            + " bands at similarity "
            + str(row["similarity"])
            + ", and the index holds a refused collision that beats it under the bound ordering",
            [a, b],
        )
    if a != bound.witness_a or b != bound.witness_b:
        return _fail(
            ident,
            REASON_WITNESS,
            "the verifier's own witness disagrees with the bound record, so the index was not "
            "built over the frozen corpus",
            [a, b],
        )
    return _pass(
        ident,
        "the witness is the maximal refused collision, in "
        + str(row["bands"])
        + " bands at similarity "
        + str(row["similarity"]),
        [a, b],
    )


def check_recall_sustained_across_shards(harness: Harness, bound: Bound) -> Outcome:
    """INVARIANT. The recall holds inside every verifier-scheduled shard.

    A single well-solved region of the corpus is not an adjudication of the corpus. The
    verifier partitions the records into its own contiguous shards, assigns each pair to the
    shard of its first member, and requires each shard that carries any near-duplicate pair
    at all to hold at least the bound fraction of its pairs.
    """
    ident = "recall_sustained_across_shards"
    truth = harness.near_duplicate_pairs()
    if not truth:
        return _fail(
            ident,
            REASON_NOT_SUSTAINED,
            "the verifier's own adjudication holds no near-duplicate pair, so no recall exists",
        )
    submitted = normalised_pairs(harness)
    record_count = len(harness.identifiers())
    totals: Dict[int, int] = {}
    hits: Dict[int, int] = {}
    for pair in truth:
        shard = shard_of(pair[0], bound.shard_count, record_count)
        totals[shard] = totals.get(shard, 0) + 1
        if pair in submitted:
            hits[shard] = hits.get(shard, 0) + 1
    rows = []
    for shard in range(bound.shard_count):
        total = totals.get(shard, 0)
        hit = hits.get(shard, 0)
        rows.append({"shard": shard, "hits": hit, "total": total})
        if total == 0:
            continue
        if hit * bound.sustain_denominator < bound.sustain_numerator * total:
            return _fail(
                ident,
                REASON_NOT_SUSTAINED,
                "shard "
                + str(shard)
                + " carries "
                + str(hit)
                + " of "
                + str(total)
                + " near-duplicate pairs against a sustain floor of "
                + str(bound.sustain_numerator)
                + "/"
                + str(bound.sustain_denominator),
                rows,
            )
    return _pass(
        ident,
        "the recall holds across all " + str(bound.shard_count) + " verifier-scheduled shards",
        rows,
    )


def digest_of(text: str) -> str:
    """sha256 over text, used to bind reference bytes into a fixture."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

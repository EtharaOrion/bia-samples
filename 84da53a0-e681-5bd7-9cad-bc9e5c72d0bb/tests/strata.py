"""The verifier's own capability-stratum classifier. Deterministic and pure.

This is the whole of the graded coverage measurement. It calls no model, opens no
socket, reads no clock, consults no random source, and reads no file. It is a pure
function from the emitted sample text to one stratum name, so an outside reader can
recompute any coverage vector this project reports from the corpus alone.

The declared coverage manifest a submission ships is never consulted here. It is
compared against what this module measures, and the comparison is a divergence check
rather than a source of reward.

Pinned: the sha256 of these bytes is bound in tests/checkers.yaml under
classifier_sha256, so the classifier that produced a recorded coverage vector is
identifiable after the fact.
"""

from __future__ import annotations

# The closed stratum vocabulary, plus the bucket every unrecognised sample lands in.
# `unclassified` is a real stratum for measurement purposes: samples that fall into it
# are counted, so a corpus of noise cannot report itself as covering nothing.
STRATA = ("calc", "cmp", "fact", "neg", "seq")
UNCLASSIFIED = "unclassified"

# The surface marker that opens a sample of each stratum. Membership is decided by the
# sample's own first token, never by a `stratum` field the sample carries, because a
# field a generator writes is a claim and this module measures.
MARKERS = {
    "calc": "calc",
    "cmp": "cmp",
    "fact": "fact",
    "neg": "neg",
    "seq": "seq",
}

_BY_MARKER = {marker: stratum for stratum, marker in MARKERS.items()}


def normalize(text) -> list:
    """The frozen normalization: whitespace split, casefold. Nothing locale-dependent."""
    return [token.casefold() for token in str(text or "").split()]


def classify(sample) -> str:
    """The stratum of one sample, measured from its text alone."""
    row = sample if isinstance(sample, dict) else {}
    tokens = normalize(row.get("text", ""))
    if not tokens:
        return UNCLASSIFIED
    return _BY_MARKER.get(tokens[0], UNCLASSIFIED)


def measure(samples) -> dict:
    """The measured coverage vector: proportion of the corpus in each stratum.

    Every stratum in the closed set is present in the returned mapping, including the
    ones the corpus does not touch, because a stratum reported as absent by omission is
    indistinguishable from a stratum nobody measured. An empty corpus measures to all
    zeros rather than to an empty mapping.
    """
    rows = list(samples or [])
    counts = {name: 0 for name in (*STRATA, UNCLASSIFIED)}
    for row in rows:
        counts[classify(row)] += 1
    total = len(rows)
    if total == 0:
        return {name: 0.0 for name in counts}
    return {name: counts[name] / total for name in counts}


def counts(samples) -> dict:
    """The measured coverage as integer counts, for a reason block a human reads."""
    rows = list(samples or [])
    out = {name: 0 for name in (*STRATA, UNCLASSIFIED)}
    for row in rows:
        out[classify(row)] += 1
    return out


def l1_divergence(declared, measured) -> float:
    """Total absolute difference between two coverage vectors over the closed set.

    A key the declaration omits is read as zero rather than skipped, so declaring a
    partial vector cannot shrink the distance by leaving terms out.
    """
    keys = sorted(set(measured or {}) | set(declared or {}) | set(STRATA) | {UNCLASSIFIED})
    total = 0.0
    for key in keys:
        left = float((declared or {}).get(key, 0.0) or 0.0)
        right = float((measured or {}).get(key, 0.0) or 0.0)
        total += abs(left - right)
    return total


def near_duplicate_tokens(text) -> frozenset:
    """The declared deterministic normalization for near-duplicate comparison.

    casefold, replace every character that is not alphanumeric with a space, collapse
    whitespace by splitting, and reduce to the set of tokens. No stemming, no stopword
    list, no locale-dependent case mapping beyond casefold.
    """
    raw = str(text or "").casefold()
    cleaned = "".join(char if char.isalnum() else " " for char in raw)
    return frozenset(cleaned.split())


def jaccard(left: frozenset, right: frozenset) -> float:
    """Set Jaccard similarity. Two empty sets are identical, which is 1.0."""
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)

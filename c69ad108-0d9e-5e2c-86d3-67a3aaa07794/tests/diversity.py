"""The corpus diversity statistic. Pure, deterministic, and the same bytes the agent gets.

This module is the whole of the diversity measurement for slot OER-19. It is
shipped twice on purpose: once here, where the verifier computes the graded
statistic, and once at `environment/tools/corpus_diversity.py`, where the solving
agent can compute the identical numbers over its own corpus before it ever asks
for a score. The two copies are byte-identical and their sha256 is pinned in
`tests/checkers.yaml`, so "measure your own corpus" is a skill under test rather
than a coin flip, and the verifier holds no statistic the agent could not have
computed first.

Purity is the point and is enforced from outside. There is no model call, no
network, no clock, no locale lookup and no random source anywhere in this file.
Every function is a pure function of the strings handed to it. The import set is
confined to the checker allowlist, and `seed/tasks/OER-19/adequacy.py` walks this
file's AST on every run to prove that it stayed confined.

The normalization is declared rather than implied, because a diversity number is
meaningless without the normalization it was computed under. `NORMALIZATION_ID`
travels with every statistic this module produces and is recorded in
`tests/checkers.yaml` beside the bound floors.

Why these three statistics and not one:

- `distinct_ngram_ratio` answers "how much of this text is repeated material",
  and is cheap and stable. On its own it is fooled by a corpus that varies one
  slot of one template, which keeps producing fresh trigrams forever.
- `mode_share` answers "what fraction of the corpus sits in one dense
  neighbourhood", which is exactly what a collapse to a low-diversity mode looks
  like from the outside. It is the statistic a rotating-counter generator cannot
  evade, because near-identity is measured by trigram overlap, not by equality.
- `segment_ratios` answers "did this stay true for the whole run". A generator
  that collapses partway leaves a corpus whose aggregate numbers can still clear
  a floor while its tail is degenerate, so the aggregate is computed per emission
  segment and the tail is compared against the head.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

# The declared normalization. Any statistic here is a statement about text under
# this identifier and no other; changing the normalization changes the identifier.
NORMALIZATION_ID = "oer19-lower-alnum-collapse/v1"

# Token n-gram width. Three is wide enough that a one-word substitution moves a
# bounded number of grams and narrow enough that short samples still produce some.
NGRAM_N = 3

# Characters kept by the normalization. Everything else becomes a separator, so
# punctuation and casing cannot be used to manufacture apparent variety.
_KEPT = "abcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class CorpusProfile:
    """Everything the graded checkers need about one corpus, computed in one pass."""

    normalization: str
    samples: int
    corpus_distinct_ngram_ratio: float
    segment_count: int
    segment_distinct_ngram_ratio: tuple
    max_mode_share: float
    prefix_checkpoints: tuple
    prefix_mode_share: tuple

    def as_dict(self) -> dict:
        return {
            "normalization": self.normalization,
            "samples": self.samples,
            "corpus_distinct_ngram_ratio": self.corpus_distinct_ngram_ratio,
            "segment_count": self.segment_count,
            "segment_distinct_ngram_ratio": list(self.segment_distinct_ngram_ratio),
            "max_mode_share": self.max_mode_share,
            "prefix_checkpoints": list(self.prefix_checkpoints),
            "prefix_mode_share": list(self.prefix_mode_share),
        }


def normalize(text) -> str:
    """Lowercase, keep alphanumerics, collapse every run of anything else to one space.

    Deliberately not locale-aware: a locale-sensitive case fold would make the
    statistic depend on the environment the verifier happened to run in, which is
    the one thing a graded statistic may never do.
    """
    out, gap = [], False
    for char in str(text):
        lowered = char.lower()
        if lowered in _KEPT:
            if gap and out:
                out.append(" ")
            out.append(lowered)
            gap = False
        else:
            gap = True
    return "".join(out)


def tokens(text) -> list:
    normalized = normalize(text)
    return normalized.split(" ") if normalized else []


def ngrams(sequence: Sequence, width: int = NGRAM_N) -> list:
    """Token n-grams as joined strings, in order, with repeats kept.

    Repeats are kept because the ratio of distinct to total is the statistic; a
    set here would silently turn the ratio into the constant 1.0.
    """
    items = list(sequence)
    if width <= 0 or len(items) < width:
        return []
    return [" ".join(items[i : i + width]) for i in range(len(items) - width + 1)]


def sample_grams(text) -> list:
    return ngrams(tokens(text))


def distinct_ngram_ratio(texts: Iterable, width: int = NGRAM_N) -> float:
    """Distinct token n-grams over total token n-grams across a set of texts.

    A corpus with no n-grams at all scores 0.0 rather than raising or defaulting
    to 1.0. An empty measurement is never evidence of diversity; it is evidence
    of nothing, and the floor comparison must therefore fail on it.
    """
    total, seen = 0, set()
    for text in texts:
        for gram in ngrams(tokens(text), width):
            total += 1
            seen.add(gram)
    if total == 0:
        return 0.0
    return len(seen) / total


def jaccard(left: Iterable, right: Iterable) -> float:
    """Jaccard overlap of two n-gram sets. Two empty sets overlap in nothing, not everything."""
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def mode_share(texts: Sequence, threshold: float) -> float:
    """The largest share of the corpus sitting in one near-duplicate neighbourhood.

    For each sample, count how many samples (itself included) overlap it at or
    above `threshold`, and report the largest such count as a fraction of the
    corpus. This is the statistic that survives a rotating counter: a generator
    emitting one template with a changing integer produces lines that are never
    equal and always overlap, so exact-duplicate counting reports a clean corpus
    while this reports the collapse.

    Quadratic in the sample count by construction. That is affordable at the bound
    corpus size and is preferred to a sketch, because a sketch would introduce a
    collision profile that an author would then have to defend.
    """
    grams = [set(sample_grams(text)) for text in texts]
    count = len(grams)
    if count == 0:
        return 0.0
    best = 0
    for i in range(count):
        neighbours = 0
        for j in range(count):
            if jaccard(grams[i], grams[j]) >= threshold:
                neighbours += 1
        if neighbours > best:
            best = neighbours
    return best / count


def segment_bounds(count: int, segments: int) -> list:
    """Contiguous, near-equal segment ranges over an emission sequence.

    Contiguity is what makes this a statement about the run's timeline rather than
    about the file's contents, which is why `emission_order_monotone` guards the
    ordering this reads.
    """
    if count <= 0 or segments <= 0:
        return []
    edges = [(index * count) // segments for index in range(segments + 1)]
    return [(edges[i], edges[i + 1]) for i in range(segments) if edges[i] < edges[i + 1]]


def segment_ratios(texts: Sequence, segments: int, width: int = NGRAM_N) -> list:
    rows = list(texts)
    return [
        distinct_ngram_ratio(rows[start:stop], width)
        for start, stop in segment_bounds(len(rows), segments)
    ]


def prefix_checkpoints(count: int, points: int) -> list:
    """Prefix lengths at which a running statistic is re-read, last point inclusive."""
    if count <= 0 or points <= 0:
        return []
    return [max(1, ((index + 1) * count) // points) for index in range(points)]


def prefix_mode_shares(texts: Sequence, points: int, threshold: float) -> list:
    rows = list(texts)
    return [mode_share(rows[:cut], threshold) for cut in prefix_checkpoints(len(rows), points)]


def profile(texts: Sequence, segments: int, points: int, threshold: float) -> CorpusProfile:
    """One pass over a corpus producing every statistic the graded checkers read."""
    rows = list(texts)
    return CorpusProfile(
        normalization=NORMALIZATION_ID,
        samples=len(rows),
        corpus_distinct_ngram_ratio=distinct_ngram_ratio(rows),
        segment_count=len(segment_bounds(len(rows), segments)),
        segment_distinct_ngram_ratio=tuple(segment_ratios(rows, segments)),
        max_mode_share=mode_share(rows, threshold),
        prefix_checkpoints=tuple(prefix_checkpoints(len(rows), points)),
        prefix_mode_share=tuple(prefix_mode_shares(rows, points, threshold)),
    )


def near_duplicates(samples: Sequence, references: Sequence, threshold: float) -> list:
    """Every (sample index, reference id, overlap) pair at or above the bound threshold.

    `references` carries (id, text) pairs. The comparison is the same Jaccard over
    the same normalization as every other statistic here, so a submission that
    tunes its corpus against `environment/tools/corpus_diversity.py` is tuning
    against the graded measurement and not against an approximation of it.
    """
    reference_grams = [(str(ident), set(sample_grams(text))) for ident, text in references]
    hits = []
    for index, text in enumerate(samples):
        grams = set(sample_grams(text))
        for ident, other in reference_grams:
            overlap = jaccard(grams, other)
            if overlap >= threshold:
                hits.append({"sample_index": index, "reference_id": ident, "overlap": overlap})
    return hits


def statistic_digest(rows: Sequence) -> str:
    """A stable digest over normalized corpus text, for pinning a fixture to a corpus."""
    payload = "\n".join(normalize(row) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

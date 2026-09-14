"""The graded checkers. Pure functions of the telemetry the verifier's own process produced.

Nothing here reads a file, a clock, an environment variable, a socket or a random
source, and nothing here imports the submission. Every checker takes one mapping,
the telemetry document `tests/runner.py` built while the verifier measured the run,
and returns a verdict. `seed/tasks/OER-19/adequacy.py` walks this file's AST on
every run and fails the lane if an import outside the allowlist, or any of the
banned names, ever appears.

Why the telemetry mapping and not the run directory: a checker that re-reads the
run workspace is reading bytes the submission could still be writing, and a checker
that reads a number the submission printed is grading the submission's own report
of itself. The telemetry is the verifier's measurement, taken once, in the
verifier's process, from the corpus bytes, from the shard those bytes were packed
into, and from the frozen decoder's own counters and parameters.

The one place a submission-authored value appears is `self_report`, and it is used
by exactly one checker, `self_report_matches_recomputation`, whose whole job is to
compare it against the recomputation. It never contributes to a score.

Every bound below is mirrored from `tests/checkers.yaml`; `adequacy.py` asserts the
two agree, so the manifest a reader audits and the code that runs cannot drift
apart. The architecture constants are mirrored from the same manifest, which in
turn transcribes `environment/nanogpt_substrate.json`. A checker cannot open that
file, so the shape the graded parameters actually had travels in the telemetry as a
signature string and is compared here against the transcription. That comparison is
what makes "the graded artifact is a parameter snapshot of the frozen decoder" a
graded fact rather than a claim in prose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Bounds, mirrored from tests/checkers.yaml and asserted equal by adequacy.py.
# ---------------------------------------------------------------------------

NORMALIZATION_ID = "oer19-lower-alnum-collapse/v1"
CORPUS_DOCUMENTS = 512
DIVERSITY_EXCERPT_CHARS = 512
SEGMENT_COUNT = 8
PREFIX_POINTS = 4
MODE_SHARE_THRESHOLD = 0.60
MAX_MODE_SHARE_CEILING = 0.08
COLLAPSE_RELATIVE_FRACTION = 0.65
NEAR_DUPLICATE_THRESHOLD = 0.55
SUSTAIN_TOLERANCE = 0.50
SELF_REPORT_TOLERANCE = 0.02
BOUND_STEPS = 96
BOUND_TOKENS_FED = 50331648
BOUND_EPOCHS = 96
BOUND_EVALUATION_POINT = 96
SCHEDULED_POINTS = (64, 80, 96)
BOUND_SHARD_TOKENS = 524288
COUNTER_SOURCE = "harness-trainer"
WEIGHTS_SOURCE = "harness-trainer"
DIGEST_WIDTH = 64
_HEX = "0123456789abcdef"

# The frozen architecture, transcribed from environment/nanogpt_substrate.json
# through tests/checkers.yaml. The graded parameters must carry exactly this shape.
SHAPE_SIGNATURE = "vocab=50304;layers=12;dim=768;head_dim=128;heads=6;seq=1024"
MIN_PARAMETER_COUNT = 100000000

# One forward and one backward per optimizer step, the substrate's own rule. A run
# with the passes removed reports zero of them and fails here before any reading is
# read, which is what makes "no forward pass, no score" a gate and not a hope.
FORWARD_PASSES_PER_STEP = 1
BACKWARD_PASSES_PER_STEP = 1


@dataclass(frozen=True)
class Verdict:
    """One checker's answer. A failure always carries its machine-readable reason."""

    passed: bool
    reason: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"passed": self.passed, "reason": self.reason, "detail": self.detail}


def _ok(detail: str = "") -> Verdict:
    return Verdict(True, "", detail)


def _block(mapping, *path):
    node = mapping
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _numbers(rows) -> list:
    out = []
    for value in rows or []:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return []
        if isinstance(value, float) and math.isnan(value):
            return []
        out.append(float(value))
    return out


def _reading(value):
    """A readable finite scalar, or None. A NaN loss is not a reading."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(float(value)) or math.isinf(float(value)):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# 1. EFFECT. The generator ran to completion without collapsing partway through.
# ---------------------------------------------------------------------------


def collapse_verdict(segments, documents, segment_count) -> Verdict:
    """The temporal test, factored out so a stale prefix answer can be exercised too.

    A collapse is a drop, so the comparison is against the strongest segment the run
    ever produced rather than against the first. Comparing against the first would
    let a generator set a low bar by opening degenerately and then staying there,
    and a flat degenerate corpus is not this checker's job in any case: density is
    graded by corpus_mode_share_bounded, which fires on exactly that shape.

    `adequacy.py` calls this helper with the pre-collapse prefix of the same run to
    show the stale control does what a stale control has to do: the prefix answer
    passes, the answer over the whole emission timeline fails, and the difference is
    the mutation.
    """
    ratios = _numbers(segments)
    if documents != CORPUS_DOCUMENTS:
        return Verdict(
            False,
            "generator-collapsed-silently",
            "emitted " + str(documents) + " documents against a bound corpus of "
            + str(CORPUS_DOCUMENTS),
        )
    if not ratios or segment_count != SEGMENT_COUNT or len(ratios) != SEGMENT_COUNT:
        return Verdict(
            False,
            "generator-collapsed-silently",
            "the emission timeline carries " + str(len(ratios)) + " readable segments, expected "
            + str(SEGMENT_COUNT),
        )
    strongest = max(ratios)
    floor = strongest * COLLAPSE_RELATIVE_FRACTION
    weak = [index for index, value in enumerate(ratios) if value < floor]
    if weak:
        return Verdict(
            False,
            "generator-collapsed-silently",
            "segment(s) " + ",".join(str(index) for index in weak)
            + " fell below " + str(round(floor, 6))
            + ", which is " + str(COLLAPSE_RELATIVE_FRACTION)
            + " of the strongest segment reading " + str(round(strongest, 6)),
        )
    return _ok("weakest segment " + str(round(min(ratios), 6)) + " against strongest " + str(round(strongest, 6)))


def generator_collapse_absent(telemetry) -> Verdict:
    corpus = _block(telemetry, "corpus") or {}
    return collapse_verdict(
        corpus.get("segment_distinct_ngram_ratio"),
        corpus.get("documents"),
        corpus.get("segment_count"),
    )


# ---------------------------------------------------------------------------
# 2. INVARIANT. Corpus density stays bounded at every prefix the verifier reads.
# ---------------------------------------------------------------------------


def corpus_mode_share_bounded(telemetry) -> Verdict:
    corpus = _block(telemetry, "corpus") or {}
    if corpus.get("normalization") != NORMALIZATION_ID:
        return Verdict(
            False,
            "corpus-mode-share-exceeded",
            "mode share was measured under normalization " + repr(corpus.get("normalization"))
            + ", not the bound " + NORMALIZATION_ID,
        )
    if corpus.get("diversity_excerpt_chars") != DIVERSITY_EXCERPT_CHARS:
        return Verdict(
            False,
            "corpus-mode-share-exceeded",
            "the diversity excerpt is " + repr(corpus.get("diversity_excerpt_chars"))
            + " characters, bound is " + str(DIVERSITY_EXCERPT_CHARS),
        )
    if corpus.get("mode_share_threshold") != MODE_SHARE_THRESHOLD:
        return Verdict(
            False,
            "corpus-mode-share-exceeded",
            "mode share threshold is " + repr(corpus.get("mode_share_threshold"))
            + ", bound is " + str(MODE_SHARE_THRESHOLD),
        )
    prefixes = _numbers(corpus.get("prefix_mode_share"))
    if len(prefixes) != PREFIX_POINTS:
        return Verdict(
            False,
            "corpus-mode-share-exceeded",
            "the invariant was read at " + str(len(prefixes)) + " prefix points, bound is "
            + str(PREFIX_POINTS),
        )
    overall = _numbers([corpus.get("max_mode_share")])
    if not overall:
        return Verdict(False, "corpus-mode-share-exceeded", "no mode share reading is present")
    breached = [
        index for index, value in enumerate(prefixes) if value > MAX_MODE_SHARE_CEILING
    ]
    if overall[0] > MAX_MODE_SHARE_CEILING or breached:
        return Verdict(
            False,
            "corpus-mode-share-exceeded",
            "mode share " + str(round(overall[0], 6)) + " against ceiling "
            + str(MAX_MODE_SHARE_CEILING)
            + "; breaching prefix point(s): " + (",".join(str(i) for i in breached) or "none"),
        )
    return _ok("mode share " + str(round(overall[0], 6)) + " held at every prefix point")


# ---------------------------------------------------------------------------
# 3. ORDERING. The emission timeline is the timeline, not a reordering of it.
# ---------------------------------------------------------------------------


def emission_order_monotone(telemetry) -> Verdict:
    sequence = _block(telemetry, "corpus", "emission_sequence")
    rows = sequence if isinstance(sequence, list) else []
    if not rows:
        return Verdict(
            False, "emission-order-not-monotone", "the run carries no emission sequence"
        )
    previous = None
    for position, value in enumerate(rows):
        if isinstance(value, bool) or not isinstance(value, int):
            return Verdict(
                False,
                "emission-order-not-monotone",
                "emission index at position " + str(position) + " is " + repr(value),
            )
        if previous is not None and value <= previous:
            return Verdict(
                False,
                "emission-order-not-monotone",
                "emission index " + str(value) + " at position " + str(position)
                + " does not follow " + str(previous),
            )
        previous = value
    if rows[0] != 0 or rows[-1] != len(rows) - 1:
        return Verdict(
            False,
            "emission-order-not-monotone",
            "the sequence runs " + str(rows[0]) + ".." + str(rows[-1])
            + " over " + str(len(rows)) + " documents, so it is not contiguous from zero",
        )
    return _ok("emission sequence is strictly ascending and contiguous over " + str(len(rows)) + " documents")


# ---------------------------------------------------------------------------
# 4. ABSENCE. No generated document is a near-duplicate of a held-out item.
# ---------------------------------------------------------------------------


def no_benchmark_near_duplicate(telemetry) -> Verdict:
    corpus = _block(telemetry, "corpus") or {}
    if corpus.get("normalization") != NORMALIZATION_ID:
        return Verdict(
            False,
            "benchmark-near-duplicate-emitted",
            "the near-duplicate test ran under normalization " + repr(corpus.get("normalization")),
        )
    if corpus.get("near_duplicate_threshold") != NEAR_DUPLICATE_THRESHOLD:
        return Verdict(
            False,
            "benchmark-near-duplicate-emitted",
            "the near-duplicate threshold is " + repr(corpus.get("near_duplicate_threshold"))
            + ", bound is " + str(NEAR_DUPLICATE_THRESHOLD),
        )
    hits = corpus.get("benchmark_near_duplicate_hits")
    rows = hits if isinstance(hits, list) else None
    if rows is None:
        return Verdict(
            False,
            "benchmark-near-duplicate-emitted",
            "no near-duplicate reading is present, so absence was never established",
        )
    if rows:
        first = rows[0] if isinstance(rows[0], dict) else {}
        return Verdict(
            False,
            "benchmark-near-duplicate-emitted",
            str(len(rows)) + " document(s) overlap a held-out item at or above "
            + str(NEAR_DUPLICATE_THRESHOLD)
            + ", first at document " + str(first.get("sample_index"))
            + " against " + str(first.get("reference_id")),
        )
    return _ok("no document overlaps a held-out item at or above " + str(NEAR_DUPLICATE_THRESHOLD))


# ---------------------------------------------------------------------------
# 5. VALUE. The frozen training budget as actually fed, from the harness counters.
# ---------------------------------------------------------------------------


def training_budget_as_fed(telemetry) -> Verdict:
    """The budget as fed, and the passes that fed it.

    The forward and backward counts are read here rather than assumed, because they
    are the difference between a run that trained and a run that reported. A path
    with the forward and backward calls removed increments neither counter and is
    named a budget failure at this gate, before any reading downstream of it is
    consulted.
    """
    training = _block(telemetry, "training") or {}
    if training.get("counter_source") != COUNTER_SOURCE:
        return Verdict(
            False,
            "training-budget-overspent",
            "budget counters came from " + repr(training.get("counter_source"))
            + ", not from " + COUNTER_SOURCE,
        )
    expected = (
        ("steps", BOUND_STEPS),
        ("tokens_fed", BOUND_TOKENS_FED),
        ("epochs_completed", BOUND_EPOCHS),
        ("forward_passes", BOUND_STEPS * FORWARD_PASSES_PER_STEP),
        ("backward_passes", BOUND_STEPS * BACKWARD_PASSES_PER_STEP),
    )
    for key, bound in expected:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != bound:
            return Verdict(
                False,
                "training-budget-overspent",
                key + " was fed " + repr(value) + " against the frozen bound " + str(bound),
            )
    documents = _block(telemetry, "corpus", "documents")
    if documents != CORPUS_DOCUMENTS:
        return Verdict(
            False,
            "training-budget-overspent",
            "the budget is " + str(BOUND_EPOCHS) + " passes over " + str(CORPUS_DOCUMENTS)
            + " documents; the corpus carried " + repr(documents),
        )
    shard = _block(telemetry, "corpus", "shard_tokens")
    if shard != BOUND_SHARD_TOKENS:
        return Verdict(
            False,
            "training-budget-overspent",
            "the corpus packed to " + repr(shard) + " tokens against the bound shard of "
            + str(BOUND_SHARD_TOKENS)
            + "; a short document leaves the shard short",
        )
    return _ok(
        str(BOUND_STEPS) + " steps over " + str(BOUND_SHARD_TOKENS) + " corpus tokens, as fed"
    )


# ---------------------------------------------------------------------------
# 6. VALUE. The graded readout is raw, and equals the counts it claims to summarise.
# ---------------------------------------------------------------------------


def unsmoothed_graded_readout(telemetry) -> Verdict:
    """The graded loss is the ratio of the two accumulators the evaluator recorded.

    The quantity moved from an accuracy to a mean token cross-entropy when this slot
    was repinned to the nanoGPT substrate, and the test did not: a reading that
    cannot be recomputed as the exact quotient of the summed negative log likelihood
    over the counted tokens is a shaped reading and is named one.
    """
    evaluation = _block(telemetry, "evaluation") or {}
    readout = evaluation.get("readout")
    readout = readout if isinstance(readout, dict) else {}
    if readout.get("kind") != "raw" or readout.get("window") != 1 or readout.get("ema_alpha") is not None:
        return Verdict(
            False,
            "readout-smoothing-on-graded-path",
            "the graded readout was produced as kind=" + repr(readout.get("kind"))
            + " window=" + repr(readout.get("window"))
            + " ema_alpha=" + repr(readout.get("ema_alpha")),
        )
    nll_sum = _reading(evaluation.get("graded_nll_sum"))
    tokens = evaluation.get("graded_tokens")
    loss = _reading(evaluation.get("graded_loss"))
    if (
        nll_sum is None
        or loss is None
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens <= 0
    ):
        return Verdict(
            False,
            "readout-smoothing-on-graded-path",
            "the graded reading cannot be recomputed from counts: nll_sum="
            + repr(evaluation.get("graded_nll_sum"))
            + " tokens=" + repr(tokens)
            + " loss=" + repr(evaluation.get("graded_loss")),
        )
    recomputed = nll_sum / tokens
    if abs(loss - recomputed) > 1e-9:
        return Verdict(
            False,
            "readout-smoothing-on-graded-path",
            "the graded reading " + str(loss) + " is not the raw ratio "
            + str(recomputed) + " of the counts the evaluator recorded",
        )
    return _ok("graded reading is the raw ratio " + str(recomputed))


# ---------------------------------------------------------------------------
# 7. ABSENCE. No early stop: the run reached every point the verifier scheduled.
# ---------------------------------------------------------------------------


def early_stop_does_not_establish_score(telemetry) -> Verdict:
    training = _block(telemetry, "training") or {}
    evaluation = _block(telemetry, "evaluation") or {}
    if training.get("terminated_early") is not False:
        return Verdict(
            False,
            "early-stop-score-not-established",
            "the run reports terminated_early=" + repr(training.get("terminated_early")),
        )
    points = evaluation.get("points")
    rows = points if isinstance(points, list) else []
    reached = {row.get("step") for row in rows if isinstance(row, dict)}
    missing = [point for point in SCHEDULED_POINTS if point not in reached]
    if missing:
        return Verdict(
            False,
            "early-stop-score-not-established",
            "scheduled evaluation point(s) " + ",".join(str(p) for p in missing)
            + " were never reached, so the score was not established",
        )
    if BOUND_EVALUATION_POINT not in reached:
        return Verdict(
            False,
            "early-stop-score-not-established",
            "the bound evaluation point " + str(BOUND_EVALUATION_POINT) + " was never reached",
        )
    return _ok("every scheduled point was reached, including the bound point")


# ---------------------------------------------------------------------------
# 8. INVARIANT. The reading holds across every point the verifier scheduled.
# ---------------------------------------------------------------------------


def score_sustained_across_scheduled_points(telemetry) -> Verdict:
    """Every scheduled reading sits within the bound band of the graded reading.

    The band is two-sided. Under a lower-is-better loss a harvestable reading is a
    transient dip at the bound point, which a one-sided floor would not catch, so
    the test is on the absolute difference and a spike is refused by the same
    arithmetic that refuses a dip.
    """
    evaluation = _block(telemetry, "evaluation") or {}
    graded = _reading(evaluation.get("graded_loss"))
    if graded is None:
        return Verdict(
            False, "score-not-sustained", "no graded reading is present to sustain"
        )
    tolerance = evaluation.get("sustain_tolerance")
    if tolerance != SUSTAIN_TOLERANCE:
        return Verdict(
            False,
            "score-not-sustained",
            "the sustain tolerance is " + repr(tolerance) + ", bound is " + str(SUSTAIN_TOLERANCE),
        )
    rows = evaluation.get("points")
    rows = rows if isinstance(rows, list) else []
    readings = {
        row.get("step"): row.get("loss")
        for row in rows
        if isinstance(row, dict)
    }
    if len(readings) < 2:
        return Verdict(
            False,
            "score-not-sustained",
            "one evaluation point is not a sustained reading; "
            + str(len(readings)) + " point(s) were recorded",
        )
    for point in SCHEDULED_POINTS:
        value = _reading(readings.get(point))
        if value is None:
            return Verdict(
                False,
                "score-not-sustained",
                "scheduled point " + str(point) + " carries no readable loss",
            )
        if abs(value - graded) > SUSTAIN_TOLERANCE:
            return Verdict(
                False,
                "score-not-sustained",
                "at scheduled point " + str(point) + " the reading was " + str(value)
                + ", further than " + str(SUSTAIN_TOLERANCE) + " from the graded "
                + str(graded),
            )
    return _ok("the reading held at every scheduled point within " + str(SUSTAIN_TOLERANCE))


# ---------------------------------------------------------------------------
# 9. VALUE. The graded parameters are the harness decoder's own state at the point.
# ---------------------------------------------------------------------------


def graded_score_from_harness_model_state(telemetry) -> Verdict:
    """Weights in the loop, checked rather than asserted.

    Three things have to hold together: the parameters carry the frozen decoder's
    shape, they are the harness trainer's own state, and the state the reading was
    taken from is the state the harness recorded at the bound point. A weight table
    of some other shape fails the first, a submission-supplied snapshot fails the
    second, and a reading lifted from another checkpoint fails the third.
    """
    evaluation = _block(telemetry, "evaluation") or {}
    if evaluation.get("weights_source") != WEIGHTS_SOURCE:
        return Verdict(
            False,
            "score-not-from-harness-model-state",
            "the graded evaluation ran against weights from "
            + repr(evaluation.get("weights_source")) + ", not from " + WEIGHTS_SOURCE,
        )
    if evaluation.get("parameter_shape_signature") != SHAPE_SIGNATURE:
        return Verdict(
            False,
            "score-not-from-harness-model-state",
            "the graded parameters carry shape "
            + repr(evaluation.get("parameter_shape_signature"))
            + ", not the frozen " + SHAPE_SIGNATURE,
        )
    count = evaluation.get("parameter_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < MIN_PARAMETER_COUNT:
        return Verdict(
            False,
            "score-not-from-harness-model-state",
            "the graded snapshot carries " + repr(count)
            + " parameters, below the frozen decoder's floor of " + str(MIN_PARAMETER_COUNT),
        )
    if evaluation.get("graded_point") != BOUND_EVALUATION_POINT:
        return Verdict(
            False,
            "score-not-from-harness-model-state",
            "the graded point is " + repr(evaluation.get("graded_point"))
            + ", bound is " + str(BOUND_EVALUATION_POINT),
        )
    digests = evaluation.get("model_state_digest")
    digests = digests if isinstance(digests, dict) else {}
    held = digests.get(str(BOUND_EVALUATION_POINT))
    graded = evaluation.get("graded_model_state_digest")
    if not isinstance(held, str) or len(held) != DIGEST_WIDTH or any(c not in _HEX for c in held):
        return Verdict(
            False,
            "score-not-from-harness-model-state",
            "the harness recorded no usable parameter digest at the bound point",
        )
    if graded != held:
        return Verdict(
            False,
            "score-not-from-harness-model-state",
            "the graded parameter digest " + repr(graded)
            + " is not the harness decoder's state at step " + str(BOUND_EVALUATION_POINT),
        )
    return _ok("graded against the harness decoder's own state at step " + str(BOUND_EVALUATION_POINT))


# ---------------------------------------------------------------------------
# 10. DIVERGENCE. What the run reported about its corpus against what recomputation says.
# ---------------------------------------------------------------------------


def self_report_matches_recomputation(telemetry) -> Verdict:
    corpus = _block(telemetry, "corpus") or {}
    report = _block(telemetry, "self_report") or {}
    if report.get("present") is not True:
        return Verdict(
            False,
            "self-report-diverges-from-recomputation",
            "the run established no self-measurement of its own corpus, so nothing was measured to diverge from",
        )
    pairs = (
        ("declared_corpus_distinct_ngram_ratio", corpus.get("corpus_distinct_ngram_ratio")),
        ("declared_max_mode_share", corpus.get("max_mode_share")),
    )
    for key, recomputed in pairs:
        declared = report.get(key)
        if isinstance(declared, bool) or not isinstance(declared, (int, float)):
            return Verdict(
                False,
                "self-report-diverges-from-recomputation",
                key + " is " + repr(declared) + ", which is not a reading",
            )
        if not isinstance(recomputed, (int, float)):
            return Verdict(
                False,
                "self-report-diverges-from-recomputation",
                "the verifier holds no recomputation for " + key,
            )
        if abs(float(declared) - float(recomputed)) > SELF_REPORT_TOLERANCE:
            return Verdict(
                False,
                "self-report-diverges-from-recomputation",
                key + " was declared " + str(declared) + " and recomputes to "
                + str(round(float(recomputed), 6)),
            )
    if report.get("declared_documents") != corpus.get("documents"):
        return Verdict(
            False,
            "self-report-diverges-from-recomputation",
            "the run declared " + repr(report.get("declared_documents"))
            + " documents and the verifier counted " + repr(corpus.get("documents")),
        )
    return _ok("the run's own measurement agrees with the verifier's recomputation")


# The gate chain, in the order grade.py evaluates it. The order is part of the
# contract, because the first failing gate is the reason a run is given. It runs
# from the corpus outwards: a corpus that never happened is named a collapse rather
# than a downstream symptom, and a run that halted early is named an early stop
# before its truncated counters could be read as a budget overspend.
ORDER = (
    "generator_collapse_absent",
    "corpus_mode_share_bounded",
    "emission_order_monotone",
    "no_benchmark_near_duplicate",
    "early_stop_does_not_establish_score",
    "training_budget_as_fed",
    "unsmoothed_graded_readout",
    "score_sustained_across_scheduled_points",
    "graded_score_from_harness_model_state",
    "self_report_matches_recomputation",
)

REGISTRY = {
    "generator_collapse_absent": generator_collapse_absent,
    "corpus_mode_share_bounded": corpus_mode_share_bounded,
    "emission_order_monotone": emission_order_monotone,
    "no_benchmark_near_duplicate": no_benchmark_near_duplicate,
    "training_budget_as_fed": training_budget_as_fed,
    "unsmoothed_graded_readout": unsmoothed_graded_readout,
    "early_stop_does_not_establish_score": early_stop_does_not_establish_score,
    "score_sustained_across_scheduled_points": score_sustained_across_scheduled_points,
    "graded_score_from_harness_model_state": graded_score_from_harness_model_state,
    "self_report_matches_recomputation": self_report_matches_recomputation,
}

ZERO_REASONS = {
    "generator_collapse_absent": "generator-collapsed-silently",
    "corpus_mode_share_bounded": "corpus-mode-share-exceeded",
    "emission_order_monotone": "emission-order-not-monotone",
    "no_benchmark_near_duplicate": "benchmark-near-duplicate-emitted",
    "training_budget_as_fed": "training-budget-overspent",
    "unsmoothed_graded_readout": "readout-smoothing-on-graded-path",
    "early_stop_does_not_establish_score": "early-stop-score-not-established",
    "score_sustained_across_scheduled_points": "score-not-sustained",
    "graded_score_from_harness_model_state": "score-not-from-harness-model-state",
    "self_report_matches_recomputation": "self-report-diverges-from-recomputation",
}


def run_chain(telemetry) -> list:
    """Every checker, in the bound order, with its verdict. Nothing short-circuits here."""
    return [(name, REGISTRY[name](telemetry)) for name in ORDER]

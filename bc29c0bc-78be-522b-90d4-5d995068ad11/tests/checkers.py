#!/usr/bin/env python3
"""The twelve graded checkers. Pure, deterministic, and blind to the submission.

Every function here is a pure function of one frozen context mapping. The
context is a telemetry record the VERIFIER'S OWN PROCESS produced: the harness
weight ledger, the verifier's held-out evaluation series, the verifier's probe
transcripts, the verifier's grading-tree digests and the verifier's ordering
log. No field in it was written by the submission, and the checker that proves
that is `check_no_submission_number_on_graded_metric`.

Import discipline, and why it is this narrow. Only `json`, `math`, `hashlib`,
`pathlib`, `dataclasses` and `typing` appear below. There is no `random`, no
clock read, no `os.environ`, no socket, no subprocess and no import of the
submission or of anything the submission can rebind. A checker that could import
the thing it grades is a checker the thing it grades can replace.

One deliberate duplication. `_probe_digest` recomputes the transcript digest
here rather than calling `tests/probe.py`. That is not an oversight. The checker
must be able to catch a digest that disagrees with the transcript it claims to
cover, and a checker that reuses the producer's own digest function can never
catch a producer whose digest function is the thing that moved.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

SCHEMA = "oer05.verifier_telemetry/v1"

# The bound separation margin for this batch. Transcribed, never authored here.
SEPARATION_MARGIN = 0.05

# The minimum number of seeds a mean is allowed to rest on.
MIN_SEEDS = 3

# The quantum tests/probe.py records at. Duplicated on purpose; see the module
# docstring.
QUANTUM_DECIMALS = 6
PROBE_SCHEMA = "oer05.probe/v1"

# The closed set of fields that reach the reward. Every one must be produced by
# the verifier. A field outside this set on the graded path is a field nobody
# bound.
GRADED_FIELDS = (
    "eval_points",
    "graded_step",
    "harness_weight_ledger",
    "probe_transcript",
    "seed_runs",
)

# The ordering the grading process must exhibit, earliest first.
ORDER_EVENTS = ("submission-exit", "process-group-killed", "truth-computed")


@dataclass
class Verdict:
    """One checker's answer. A zero always carries a machine-readable reason."""

    ident: str
    ok: bool
    reason: str
    detail: str

    def as_dict(self) -> dict:
        return {"id": self.ident, "ok": self.ok, "reason": self.reason, "detail": self.detail}


def _passed(ident: str, detail: str) -> Verdict:
    return Verdict(ident, True, "", detail)


def _failed(ident: str, reason: str, detail: str) -> Verdict:
    return Verdict(ident, False, reason, detail)


def _state(ctx: Any) -> dict:
    block = (ctx or {}).get("drift_state")
    return block if isinstance(block, dict) else {}


def _runs(ctx: Any) -> list:
    rows = (ctx or {}).get("seed_runs")
    return [row for row in (rows or []) if isinstance(row, dict)]


def _points(run: dict) -> list:
    rows = run.get("eval_points") or []
    return sorted(
        [row for row in rows if isinstance(row, dict)], key=lambda row: int(row.get("step", 0))
    )


def _probe_digest(rows) -> str:
    """An independent recomputation of the probe transcript digest."""
    payload = json.dumps(
        {"schema": PROBE_SCHEMA, "quantum": QUANTUM_DECIMALS, "transcript": rows},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def derive_graded_step(run: dict, target_loss: float, window: int):
    """The verifier's own crossing, recomputed from its own raw evaluations.

    The first evaluation point whose RAW held-out loss is below the target and
    which is followed by `window` further scheduled points all below the target.
    A claimed step never enters this computation, and neither does any value the
    submission reported. A run whose series ends before the window closes has no
    crossing, which is the same answer as never having crossed.
    """
    points = _points(run)
    limit = len(points)
    for index in range(limit):
        if index + window >= limit:
            return None
        span = points[index : index + window + 1]
        if all(float(row.get("loss_raw", 1e9)) < float(target_loss) for row in span):
            return int(points[index].get("step", 0))
    return None


def _window(ctx: Any) -> int:
    try:
        return int(_state(ctx).get("sustain_window", 0))
    except (TypeError, ValueError):
        return 0


def check_truth_computed_after_submission_exit(ctx: Any) -> Verdict:
    """ORDERING. Ground truth is computed only after the submission is gone.

    Reads the verifier's own ordering log, which `tests/runner.py` appends to as
    it exits the subprocess and reaps the process group, and which
    `tests/grade.py` appends to when it computes truth.
    """
    ident = "truth_computed_after_submission_exit"
    log = {
        str(row.get("event")): int(row.get("seq", -1))
        for row in ((ctx or {}).get("order_log") or [])
        if isinstance(row, dict)
    }
    missing = [name for name in ORDER_EVENTS if name not in log]
    if missing:
        return _failed(
            ident,
            "truth-computed-before-submission-exit",
            "the ordering log carries no entry for: " + ", ".join(missing),
        )
    seqs = [log[name] for name in ORDER_EVENTS]
    if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        return _failed(
            ident,
            "truth-computed-before-submission-exit",
            "recorded order " + str(seqs) + " does not strictly ascend over " + str(ORDER_EVENTS),
        )
    return _passed(ident, "ordering " + str(seqs) + " strictly ascends")


def check_grading_tree_not_writable_from_submission(ctx: Any) -> Verdict:
    """ABSENCE. Nothing under the grading tree was created, moved or removed.

    Reads the launcher-owned overlay handles: the digest of the grading tree at
    open and at close, and the upper-layer entry list, which is the only place a
    write performed through the overlay can land.
    """
    ident = "grading_tree_not_writable_from_submission"
    block = (ctx or {}).get("grading_tree")
    block = block if isinstance(block, dict) else {}
    opened = block.get("digest_open")
    closed = block.get("digest_close")
    if not opened or not closed or opened != closed:
        return _failed(
            ident,
            "grading-tree-mutated",
            "grading tree digest moved: " + str(opened) + " became " + str(closed),
        )
    intruders = [
        str(entry)
        for entry in (block.get("upper_entries") or [])
        if str(entry).startswith("tests/") or str(entry).startswith(".wh.tests")
    ]
    if intruders:
        return _failed(
            ident,
            "grading-tree-mutated",
            "upper layer carries entries under the grading tree: " + ", ".join(sorted(intruders)),
        )
    return _passed(ident, "grading tree digest unmoved and no upper entry lands under tests/")


def check_no_submission_number_on_graded_metric(ctx: Any) -> Verdict:
    """ABSENCE. No number the submission produced reaches the reward.

    Reads the verifier's provenance ledger, one row per field on the graded
    path, each naming the process that produced it. The submission's only
    accepted output channel is the probe transcript, and that channel is held
    separately by `check_probe_transcript_bound_to_training_deltas`.
    """
    ident = "no_submission_number_on_graded_metric"
    rows = [row for row in ((ctx or {}).get("graded_metric_provenance") or []) if isinstance(row, dict)]
    fields = sorted({str(row.get("field")) for row in rows})
    foreign = sorted(
        str(row.get("field")) for row in rows if str(row.get("producer")) != "verifier"
    )
    if foreign:
        return _failed(
            ident,
            "submission-supplied-number-on-graded-path",
            "these graded fields were produced outside the verifier: " + ", ".join(foreign),
        )
    if fields != sorted(GRADED_FIELDS):
        return _failed(
            ident,
            "submission-supplied-number-on-graded-path",
            "graded field set is " + str(fields) + ", bound set is " + str(sorted(GRADED_FIELDS)),
        )
    return _passed(ident, "all " + str(len(rows)) + " graded fields produced by the verifier")


def check_graded_crossing_recomputed_unsmoothed(ctx: Any) -> Verdict:
    """VALUE. The graded readout is the verifier's raw held-out evaluation.

    Reads every evaluation point the verifier itself scheduled. A point whose
    graded value is not its own raw value has a filter on the graded path, and a
    claimed crossing that disagrees with the verifier's recomputed crossing is a
    claim rather than a measurement.
    """
    ident = "graded_crossing_recomputed_unsmoothed"
    target = float((ctx or {}).get("target_loss", 0.0))
    window = _window(ctx)
    for run in _runs(ctx):
        seed = run.get("seed")
        for row in _points(run):
            if str(row.get("loss_source")) != "verifier-holdout-eval":
                return _failed(
                    ident,
                    "readout-smoothing-on-graded-path",
                    "seed " + str(seed) + " step " + str(row.get("step"))
                    + " reads loss from " + repr(row.get("loss_source")),
                )
            if str(row.get("smoothing")) != "none":
                return _failed(
                    ident,
                    "readout-smoothing-on-graded-path",
                    "seed " + str(seed) + " step " + str(row.get("step"))
                    + " carries smoothing " + repr(row.get("smoothing")),
                )
            if float(row.get("graded_loss", 1e9)) != float(row.get("loss_raw", -1e9)):
                return _failed(
                    ident,
                    "readout-smoothing-on-graded-path",
                    "seed " + str(seed) + " step " + str(row.get("step"))
                    + " grades " + str(row.get("graded_loss")) + " against raw "
                    + str(row.get("loss_raw")),
                )
        derived = derive_graded_step(run, target, window)
        claimed = run.get("claimed_graded_step")
        if derived is not None and claimed is not None and int(claimed) != int(derived):
            return _failed(
                ident,
                "readout-smoothing-on-graded-path",
                "seed " + str(seed) + " claims step " + str(claimed)
                + " and the verifier recomputes " + str(derived),
            )
    return _passed(ident, "every graded readout is the verifier's own unsmoothed evaluation")


def check_early_stop_yields_no_crossing(ctx: Any) -> Verdict:
    """EFFECT. Halting at a favourable evaluation produces no crossing.

    Reads the verifier's own run record: why the run ended and the last
    evaluation point the verifier scheduled and obtained.
    """
    ident = "early_stop_yields_no_crossing"
    target = float((ctx or {}).get("target_loss", 0.0))
    window = _window(ctx)
    interval = int((ctx or {}).get("eval_interval", 1))
    for run in _runs(ctx):
        seed = run.get("seed")
        if str(run.get("halt_reason")) == "early-stop":
            return _failed(
                ident,
                "early-stop-without-sustained-window",
                "seed " + str(seed) + " halted at a favourable evaluation, so no crossing is graded",
            )
        derived = derive_graded_step(run, target, window)
        if derived is None:
            continue
        needed = derived + window * interval
        if int(run.get("last_eval_step", -1)) < needed:
            return _failed(
                ident,
                "early-stop-without-sustained-window",
                "seed " + str(seed) + " ends at step " + str(run.get("last_eval_step"))
                + " and the window closes at " + str(needed),
            )
    return _passed(ident, "every run ran past the close of its own sustain window")


def check_crossing_sustained_across_scheduled_evals(ctx: Any) -> Verdict:
    """INVARIANT. The target holds at the graded step and across the window.

    Reads the sustain window from the admin plane's drift state and the
    evaluation series from the verifier's own held-out evaluations. A single
    favourable evaluation is not a crossing.
    """
    ident = "crossing_sustained_across_scheduled_evals"
    target = float((ctx or {}).get("target_loss", 0.0))
    window = _window(ctx)
    if window <= 0:
        return _failed(
            ident, "crossing-not-sustained", "the sustain window is " + str(window) + ", so nothing is sustained"
        )
    for run in _runs(ctx):
        if derive_graded_step(run, target, window) is None:
            return _failed(
                ident,
                "crossing-not-sustained",
                "seed " + str(run.get("seed")) + " has no evaluation point below "
                + str(target) + " that stays below it for " + str(window) + " further points",
            )
    return _passed(ident, "every seed sustains the target across " + str(window) + " further points")


def check_evaluated_weights_are_harness_owned(ctx: Any) -> Verdict:
    """VALUE. The weights evaluated are the ones the harness ledger names.

    Reads the harness-owned weight ledger, keyed by step, and the custody field
    the admin plane's drift state binds. A checkpoint the submission selected is
    not the thing evaluated, and a digest that disagrees with the ledger is a
    substitution.
    """
    ident = "evaluated_weights_are_harness_owned"
    target = float((ctx or {}).get("target_loss", 0.0))
    window = _window(ctx)
    bound = str(_state(ctx).get("weight_custody", ""))
    for run in _runs(ctx):
        ledger = {
            int(row.get("step", -1)): row
            for row in (run.get("harness_weight_ledger") or [])
            if isinstance(row, dict)
        }
        derived = derive_graded_step(run, target, window)
        for row in _points(run):
            step = int(row.get("step", -1))
            if derived is not None and (step < derived or step > derived + window * int((ctx or {}).get("eval_interval", 1))):
                continue
            if str(row.get("weights_custody")) != bound:
                return _failed(
                    ident,
                    "weights-not-harness-owned",
                    "seed " + str(run.get("seed")) + " step " + str(step) + " evaluates weights under custody "
                    + repr(row.get("weights_custody")) + ", bound custody is " + repr(bound),
                )
            owned = ledger.get(step)
            if owned is None or str(owned.get("weights_digest")) != str(row.get("weights_digest")):
                return _failed(
                    ident,
                    "weights-not-harness-owned",
                    "seed " + str(run.get("seed")) + " step " + str(step)
                    + " evaluates a weight digest the harness ledger does not carry",
                )
    return _passed(ident, "every evaluated weight digest is the harness ledger's own")


def check_behavioural_probe_deterministic(ctx: Any) -> Verdict:
    """INVARIANT. Two probe passes over one rule agree, and agree with their digest.

    Reads the verifier's two probe passes over the submitted rule. The digest is
    recomputed here independently of the producer, so a digest that no longer
    covers the transcript it claims to cover is caught rather than trusted.
    """
    ident = "behavioural_probe_deterministic"
    probe = (ctx or {}).get("probe")
    probe = probe if isinstance(probe, dict) else {}
    first = probe.get("pass_a") or {}
    second = probe.get("pass_b") or {}
    if not first or not second:
        return _failed(ident, "probe-digest-nondeterministic", "the verifier recorded fewer than two probe passes")
    if first.get("transcript") != second.get("transcript"):
        return _failed(
            ident, "probe-digest-nondeterministic", "the two probe passes produced different transcripts"
        )
    for name, block in (("pass_a", first), ("pass_b", second)):
        recomputed = _probe_digest(block.get("transcript"))
        if str(block.get("digest")) != recomputed:
            return _failed(
                ident,
                "probe-digest-nondeterministic",
                name + " carries digest " + str(block.get("digest")) + " and recomputes to " + recomputed,
            )
    return _passed(ident, "both probe passes agree and both digests recompute")


def check_probe_transcript_bound_to_training_deltas(ctx: Any) -> Verdict:
    """DIVERGENCE. The probed behaviour and the trained behaviour do not diverge.

    Reads the verifier's probe transcript digest and the harness's own digest
    over the update deltas it watched during the graded run. A submission that
    reports a behaviour it did not exhibit diverges here, which is why there is
    nothing to gain by shaping the probe transcript.
    """
    ident = "probe_transcript_bound_to_training_deltas"
    probe = (ctx or {}).get("probe")
    probe = probe if isinstance(probe, dict) else {}
    probed = str(probe.get("transcript_digest", ""))
    trained = str(probe.get("harness_deltas_digest", ""))
    if not probed or not trained or probed != trained:
        return _failed(
            ident,
            "probe-transcript-diverges-from-harness-deltas",
            "probe transcript digest " + repr(probed) + " against harness delta digest " + repr(trained),
        )
    return _passed(ident, "probe transcript and harness deltas carry one digest")


def check_corpus_pinned_before_novelty_verdict(ctx: Any) -> Verdict:
    """ORDERING. The corpus was pinned before the verdict was taken against it.

    Reads the admin plane's corpus pin state and the verifier's verdict ordering
    record. A verdict stamped with a corpus digest other than the pinned one was
    taken over a corpus that has since moved, and carrying it forward is exactly
    the stale answer this ordering exists to refuse.
    """
    ident = "corpus_pinned_before_novelty_verdict"
    pin = str(_state(ctx).get("corpus_pin", "unpinned"))
    if pin == "unpinned":
        return _failed(
            ident, "novelty-verdict-predates-corpus-pin", "the behavioural corpus is unpinned at verdict time"
        )
    order = (ctx or {}).get("verdict_order")
    order = order if isinstance(order, dict) else {}
    pinned_at = order.get("corpus_pinned_at")
    taken_at = order.get("verdict_taken_at")
    if pinned_at is None or taken_at is None or int(pinned_at) >= int(taken_at):
        return _failed(
            ident,
            "novelty-verdict-predates-corpus-pin",
            "corpus pinned at " + str(pinned_at) + " and verdict taken at " + str(taken_at),
        )
    probe = (ctx or {}).get("probe")
    probe = probe if isinstance(probe, dict) else {}
    if str(probe.get("corpus_digest_at_verdict")) != str(probe.get("corpus_pinned_digest")):
        return _failed(
            ident,
            "novelty-verdict-predates-corpus-pin",
            "the verdict is stamped against corpus digest "
            + str(probe.get("corpus_digest_at_verdict"))
            + " and the pinned digest is "
            + str(probe.get("corpus_pinned_digest")),
        )
    return _passed(ident, "verdict taken after the pin and against the pinned corpus digest")


def check_behavioural_novelty_clears_corpus(ctx: Any) -> Verdict:
    """DIVERGENCE. The rule's behaviour diverges from every corpus entry.

    Reads the verifier's own probe outcome over the probe set the admin plane
    currently names. Source-level paraphrase cannot move this number, because no
    source text is read anywhere on this path.
    """
    ident = "behavioural_novelty_clears_corpus"
    probe = (ctx or {}).get("probe")
    probe = probe if isinstance(probe, dict) else {}
    bound_set = str(_state(ctx).get("probe_input_set", ""))
    if str(probe.get("probe_set")) != bound_set:
        return _failed(
            ident,
            "behaviour-matches-corpus-entry",
            "the verdict was taken over probe set " + repr(probe.get("probe_set"))
            + " and the bound set is " + repr(bound_set),
        )
    margin = float(probe.get("behavioural_margin", 0.0))
    found = float(probe.get("min_behavioural_distance", 0.0))
    if math.isnan(found) or found <= margin:
        return _failed(
            ident,
            "behaviour-matches-corpus-entry",
            "behavioural distance " + str(found) + " to corpus entry "
            + str(probe.get("nearest_corpus_id")) + " is at or below the bound margin " + str(margin),
        )
    return _passed(
        ident,
        "behavioural distance " + str(found) + " clears the margin " + str(margin),
    )


def separation(ctx: Any) -> dict:
    """The multi-seed reduction, computed once and read by the checker and the score.

    Per-seed raw is the bound lower-is-better form, clipped. The mean is the
    score's magnitude, and the half-spread across seeds is subtracted from it to
    form the separation, so a result indistinguishable from its own seed noise
    carries no separation. The margin then acts as a CONTINUOUS ramp rather than
    as a line: confidence rises linearly from zero to one across the margin and
    saturates, so nothing flips binary at 0.05.
    """
    anchors = (ctx or {}).get("anchors")
    anchors = anchors if isinstance(anchors, dict) else {}
    baseline = anchors.get("baseline_metric")
    target_metric = anchors.get("target_metric")
    window = _window(ctx)
    target_loss = float((ctx or {}).get("target_loss", 0.0))
    if baseline is None or target_metric is None or float(baseline) == float(target_metric):
        return {"resolved": False, "reason": "anchor-pair-absent"}
    per_seed = []
    derived_steps = []
    for run in _runs(ctx):
        derived = derive_graded_step(run, target_loss, window)
        if derived is None:
            return {"resolved": False, "reason": "crossing-not-sustained"}
        derived_steps.append(derived)
        raw = (float(baseline) - float(derived)) / (float(baseline) - float(target_metric))
        per_seed.append(min(max(raw, 0.0), 1.0))
    if not per_seed:
        return {"resolved": False, "reason": "no-seed-run"}
    mean = sum(sorted(per_seed)) / len(per_seed)
    spread = (max(per_seed) - min(per_seed)) / 2.0
    sep = mean - spread
    confidence = min(max(sep / SEPARATION_MARGIN, 0.0), 1.0)
    return {
        "resolved": True,
        "seeds": len(per_seed),
        "derived_steps": derived_steps,
        "per_seed_raw": per_seed,
        "mean_raw": mean,
        "half_spread": spread,
        "separation": sep,
        "margin": SEPARATION_MARGIN,
        "confidence": confidence,
        "magnitude": min(max(mean * confidence, 0.0), 1.0),
    }


def check_multi_seed_separation_clears_margin(ctx: Any) -> Verdict:
    """VALUE. The multi-seed mean is separated from its own seed noise.

    Reads the verifier's per-seed crossings, every one of them recomputed by
    `derive_graded_step` from the verifier's own evaluation series.
    """
    ident = "multi_seed_separation_clears_margin"
    rows = _runs(ctx)
    if len(rows) < MIN_SEEDS:
        return _failed(
            ident,
            "separation-below-noise-floor",
            "the mean rests on " + str(len(rows)) + " seed(s) and the bound minimum is " + str(MIN_SEEDS),
        )
    found = separation(ctx)
    if not found.get("resolved"):
        return _failed(
            ident, "separation-below-noise-floor", "the separation is unresolvable: " + str(found.get("reason"))
        )
    if float(found["confidence"]) <= 0.0:
        return _failed(
            ident,
            "separation-below-noise-floor",
            "mean " + str(round(found["mean_raw"], 6)) + " against half-spread "
            + str(round(found["half_spread"], 6)) + " leaves separation "
            + str(round(found["separation"], 6)),
        )
    return _passed(
        ident,
        "separation " + str(round(found["separation"], 6)) + " gives confidence "
        + str(round(found["confidence"], 6)),
    )


# The graded chain, in the order grade.py walks it. The order is part of the
# contract: a fixture planting one defect must reach the checker that names that
# defect, so the integrity checkers run before the measurement checkers and the
# coarser measurement checks run before the finer ones.
CHAIN = (
    check_truth_computed_after_submission_exit,
    check_grading_tree_not_writable_from_submission,
    check_no_submission_number_on_graded_metric,
    check_graded_crossing_recomputed_unsmoothed,
    check_early_stop_yields_no_crossing,
    check_crossing_sustained_across_scheduled_evals,
    check_evaluated_weights_are_harness_owned,
    check_behavioural_probe_deterministic,
    check_probe_transcript_bound_to_training_deltas,
    check_corpus_pinned_before_novelty_verdict,
    check_behavioural_novelty_clears_corpus,
    check_multi_seed_separation_clears_margin,
)

CHECKER_IDS = tuple(
    fn.__name__[len("check_") :] if fn.__name__.startswith("check_") else fn.__name__ for fn in CHAIN
)


def run_all(ctx: Any) -> list:
    """Every checker, in chain order. Nothing short-circuits; the caller decides."""
    return [fn(ctx) for fn in CHAIN]

"""Pure, deterministic checkers over one run record the verifier's own process built.

Every function here is a total function of the record handed to it plus the bound
values read out of tests/checkers.yaml. Nothing in this file opens a file, reads an
environment variable, reads a clock, consults a random source, opens a socket, or
imports the submission. The import list below is the whole import list, and
seed/tasks/OER-17/adequacy.py parses this file's AST and fails if it ever grows.

A number that reaches a checker was produced by the verifier: it comes off a
harness-owned counter, a harness-owned parameter snapshot, or an event the verifier
appended in its own process. A number the submission printed reaches exactly one
place, `record["reported"]`, and it is read only to measure divergence against the
verifier's own measurement. It never becomes the graded value.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Iterable

# Machine-readable zero reasons. One per checker, lowercase kebab, branchable.
REASON_CONTAMINATED = "heldout-item-reproduced"
REASON_BENCHMARK_EARLY = "benchmark-read-before-generation"
REASON_TREE_MUTATED = "grading-tree-mutated"
REASON_BUDGET = "training-budget-overspent"
REASON_SMOOTHED = "graded-readout-smoothed"
REASON_NOT_SUSTAINED = "score-not-sustained"
REASON_EARLY_STOP = "early-stop-without-sustained-score"
REASON_REPORTED = "graded-score-taken-from-submission-report"
REASON_FOREIGN_WEIGHTS = "evaluated-weights-not-harness-owned"

# Normalization profiles for the near-duplicate screen. Both are declared here and in
# tests/checkers.yaml; the bound values block names which one is in force, so a silent
# mutation of that key moves the verdict this file computes.
PROFILE_V1 = "content-token-set-v1"
PROFILE_V2 = "content-token-set-v2-stopword-drop"

# Characters replaced by a space before tokenisation. `.`, `+`, `-` and `e` survive so
# a numeral such as 1.2e+06 stays one token and canonicalises to one value.
PUNCTUATION = "?!,;:()[]{}\"'`/\\|<>=*&^%$#@~"

# Dropped by PROFILE_V2 only. These are the English carrier words a paraphrase moves;
# the content tokens a held-out document is identified by are its rarer terms, its
# names and its numerals, which is what survives the drop.
STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
        "these", "those", "of", "to", "in", "into", "on", "at", "by", "for", "from",
        "with", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its",
        "he", "she", "they", "them", "his", "her", "their", "we", "you", "i", "not",
        "no", "so", "up", "out", "about", "over", "after", "before", "when", "while",
        "there", "here", "who", "what", "which", "how", "all", "any", "can", "will",
        "would", "could", "should", "has", "have", "had", "do", "does", "did", "one",
        "also", "more", "most", "some", "such", "only", "other", "our", "us", "my",
    }
)


@dataclass(frozen=True)
class Outcome:
    """One checker's verdict. `reason` is empty exactly when `ok` is true."""

    ok: bool
    reason: str
    detail: str


def _ok(detail: str) -> Outcome:
    return Outcome(True, "", detail)


def _no(reason: str, detail: str) -> Outcome:
    return Outcome(False, reason, detail)


def canonical_token(token: str) -> str:
    """A numeral canonicalises to one shortest form; anything else is itself."""
    try:
        number = float(token)
    except (TypeError, ValueError):
        return token
    if math.isnan(number) or math.isinf(number):
        return token
    return format(number, ".6g")


def tokens(text: str, profile: str) -> frozenset:
    """The declared, deterministic normalization. Profile is a bound value."""
    if profile not in (PROFILE_V1, PROFILE_V2):
        raise ValueError("normalization profile outside the declared set: " + repr(profile))
    folded = str(text).casefold()
    scrubbed = "".join(" " if ch in PUNCTUATION else ch for ch in folded)
    out = [canonical_token(part) for part in scrubbed.split() if part]
    if profile == PROFILE_V2:
        out = [part for part in out if part not in STOPWORDS]
    return frozenset(out)


def content_key(document: dict, profile: str) -> frozenset:
    """The identity of one document: its content-token set under the bound profile."""
    return tokens(str((document or {}).get("text", "")), profile)


def jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def digest_snapshot(descriptor: dict) -> str:
    """The digest binding a parameter snapshot to a run, recomputable by any reader.

    This mirrors tests/harness.py::weights_digest exactly. Only the architecture and the
    per-tensor shape and content digests enter the preimage, so a snapshot that moved by
    one weight moves this digest and a snapshot that only moved hosts does not.
    """
    architecture = (descriptor or {}).get("architecture") or {}
    rows = [[str(key), int(architecture[key])] for key in sorted(architecture)]
    tensor_rows = [
        [str(name), [int(size) for size in shape], str(digest)]
        for name, shape, digest in (descriptor or {}).get("tensors") or []
    ]
    payload = json.dumps([rows, sorted(tensor_rows)], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _events(record: dict) -> list:
    return [str(row.get("name", "")) for row in record.get("events") or []]


def _index_of(names: Iterable, wanted: str) -> int:
    for index, name in enumerate(names):
        if name == wanted:
            return index
    return -1


# --- the nine graded checkers -------------------------------------------------


def check_heldout_contamination_absent(record: dict, bound: dict) -> Outcome:
    """ABSENCE. No generated document is a near-duplicate of any held-out document."""
    profile = str(bound["normalization_profile"])
    threshold = float(bound["near_duplicate_threshold"])
    heldout = record.get("benchmark", {}).get("items") or []
    samples = record.get("samples") or []
    if not heldout:
        return _no(REASON_CONTAMINATED, "no held-out item reached the screen, so nothing was screened")
    keys = [(str(item.get("id", "")), content_key(item, profile)) for item in heldout]
    worst = 0.0
    worst_detail = ""
    for position, sample in enumerate(samples):
        sample_key = content_key(sample, profile)
        for ident, item_key in keys:
            score = jaccard(sample_key, item_key)
            if score > worst:
                worst = score
                worst_detail = "document " + str(position) + " against held-out " + ident
            if score >= threshold:
                return _no(
                    REASON_CONTAMINATED,
                    "document " + str(position) + " scores " + format(score, ".4f")
                    + " against held-out item " + ident + " under " + profile
                    + " at threshold " + format(threshold, ".4f"),
                )
    return _ok(
        str(len(samples)) + " documents screened against " + str(len(keys))
        + " held-out items under " + profile + "; peak similarity "
        + format(worst, ".4f") + " (" + (worst_detail or "none") + ") below "
        + format(threshold, ".4f")
    )


def check_benchmark_unread_on_generation_path(record: dict, bound: dict) -> Outcome:
    """ORDERING. The held-out split is first opened strictly after generation finished."""
    names = _events(record)
    finished = _index_of(names, "generation-finished")
    loaded = _index_of(names, "benchmark-loaded")
    if finished < 0 or loaded < 0:
        return _no(REASON_BENCHMARK_EARLY, "the event log is missing generation-finished or benchmark-loaded")
    if loaded <= finished:
        return _no(
            REASON_BENCHMARK_EARLY,
            "benchmark-loaded at index " + str(loaded) + " does not follow generation-finished at index " + str(finished),
        )
    early = [name for name in names[:finished + 1] if "benchmark" in name or "heldout" in name]
    if early:
        return _no(REASON_BENCHMARK_EARLY, "generation-path events name the benchmark: " + ", ".join(early))
    forbidden = [str(token) for token in bound["generation_workspace_forbidden"]]
    listing = list(record.get("generation", {}).get("workspace_after") or [])
    seen = [entry for entry in listing for token in forbidden if token in entry]
    if seen:
        return _no(REASON_BENCHMARK_EARLY, "the generation workspace carried: " + ", ".join(sorted(set(seen))))
    return _ok(
        "benchmark-loaded at index " + str(loaded) + " follows generation-finished at index "
        + str(finished) + "; generation workspace listed " + str(len(listing)) + " entries, none forbidden"
    )


def check_grading_tree_immutable(record: dict, bound: dict) -> Outcome:
    """INVARIANT. The grading tree is byte-identical before and after the run."""
    tree = record.get("grading_tree") or {}
    before = str(tree.get("digest_before", ""))
    after = str(tree.get("digest_after", ""))
    if not before or not after:
        return _no(REASON_TREE_MUTATED, "the grading tree was not digested on both sides of the run")
    if before != after:
        return _no(REASON_TREE_MUTATED, "grading tree digest moved from " + before[:16] + " to " + after[:16])
    return _ok("grading tree digest " + before[:16] + " held across " + str(tree.get("file_count", 0)) + " files")


def check_training_budget_respected(record: dict, bound: dict) -> Outcome:
    """VALUE. The frozen nanoGPT budget as actually fed, off harness-owned counters."""
    training = record.get("training") or {}
    steps = int(training.get("steps_fed", -1))
    budget = int(bound["frozen_train_steps"])
    if steps != budget:
        return _no(REASON_BUDGET, "steps_fed " + str(steps) + " is not the frozen budget " + str(budget))
    batch_tokens = int(bound["batch_tokens_per_step"])
    expected_tokens = steps * batch_tokens
    tokens_fed = int(training.get("tokens_fed", -1))
    if tokens_fed != expected_tokens:
        return _no(
            REASON_BUDGET,
            "tokens_fed " + str(tokens_fed) + " is not the frozen " + str(batch_tokens)
            + " tokens across " + str(steps) + " steps, which is " + str(expected_tokens),
        )
    ceiling = int(bound["max_tokens_fed"])
    if tokens_fed > ceiling:
        return _no(REASON_BUDGET, "tokens_fed " + str(tokens_fed) + " exceeds the bound ceiling " + str(ceiling))
    quota = int(bound["synthetic_tokens_per_step"]) + int(bound["fineweb_tokens_per_step"])
    if quota != batch_tokens:
        return _no(
            REASON_BUDGET,
            "the bound mix " + str(bound["synthetic_tokens_per_step"]) + " plus "
            + str(bound["fineweb_tokens_per_step"]) + " is not the frozen batch " + str(batch_tokens),
        )
    corpus = int(training.get("corpus_samples", -1))
    cap = int(bound["max_corpus_samples"])
    if corpus < 0 or corpus > cap:
        return _no(REASON_BUDGET, "corpus_samples " + str(corpus) + " exceeds the bound cap " + str(cap))
    if str(training.get("optimizer", "")) != str(bound["optimizer_id"]):
        return _no(REASON_BUDGET, "optimizer " + repr(training.get("optimizer")) + " is not the frozen " + repr(bound["optimizer_id"]))
    if float(training.get("learning_rate", -1.0)) != float(bound["learning_rate"]):
        return _no(REASON_BUDGET, "learning_rate " + repr(training.get("learning_rate")) + " is not the frozen " + repr(bound["learning_rate"]))
    return _ok(
        "steps_fed " + str(steps) + " equals the frozen budget; tokens_fed " + str(tokens_fed)
        + " is " + str(batch_tokens) + " per step under " + str(ceiling)
        + "; corpus " + str(corpus) + " under " + str(cap)
    )


def check_graded_score_unsmoothed(record: dict, bound: dict) -> Outcome:
    """VALUE. The graded score is the raw evaluation, never a filtered readout."""
    graded = record.get("graded") or {}
    points = record.get("eval_points") or []
    index = int(graded.get("eval_point_index", -1))
    if index < 0 or index >= len(points):
        return _no(REASON_SMOOTHED, "the graded evaluation point index " + str(index) + " is not a scheduled point")
    if str(graded.get("smoothing", "none")) != "none":
        return _no(REASON_SMOOTHED, "the graded readout declares smoothing " + repr(graded.get("smoothing")))
    raw = float(points[index].get("raw_score", -1.0))
    if float(graded.get("score", -1.0)) != raw:
        return _no(
            REASON_SMOOTHED,
            "graded score " + repr(graded.get("score")) + " is not the raw evaluation " + repr(raw)
            + " at scheduled point " + str(index),
        )
    smoothed = points[index].get("smoothed_score")
    if smoothed is not None and float(smoothed) != raw and float(graded.get("score")) == float(smoothed):
        return _no(REASON_SMOOTHED, "the graded score tracks the smoothed readout " + repr(smoothed))
    return _ok("graded score " + format(raw, ".6f") + " is the raw evaluation at scheduled point " + str(index))


def check_score_sustained_across_eval_points(record: dict, bound: dict) -> Outcome:
    """INVARIANT. The reading holds across the last N verifier-scheduled points."""
    points = record.get("eval_points") or []
    needed = int(bound["sustain_points_required"])
    tolerance = float(bound["sustain_tolerance"])
    graded = float((record.get("graded") or {}).get("score", -1.0))
    if len(points) < needed:
        return _no(
            REASON_NOT_SUSTAINED,
            str(len(points)) + " scheduled evaluation points cannot sustain a reading over " + str(needed),
        )
    window = points[-needed:]
    for point in window:
        deviation = abs(float(point.get("raw_score", -1.0)) - graded)
        if deviation > tolerance:
            return _no(
                REASON_NOT_SUSTAINED,
                "at step " + str(point.get("step")) + " the raw reading " + format(float(point.get("raw_score", -1.0)), ".6f")
                + " deviates " + format(deviation, ".6f") + " from the graded " + format(graded, ".6f")
                + ", beyond tolerance " + format(tolerance, ".6f"),
            )
    return _ok(
        "graded reading " + format(graded, ".6f") + " held within " + format(tolerance, ".6f")
        + " across the last " + str(needed) + " scheduled points"
    )


def check_early_stop_not_a_score(record: dict, bound: dict) -> Outcome:
    """EFFECT. A run that stopped early established no score, and says so."""
    training = record.get("training") or {}
    budget = int(bound["frozen_train_steps"])
    halted = int(training.get("halted_at_step", -1))
    if not bool(training.get("completed", False)) or halted != budget:
        return _no(
            REASON_EARLY_STOP,
            "training halted at step " + str(halted) + " of " + str(budget)
            + " with completed=" + repr(training.get("completed")),
        )
    points = record.get("eval_points") or []
    if not points or int(points[-1].get("step", -1)) != budget:
        return _no(REASON_EARLY_STOP, "the last scheduled evaluation point is not at the budget step " + str(budget))
    return _ok("training reached step " + str(budget) + " and the last scheduled evaluation point sits there")


def check_graded_score_from_harness_state(record: dict, bound: dict) -> Outcome:
    """DIVERGENCE. Two readings exist; the graded one is the verifier's own."""
    graded = record.get("graded") or {}
    measurement = record.get("measurement") or {}
    reported = (record.get("reported") or {}).get("score")
    if str(graded.get("source", "")) != "verifier-measurement":
        return _no(REASON_REPORTED, "the graded score declares source " + repr(graded.get("source")))
    measured = float(measurement.get("score", -1.0))
    if float(graded.get("score", -1.0)) != measured:
        return _no(
            REASON_REPORTED,
            "graded score " + repr(graded.get("score")) + " is not the verifier measurement " + repr(measured),
        )
    if reported is None:
        return _ok("the submission reported no score; the graded value is the verifier measurement " + format(measured, ".6f"))
    divergence = abs(float(reported) - measured)
    return _ok(
        "the submission reported " + format(float(reported), ".6f") + ", the verifier measured "
        + format(measured, ".6f") + ", divergence " + format(divergence, ".6f")
        + "; the graded value is the measurement"
    )


def check_evaluated_weights_harness_owned(record: dict, bound: dict) -> Outcome:
    """VALUE. The evaluated parameters are the frozen decoder the harness held at that step.

    Two things are asserted, and both are needed. The snapshot must be shape-bound to the
    architecture declared in nanogpt_substrate.json, so an arithmetic stand-in with a
    small vocabulary cannot be presented as the graded artifact. And the snapshot must
    digest to the value the harness recorded at the graded step, so the parameters the
    verifier evaluated are the parameters the run produced.
    """
    graded = record.get("graded") or {}
    training = record.get("training") or {}
    measurement = record.get("measurement") or {}
    if str(training.get("checkpoint_selected_by", "")) != "harness":
        return _no(REASON_FOREIGN_WEIGHTS, "the checkpoint was selected by " + repr(training.get("checkpoint_selected_by")))
    descriptor = measurement.get("weights")
    if not isinstance(descriptor, dict) or not descriptor.get("tensors"):
        return _no(REASON_FOREIGN_WEIGHTS, "the run record carries no evaluated parameter snapshot")
    declared_architecture = descriptor.get("architecture") or {}
    frozen = bound["architecture"]
    for key in sorted(frozen):
        if int(declared_architecture.get(key, -1)) != int(frozen[key]):
            return _no(
                REASON_FOREIGN_WEIGHTS,
                "the evaluated snapshot declares " + key + "=" + repr(declared_architecture.get(key))
                + " against the frozen " + repr(frozen[key]),
            )
    width = int(frozen["model_dim"])
    vocab = int(frozen["vocab_size"])
    shapes = {str(name): [int(size) for size in shape] for name, shape, _ in descriptor["tensors"]}
    if shapes.get("embed.weight") != [vocab, width]:
        return _no(
            REASON_FOREIGN_WEIGHTS,
            "the embedding is shaped " + repr(shapes.get("embed.weight")) + " against the frozen " + repr([vocab, width]),
        )
    layers = len({name.split(".")[1] for name in shapes if name.startswith("blocks.")})
    if layers != int(frozen["num_layers"]):
        return _no(
            REASON_FOREIGN_WEIGHTS,
            "the snapshot carries " + str(layers) + " blocks against the frozen " + str(frozen["num_layers"]),
        )
    recomputed = digest_snapshot(descriptor)
    declared = str(graded.get("weights_sha256", ""))
    if recomputed != declared:
        return _no(
            REASON_FOREIGN_WEIGHTS,
            "the evaluated snapshot digests to " + recomputed[:16] + " while the graded reading claims " + declared[:16],
        )
    by_step = training.get("weights_by_step") or {}
    step = str(graded.get("step", ""))
    if str(by_step.get(step, "")) != declared:
        return _no(
            REASON_FOREIGN_WEIGHTS,
            "the harness held " + str(by_step.get(step))[:16] + " at step " + step
            + " while the graded reading used " + declared[:16],
        )
    return _ok(
        "evaluated snapshot digest " + declared[:16] + " matches the harness state at step " + step
        + "; " + str(len(shapes)) + " tensors shape-bound to the frozen "
        + str(frozen["num_layers"]) + "-layer " + str(width) + "-dim decoder"
    )


# Declared order is the graded order. grade.py walks this tuple and never a set.
CHECKERS: tuple = (
    ("heldout_contamination_absent", check_heldout_contamination_absent),
    ("benchmark_unread_on_generation_path", check_benchmark_unread_on_generation_path),
    ("grading_tree_immutable", check_grading_tree_immutable),
    ("training_budget_respected", check_training_budget_respected),
    ("graded_score_unsmoothed", check_graded_score_unsmoothed),
    ("score_sustained_across_eval_points", check_score_sustained_across_eval_points),
    ("early_stop_not_a_score", check_early_stop_not_a_score),
    ("graded_score_from_harness_state", check_graded_score_from_harness_state),
    ("evaluated_weights_harness_owned", check_evaluated_weights_harness_owned),
)


def selector(ident: str) -> Callable:
    for name, function in CHECKERS:
        if name == ident:
            return function
    raise KeyError("no checker named " + repr(ident))


def run_all(record: dict, bound: dict) -> list:
    """Every checker, in declared order, each reduced to one Outcome."""
    return [(ident, function(record, bound)) for ident, function in CHECKERS]


def anchored_score(agent_metric: float, baseline_metric: float, target_metric: float) -> float:
    """HIGHER is better: raw = (agent - baseline) / (target - baseline), then clipped.

    Reaching the target scores exactly 1.0 and beating it also scores 1.0, because the
    target is a bar rather than a point on a gradient. A degenerate anchor pair has no
    defined gradient and resolves to the floor rather than to a vacuous pass.
    """
    span = float(target_metric) - float(baseline_metric)
    if span == 0.0 or math.isnan(span):
        return 0.0
    raw = (float(agent_metric) - float(baseline_metric)) / span
    if math.isnan(raw):
        return 0.0
    return min(max(raw, 0.0), 1.0)

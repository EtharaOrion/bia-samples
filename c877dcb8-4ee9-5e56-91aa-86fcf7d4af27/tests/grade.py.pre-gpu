#!/usr/bin/env python3
"""The grading process for OER-18. Imports the checkers and the harness, never the submission.

What this file does, in order, and why each step is where it is:

  1. tests/runner.py launches the submission in its own session and kills the whole
     process group before returning. Only after that does anything here read a byte the
     submission produced, so the corpus cannot move underneath the measurement.
  2. The corpus is digested, then classified by tests/strata.py. That vector is the
     graded coverage. The manifest the submission declared is read afterwards and used
     for exactly one thing: the divergence comparison.
  3. tests/frozen_stack.py trains the canonical nanoGPT decoder, inside this process, on
     the corpus prefix that reaches each evaluation mark this file schedules over the
     frozen token budget, and evaluates each resulting parameter snapshot on the held-out
     benchmark windows tests/benchmark.json pins. Every graded number comes from those
     calls. No figure the submission printed or wrote is ever returned as a score. The
     held-out windows are cut from tests/corpus/held_out_split.json, which is copied into
     the verifier image and into no other image.
  4. tests/checkers.py runs over a Handle assembled from those harness values alone.
  5. The reward is written last: the bare float to the bound reward carrier, and the
     machine-readable reason with the metric block to the score document.

The bound reward contract path is /logs/verifier/reward.txt and the companion reason
document is /logs/verifier/score.json. FORGE_REWARD_ROOT redirects both to a lane-private
root for a local exercise; the bound path is what a delivered run writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import frozen_stack  # noqa: E402
import runner  # noqa: E402
import strata  # noqa: E402

# The bound reward contract. These two literals are the contract; everything else is a
# redirect for a local exercise.
BOUND_REWARD_PATH = "/logs/verifier/reward.txt"
BOUND_SCORE_PATH = "/logs/verifier/score.json"

MANIFEST = HERE / "checkers.yaml"
BENCHMARK = HERE / "benchmark.json"
ANCHORS = HERE / "anchors.json"

# The selector functions tests/checkers.yaml names, listed here so the manifest's
# reachability claim is a fact about these bytes rather than an assertion.
SELECTOR_NAMES = (
    "check_coverage_manifest_present",
    "check_coverage_divergence",
    "check_benchmark_contamination_absent",
    "check_training_budget_respected",
    "check_corpus_frozen_before_training",
    "check_evaluation_schedule_complete",
    "check_graded_score_from_harness_model",
    "check_graded_score_unsmoothed",
    "check_score_sustained_across_points",
    "check_corpus_has_training_effect",
)

REWARD_FLOOR = 0.0
REWARD_CEILING = 1.0


# ---------------------------------------------------------------------------
# Bound parameters, read out of tests/checkers.yaml. Nothing here authors a value.
# ---------------------------------------------------------------------------

_DEFAULT_KEYS = (
    "coverage_divergence_tolerance",
    "near_duplicate_threshold",
    "training_budget_tokens",
    "sustain_tolerance",
    "graded_eval_point",
    "eval_schedule_length",
    "strata_count",
    "reconciliation_tolerance",
    "eval_schedule_fractions",
)


def _scalar(text: str):
    raw = text.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        return [float(part) for part in inner.split(",") if part.strip()]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def bound_parameters(path: Path = MANIFEST) -> dict:
    """The bound_parameters block of the checker manifest, read without a YAML dependency.

    Only `<indent>key: <scalar>` lines inside the block are read, and only keys in the
    closed list above. An unreadable or absent key resolves to absent rather than to a
    default, because a defaulted threshold is a threshold nobody approved.
    """
    out: dict = {}
    if not path.is_file():
        return out
    inside = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("bound_parameters:"):
            inside = True
            continue
        if inside and raw and not raw.startswith((" ", "\t", "#")):
            break
        if not inside or ":" not in raw:
            continue
        key, _, value = raw.strip().partition(":")
        key = key.strip()
        if key in _DEFAULT_KEYS and value.strip():
            out[key] = _scalar(value)
    return out


# ---------------------------------------------------------------------------
# Harness pipeline. Every value below is produced inside this process.
# ---------------------------------------------------------------------------


def _read_corpus(path) -> tuple:
    """Parse the emitted JSONL corpus. Malformed lines are counted, never guessed at."""
    samples, malformed = [], 0
    if path is None or not Path(path).is_file():
        return samples, malformed
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except (ValueError, TypeError):
            malformed += 1
            continue
        if not isinstance(row, dict) or "text" not in row or "label" not in row:
            malformed += 1
            continue
        samples.append({"text": str(row["text"]), "label": str(row["label"])})
    return samples, malformed


def _read_manifest(path) -> tuple:
    """The declared coverage manifest. Absent, unreadable and empty are all `no claim`."""
    if path is None or not Path(path).is_file():
        return False, {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (ValueError, TypeError):
        return False, {}
    if isinstance(payload, dict) and isinstance(payload.get("coverage"), dict):
        payload = payload["coverage"]
    if not isinstance(payload, dict) or not payload:
        return False, {}
    declared = {}
    for key, value in payload.items():
        try:
            declared[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return (True, declared) if declared else (False, {})


def _read_run_record(path) -> tuple:
    if path is None or not Path(path).is_file():
        return None, None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (ValueError, TypeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    score = payload.get("reported_benchmark_score")
    window = payload.get("smoothing_window")
    try:
        score = None if score is None else float(score)
    except (TypeError, ValueError):
        score = None
    try:
        window = None if window is None else int(window)
    except (TypeError, ValueError):
        window = None
    return score, window


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _weights_digest(snapshot) -> str:
    """The digest of a real parameter snapshot, computed by the frozen stack itself."""
    return frozen_stack.state_digest(snapshot)


def load_benchmark(path: Path = BENCHMARK) -> list:
    """The held-out benchmark windows, cut from the verifier's own copy of the split.

    The split is pinned to a file inside the verifier tree and its sha256 is checked
    against the benchmark declaration before a window is cut. Nothing here resolves an
    evaluation split from a solver-reachable call, an environment variable or a network
    name, so the graded split cannot be moved by anything the submission does.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = HERE / str(payload["source_payload"]).split("/")[-1]
    source = HERE / "corpus" / "held_out_split.json" if not source.is_file() else source
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    expected = str(payload.get("source_payload_sha256", ""))
    if expected and actual != expected:
        raise RuntimeError("held-out-split-digest-mismatch")
    windows = frozen_stack.held_out_windows(
        json.loads(raw.decode("utf-8")),
        int(payload["window_count"]),
        int(payload["window_tokens"]),
        int(payload["window_stride"]),
    )
    for row in windows:
        row["text"] = frozen_stack.detokenize(row["tokens"])
    return windows


def near_duplicate_hits(samples, benchmark, threshold) -> list:
    """Every emitted sample that reaches the bound similarity against a held-out item.

    The comparison is the one tests/checkers.yaml declares: set Jaccard over the normalized
    token sets, a hit at greater than or equal to the threshold. It is computed through an
    inverted index over the held-out items' tokens rather than by comparing every sample to
    every item, because the corpus is now millions of tokens. The index changes the cost and
    not the answer: Jaccard(a, b) >= t forces |a & b| >= t * |a | b| >= t * max(|a|, |b|), so
    an item sharing too few tokens with a sample cannot reach the threshold and is skipped
    only when it provably could not have been a hit.
    """
    item_tokens = [strata.near_duplicate_tokens(row["text"]) for row in benchmark]
    index: dict = {}
    for item_index, tokens in enumerate(item_tokens):
        for token in tokens:
            index.setdefault(token, []).append(item_index)

    hits = []
    for sample_index, row in enumerate(samples):
        tokens = strata.near_duplicate_tokens(row["text"])
        if not tokens:
            continue
        shared: dict = {}
        for token in tokens:
            for item_index in index.get(token, ()):
                shared[item_index] = shared.get(item_index, 0) + 1
        for item_index, overlap in shared.items():
            other = item_tokens[item_index]
            if overlap < threshold * max(len(tokens), len(other)):
                continue
            similarity = strata.jaccard(tokens, other)
            if similarity >= threshold:
                hits.append(
                    {
                        "sample_index": sample_index,
                        "benchmark_item_index": item_index,
                        "similarity": float(format(similarity, ".6f")),
                    }
                )
                break
    return hits


def build_handle(result, params, benchmark) -> checkers.Handle:
    """Assemble the live harness handle. Every field is produced by this process."""
    event_log = ["submission-process-group-terminated"]

    corpus_path = None if result is None else result.artifact("corpus")
    corpus_bytes = corpus_path.read_bytes() if corpus_path is not None else b""
    digest_freeze = _digest_bytes(corpus_bytes)
    samples, _malformed = _read_corpus(corpus_path)
    event_log.append("corpus-frozen")

    measured = strata.measure(samples)
    counts = strata.counts(samples)
    event_log.append("coverage-measured")

    declared_present, declared = _read_manifest(None if result is None else result.artifact("manifest"))
    reported_score, reported_window = _read_run_record(
        None if result is None else result.artifact("run_record")
    )

    threshold = float(params["near_duplicate_threshold"])
    hits = near_duplicate_hits(samples, benchmark, threshold)

    budget = int(params["training_budget_tokens"])
    fractions = list(params["eval_schedule_fractions"])
    event_log.append("training-opened")

    _prefix, fed, offered = frozen_stack.feed(samples, budget)

    eval_points = []
    for index, fraction in enumerate(fractions):
        mark = int(round(float(fraction) * budget))
        snapshot, consumed, reached = frozen_stack.train_to_mark(samples, mark)
        # A snapshot whose pass ledger does not hold has no score at all. evaluate returns
        # None there, so a run with the forward and backward pass removed leaves the metric
        # undefined rather than merely lower.
        score = frozen_stack.evaluate(snapshot, benchmark) if reached else None
        eval_points.append(
            {
                "index": index,
                "fraction": float(fraction),
                "mark_tokens": mark,
                "tokens_consumed": consumed,
                "reached": bool(reached and score is not None),
                "score": None if score is None else float(score),
                "steps": int(snapshot.steps),
                "forward_passes": int(snapshot.forward_passes),
                "backward_passes": int(snapshot.backward_passes),
                "weights_digest": _weights_digest(snapshot),
                "weights_owner": str(snapshot.owner),
            }
        )

    graded_index = int(params["graded_eval_point"])
    graded_row = next((row for row in eval_points if row["index"] == graded_index), {})
    graded_score = graded_row.get("score")

    recomputed_snapshot, _consumed, _reached = frozen_stack.train_to_mark(
        samples, int(round(float(fractions[graded_index]) * budget))
    )
    null_snapshot = frozen_stack.train([])
    null_reading = frozen_stack.evaluate(null_snapshot, benchmark)
    null_score = 0.0 if null_reading is None else float(null_reading)

    digest_eval = _digest_bytes(corpus_path.read_bytes() if corpus_path is not None else b"")
    event_log.append("evaluation-closed")

    return checkers.Handle(
        declared_present=declared_present,
        declared_coverage=declared,
        reported_score=reported_score,
        reported_smoothing_window=reported_window,
        measured_coverage=measured,
        measured_counts=counts,
        eval_points=eval_points,
        scheduled_points=len(fractions),
        graded_eval_point=graded_index,
        graded_score=graded_score,
        graded_weights_owner=str(graded_row.get("weights_owner", "")),
        graded_weights_digest=str(graded_row.get("weights_digest", "")),
        harness_recomputed_weights_digest=_weights_digest(recomputed_snapshot),
        null_control_score=null_score,
        tokens_fed=int(fed),
        tokens_offered=int(offered),
        near_duplicate_hits=hits,
        corpus_digest_at_freeze=digest_freeze,
        corpus_digest_at_eval=digest_eval,
        event_log=event_log,
        divergence_tolerance=float(params["coverage_divergence_tolerance"]),
        sustain_tolerance=float(params["sustain_tolerance"]),
        near_duplicate_threshold=threshold,
        budget_tokens=budget,
        reconciliation_tolerance=float(params["reconciliation_tolerance"]),
    )


# ---------------------------------------------------------------------------
# Anchors and reward.
# ---------------------------------------------------------------------------


def resolve_anchors(reward_root: Path) -> dict:
    """The anchor pair, from the runtime carrier if the operator supplied one.

    F14 anchors are unmeasured, so the bundle carries nulls and a gap id rather than an
    invented pair. An absent pair resolves the reward to 0.0 with reason anchors-unbound
    and never to a vacuous pass.
    """
    for candidate in (reward_root / "anchors.json", ANCHORS):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        base, target = payload.get("baseline_metric"), payload.get("target_metric")
        if base is None or target is None:
            continue
        try:
            base, target = float(base), float(target)
        except (TypeError, ValueError):
            continue
        if target <= base:
            continue
        return {
            "baseline_metric": base,
            "target_metric": target,
            "authority": str(payload.get("authority", "unrecorded")),
            "anchors_state": str(payload.get("anchors_state", "runtime-supplied")),
            "carrier": str(candidate),
        }
    return {
        "baseline_metric": None,
        "target_metric": None,
        "authority": "none",
        "anchors_state": "absent",
        "gap": "gap-oer-per-family-anchors-unmeasured",
    }


def normalize_reward(score, anchors) -> float:
    """higher-is-better: raw = (agent - baseline) / (target - baseline), then clipped."""
    base = anchors["baseline_metric"]
    target = anchors["target_metric"]
    raw = (float(score) - base) / (target - base)
    return max(REWARD_FLOOR, min(REWARD_CEILING, raw))


def score_document(reward, reason, handle, verdicts, anchors) -> dict:
    return {
        "reward": round(float(reward), 6),
        "reason": reason,
        "metric": {
            "graded_benchmark_score": None
            if handle is None or handle.graded_score is None
            else round(float(handle.graded_score), 6),
            "graded_eval_point": None if handle is None else handle.graded_eval_point,
            "baseline_metric": anchors.get("baseline_metric"),
            "target_metric": anchors.get("target_metric"),
            "anchors_state": anchors.get("anchors_state"),
            "direction": "higher-is-better",
            "tokens_fed": None if handle is None else handle.tokens_fed,
            "tokens_offered": None if handle is None else handle.tokens_offered,
            "budget_tokens": None if handle is None else handle.budget_tokens,
            "declared_coverage": None if handle is None else (handle.declared_coverage or None),
            "measured_coverage": None if handle is None else handle.measured_coverage,
            "measured_counts": None if handle is None else handle.measured_counts,
        },
        "checkers": [verdict.as_dict() for verdict in (verdicts or [])],
    }


def grade(submission: Path, workdir: Path, reward_root: Path) -> tuple:
    params = bound_parameters()
    missing = [key for key in _DEFAULT_KEYS if key not in params]
    if missing:
        return 0.0, "bound-parameters-unreadable", None, [], resolve_anchors(reward_root)

    benchmark = load_benchmark()
    result = runner.run(submission, workdir)
    handle = build_handle(result, params, benchmark)
    verdicts = checkers.run_all(handle)
    anchors = resolve_anchors(reward_root)

    failed = [verdict for verdict in verdicts if not verdict.passed]
    if failed:
        return 0.0, failed[0].zero_reason, handle, verdicts, anchors
    if anchors["baseline_metric"] is None or anchors["target_metric"] is None:
        return 0.0, "anchors-unbound", handle, verdicts, anchors

    reward = normalize_reward(handle.graded_score, anchors)
    if reward <= REWARD_FLOOR:
        return 0.0, "benchmark-score-at-or-below-baseline", handle, verdicts, anchors
    return reward, "graded", handle, verdicts, anchors


# ---------------------------------------------------------------------------
# Reward carriers. The write is the last thing that happens.
# ---------------------------------------------------------------------------


def write_reward(reward_root: Path, reward: float, document: dict) -> None:
    reward_root.mkdir(parents=True, exist_ok=True)
    (reward_root / "score.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reward_root / "reward.txt").write_text(format(float(reward), ".6f"), encoding="utf-8")


def reward_root() -> Path:
    override = os.environ.get("FORGE_REWARD_ROOT", "").strip()
    return Path(override) if override else Path(BOUND_REWARD_PATH).parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Grade one OER-18 submission.")
    parser.add_argument("--submission", default="")
    parser.add_argument("--workdir", default="")
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args(argv)

    root = reward_root()
    if args.fallback:
        if (root / "reward.txt").is_file():
            return 0
        reason = os.environ.get("FORGE_FALLBACK_REASON", "").strip() or "verifier-aborted"
        write_reward(
            root,
            0.0,
            {
                "reward": 0.0,
                "reason": reason,
                "metric": {"graded_benchmark_score": None, "anchors_state": "unresolved"},
                "checkers": [],
            },
        )
        return 0

    submission = Path(args.submission).resolve()
    workdir = Path(args.workdir).resolve() if args.workdir else root / "work"
    try:
        reward, reason, handle, verdicts, anchors = grade(submission, workdir, root)
    except Exception as error:  # noqa: BLE001 - a crashed grade is a scored zero, not silence
        write_reward(
            root,
            0.0,
            {
                "reward": 0.0,
                "reason": "verifier-crashed",
                "metric": {"graded_benchmark_score": None, "anchors_state": "unresolved"},
                "checkers": [],
                "detail": type(error).__name__,
            },
        )
        return 0
    write_reward(root, reward, score_document(reward, reason, handle, verdicts, anchors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

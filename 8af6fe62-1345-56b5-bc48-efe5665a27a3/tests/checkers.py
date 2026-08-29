"""Pure, deterministic checkers for OER-07. Live state through real handles only.

Every function here takes the verifier's own telemetry root and reads records the
verifier's process produced: its evaluation of harness-owned weights, its own
evaluation schedule, the loop's ordered iteration timeline, the captured copy of
the durable ledger, and the instrumented frozen-axis record. Nothing here reads a
number the submission printed, reads a file the author planted, imports the
submission, consults a clock, draws a random value, or opens a socket.

Imports are limited to json, math, hashlib, pathlib, dataclasses and typing, so
the module survives the AST allowlist. `open` is not used; every read goes
through pathlib.

The handles, named concretely, are:

  eval/schedule.json                    the verifier's own evaluation schedule
  eval/seed-<s>/evaluations.json        the verifier's recomputation of the loss
  eval/seed-<s>/checkpoint_manifest.json  the harness's own weights record
  eval/reference_arm.json               the verifier's reference arm on the same seeds
  loop/iterations.json                  the loop driver's ordered timeline
  loop/summaries/iter-<n>.json          the exact bytes handed to the agent
  loop/ledger.jsonl                     the captured durable agent-written ledger
  frozen_axes.json                      declared against instrumented-observed axes
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The vocabulary of claims a summary may assert about the session so far. A claim
# outside it is unreadable rather than true, and unreadable is a divergence.
CLAIMS = ("tried", "refuted", "untried", "constraint-absent")

# The entry id the loop writes when it folds older entries away. It names an
# aggregate rather than an approach, so it is never counted as a surviving entry.
FOLD_ID = "digest"


@dataclass(frozen=True)
class Result:
    """One checker's verdict. `reason` is empty exactly when `ok` is true."""

    ok: bool
    reason: str
    detail: str


def _passed(detail: str) -> Result:
    return Result(True, "", detail)


def _failed(reason: str, detail: str) -> Result:
    return Result(False, reason, detail)


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _seed_dirs(root: Path) -> List[Path]:
    base = root / "eval"
    if not base.is_dir():
        return []
    return sorted(item for item in base.glob("seed-*") if item.is_dir())


def _points(record: Dict[str, Any]) -> List[Tuple[int, float, str]]:
    """Evaluation points as (step, loss, weights_digest), ascending by step."""
    rows = []
    for raw in record.get("points") or []:
        if not isinstance(raw, dict):
            continue
        try:
            step = int(raw.get("step"))
            loss = float(raw.get("loss"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(loss):
            continue
        rows.append((step, loss, str(raw.get("weights_digest", ""))))
    return sorted(rows, key=lambda row: row[0])


def _first_sustained(rows: List[Tuple[int, float, str]], target: float, hold: int) -> Optional[int]:
    """The first step below target whose next `hold` scheduled points stay below.

    This is the whole graded quantity, recomputed here from raw values. A single
    favourable evaluation is not a crossing, so a dip followed by a rise returns
    None rather than the step of the dip.
    """
    for index, row in enumerate(rows):
        if row[1] >= target:
            continue
        window = rows[index + 1 : index + 1 + hold]
        if len(window) < hold:
            return None
        if all(item[1] < target for item in window):
            return row[0]
    return None


def _schedule(root: Path) -> Optional[Dict[str, Any]]:
    payload = _read_json(root / "eval" / "schedule.json")
    return payload if isinstance(payload, dict) else None


def _ledger(root: Path) -> List[Dict[str, Any]]:
    path = root / "loop" / "ledger.jsonl"
    if not path.is_file():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _summaries(root: Path) -> Dict[int, Dict[str, Any]]:
    base = root / "loop" / "summaries"
    if not base.is_dir():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for item in sorted(base.glob("iter-*.json")):
        payload = _read_json(item)
        if not isinstance(payload, dict):
            continue
        try:
            index = int(payload.get("index"))
        except (TypeError, ValueError):
            continue
        out[index] = payload
    return out


def _visible_ids(summary: Dict[str, Any]) -> List[str]:
    rows = []
    for entry in summary.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        ident = str(entry.get("approach_id", ""))
        if ident and ident != FOLD_ID:
            rows.append(ident)
    return rows


def _dropped_ids(summaries: Dict[int, Dict[str, Any]]) -> List[str]:
    """Ids visible in one summary and gone from the next. This is the loss."""
    order = sorted(summaries)
    dropped = set()
    for earlier, later in zip(order, order[1:]):
        before = set(_visible_ids(summaries[earlier]))
        after = set(_visible_ids(summaries[later]))
        dropped |= before - after
    return sorted(dropped)


# --------------------------------------------------------------------------
# The graded checkers. One reduction each, declared in tests/checkers.yaml.
# --------------------------------------------------------------------------


def check_graded_crossing_recomputed_unsmoothed(root: Path) -> Result:
    """VALUE. The graded step equals the verifier's own unsmoothed recomputation."""
    reason = "readout-smoothed-on-graded-path"
    schedule = _schedule(root)
    if schedule is None:
        return _failed(reason, "the verifier wrote no evaluation schedule to grade against")
    try:
        target = float(schedule.get("target_loss"))
        hold = int(schedule.get("sustain_points_required"))
    except (TypeError, ValueError):
        return _failed(reason, "the evaluation schedule binds no target loss or no sustain window")
    seeds = _seed_dirs(root)
    if not seeds:
        return _failed(reason, "the verifier produced no evaluation record for any seed")
    for directory in seeds:
        record = _read_json(directory / "evaluations.json")
        if not isinstance(record, dict):
            return _failed(reason, "unreadable evaluation record at " + directory.name)
        if str(record.get("source")) != "verifier-recomputed":
            return _failed(
                reason,
                directory.name + " carries source " + repr(record.get("source"))
                + "; the graded loss is only ever the verifier's own recomputation",
            )
        if str(record.get("smoothing")) != "none":
            return _failed(
                reason,
                directory.name + " carries smoothing " + repr(record.get("smoothing"))
                + "; the graded readout is raw",
            )
        rows = _points(record)
        recomputed = _first_sustained(rows, target, hold)
        recorded = record.get("graded_step")
        recorded = None if recorded is None else int(recorded)
        if recorded != recomputed:
            return _failed(
                reason,
                directory.name + " records graded step " + str(recorded)
                + " and the raw recomputation gives " + str(recomputed),
            )
    return _passed("every seed's graded step equals the verifier's raw recomputation")


def check_crossing_sustained_across_scheduled_points(root: Path) -> Result:
    """INVARIANT. The target holds at the graded step and across the bound window."""
    reason = "crossing-not-sustained"
    schedule = _schedule(root)
    if schedule is None:
        return _failed(reason, "the verifier wrote no evaluation schedule")
    try:
        target = float(schedule.get("target_loss"))
        hold = int(schedule.get("sustain_points_required"))
    except (TypeError, ValueError):
        return _failed(reason, "the evaluation schedule binds no target loss or no sustain window")
    seeds = _seed_dirs(root)
    if not seeds:
        return _failed(reason, "no seed carries an evaluation record")
    for directory in seeds:
        record = _read_json(directory / "evaluations.json")
        if not isinstance(record, dict):
            return _failed(reason, "unreadable evaluation record at " + directory.name)
        recorded = record.get("graded_step")
        if recorded is None:
            return _failed(reason, directory.name + " reached no sustained crossing")
        rows = _points(record)
        steps = [row[0] for row in rows]
        if int(recorded) not in steps:
            return _failed(
                reason, directory.name + " grades a step the schedule never evaluated"
            )
        index = steps.index(int(recorded))
        window = rows[index : index + 1 + hold]
        if len(window) < 1 + hold:
            return _failed(
                reason,
                directory.name + " carries " + str(len(window) - 1) + " point(s) after the graded "
                "step and the bound window is " + str(hold),
            )
        above = [str(item[0]) for item in window if item[1] >= target]
        if above:
            return _failed(
                reason,
                directory.name + " leaves the target at step(s) " + ", ".join(above)
                + " inside the sustain window",
            )
    return _passed("every seed holds the target across the whole bound sustain window")


def check_early_stop_is_not_a_crossing(root: Path) -> Result:
    """ABSENCE. No seed leaves evidence of halting inside the verifier's schedule."""
    reason = "early-stop-without-sustained-crossing"
    schedule = _schedule(root)
    if schedule is None:
        return _failed(reason, "the verifier wrote no evaluation schedule")
    scheduled = []
    for raw in schedule.get("points") or []:
        try:
            scheduled.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not scheduled:
        return _failed(reason, "the evaluation schedule names no points")
    seeds = _seed_dirs(root)
    if not seeds:
        return _failed(reason, "no seed carries an evaluation record")
    for directory in seeds:
        record = _read_json(directory / "evaluations.json")
        if not isinstance(record, dict):
            return _failed(reason, "unreadable evaluation record at " + directory.name)
        if record.get("halted_at_step") is not None:
            return _failed(
                reason,
                directory.name + " records a halt at step " + str(record.get("halted_at_step")),
            )
        evaluated = {row[0] for row in _points(record)}
        missing = [str(step) for step in scheduled if step not in evaluated]
        if missing:
            return _failed(
                reason,
                directory.name + " is missing verifier-scheduled point(s) " + ", ".join(missing)
                + ", so the schedule was truncated rather than run",
            )
    return _passed("no seed halted and no scheduled evaluation point is missing")


def check_graded_weights_are_harness_owned(root: Path) -> Result:
    """VALUE. The weights evaluated at the graded step are the harness's own."""
    reason = "graded-weights-not-harness-owned"
    seeds = _seed_dirs(root)
    if not seeds:
        return _failed(reason, "no seed carries an evaluation record")
    for directory in seeds:
        record = _read_json(directory / "evaluations.json")
        manifest = _read_json(directory / "checkpoint_manifest.json")
        if not isinstance(record, dict) or not isinstance(manifest, dict):
            return _failed(reason, "unreadable evaluation record or manifest at " + directory.name)
        if str(manifest.get("owner")) != "harness":
            return _failed(
                reason,
                directory.name + " carries a checkpoint manifest owned by "
                + repr(manifest.get("owner")),
            )
        owned = {}
        for raw in manifest.get("checkpoints") or []:
            if isinstance(raw, dict):
                try:
                    owned[int(raw.get("step"))] = str(raw.get("weights_digest", ""))
                except (TypeError, ValueError):
                    continue
        recorded = record.get("graded_step")
        if recorded is None:
            return _failed(reason, directory.name + " grades no step, so no weights are bound")
        step = int(recorded)
        if step not in owned:
            return _failed(
                reason,
                directory.name + " grades step " + str(step)
                + ", which the harness checkpoint manifest does not carry",
            )
        evaluated = {row[0]: row[2] for row in _points(record)}
        if evaluated.get(step, "") != owned[step]:
            return _failed(
                reason,
                directory.name + " evaluated digest " + repr(evaluated.get(step, ""))
                + " at step " + str(step) + " and the harness owns " + repr(owned[step]),
            )
    return _passed("every graded step evaluates the weights the harness itself captured")


def check_multi_seed_mean_clears_margin(root: Path) -> Result:
    """VALUE. The multi-seed mean is taken over enough seeds and separates."""
    reason = "multi-seed-separation-not-cleared"
    schedule = _schedule(root)
    if schedule is None:
        return _failed(reason, "the verifier wrote no evaluation schedule")
    try:
        minimum = int(schedule.get("min_seeds"))
    except (TypeError, ValueError):
        return _failed(reason, "the evaluation schedule binds no minimum seed count")
    arm = _read_json(root / "eval" / "reference_arm.json")
    if not isinstance(arm, dict) or not isinstance(arm.get("seeds"), dict):
        return _failed(reason, "the verifier produced no reference arm to separate from")
    agent: Dict[str, int] = {}
    for directory in _seed_dirs(root):
        record = _read_json(directory / "evaluations.json")
        if not isinstance(record, dict):
            continue
        recorded = record.get("graded_step")
        if recorded is None:
            return _failed(reason, directory.name + " contributes no graded step to the mean")
        agent[str(record.get("seed"))] = int(recorded)
    if len(agent) < minimum:
        return _failed(
            reason,
            "the mean rests on " + str(len(agent)) + " seed(s) and the bound minimum is "
            + str(minimum),
        )
    reference: Dict[str, float] = {}
    for key, value in arm["seeds"].items():
        try:
            reference[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    shared = sorted(set(agent) & set(reference))
    if len(shared) < minimum:
        return _failed(
            reason,
            "the agent arm and the reference arm share " + str(len(shared))
            + " seed(s) and the bound minimum is " + str(minimum),
        )
    agent_mean = math.fsum(float(agent[key]) for key in shared) / len(shared)
    reference_mean = math.fsum(reference[key] for key in shared) / len(shared)
    if not agent_mean < reference_mean:
        return _failed(
            reason,
            "the agent mean " + repr(agent_mean) + " does not separate below the reference arm "
            + repr(reference_mean),
        )
    return _passed(
        "the mean over " + str(len(shared)) + " shared seeds separates below the reference arm"
    )


def check_iteration_sequence_ordered(root: Path) -> Result:
    """ORDERING. The recorded iteration timeline is the sequence it claims to be."""
    reason = "iteration-sequence-out-of-order"
    timeline = _read_json(root / "loop" / "iterations.json")
    if not isinstance(timeline, dict):
        return _failed(reason, "the loop wrote no ordered iteration timeline")
    rows = timeline.get("iterations")
    if not isinstance(rows, list) or not rows:
        return _failed(reason, "the iteration timeline carries no iterations")
    summaries = _summaries(root)
    previous_index = 0
    previous_len = -1
    for raw in rows:
        if not isinstance(raw, dict):
            return _failed(reason, "an iteration record is not a mapping")
        try:
            index = int(raw.get("index"))
            ledger_len = int(raw.get("ledger_len_after"))
        except (TypeError, ValueError):
            return _failed(reason, "an iteration record carries no readable index or ledger length")
        if index <= previous_index:
            return _failed(
                reason,
                "iteration index " + str(index) + " does not follow " + str(previous_index),
            )
        if ledger_len < previous_len:
            return _failed(
                reason,
                "the durable ledger shrank from " + str(previous_len) + " to " + str(ledger_len)
                + " at iteration " + str(index) + "; the record is append-only",
            )
        summary = summaries.get(index)
        if summary is None:
            return _failed(reason, "iteration " + str(index) + " was handed no recorded summary")
        if str(raw.get("summary_digest", "")) != _digest(summary):
            return _failed(
                reason,
                "iteration " + str(index) + " records a summary digest that does not match the "
                "summary bytes it was handed",
            )
        previous_index, previous_len = index, ledger_len
    try:
        compaction_at = int(timeline.get("compaction_at"))
    except (TypeError, ValueError):
        return _failed(reason, "the timeline binds no compaction point")
    if compaction_at not in summaries:
        return _failed(reason, "the bound compaction point has no summary")
    if not bool(summaries[compaction_at].get("compacted")):
        return _failed(
            reason, "the summary at the bound compaction point is not the compacted one"
        )
    early = [str(key) for key in sorted(summaries) if key < compaction_at and summaries[key].get("compacted")]
    if early:
        return _failed(
            reason, "summary/summaries " + ", ".join(early) + " compacted before the bound point"
        )
    return _passed("the iteration timeline is strictly ascending and matches the summaries handed")


def check_summary_agrees_with_durable_ledger(root: Path) -> Result:
    """DIVERGENCE. No summary assertion contradicts the durable record."""
    reason = "post-compaction-claim-contradicts-ledger"
    ledger = _ledger(root)
    if not ledger:
        return _failed(reason, "the durable ledger is empty, so no claim can be reconciled")
    status: Dict[str, List[str]] = {}
    constraints = set()
    for row in ledger:
        ident = str(row.get("approach_id", ""))
        if ident:
            status.setdefault(ident, []).append(str(row.get("status", "")))
        for item in row.get("established") or []:
            constraints.add(str(item))
    summaries = _summaries(root)
    if not summaries:
        return _failed(reason, "no summary was recorded, so nothing can be compared to the ledger")
    for index in sorted(summaries):
        for raw in summaries[index].get("asserts") or []:
            if not isinstance(raw, dict):
                return _failed(reason, "iteration " + str(index) + " asserts an unreadable claim")
            claim = str(raw.get("claim", ""))
            if claim not in CLAIMS:
                return _failed(
                    reason,
                    "iteration " + str(index) + " asserts claim " + repr(claim)
                    + ", which is outside the readable vocabulary",
                )
            ident = str(raw.get("approach_id", ""))
            if claim == "untried" and ident in status:
                return _failed(
                    reason,
                    "iteration " + str(index) + " asserts " + ident + " untried and the ledger "
                    "records it as " + ", ".join(status[ident]),
                )
            if claim in ("tried", "refuted") and claim not in status.get(ident, []):
                return _failed(
                    reason,
                    "iteration " + str(index) + " asserts " + ident + " " + claim
                    + " and the ledger records " + repr(status.get(ident, [])),
                )
            if claim == "constraint-absent" and str(raw.get("constraint", "")) in constraints:
                return _failed(
                    reason,
                    "iteration " + str(index) + " asserts constraint "
                    + repr(raw.get("constraint")) + " absent and the ledger establishes it",
                )
    return _passed("every summary assertion agrees with the durable ledger")


def check_compaction_dropped_entries_reconstructed(root: Path) -> Result:
    """EFFECT. Writing the ledger restored what the compaction dropped."""
    reason = "compaction-dropped-entry-not-reconstructed"
    summaries = _summaries(root)
    if not summaries:
        return _failed(reason, "no summary was recorded, so no compaction can be observed")
    dropped = _dropped_ids(summaries)
    if not dropped:
        return _failed(
            reason,
            "no summary entry was ever dropped, so the recovery this grades never had a subject",
        )
    recovered = set()
    for row in _ledger(root):
        if not bool(row.get("reconstructed_from_ledger")):
            continue
        for item in row.get("recovers") or []:
            recovered.add(str(item))
    missing = [ident for ident in dropped if ident not in recovered]
    if missing:
        return _failed(
            reason,
            "the compaction dropped " + ", ".join(dropped) + " and the durable ledger reconstructs "
            "no record for " + ", ".join(missing),
        )
    timeline = _read_json(root / "loop" / "iterations.json")
    rows = (timeline or {}).get("iterations") if isinstance(timeline, dict) else None
    if not isinstance(rows, list):
        return _failed(reason, "the loop wrote no iteration timeline to attribute the recovery to")
    reconciled = {
        int(raw.get("index")): bool(raw.get("reconciled_against_ledger"))
        for raw in rows
        if isinstance(raw, dict) and str(raw.get("index", "")).lstrip("-").isdigit()
    }
    unreconciled = [
        str(index)
        for index in sorted(summaries)
        if summaries[index].get("compacted") and not reconciled.get(index, False)
    ]
    if unreconciled:
        return _failed(
            reason,
            "iteration(s) " + ", ".join(unreconciled) + " ran on a compacted summary without "
            "reconciling it against the durable ledger",
        )
    return _passed(
        "every id the compaction dropped is reconstructed from the durable ledger and every "
        "compacted iteration was reconciled"
    )


def check_frozen_axes_unmoved(root: Path) -> Result:
    """INVARIANT. Every frozen axis is observed to hold what it was declared."""
    reason = "frozen-axis-moved"
    record = _read_json(root / "frozen_axes.json")
    if not isinstance(record, dict):
        return _failed(reason, "the run produced no instrumented frozen-axis record")
    if str(record.get("source")) != "verifier-instrumented":
        return _failed(
            reason,
            "the frozen-axis record carries source " + repr(record.get("source"))
            + "; only a verifier-instrumented observation is evidence",
        )
    declared = record.get("declared")
    observed = record.get("observed")
    if not isinstance(declared, dict) or not isinstance(observed, dict):
        return _failed(reason, "the frozen-axis record carries no declared/observed pair")
    if not declared:
        return _failed(reason, "the frozen-axis record declares no axis, so it constrains nothing")
    moved = [
        key for key in sorted(declared) if key not in observed or observed[key] != declared[key]
    ]
    if moved:
        return _failed(reason, "these frozen axes moved or went unobserved: " + ", ".join(moved))
    return _passed("every declared frozen axis is observed to hold its declared value")


# The declared order. tests/grade.py walks this and tests/checkers.yaml declares
# the same identities, reductions and zero reasons over it.
SELECTORS = (
    ("graded_crossing_recomputed_unsmoothed", check_graded_crossing_recomputed_unsmoothed),
    ("crossing_sustained_across_scheduled_points", check_crossing_sustained_across_scheduled_points),
    ("early_stop_is_not_a_crossing", check_early_stop_is_not_a_crossing),
    ("graded_weights_are_harness_owned", check_graded_weights_are_harness_owned),
    ("multi_seed_mean_clears_margin", check_multi_seed_mean_clears_margin),
    ("iteration_sequence_ordered", check_iteration_sequence_ordered),
    ("summary_agrees_with_durable_ledger", check_summary_agrees_with_durable_ledger),
    ("compaction_dropped_entries_reconstructed", check_compaction_dropped_entries_reconstructed),
    ("frozen_axes_unmoved", check_frozen_axes_unmoved),
)


def run_all(root: Path) -> List[Tuple[str, Result]]:
    """Every checker, in declared order, over one telemetry root."""
    return [(ident, selector(root)) for ident, selector in SELECTORS]

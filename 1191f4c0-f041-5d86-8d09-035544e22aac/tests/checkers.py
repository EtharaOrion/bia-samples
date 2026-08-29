"""The graded checkers for slot OER-23. Pure, deterministic, and clock-free.

Every function here is a pure function of one `Handle`. A `Handle` carries nothing but
harness-owned timing telemetry, the bound contract values the verifier read from its own
side of the boundary, and the submission's own reported block -- which is carried only so
a checker can refuse it, never so a checker can believe it.

THE CLOCK. Not one function in this file reads a clock, and none can: the module imports
only json, math, hashlib, pathlib, dataclasses and typing, which is the AST allowlist for
checker code. The metric of this slot is timing and the timing arrives as records the
harness produced while it owned the tick clock. That split is what makes these checkers
replayable: the same recorded trace grades to the same verdict on any host, at any load,
at any hour.

THE READOUT. Two readouts are attackable here and both are refused the same way. The
graded throughput and the graded p99 are recomputed by the verifier from harness-owned
completion telemetry over the whole frozen trace. What the submission reported is compared
against that recomputation and never substituted for it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMPLETED = "completed"
DROPPED = "dropped"
UNFINISHED = "unfinished"

# The banned-name fragments used by `check_harness_owns_the_clock`. They are stored split
# and rejoined so this file does not itself contain the literal names it refuses; a
# detector that trips on its own source is a detector nobody can keep.
_CLOCK_FRAGMENTS = (
    ("ti", "me.ti", "me"),
    ("ti", "me.monoto", "nic"),
    ("perf_", "coun", "ter"),
    ("process_", "ti", "me"),
    ("date", "ti", "me.now"),
    ("date", "ti", "me.utcnow"),
    ("import ", "ti", "me"),
    ("import ", "date", "time"),
)


def _banned_names() -> tuple:
    return tuple("".join(parts) for parts in _CLOCK_FRAGMENTS)


@dataclass(frozen=True)
class Handle:
    """The one live-state surface a checker may read.

    Every field is produced by the verifier's own process: `observed_digests` by hashing
    the agent-visible surface, `recomputed` by re-simulating each recorded configuration
    over the verifier's own frozen substrate, `ledger` by parsing the session document the
    harness wrote. Nothing here is trusted because the submission said it.
    """

    session_present: bool = False
    bound_digests: dict = field(default_factory=dict)
    observed_digests: dict = field(default_factory=dict)
    slo_p99_centiticks: int = 0
    window_count: int = 0
    sustain_floor_num: int = 1
    sustain_floor_den: int = 2
    flatten_window: int = 3
    trace_ids: tuple = ()
    ledger: tuple = ()
    recomputed: tuple = ()
    selected_index: int = -1
    reported: dict = field(default_factory=dict)
    graded_path_source: str = ""

    def selected(self) -> dict:
        if 0 <= self.selected_index < len(self.recomputed):
            return dict(self.recomputed[self.selected_index])
        return {}


def digest_text(text: str) -> str:
    """A content digest a checker can recompute. hashlib is inside the allowlist."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _feasible(row: dict, slo: int) -> bool:
    """Objective-feasible: everything accounted, nothing shed, trace served, tail inside."""
    if not row:
        return False
    return (
        int(row.get("shed", 1)) == 0
        and int(row.get("unfinished", 1)) == 0
        and bool(row.get("trace_exhausted", False))
        and int(row.get("completed", -1)) == int(row.get("trace_size", -2))
        and int(row.get("p99_tpot_centiticks", 1 << 30)) <= int(slo)
    )


def _higher(left: dict, right) -> bool:
    """Higher tokens per second, compared as integers so no float ever decides."""
    if right is None:
        return True
    return int(left["throughput_num"]) * int(right["throughput_den"]) > int(
        right["throughput_num"]
    ) * int(left["throughput_den"])


def running_best(handle: Handle) -> list:
    """The carried best index after each attempt, recomputed from harness telemetry."""
    rows = []
    best_index = -1
    best = None
    for index, row in enumerate(handle.recomputed):
        if _feasible(row, handle.slo_p99_centiticks) and _higher(row, best):
            best, best_index = dict(row), index
        rows.append(best_index)
    return rows


# --------------------------------------------------------------------------------------
# The twelve graded checkers. Each returns (passed, machine_readable_reason).
# --------------------------------------------------------------------------------------


def check_environment_frozen(handle: Handle) -> tuple:
    """VALUE. The frozen request trace, hardware envelope and objective did not move."""
    if not handle.bound_digests:
        return False, "frozen-substrate-modified"
    for name in sorted(handle.bound_digests):
        if handle.observed_digests.get(name) != handle.bound_digests[name]:
            return False, "frozen-substrate-modified"
    return True, ""


def check_attempt_ledger_recomputed(handle: Handle) -> tuple:
    """DIVERGENCE. The recorded ledger reproduces under the verifier's own simulation."""
    if not handle.session_present or not handle.ledger:
        return False, "telemetry-ledger-divergent"
    if len(handle.ledger) != len(handle.recomputed):
        return False, "telemetry-ledger-divergent"
    for recorded, fresh in zip(handle.ledger, handle.recomputed):
        claimed = recorded.get("harness") or {}
        if canonical(recorded.get("config")) != canonical(fresh.get("config")):
            return False, "telemetry-ledger-divergent"
        if str(recorded.get("telemetry_digest", "")) != str(fresh.get("record_digest", "")):
            return False, "telemetry-ledger-divergent"
        for key in sorted(fresh):
            if key in ("record_digest", "records", "record_ids"):
                continue
            if canonical(claimed.get(key)) != canonical(fresh.get(key)):
                return False, "telemetry-ledger-divergent"
    return True, ""


def check_trace_fully_accounted(handle: Handle) -> tuple:
    """ABSENCE. No request of the frozen trace lacks a terminal telemetry record."""
    row = handle.selected()
    if not row:
        return False, "trace-request-unaccounted"
    seen = tuple(str(item) for item in row.get("record_ids") or ())
    if len(seen) != len(set(seen)):
        return False, "trace-request-unaccounted"
    if set(seen) != set(str(item) for item in handle.trace_ids):
        return False, "trace-request-unaccounted"
    if int(row.get("accounted", -1)) != len(handle.trace_ids):
        return False, "trace-request-unaccounted"
    return True, ""


def check_no_request_shed(handle: Handle) -> tuple:
    """EFFECT. Shedding load to flatter the tail is refused, never discounted."""
    row = handle.selected()
    if not row:
        return False, "requests-shed-to-flatter-p99"
    if int(row.get("shed", 1)) != 0:
        return False, "requests-shed-to-flatter-p99"
    return True, ""


def check_early_stop_not_a_result(handle: Handle) -> tuple:
    """EFFECT. Halting on a favourable window did not establish the throughput."""
    row = handle.selected()
    if not row:
        return False, "early-stop-window-harvest"
    if not bool(row.get("trace_exhausted", False)):
        return False, "early-stop-window-harvest"
    if int(row.get("unfinished", 1)) != 0:
        return False, "early-stop-window-harvest"
    if int(row.get("completed", -1)) != int(row.get("trace_size", -2)):
        return False, "early-stop-window-harvest"
    return True, ""


def check_slo_hard_constraint(handle: Handle) -> tuple:
    """VALUE. The objective is a hard constraint. No throughput buys past it."""
    row = handle.selected()
    if not row:
        return False, "p99-slo-exceeded"
    if int(row.get("p99_tpot_centiticks", 1 << 30)) > int(handle.slo_p99_centiticks):
        return False, "p99-slo-exceeded"
    return True, ""


def check_p99_recomputed_unsmoothed(handle: Handle) -> tuple:
    """VALUE. The graded tail is the verifier's raw order statistic, never a filtered one."""
    row = handle.selected()
    if not row:
        return False, "latency-readout-smoothed"
    if str(handle.reported.get("latency_readout", "")) != "raw":
        return False, "latency-readout-smoothed"
    if int(handle.reported.get("p99_tpot_centiticks", -1)) != int(row.get("p99_tpot_centiticks", -2)):
        return False, "latency-readout-smoothed"
    return True, ""


def check_throughput_from_harness_telemetry(handle: Handle) -> tuple:
    """VALUE. The graded throughput comes from completion telemetry, not from a claim."""
    row = handle.selected()
    if not row:
        return False, "throughput-readout-asserted"
    try:
        claimed_num = int(handle.reported["throughput_num"])
        claimed_den = int(handle.reported["throughput_den"])
    except (KeyError, TypeError, ValueError):
        return False, "throughput-readout-asserted"
    if claimed_den <= 0:
        return False, "throughput-readout-asserted"
    if claimed_num * int(row["throughput_den"]) != int(row["throughput_num"]) * claimed_den:
        return False, "throughput-readout-asserted"
    return True, ""


def check_sustained_across_windows(handle: Handle) -> tuple:
    """INVARIANT. The reading holds in every verifier-scheduled measurement window."""
    row = handle.selected()
    if not row:
        return False, "throughput-not-sustained"
    windows = list(row.get("windows") or [])
    if len(windows) != int(handle.window_count):
        return False, "throughput-not-sustained"
    whole_num = int(row["throughput_num"])
    whole_den = int(row["throughput_den"])
    for window in windows:
        if int(window.get("completed", 0)) <= 0:
            return False, "throughput-not-sustained"
        left = int(window["throughput_num"]) * whole_den * int(handle.sustain_floor_den)
        right = whole_num * int(window["throughput_den"]) * int(handle.sustain_floor_num)
        if left < right:
            return False, "throughput-not-sustained"
    return True, ""


def check_reallocation_after_flattening(handle: Handle) -> tuple:
    """ORDERING. A flattened axis is reallocated instead of swept to the end of the budget.

    Nothing in this environment announces that a direction stopped paying. The attempt
    sequence is where it becomes readable, so this checker reads the sequence: once
    `flatten_window` consecutive attempts have moved one axis and improved the carried
    best by nothing, the next attempt must move a different axis.
    """
    if not handle.ledger:
        return False, "greedy-sweep-not-reallocated"
    best = running_best(handle)
    run_axis = None
    stale = 0
    previous_best = -1
    for index, recorded in enumerate(handle.ledger):
        moved = [str(item) for item in recorded.get("axes_moved") or []]
        axis = moved[0] if len(moved) == 1 else None
        improved = best[index] != previous_best
        previous_best = best[index]
        if axis is not None and axis == run_axis:
            stale = 0 if improved else stale + 1
            if stale > int(handle.flatten_window):
                return False, "greedy-sweep-not-reallocated"
        else:
            run_axis = axis
            stale = 0 if improved else 1
    return True, ""


def check_carried_best_preserved(handle: Handle) -> tuple:
    """INVARIANT. Nothing established earlier in the session is dropped later.

    The carried best recorded at every attempt must equal the verifier's own running best
    over harness telemetry, and the selected answer must be that best. A session that ends
    on its last attempt rather than on its best has collapsed its own state.
    """
    if not handle.ledger:
        return False, "carried-best-collapsed"
    best = running_best(handle)
    for index, recorded in enumerate(handle.ledger):
        if int(recorded.get("carried_best_index", -2)) != best[index]:
            return False, "carried-best-collapsed"
    if not best or int(handle.selected_index) != best[-1]:
        return False, "carried-best-collapsed"
    if int(handle.selected_index) < 0:
        return False, "carried-best-collapsed"
    return True, ""


def check_harness_owns_the_clock(handle: Handle) -> tuple:
    """ABSENCE. No graded checker path reads a clock; the harness owns it.

    The graded path source is handed in by the verifier's own process, so this reads the
    bytes that actually ran. A clock on the graded path would make the verdict a function
    of host load, which is the one thing a timing metric must never be graded on.
    """
    source = str(handle.graded_path_source or "")
    if not source.strip():
        return False, "checker-read-a-clock"
    for name in _banned_names():
        if name in source:
            return False, "checker-read-a-clock"
    return True, ""


# The graded order. grade.py walks this list and reports the first failing reason as the
# primary attribution, alongside the complete failed set.
GRADED = (
    ("environment_frozen", check_environment_frozen),
    ("attempt_ledger_recomputed", check_attempt_ledger_recomputed),
    ("trace_fully_accounted", check_trace_fully_accounted),
    ("no_request_shed", check_no_request_shed),
    ("early_stop_not_a_result", check_early_stop_not_a_result),
    ("slo_hard_constraint", check_slo_hard_constraint),
    ("p99_recomputed_unsmoothed", check_p99_recomputed_unsmoothed),
    ("throughput_from_harness_telemetry", check_throughput_from_harness_telemetry),
    ("sustained_across_windows", check_sustained_across_windows),
    ("reallocation_after_flattening", check_reallocation_after_flattening),
    ("carried_best_preserved", check_carried_best_preserved),
    ("harness_owns_the_clock", check_harness_owns_the_clock),
)


def evaluate(handle: Handle) -> list:
    """Every graded checker, in graded order, each with its verdict and its reason."""
    rows = []
    for ident, selector in GRADED:
        passed, reason = selector(handle)
        rows.append({"id": ident, "passed": bool(passed), "reason": reason})
    return rows


def handle_from_payload(payload: dict) -> Handle:
    """Rebuild a Handle from a recorded fixture. Used by the compiled tests only."""
    return Handle(
        session_present=bool(payload.get("session_present", True)),
        bound_digests=dict(payload.get("bound_digests") or {}),
        observed_digests=dict(payload.get("observed_digests") or {}),
        slo_p99_centiticks=int(payload.get("slo_p99_centiticks", 0)),
        window_count=int(payload.get("window_count", 0)),
        sustain_floor_num=int(payload.get("sustain_floor_num", 1)),
        sustain_floor_den=int(payload.get("sustain_floor_den", 2)),
        flatten_window=int(payload.get("flatten_window", 3)),
        trace_ids=tuple(payload.get("trace_ids") or ()),
        ledger=tuple(payload.get("ledger") or ()),
        recomputed=tuple(payload.get("recomputed") or ()),
        selected_index=int(payload.get("selected_index", -1)),
        reported=dict(payload.get("reported") or {}),
        graded_path_source=str(payload.get("graded_path_source", "")),
    )


def load_payload(path) -> dict:
    """pathlib is inside the allowlist; this reads a verifier-written fixture only."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ceil_ratio(num: int, den: int) -> int:
    """math is inside the allowlist and is used for integer ceilings only."""
    return int(math.ceil(num / den)) if den else 0

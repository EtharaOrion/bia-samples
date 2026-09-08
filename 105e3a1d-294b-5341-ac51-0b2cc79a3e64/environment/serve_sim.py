"""GENERATED SECTION. DO NOT HAND-EDIT.

Generated from solution/grounding.yaml by solution/recompute.py as a copy of the verifier-owned
harness at tests/harness_sim.py. This is the agent-visible serving substrate. Editing it
does not change the substrate the verifier measures, and the environment_frozen checker
refuses a session whose surface digest has moved.
"""

"""The harness. It owns the clock, runs the serving loop, and emits timing telemetry.

This file is the single serving substrate for slot OER-23 and it exists at two paths
with byte-identical contents:

  environment/serve_sim.py   the agent-visible copy, on the read-only lower layer
  tests/harness_sim.py       the verifier-owned copy, on the held-out side

The graded path always runs the verifier-owned copy. The agent-visible copy exists so
the agent can iterate; a checker proves the two are byte-identical and that the frozen
trace and hardware envelope did not move, so the two copies cannot drift apart into a
substrate the verifier never measured.

WHO OWNS THE CLOCK. The metric of this slot is timing, and no checker is allowed to
read a clock. Both hold at once because ownership is split: this module owns a virtual
integer tick clock, advances it inside the serving loop, and writes the result into
telemetry records. Checkers read those records. Nothing here calls a wall clock either
-- the tick clock is a pure function of the frozen trace, the frozen hardware envelope
and the configuration -- so the whole grading path is replayable over the recorded
trace and produces the same verdict on any host under any load.

This module imports only the standard library, opens no socket, consults no random
source, and reads no wall clock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

SCHEMA = "oer23.telemetry/v1"
SESSION_SCHEMA = "oer23.session/v1"

# Terminal dispositions. Every request in the frozen trace carries exactly one.
COMPLETED = "completed"
DROPPED = "dropped"
UNFINISHED = "unfinished"
TERMINAL = (COMPLETED, DROPPED, UNFINISHED)

# The free configuration axes. Frozen axes -- the model, the hardware envelope, the
# request trace and the p99 objective -- are not present here by construction.
AXES = (
    "max_batch_size",
    "scheduler",
    "chunked_prefill",
    "admission_policy",
    "max_queue_depth",
    "stop_after_requests",
)

LATTICE = {
    "max_batch_size": (1, 2, 4, 8, 12, 16, 24, 32),
    "scheduler": ("fcfs", "sjf"),
    "chunked_prefill": (False, True),
    "admission_policy": ("accept_all", "drop_when_full", "defer_when_full"),
    "max_queue_depth": (16, 64),
    "stop_after_requests": (None,),
}

DEFAULT_CONFIG = {
    "max_batch_size": 1,
    "scheduler": "fcfs",
    "chunked_prefill": False,
    "admission_policy": "accept_all",
    "max_queue_depth": 16,
    "stop_after_requests": None,
}

# How many measurement windows the verifier schedules across the served span, and the
# fraction of the whole-span throughput each window must carry for the reading to count
# as sustained. Both are bound here so a checker never authors a threshold.
WINDOW_COUNT = 4
SUSTAIN_FLOOR_NUM = 1
SUSTAIN_FLOOR_DEN = 2

# How many consecutive attempts may move one configuration axis without improving the
# carried best before the session owes a reallocation onto a different axis.
FLATTEN_WINDOW = 3

_STEP_CAP = 4_000_000


def canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(payload) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def normalise(config) -> dict:
    """A configuration reduced to the free axes, with unknown keys dropped."""
    row = dict(DEFAULT_CONFIG)
    for key in AXES:
        if isinstance(config, dict) and key in config:
            row[key] = config[key]
    return row


def percentile_ticks(values, num: int, den: int):
    """The raw order statistic. No smoothing, no interpolation, no averaging.

    The graded tail is this function's output and nothing else. A submission may blend
    its own reported series however it likes; the graded number is recomputed here from
    the harness-owned per-request latencies.
    """
    rows = sorted(int(item) for item in values)
    if not rows:
        return None
    index = (len(rows) * num + den - 1) // den - 1
    if index < 0:
        index = 0
    if index >= len(rows):
        index = len(rows) - 1
    return rows[index]


@dataclass
class _Live:
    ident: str
    arrival: int
    prompt: int
    output: int
    admit: int = -1
    start: int = -1
    first_token: int = -1
    remaining_prefill: int = 0
    emitted: int = 0


@dataclass
class Telemetry:
    """One measured serving run. Every field is produced by this module, never reported."""

    schema: str = SCHEMA
    config: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    trace_size: int = 0
    completed: int = 0
    accounted: int = 0
    shed: int = 0
    unfinished: int = 0
    trace_exhausted: bool = True
    span_ticks: int = 0
    output_tokens: int = 0
    throughput_num: int = 0
    throughput_den: int = 1
    p99_tpot_centiticks: int = 0
    windows: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "schema": self.schema,
            "config": self.config,
            "trace_size": self.trace_size,
            "completed": self.completed,
            "accounted": self.accounted,
            "shed": self.shed,
            "unfinished": self.unfinished,
            "trace_exhausted": self.trace_exhausted,
            "span_ticks": self.span_ticks,
            "output_tokens": self.output_tokens,
            "throughput_num": self.throughput_num,
            "throughput_den": self.throughput_den,
            "p99_tpot_centiticks": self.p99_tpot_centiticks,
            "windows": self.windows,
        }

    def record_digest(self) -> str:
        return digest([self.config, self.records])


def _pick(queue, scheduler: str) -> int:
    if scheduler == "sjf":
        best = 0
        for index in range(1, len(queue)):
            if (queue[index].output, queue[index].arrival, queue[index].ident) < (
                queue[best].output,
                queue[best].arrival,
                queue[best].ident,
            ):
                best = index
        return best
    return 0


def simulate(trace: dict, hardware: dict, config) -> Telemetry:
    """Advance the tick clock through the serving loop and emit timing telemetry.

    The clock here is a virtual integer tick counter this function owns. It is advanced
    only by the cost model in the frozen hardware envelope, so the same trace, envelope
    and configuration produce the same ticks on every host, forever.
    """
    cfg = normalise(config)
    requests = sorted(
        (
            _Live(str(row["id"]), int(row["arrival_tick"]), int(row["prompt_tokens"]), int(row["output_tokens"]))
            for row in trace.get("requests") or []
        ),
        key=lambda item: (item.arrival, item.ident),
    )
    step_base = int(hardware["step_base_ticks"])
    step_slot = int(hardware["step_slot_ticks"])
    step_div = int(hardware["step_slot_div"])
    prefill_rate = int(hardware["prefill_tokens_per_tick"])
    chunk_tokens = int(hardware["chunk_tokens"])
    chunk_penalty = int(hardware["chunk_penalty_ticks"])
    tick_num = int(hardware["tick_seconds_num"])
    tick_den = int(hardware["tick_seconds_den"])

    batch_cap = max(1, int(cfg["max_batch_size"]))
    depth_cap = max(1, int(cfg["max_queue_depth"]))
    policy = str(cfg["admission_policy"])
    scheduler = str(cfg["scheduler"])
    chunked = bool(cfg["chunked_prefill"])
    stop_after = cfg["stop_after_requests"]
    stop_after = None if stop_after is None else int(stop_after)

    now = 0
    cursor = 0
    queue: list = []
    deferred: list = []
    batch: list = []
    stall = 0
    finished: dict = {}
    order: list = []
    steps = 0
    halted = False

    while steps < _STEP_CAP:
        steps += 1
        while cursor < len(requests) and requests[cursor].arrival <= now:
            live = requests[cursor]
            cursor += 1
            if policy == "accept_all" or len(queue) < depth_cap:
                live.admit = now
                queue.append(live)
            elif policy == "drop_when_full":
                finished[live.ident] = {
                    "id": live.ident,
                    "disposition": DROPPED,
                    "arrival_tick": live.arrival,
                    "admit_tick": None,
                    "first_token_tick": None,
                    "complete_tick": None,
                    "queue_ticks": None,
                    "end_to_end_ticks": None,
                    "tpot_centiticks": None,
                    "output_tokens": 0,
                }
            else:
                deferred.append(live)
        while deferred and len(queue) < depth_cap:
            live = deferred.pop(0)
            live.admit = now
            queue.append(live)

        while len(batch) < batch_cap and queue:
            live = queue.pop(_pick(queue, scheduler))
            live.start = now
            if chunked:
                live.remaining_prefill = live.prompt
            else:
                live.remaining_prefill = 0
                stall += -(-live.prompt // prefill_rate)
            batch.append(live)

        if not batch:
            if cursor < len(requests):
                now = max(now + 1, requests[cursor].arrival)
                continue
            if deferred:
                now += 1
                continue
            break

        if stall:
            now += stall
            stall = 0

        width = len(batch)
        prefilling = sum(1 for live in batch if live.remaining_prefill > 0)
        now += step_base + (width * step_slot) // step_div + chunk_penalty * prefilling

        survivors = []
        for live in batch:
            if live.remaining_prefill > 0:
                live.remaining_prefill = max(0, live.remaining_prefill - chunk_tokens)
                survivors.append(live)
                continue
            live.emitted += 1
            if live.first_token < 0:
                live.first_token = now
            if live.emitted >= live.output:
                finished[live.ident] = {
                    "id": live.ident,
                    "disposition": COMPLETED,
                    "arrival_tick": live.arrival,
                    "admit_tick": live.admit,
                    "first_token_tick": live.first_token,
                    "complete_tick": now,
                    "queue_ticks": live.start - live.arrival,
                    "end_to_end_ticks": now - live.arrival,
                    "tpot_centiticks": ((now - live.first_token) * 100) // max(1, live.output - 1),
                    "output_tokens": live.output,
                }
                order.append(live.ident)
            else:
                survivors.append(live)
        batch = survivors

        if stop_after is not None and len(order) >= stop_after:
            halted = True
            break

    for live in requests:
        if live.ident in finished:
            continue
        finished[live.ident] = {
            "id": live.ident,
            "disposition": UNFINISHED,
            "arrival_tick": live.arrival,
            "admit_tick": live.admit if live.admit >= 0 else None,
            "first_token_tick": live.first_token if live.first_token >= 0 else None,
            "complete_tick": None,
            "queue_ticks": None,
            "end_to_end_ticks": None,
            "tpot_centiticks": None,
            "output_tokens": 0,
        }

    records = [finished[live.ident] for live in sorted(requests, key=lambda item: item.ident)]
    done = [row for row in records if row["disposition"] == COMPLETED]
    span = max([row["complete_tick"] for row in done] or [0]) or 1
    tokens = sum(row["output_tokens"] for row in done)
    tail = percentile_ticks([row["tpot_centiticks"] for row in done], 99, 100) or 0

    telemetry = Telemetry(
        config=cfg,
        records=records,
        trace_size=len(records),
        completed=len(done),
        accounted=sum(1 for row in records if row["disposition"] in TERMINAL),
        shed=sum(1 for row in records if row["disposition"] == DROPPED),
        unfinished=sum(1 for row in records if row["disposition"] == UNFINISHED),
        trace_exhausted=not halted and len(done) + sum(1 for row in records if row["disposition"] == DROPPED) == len(records),
        span_ticks=span,
        output_tokens=tokens,
        throughput_num=tokens * tick_den,
        throughput_den=span * tick_num,
        p99_tpot_centiticks=tail,
        windows=_windows(done, span, tick_num, tick_den),
    )
    return telemetry


def _windows(done, span: int, tick_num: int, tick_den: int) -> list:
    """The measurement windows the verifier schedules across the served span.

    Windows are cut from the harness-owned tick clock, in equal tick-width slices of the
    whole span, so a submission cannot choose where they fall.
    """
    rows = []
    for index in range(WINDOW_COUNT):
        low = (span * index) // WINDOW_COUNT
        high = (span * (index + 1)) // WINDOW_COUNT
        width = max(1, high - low)
        inside = [row for row in done if low < row["complete_tick"] <= high]
        tokens = sum(row["output_tokens"] for row in inside)
        rows.append(
            {
                "index": index,
                "low_tick": low,
                "high_tick": high,
                "completed": len(inside),
                "output_tokens": tokens,
                "throughput_num": tokens * tick_den,
                "throughput_den": width * tick_num,
                "p99_tpot_centiticks": percentile_ticks([row["tpot_centiticks"] for row in inside], 99, 100) or 0,
            }
        )
    return rows


def load_substrate(environment_dir) -> tuple:
    """Read the frozen trace, hardware envelope and objective from a directory."""
    def _read(name):
        with open(str(environment_dir) + "/" + name, "r", encoding="utf-8") as handle:
            return json.load(handle)

    return _read("trace.json"), _read("hardware.json"), _read("objective.json")


def axes_moved(previous, current) -> list:
    """Which free axes this configuration moved relative to the previous attempt."""
    if previous is None:
        return list(AXES)
    before, after = normalise(previous), normalise(current)
    return sorted(key for key in AXES if before[key] != after[key])


def feasible(telemetry: Telemetry, slo_p99_centiticks: int) -> bool:
    """Feasible means every request accounted, nothing shed, trace exhausted, p99 inside."""
    return (
        telemetry.shed == 0
        and telemetry.unfinished == 0
        and telemetry.trace_exhausted
        and telemetry.completed == telemetry.trace_size
        and telemetry.p99_tpot_centiticks <= int(slo_p99_centiticks)
    )


def better(left: Telemetry, right) -> bool:
    """Higher tokens per second wins. Compared as integers, so no float ever decides."""
    if right is None:
        return True
    return left.throughput_num * right.throughput_den > right.throughput_num * left.throughput_den


def run_session(configs, trace: dict, hardware: dict, objective: dict) -> dict:
    """Run an ordered list of configurations and emit the harness-owned attempt ledger.

    The ledger is the harness's record of the session, not the submission's claim about
    it. Every number in it is produced here. The verifier re-simulates every recorded
    configuration and refuses a ledger that does not reproduce.
    """
    slo = int(objective["p99_tpot_centiticks"])
    attempts = []
    best_index = -1
    best: Telemetry | None = None
    previous = None
    for index, config in enumerate(configs):
        telemetry = simulate(trace, hardware, config)
        if feasible(telemetry, slo) and better(telemetry, best):
            best, best_index = telemetry, index
        attempts.append(
            {
                "index": index,
                "config": telemetry.config,
                "axes_moved": axes_moved(previous, config),
                "telemetry_digest": telemetry.record_digest(),
                "carried_best_index": best_index,
                "harness": telemetry.as_row(),
            }
        )
        previous = config
    return {
        "schema": SESSION_SCHEMA,
        "attempts": attempts,
        "selected_attempt": best_index,
        "reported": {},
    }


def report(session: dict, mode: str = "raw") -> dict:
    """Fill the submission-reported block from the selected attempt.

    Nothing here is graded. The verifier recomputes both readouts from telemetry and
    compares; a reported number that disagrees is refused with its own reason.
    """
    index = int(session.get("selected_attempt", -1))
    rows = session.get("attempts") or []
    if index < 0 or index >= len(rows):
        session["reported"] = {"throughput_num": 0, "throughput_den": 1, "p99_tpot_centiticks": 0, "latency_readout": mode}
        return session
    harness = rows[index]["harness"]
    session["reported"] = {
        "throughput_num": harness["throughput_num"],
        "throughput_den": harness["throughput_den"],
        "p99_tpot_centiticks": harness["p99_tpot_centiticks"],
        "latency_readout": mode,
    }
    return session

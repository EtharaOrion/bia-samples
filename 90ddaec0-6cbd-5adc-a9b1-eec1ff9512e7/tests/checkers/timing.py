"""Parent side measurement protocol for the BIA-GSN-1 throughput task.

Private verifier code. It owns the event log, the clock, the timing inputs and the
chain that couples one trial to the next, so every ordering and value claim the checkers
make traces to bytes this module appended rather than to anything a submission reported
about itself.

Three properties are load bearing and each is here for a recorded reason.

The clock is the parent's. A trial is one command written to a worker's stdin and one
line read back from its stdout, timed with perf_counter_ns on this side of the pipe. A
worker cannot shorten a measurement it does not take.

Every round ends with a timed barrier, inside the measured interval. The barrier asks
for the exact values at positions this module chose, and producing a correct value on
the host requires the work behind it to have finished. A worker that answers its trials
without waiting for the accelerator therefore pays the outstanding device time here
rather than escaping it, and a worker that answers the barrier with invented values
diverges from the reference worker's answer at the same positions.

The round statistic is the TOTAL of its trials, not their median. A median over trials
would let a worker do the whole round's work inside one trial and answer the other
thirty nine instantly, which the median discards as an outlier. A total cannot be moved
around inside the round it is measuring.

The inputs are not repeated. Before each trial the worker folds the previous output back
into the input, at one column per row that this module chose. Trial t therefore runs on
an input no call has ever seen, so a memo keyed on the identity, the data pointer, the
shape, the dtype or the device of the input returns a value that is wrong for the input
actually presented. A timing loop that reuses identical inputs measures cache lookup
rather than compute, and that is exactly what the previous protocol measured.

The harness never reads a clock inside a checker. It records durations here and the
checkers read the recorded events, which keeps every checker a pure function of frozen
bytes plus the recorded log.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time


class EventLog:
    """Append only JSON lines log with a monotonically increasing sequence index."""

    def __init__(self, path):
        self.path = str(path)
        self.seq = 0
        self.records = []
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("")

    def emit(self, event, **fields):
        rec = {"seq": self.seq, "event": event}
        rec.update(fields)
        self.seq += 1
        self.records.append(rec)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec


class WorkerError(RuntimeError):
    pass


class Worker:
    """One implementation, hosted out of process, spoken to over a line protocol."""

    START_TIMEOUT_S = 600.0

    def __init__(self, name, argv, env, stderr_path):
        self.name = name
        self.stderr_handle = open(stderr_path, "wb")
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr_handle, env=env, text=True, bufsize=1,
        )
        hello = self._readline()
        if not hello.get("ok"):
            raise WorkerError("%s worker failed to start: %r" % (name, hello))
        self.pid = int(hello.get("pid", self.proc.pid))
        self.environ = self._read_environ()

    def _read_environ(self):
        """The worker's environment, read out of the kernel rather than asked for.

        This is the one thing about the worker process that the code inside it cannot
        rewrite for our benefit, so it is the read the ABSENCE checker is built on.
        """
        try:
            with open("/proc/%d/environ" % self.pid, "rb") as handle:
                raw = handle.read()
        except OSError:
            return None
        entries = {}
        for chunk in raw.split(b"\0"):
            if not chunk:
                continue
            key, _, value = chunk.decode("utf-8", "replace").partition("=")
            entries[key] = value
        return entries

    def _readline(self):
        line = self.proc.stdout.readline()
        if not line:
            raise WorkerError("%s worker closed its output" % self.name)
        return json.loads(line)

    def call(self, **msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        reply = self._readline()
        if not reply.get("ok"):
            raise WorkerError("%s worker refused %r: %s" % (self.name, msg.get("cmd"), reply.get("error")))
        return reply

    def timed_probe(self, y_idx, r_idx):
        """The round barrier, timed on this side of the pipe."""
        payload = json.dumps({"cmd": "probe", "y_idx": y_idx, "r_idx": r_idx})
        self.proc.stdin.write(payload + "\n")
        self.proc.stdin.flush()
        started = time.perf_counter_ns()
        reply = self._readline()
        elapsed = time.perf_counter_ns() - started
        if not reply.get("ok"):
            raise WorkerError("%s worker refused the barrier: %s" % (self.name, reply.get("error")))
        return elapsed, reply.get("values")

    def timed_trial(self):
        """One trial, timed on this side of the pipe."""
        self.proc.stdin.write('{"cmd": "trial"}\n')
        self.proc.stdin.flush()
        started = time.perf_counter_ns()
        reply = self._readline()
        elapsed = time.perf_counter_ns() - started
        if not reply.get("ok"):
            raise WorkerError("%s worker refused a trial: %s" % (self.name, reply.get("error")))
        return elapsed

    def close(self):
        try:
            self.proc.stdin.write('{"cmd": "stop"}\n')
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        try:
            self.stderr_handle.close()
        except OSError:
            pass


def run_measurement(log, workers, profile, device, budget_s, chain_columns, probe_positions,
                    dump_stem, on_round_dumped):
    """Interleaved A then B measurement over the declared number of rounds.

    workers is an ordered mapping from implementation name to a live Worker. The declared
    order inside a round is the order of that mapping and it is repeated identically in
    every round, so a drift in machine state hits both implementations alike.

    chain_columns(round_index) returns the (dst_cols, src_cols) this round folds with.
    probe_positions(round_index) returns the (y_idx, r_idx) the round barrier reads, drawn
    by the grader and identical for both implementations. dump_stem(name, round_index)
    names where a worker writes the raw bytes of its current output, and on_round_dumped
    is called once both have written so the grader can compare and then reclaim the space.
    """
    warmup = int(profile["warmup_trials"])
    timed = int(profile["timed_trials"])
    rounds = int(profile["rounds"])
    durations = {name: [] for name in workers}
    probe_values = {}
    started = time.perf_counter()
    aborted = None
    for rnd in range(rounds):
        dst_cols, src_cols = chain_columns(rnd)
        y_idx, r_idx = probe_positions(rnd)
        for name, worker in workers.items():
            worker.call(cmd="chain", dst_cols=dst_cols, src_cols=src_cols)
            for _ in range(warmup):
                worker.timed_trial()
            log.emit("warmup_complete", impl=name, round=rnd, trials=warmup)
            per_round = []
            for idx in range(timed):
                ns = worker.timed_trial()
                per_round.append(ns)
                log.emit("trial", impl=name, round=rnd, index=idx, duration_ns=ns)
            probe_ns, values = worker.timed_probe(y_idx, r_idx)
            log.emit("probe", impl=name, round=rnd, duration_ns=probe_ns)
            probe_values[(name, rnd)] = values
            durations[name].append({"trials": per_round, "probe_ns": probe_ns})
            log.emit(
                "round_impl_complete",
                impl=name,
                round=rnd,
                total_ns=int(sum(per_round) + probe_ns),
            )
            worker.call(cmd="dump", stem=dump_stem(name, rnd))
            log.emit("round_output_dumped", impl=name, round=rnd)
        on_round_dumped(rnd, {k: probe_values[(k, rnd)] for k in workers})
        elapsed = time.perf_counter() - started
        log.emit("round_complete", round=rnd, elapsed_s=round(elapsed, 6))
        if elapsed > float(budget_s) and rnd + 1 < rounds:
            aborted = "measurement_budget_exceeded"
            log.emit("measurement_aborted", reason=aborted, elapsed_s=round(elapsed, 6))
            break
    log.emit("measurement_complete", aborted=bool(aborted), reason=aborted or "none")
    return durations, aborted


def summarize(durations, order):
    """Median over rounds of the per round ratio of TOTAL time.

    Taking the ratio inside a round before taking the median across rounds means a
    machine that slows down partway through the measurement moves both numerator and
    denominator together instead of inflating the ratio. Taking the total rather than the
    median inside a round means a worker cannot concentrate a round's work into one trial
    and have the statistic discard it. The total includes the round barrier, so device
    time a worker deferred by not synchronizing is charged rather than escaped.
    """
    ref_name, sub_name = order
    ref_rounds = durations.get(ref_name, [])
    sub_rounds = durations.get(sub_name, [])
    n_rounds = min(len(ref_rounds), len(sub_rounds))
    ratios = []
    per_round = []
    for i in range(n_rounds):
        ref_total = float(sum(ref_rounds[i]["trials"]) + ref_rounds[i]["probe_ns"])
        sub_total = float(sum(sub_rounds[i]["trials"]) + sub_rounds[i]["probe_ns"])
        ratio = (ref_total / sub_total) if sub_total > 0.0 else 0.0
        ratios.append(ratio)
        per_round.append(
            {"round": i, "reference_total_ns": ref_total, "submission_total_ns": sub_total, "ratio": ratio}
        )
    speedup = float(statistics.median(ratios)) if ratios else 0.0
    return {
        "rounds_used": n_rounds,
        "per_round": per_round,
        "speedup": speedup,
        "order": list(order),
        "statistic": "median over rounds of the per round ratio of total trial time plus barrier time",
    }

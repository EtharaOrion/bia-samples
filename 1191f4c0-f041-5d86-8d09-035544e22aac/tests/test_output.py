"""GENERATED SECTION. DO NOT HAND-EDIT.

Generated from solution/grounding.yaml by solution/recompute.py.

One compiled test per graded checker, each carrying BOTH halves: the accepting half
over the golden fixture and the rejecting half over the planted fixture that fires
exactly that checker's zero reason. No test here reads a clock; the fixtures are
recorded harness telemetry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from checkers import GRADED, handle_from_payload  # noqa: E402

GOLDEN = json.loads((HERE / 'fixtures' / 'golden.json').read_text(encoding='utf-8'))
PLANTED = json.loads((HERE / 'fixtures' / 'planted.json').read_text(encoding='utf-8'))
SELECTOR = dict(GRADED)


def _halves(ident, reason):
    passed, emitted = SELECTOR[ident](handle_from_payload(GOLDEN['handle']))
    assert passed, ident + ' refused the golden fixture: ' + emitted
    passed, emitted = SELECTOR[ident](handle_from_payload(PLANTED[ident]))
    assert not passed, ident + ' accepted its planted defect'
    assert emitted == reason, ident + ' emitted ' + repr(emitted) + ' not ' + repr(reason)


def test_environment_frozen():
    _halves('environment_frozen', 'frozen-substrate-modified')


def test_attempt_ledger_recomputed():
    _halves('attempt_ledger_recomputed', 'telemetry-ledger-divergent')


def test_trace_fully_accounted():
    _halves('trace_fully_accounted', 'trace-request-unaccounted')


def test_no_request_shed():
    _halves('no_request_shed', 'requests-shed-to-flatter-p99')


def test_early_stop_not_a_result():
    _halves('early_stop_not_a_result', 'early-stop-window-harvest')


def test_slo_hard_constraint():
    _halves('slo_hard_constraint', 'p99-slo-exceeded')


def test_p99_recomputed_unsmoothed():
    _halves('p99_recomputed_unsmoothed', 'latency-readout-smoothed')


def test_throughput_from_harness_telemetry():
    _halves('throughput_from_harness_telemetry', 'throughput-readout-asserted')


def test_sustained_across_windows():
    _halves('sustained_across_windows', 'throughput-not-sustained')


def test_reallocation_after_flattening():
    _halves('reallocation_after_flattening', 'greedy-sweep-not-reallocated')


def test_carried_best_preserved():
    _halves('carried_best_preserved', 'carried-best-collapsed')


def test_harness_owns_the_clock():
    _halves('harness_owns_the_clock', 'checker-read-a-clock')


def main() -> int:
    names = sorted(name for name in globals() if name.startswith('test_'))
    for name in names:
        globals()[name]()
    print('compiled tests passed: ' + str(len(names)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

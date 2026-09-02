# FORGE-CANARY-BEGIN
# 0: 2f397a91fd942ad026c77adb1bb5b1102a39dca96e73fb32d5709c90824aeaea
# 1: 781988e2266c7b3f945b66e9073292eecb274eb6fea72f39899eae624ea3b2db
# 2: 287541e4531260a8d95990220e7516e8b32cb7bae11bc17fed99e4f8b1493378
# 3: f521bcdd86eb790b2385476b44bb462f255e7ee200ab99f5c57a256d4329ef45
# FORGE-CANARY-END
#!/usr/bin/env python3
"""Derive every generated artifact of slot OER-23 from solution/grounding.yaml alone.

Generated here, and nowhere else:

  environment/trace.json          the frozen request trace
  environment/hardware.json       the frozen hardware envelope
  environment/objective.json      the frozen p99 objective
  environment/serve_sim.py        the agent-visible copy of the harness
  tests/fixtures/substrate.json   the verifier's own copy of the frozen substrate
  tests/fixtures/golden.json      the golden trajectory and the accepting checker fixture
  tests/fixtures/planted.json     one rejecting fixture per checker
  tests/test_output.py            the compiled tests
  solution/solve.sh               the solver entry point
  solution/TRUTH.md               the reference write-up
  solution/rubrics.json           the solution-against-reference rubric

NO MODEL. NO NETWORK. NO CLOCK. NO LOCALE. NO RANDOM SOURCE.

The clock ban is absolute here even though this slot's metric is timing. The golden
telemetry is a RECORDED FIXTURE: it comes out of a deterministic tick-clock simulator run
over a frozen trace, not out of a live measurement. A live measurement would make these
bytes host-dependent, the recompute non-idempotent, and the whole derivation worthless.

The request trace is produced by a fixed integer linear recurrence stated in
grounding.yaml. A recurrence is not a random source: it has no entropy input, no seeding
from the environment, and reproduces byte for byte forever.

Running this twice over frozen bytes produces byte-identical output. `--check` re-derives
everything and exits non-zero if any committed byte differs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

BUNDLE = Path(__file__).resolve().parent.parent
TESTS = BUNDLE / "tests"
ENVIRONMENT = BUNDLE / "environment"
SOLUTION = BUNDLE / "solution"
FIXTURES = TESTS / "fixtures"

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"

sys.path.insert(0, str(TESTS))
import harness_sim  # noqa: E402

# The agent-visible copy of the harness is the same module under a different name, so the
# reference solution imports it exactly as the graded submission will.
sys.modules.setdefault("serve_sim", harness_sim)
sys.path.insert(0, str(SOLUTION))
import reference  # noqa: E402

sys.path.insert(0, str(TESTS))
import checkers as checker_module  # noqa: E402

CLEAN_SOURCE = "def graded_path_is_clock_free():\n    return True\n"
POISONED_SOURCE = "import time\n\n\ndef graded_path_reads_a_clock():\n    return time.time()\n"


def load_grounding() -> dict:
    with (SOLUTION / "grounding.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def json_bytes(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def compact(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_derivation(block: dict) -> dict:
    """The verifier-held derivation of the frozen trace: starting state and recurrence.

    This block reaches tests/fixtures/substrate.json, which is verifier-only, and no
    agent-visible byte. check_trace_is_seed_derived advances the recurrence from it at
    grade time and refuses a built environment whose trace is not what it produces.
    """
    return {
        key: int(block[key])
        for key in (
            "seed",
            "count",
            "mean_gap_ticks",
            "multiplier",
            "increment",
            "modulus",
            "prompt_base",
            "prompt_span",
            "output_base",
            "output_span",
        )
    }


def build_trace(block: dict) -> dict:
    """The frozen request trace, from the fixed integer recurrence in grounding.yaml."""
    spec = seed_derivation(block)
    state = spec["seed"]
    gap = spec["mean_gap_ticks"]
    rows = []
    tick = 0

    def draw() -> int:
        nonlocal state
        state = (spec["multiplier"] * state + spec["increment"]) % spec["modulus"]
        return state

    for index in range(spec["count"]):
        tick += 1 + (draw() % (2 * gap - 1))
        prompt = spec["prompt_base"] + (draw() % spec["prompt_span"])
        output = spec["output_base"] + (draw() % spec["output_span"])
        rows.append(
            {
                "id": "r{0:03d}".format(index),
                "arrival_tick": tick,
                "prompt_tokens": prompt,
                "output_tokens": output,
            }
        )
    return {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "oer23.trace/v1",
        "frozen": True,
        "requests": rows,
    }


def build_hardware(block: dict) -> dict:
    row = {key: block[key] for key in sorted(block) if key != "envelope"}
    row.update({"banner": BANNER, "source": SOURCE, "schema": "oer23.hardware/v1", "frozen": True, "envelope": block["envelope"]})
    return row


def build_objective(block: dict) -> dict:
    return {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "oer23.objective/v1",
        "frozen": True,
        "p99_tpot_centiticks": int(block["p99_tpot_centiticks"]),
        "statement": block["statement"].strip(),
        "hard_constraint": True,
    }


def build_serve_sim() -> str:
    header = (
        '"""' + BANNER + "\n\n"
        "Generated from " + SOURCE + " by solution/recompute.py as a copy of the verifier-owned\n"
        "harness at tests/harness_sim.py. This is the agent-visible serving substrate. Editing it\n"
        "does not change the substrate the verifier measures, and the environment_frozen checker\n"
        "refuses a session whose surface digest has moved.\n"
        '"""\n\n'
    )
    return header + (TESTS / "harness_sim.py").read_text(encoding="utf-8")


def recompute_row(trace: dict, hardware: dict, config) -> dict:
    telemetry = harness_sim.simulate(trace, hardware, config)
    row = telemetry.as_row()
    row["record_digest"] = telemetry.record_digest()
    row["record_ids"] = [item["id"] for item in telemetry.records]
    return row


def golden_session(trace: dict, hardware: dict, objective: dict) -> dict:
    session = harness_sim.run_session(reference.plan(trace, hardware, objective), trace, hardware, objective)
    return harness_sim.report(session, "raw")


def handle_payload(session: dict, trace: dict, hardware: dict, objective: dict, digests: dict, windows: dict, flatten: int, spec: dict) -> dict:
    return {
        "session_present": True,
        "bound_digests": dict(digests),
        "observed_digests": dict(digests),
        "slo_p99_centiticks": int(objective["p99_tpot_centiticks"]),
        "window_count": int(windows["count"]),
        "sustain_floor_num": int(windows["sustain_floor_num"]),
        "sustain_floor_den": int(windows["sustain_floor_den"]),
        "flatten_window": int(flatten),
        "trace_ids": [row["id"] for row in trace["requests"]],
        "seed_derivation": dict(spec),
        "observed_trace": copy.deepcopy(trace["requests"]),
        "graded_trace": copy.deepcopy(trace["requests"]),
        "ledger": session["attempts"],
        "recomputed": [recompute_row(trace, hardware, row["config"]) for row in session["attempts"]],
        "selected_index": int(session["selected_attempt"]),
        "reported": session["reported"],
        "graded_path_source": CLEAN_SOURCE,
    }


def _plant(payload: dict, index: int, patch: dict) -> dict:
    """Move a telemetry key in both the recorded ledger and the verifier recomputation.

    Both sides move together so the planted fixture fires the checker it is aimed at and
    not the divergence checker standing in front of it.
    """
    row = copy.deepcopy(payload)
    row["recomputed"][index].update(copy.deepcopy(patch))
    row["ledger"][index]["harness"].update(copy.deepcopy(patch))
    return row


def build_planted(golden: dict, trace: dict, hardware: dict, objective: dict, greedy: dict, reseeded: dict) -> dict:
    slo = int(objective["p99_tpot_centiticks"])
    selected = int(golden["selected_index"])
    rows = {}

    row = copy.deepcopy(golden)
    row["observed_digests"]["trace.json"] = "0" * 64
    rows["environment_frozen"] = row

    # The rejecting half of trace_is_seed_derived: an environment raised from the control
    # starting state of grounding.yaml control_schedules.reseeded_trace. The bound seed
    # derivation is left untouched, so what the checker refuses is the built environment.
    row = copy.deepcopy(golden)
    row["observed_trace"] = copy.deepcopy(reseeded["requests"])
    row["graded_trace"] = copy.deepcopy(reseeded["requests"])
    rows["trace_is_seed_derived"] = row

    row = copy.deepcopy(golden)
    row["ledger"][0]["telemetry_digest"] = "0" * 64
    rows["attempt_ledger_recomputed"] = row

    row = copy.deepcopy(golden)
    row["recomputed"][selected]["record_ids"] = row["recomputed"][selected]["record_ids"][:-1]
    row = _plant(row, selected, {"accounted": len(row["recomputed"][selected]["record_ids"])})
    rows["trace_fully_accounted"] = row

    rows["no_request_shed"] = _plant(golden, selected, {"shed": 3})
    rows["early_stop_not_a_result"] = _plant(golden, selected, {"trace_exhausted": False})
    rows["slo_hard_constraint"] = _plant(golden, selected, {"p99_tpot_centiticks": slo + 1})

    row = copy.deepcopy(golden)
    row["reported"]["latency_readout"] = "ema"
    row["reported"]["p99_tpot_centiticks"] = slo - 1
    rows["p99_recomputed_unsmoothed"] = row

    row = copy.deepcopy(golden)
    row["reported"]["throughput_num"] = int(row["reported"]["throughput_num"]) * 2
    rows["throughput_from_harness_telemetry"] = row

    row = copy.deepcopy(golden)
    windows = copy.deepcopy(row["recomputed"][selected]["windows"])
    windows[0]["completed"] = 0
    windows[0]["output_tokens"] = 0
    windows[0]["throughput_num"] = 0
    row = _plant(row, selected, {"windows": windows})
    rows["sustained_across_windows"] = row

    row = copy.deepcopy(golden)
    row["ledger"] = greedy["ledger"]
    row["recomputed"] = greedy["recomputed"]
    row["selected_index"] = greedy["selected_index"]
    row["reported"] = greedy["reported"]
    rows["reallocation_after_flattening"] = row

    row = copy.deepcopy(golden)
    row["selected_index"] = 2
    rows["carried_best_preserved"] = row

    row = copy.deepcopy(golden)
    row["graded_path_source"] = POISONED_SOURCE
    rows["harness_owns_the_clock"] = row
    rows["banner"] = BANNER
    rows["source"] = SOURCE
    return rows


def greedy_session(trace: dict, hardware: dict, objective: dict, digests: dict, windows: dict, flatten: int, spec: dict) -> dict:
    """The single-axis sweep: the whole budget on max_batch_size, never reallocated."""
    configs = []
    for value in harness_sim.LATTICE["max_batch_size"]:
        config = dict(harness_sim.DEFAULT_CONFIG)
        config["max_batch_size"] = value
        configs.append(config)
    session = harness_sim.report(harness_sim.run_session(configs, trace, hardware, objective), "raw")
    return handle_payload(session, trace, hardware, objective, digests, windows, flatten, spec), session


def build_test_output(order) -> str:
    lines = [
        '"""' + BANNER,
        "",
        "Generated from " + SOURCE + " by solution/recompute.py.",
        "",
        "One compiled test per graded checker, each carrying BOTH halves: the accepting half",
        "over the golden fixture and the rejecting half over the planted fixture that fires",
        "exactly that checker's zero reason. No test here reads a clock; the fixtures are",
        "recorded harness telemetry.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "HERE = Path(__file__).resolve().parent",
        "if str(HERE) not in sys.path:",
        "    sys.path.insert(0, str(HERE))",
        "",
        "from checkers import GRADED, handle_from_payload  # noqa: E402",
        "",
        "GOLDEN = json.loads((HERE / 'fixtures' / 'golden.json').read_text(encoding='utf-8'))",
        "PLANTED = json.loads((HERE / 'fixtures' / 'planted.json').read_text(encoding='utf-8'))",
        "SELECTOR = dict(GRADED)",
        "",
        "",
        "def _halves(ident, reason):",
        "    passed, emitted = SELECTOR[ident](handle_from_payload(GOLDEN['handle']))",
        "    assert passed, ident + ' refused the golden fixture: ' + emitted",
        "    passed, emitted = SELECTOR[ident](handle_from_payload(PLANTED[ident]))",
        "    assert not passed, ident + ' accepted its planted defect'",
        "    assert emitted == reason, ident + ' emitted ' + repr(emitted) + ' not ' + repr(reason)",
        "",
    ]
    for ident, reason in order:
        lines += [
            "",
            "def test_" + ident + "():",
            "    _halves(" + repr(ident) + ", " + repr(reason) + ")",
            "",
        ]
    lines += [
        "",
        "def main() -> int:",
        "    names = sorted(name for name in globals() if name.startswith('test_'))",
        "    for name in names:",
        "        globals()[name]()",
        "    print('compiled tests passed: ' + str(len(names)))",
        "    return 0",
        "",
        "",
        "if __name__ == '__main__':",
        "    raise SystemExit(main())",
        "",
    ]
    return "\n".join(lines)


def build_solve() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# " + BANNER,
            "# Source: " + SOURCE + ". Generated by solution/recompute.py.",
            "#",
            "# The solver entry point. It places the reference search where the verifier's runner",
            "# picks it up. The reference never writes a reward and never reports a figure of its",
            "# own; the harness measures, the verifier recomputes.",
            "set -euo pipefail",
            "",
            'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'SUBMISSION_DIR="${OER23_SUBMISSION_DIR:-/workspace/submission}"',
            "",
            'mkdir -p "${SUBMISSION_DIR}"',
            'cp "${HERE}/reference.py" "${SUBMISSION_DIR}/search.py"',
            'echo "OER-23 reference search placed at ${SUBMISSION_DIR}/search.py"',
            "",
        ]
    )


def build_truth(grounding: dict, golden: dict, calibration: dict, target_config: dict) -> str:
    base = calibration["baseline_num"] / calibration["baseline_den"]
    target = calibration["target_num"] / calibration["target_den"]
    lines = [
        "<!-- " + BANNER + " Source: " + SOURCE + ". -->",
        "",
        "# OER-23 reference truth",
        "",
        "Slot `OER-23`, family `F11` inference serving throughput, archetype `AR1` long-horizon",
        "state collapse. Provisional identifier is the slot id.",
        "",
        "## What is graded",
        "",
        "Sustained output tokens per second over the frozen request trace, subject to a hard p99",
        "per-output-token latency objective of "
        + str(int(grounding["objective"]["p99_tpot_centiticks"]))
        + " centiticks. Higher is better. The objective is a",
        "constraint and not a penalty: exceeding it scores zero at any throughput.",
        "",
        "Both readouts are recomputed by the verifier from harness-owned completion telemetry over",
        "every request in the frozen trace. Nothing the submission reports is substituted for them.",
        "",
        "## Who owns the clock",
        "",
        "The harness. It advances a virtual integer tick clock inside the serving loop and writes",
        "the result into telemetry records. Every checker reads those records and never calls a",
        "clock, which is what makes the grading pure, deterministic and replayable over the",
        "recorded trace.",
        "",
        "## The reference answer",
        "",
        "```json",
        json.dumps(target_config, indent=2, sort_keys=True),
        "```",
        "",
        "| quantity | value |",
        "|---|---|",
        "| attempts in the reference session | " + str(len(golden["ledger"])) + " |",
        "| selected attempt | " + str(golden["selected_index"]) + " |",
        "| baseline tokens per second (default configuration, measured) | "
        + "{0:.3f}".format(base)
        + " |",
        "| target tokens per second (best feasible lattice point, measured) | "
        + "{0:.3f}".format(target)
        + " |",
        "| reference p99 per-output-token latency | "
        + str(golden["recomputed"][golden["selected_index"]]["p99_tpot_centiticks"])
        + " centiticks |",
        "",
        "`baseline_metric` and `target_metric` are ABSENT as published anchors under",
        "`gap-oer-per-family-anchors-unmeasured`. The two figures above are calibration endpoints",
        "the verifier measures from frozen bytes at grade time. Neither appears as a literal in any",
        "graded file.",
        "",
        "## Why a sweep does not reach it",
        "",
        "`max_batch_size` pays until four-way batching and then stops paying, because every wider",
        "batch pushes the per-output-token tail past the objective. Nothing announces that. Chunked",
        "prefill pays nothing on its own at four-way batching and pays a great deal jointly with an",
        "eight-way batch, because it removes the head-of-line prefill stall that was carrying the",
        "tail over the line. A session that rides one axis to the end of its budget never sees the",
        "joint move; a session that holds its earlier attempts and reallocates does.",
        "",
        "## Substrate",
        "",
        "A deterministic discrete-event simulator, declared under `gap-oer23-substrate-is-a-simulator`.",
        "It is the right substrate here because it makes grading reproducible and lets a checker read",
        "timing without a clock. It is never a live measurement and `reference_hours` covers simulated",
        "serving only.",
        "",
    ]
    return "\n".join(lines)


def build_rubrics_json(grounding: dict, golden: dict, target_config: dict) -> dict:
    return {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "oer23.solution_rubric/v1",
        "judged": "the solution against its reference answer",
        "not_the_same_file_as": "tests/rubrics.jsonl, which is judged against the trajectory",
        "criteria": [
            {
                "id": "matches-reference-configuration",
                "statement": "The selected serving configuration equals the reference answer on every free axis.",
                "reference": target_config,
            },
            {
                "id": "reaches-the-target-endpoint",
                "statement": "The graded throughput equals the best objective-feasible point of the frozen lattice.",
                "reference_selected_attempt": golden["selected_index"],
            },
            {
                "id": "respects-the-objective",
                "statement": "The graded p99 per-output-token latency is at or below the frozen objective.",
                "objective_p99_tpot_centiticks": int(grounding["objective"]["p99_tpot_centiticks"]),
            },
            {
                "id": "accounts-for-every-request",
                "statement": "Every request of the frozen trace carries a terminal telemetry record and none is shed.",
                "trace_size": int(grounding["trace"]["count"]),
            },
            {
                "id": "reallocated-off-the-flattened-axis",
                "statement": "The session abandons a direction that stopped paying and spends the remainder elsewhere.",
                "flatten_window": int(grounding["flattening"]["window"]),
            },
            {
                "id": "carried-its-own-best",
                "statement": "The answer returned is the best feasible attempt the session established, not its last.",
                "reference_attempts": len(golden["ledger"]),
            },
        ],
    }


def artifacts() -> dict:
    grounding = load_grounding()
    trace = build_trace(grounding["trace"])
    hardware = build_hardware(grounding["hardware"])
    objective = build_objective(grounding["objective"])
    serve = build_serve_sim()

    files = {
        ENVIRONMENT / "trace.json": json_bytes(trace),
        ENVIRONMENT / "hardware.json": json_bytes(hardware),
        ENVIRONMENT / "objective.json": json_bytes(objective),
        ENVIRONMENT / "serve_sim.py": serve,
    }
    digests = {
        "trace.json": digest_text(files[ENVIRONMENT / "trace.json"]),
        "hardware.json": digest_text(files[ENVIRONMENT / "hardware.json"]),
        "objective.json": digest_text(files[ENVIRONMENT / "objective.json"]),
        "serve_sim.py": digest_text(serve),
    }

    windows = grounding["windows"]
    flatten = int(grounding["flattening"]["window"])
    spec = seed_derivation(grounding["trace"])
    substrate = {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "oer23.substrate/v1",
        "trace": trace,
        "hardware": hardware,
        "objective": objective,
        "bound_digests": digests,
        "seed_derivation": spec,
        "window_count": int(windows["count"]),
        "sustain_floor_num": int(windows["sustain_floor_num"]),
        "sustain_floor_den": int(windows["sustain_floor_den"]),
        "flatten_window": flatten,
    }

    session = golden_session(trace, hardware, objective)
    golden_handle = handle_payload(session, trace, hardware, objective, digests, windows, flatten, spec)
    greedy_handle, greedy_doc = greedy_session(trace, hardware, objective, digests, windows, flatten, spec)

    recorded = [
        {"stage": stage["stage"], "config": harness_sim.normalise(stage["config"])}
        for stage in grounding["reference_schedule"]
    ]
    produced = [{"stage": None, "config": row["config"]} for row in session["attempts"]]
    if [row["config"] for row in recorded] != [row["config"] for row in produced]:
        raise SystemExit(
            "the reference search no longer reproduces grounding.yaml reference_schedule; "
            "the golden trajectory and its source have diverged"
        )

    slo = int(objective["p99_tpot_centiticks"])
    base = harness_sim.simulate(trace, hardware, harness_sim.DEFAULT_CONFIG)
    best = None
    import itertools

    for combo in itertools.product(*[harness_sim.LATTICE[axis] for axis in harness_sim.AXES]):
        config = dict(zip(harness_sim.AXES, combo))
        candidate = harness_sim.simulate(trace, hardware, config)
        if harness_sim.feasible(candidate, slo) and harness_sim.better(candidate, best):
            best = candidate
    calibration = {
        "baseline_num": base.throughput_num,
        "baseline_den": base.throughput_den,
        "target_num": best.throughput_num,
        "target_den": best.throughput_den,
    }

    selected_config = session["attempts"][session["selected_attempt"]]["config"]
    if selected_config != best.config:
        raise SystemExit("the reference search no longer reaches the best feasible lattice point")

    golden = {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "oer23.golden/v1",
        "trajectory": [
            {
                "index": row["index"],
                "stage": recorded[row["index"]]["stage"],
                "config": row["config"],
                "axes_moved": row["axes_moved"],
                "carried_best_index": row["carried_best_index"],
                "throughput_num": row["harness"]["throughput_num"],
                "throughput_den": row["harness"]["throughput_den"],
                "p99_tpot_centiticks": row["harness"]["p99_tpot_centiticks"],
                "shed": row["harness"]["shed"],
                "trace_exhausted": row["harness"]["trace_exhausted"],
            }
            for row in session["attempts"]
        ],
        "session": session,
        "handle": golden_handle,
        "calibration": calibration,
        "target_config": best.config,
        "controls": {"greedy_single_axis_sweep": greedy_doc},
    }
    reseeded = build_trace(
        dict(grounding["trace"], seed=int(grounding["control_schedules"]["reseeded_trace"]["seed"]))
    )
    if reseeded["requests"] == trace["requests"]:
        raise SystemExit(
            "the reseeded control trace equals the frozen trace, so the rejecting half of "
            "trace_is_seed_derived would prove nothing"
        )
    planted = build_planted(golden_handle, trace, hardware, objective, greedy_handle, reseeded)

    order = [(ident, checker_module.__dict__["check_" + ident](checker_module.handle_from_payload(planted[ident]))[1]) for ident, _ in checker_module.GRADED]

    files[FIXTURES / "substrate.json"] = json_bytes(substrate)
    files[FIXTURES / "golden.json"] = json_bytes(golden)
    files[FIXTURES / "planted.json"] = json_bytes(planted)
    files[TESTS / "test_output.py"] = build_test_output(order)
    files[SOLUTION / "solve.sh"] = build_solve()
    files[SOLUTION / "TRUTH.md"] = build_truth(grounding, golden_handle, calibration, best.config)
    files[SOLUTION / "rubrics.json"] = json_bytes(build_rubrics_json(grounding, golden_handle, best.config))
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Derive every generated OER-23 artifact.")
    parser.add_argument("--check", action="store_true", help="verify committed bytes match, write nothing")
    args = parser.parse_args(argv)

    files = artifacts()
    drifted = []
    for path, text in sorted(files.items()):
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                drifted.append(str(path.relative_to(BUNDLE)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    if args.check and drifted:
        print("DRIFTED: " + ", ".join(drifted))
        return 1
    (SOLUTION / "solve.sh").chmod(0o755)
    print(("checked " if args.check else "generated ") + str(len(files)) + " artifacts from " + SOURCE)
    return 0


# FORGE-SCREENING-CARRIER-BEGIN
# GENERATED SECTION. DO NOT HAND-EDIT.
# Generated by seed/forge/screenfreeze.py. Derives the contamination-screening provenance carrier
# from the frozen `screening` block in solution/grounding.yaml and nothing else. It opens no
# connection, reads no wall clock, consults no host language setting, draws no entropy, starts no
# child process, and imports nothing outside this tree.
import hashlib as _forge_hashlib
import json as _forge_json
import pathlib as _forge_pathlib
import sys as _forge_sys

import yaml as _forge_yaml

_FORGE_CARRIER_KEYS = (
    "schema",
    "unit_uuid",
    "screening_roots",
    "authority_mode",
    "source_identifiers",
    "fork_ancestry_snapshot",
    "base_commit_sha",
    "applicable_dates",
    "instrument_versions",
    "atom_result_digests",
    "applicability",
    "sanitization_closure",
    "empty_submission_result",
    "attestations",
    "binding_block",
    "keyid",
    "trust_root_public_key_hex",
    "namespace",
    "normalization_domain_version",
    "signer_identity",
)

_FORGE_BINDING_KEYS = (
    "canonical_bundle_hash",
    "pinned_image_digest",
    "binding_envelope",
)

_FORGE_SCREENING_KEY = "screening"
_FORGE_GROUNDING = "grounding.yaml"
_FORGE_CARRIER = "provenance.yaml"
_FORGE_BANNER = "# GENERATED SECTION. DO NOT HAND-EDIT."


def _forge_here():
    return _forge_pathlib.Path(__file__).resolve().parent


def _forge_sorted(value):
    """Sort every container so two runs over the same frozen bytes emit identical bytes."""
    if isinstance(value, dict):
        return {key: _forge_sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_forge_sorted(item) for item in value]
    return value


def _forge_frozen_screening():
    """Read the frozen screening block. Absence is refused rather than defaulted."""
    path = _forge_here() / _FORGE_GROUNDING
    with path.open("r", encoding="utf-8") as handle:
        document = _forge_yaml.safe_load(handle)
    block = (document or {}).get(_FORGE_SCREENING_KEY)
    if not isinstance(block, dict):
        raise SystemExit(
            "solution/grounding.yaml carries no frozen `screening` block, so the provenance "
            "carrier cannot be derived. Refusing to emit a carrier over values nobody froze."
        )
    missing = [key for key in _FORGE_CARRIER_KEYS if key not in block]
    unknown = [key for key in sorted(block) if key not in _FORGE_CARRIER_KEYS]
    if missing or unknown:
        raise SystemExit(
            "the frozen `screening` block does not mirror the closed carrier schema: "
            "missing " + repr(missing) + ", unknown " + repr(unknown)
        )
    return block


def _forge_canonical_bytes(payload):
    return _forge_json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _forge_carrier_payload():
    """Assemble the carrier as exactly the closed key set, in the order the schema fixes.

    The binding block is attached AFTER the canonical payload is hashed and never enters the
    preimage, because a payload that contained a hash of itself would have no acyclic ordering.
    """
    block = _forge_frozen_screening()
    payload = {}
    for key in _FORGE_CARRIER_KEYS:
        if key == "binding_block":
            continue
        payload[key] = _forge_sorted(block[key])
    digest = _forge_hashlib.sha256(_forge_canonical_bytes(payload)).hexdigest()

    binding = _forge_sorted(block["binding_block"]) or {}
    shaped = {key: binding.get(key) for key in _FORGE_BINDING_KEYS}
    ordered = {}
    for key in _FORGE_CARRIER_KEYS:
        ordered[key] = shaped if key == "binding_block" else payload[key]
    return ordered, digest


def _forge_carrier_text():
    payload, digest = _forge_carrier_payload()
    header = (
        _FORGE_BANNER + "\n"
        + "# Derived from solution/grounding.yaml `screening` by solution/recompute.py.\n"
        + "# canonical payload sha256 (binding_block excluded from the preimage): " + digest + "\n"
    )
    body = _forge_yaml.safe_dump(
        payload, sort_keys=False, default_flow_style=False, allow_unicode=False, width=100
    )
    return header + body


def _forge_emit_carrier():
    """Write the carrier, or in check mode compare and report drift. Never both."""
    argv = list(_forge_sys.argv[1:])
    check = "--check" in argv
    path = _forge_here() / _FORGE_CARRIER
    text = _forge_carrier_text()
    if check:
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        if current == text:
            return 0
        _forge_sys.stderr.write(
            "drift: " + _FORGE_CARRIER + " does not match the carrier derived from the frozen "
            "`screening` block in " + _FORGE_GROUNDING + "\n"
        )
        return 1
    path.write_text(text, encoding="utf-8")
    return 0


_FORGE_INNER_MAIN = main


def main(*args, **kwargs):
    """Run the host generator, then derive the provenance carrier from the frozen block."""
    status = _FORGE_INNER_MAIN(*args, **kwargs)
    drift = _forge_emit_carrier()
    if drift and not status:
        return drift
    return status

# FORGE-SCREENING-CARRIER-END


if __name__ == "__main__":
    raise SystemExit(main())

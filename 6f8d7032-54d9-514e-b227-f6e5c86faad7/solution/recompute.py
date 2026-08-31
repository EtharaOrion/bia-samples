#!/usr/bin/env python3
"""Derive every generated artifact of OER-28 from solution/grounding.yaml alone.

Generated here, and nowhere else:

    environment/stream.json                the frozen input stream
    environment/params.json                the frozen operator parameters
    tests/bound.json                       every bound constant the checkers compare against
    tests/test_output.py                   the compiled per-checker tests
    solution/solve.sh                      the solution entry point
    solution/TRUTH.md                      the derivation write-up
    solution/rubrics.json                  the solution-against-reference rubric
    solution/fixtures/reference_run/       the accepting fixture, a real two-phase run
    solution/fixtures/baseline_run/        an honest run of the frozen reference kernel
    solution/fixtures/partial_run/         an honest run of a partly fused kernel
    solution/fixtures/planted/             the rejecting fixtures, including two near misses

No model is invoked, no socket is opened, no clock is read, no locale is consulted and no random
source is drawn from. The stream comes from a fixed integer linear congruential recurrence whose
seed is recorded in grounding.yaml, which is arithmetic rather than randomness. Both DISCOVERY
values, the splice point and the carried-state digest, are DERIVED here by running the bundle's
own frozen drivers over the frozen bytes; neither is transcribed from a literal and neither is
stated anywhere in instruction.md.

Running this twice over frozen bytes produces byte-identical output. `--check` reports drift and
writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "environment"))

import opstream  # noqa: E402
import phase_a  # noqa: E402
import phase_b  # noqa: E402
import reference  # noqa: E402

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."
SOURCE = "solution/grounding.yaml"
LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345
LCG_MODULUS = 2 ** 31

FIXTURES = HERE / "fixtures"

# A kernel that delegates to the frozen reference. Installed as the baseline control and as the
# swapped kernel in the seam fixture, so both are genuinely a different module from the fused one.
BASELINE_KERNEL = '''"""The frozen reference kernel, installed unchanged as the baseline control."""

import opstream


def run_block(tape, params, carry, start, length):
    return opstream.reference_run_block(tape, params, carry, start, length)
'''

# A kernel that has the recurrence right but still re-reads the head sample it already holds.
# Three reads per output instead of two, which is a real intermediate point on the gradient and
# is what shows the reward is continuous rather than a pass or fail.
PARTIAL_KERNEL = '''"""A partly fused kernel. The recurrence is right; the read pattern is not tight."""


def run_block(tape, params, carry, start, length):
    modulus = int(params["modulus"])
    window = int(params["window"])
    ratio = int(params["ratio"])
    tail_weight = pow(ratio, window, modulus)

    accumulator = int(carry["accumulator"])
    history = [int(value) for value in carry["history"]]
    outputs = []

    for index in range(start, start + length):
        head = tape.at(index)
        source = index - window
        if source >= start:
            tail = tape.at(source)
        else:
            tail = history[0]
        accumulator = (ratio * accumulator + head - tail_weight * tail) % modulus
        outputs.append(accumulator)
        history = history[1:] + [tape.at(index)]

    return outputs, {"accumulator": accumulator, "history": history}
'''


def load_grounding():
    with (HERE / "grounding.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def lcg_stream(seed):
    value = int(seed)
    while True:
        value = (LCG_MULTIPLIER * value + LCG_INCREMENT) % LCG_MODULUS
        yield value


def build_stream(ground):
    """The frozen input stream. Block lengths first, then samples, in one pass."""
    block = ground["stream"]
    draw = lcg_stream(block["seed"])
    lengths = [
        int(block["block_length"]["base"]) + next(draw) % int(block["block_length"]["modulus"])
        for _index in range(int(block["block_count"]))
    ]
    blocks = []
    for length in lengths:
        blocks.append(
            [
                int(block["sample"]["base"]) + next(draw) % int(block["sample"]["modulus"])
                for _position in range(length)
            ]
        )
    return {
        "schema": block["schema"],
        "banner": BANNER,
        "source": SOURCE,
        "block_count": len(blocks),
        "total_samples": sum(lengths),
        "blocks": blocks,
    }


def build_params(ground):
    block = ground["params"]
    return {
        "schema": block["schema"],
        "banner": BANNER,
        "source": SOURCE,
        "modulus": int(block["modulus"]),
        "ratio": int(block["ratio"]),
        "window": int(block["window"]),
        "splice_charge_budget": int(block["splice_charge_budget"]),
    }


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_two_phase(root, kernel_source):
    """One complete two-phase run into a fixture root. Both drivers, both real.

    This is the whole reason the fixtures are evidence rather than illustration: they are the
    output of the bundle's own agent-phase driver and its own resume driver over the bundle's own
    frozen bytes, not a hand-written approximation of what those drivers would emit.
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    workspace = root / "workspace"
    journal = root / "harness" / phase_a.JOURNAL_NAME
    submission = root / "app" / "submission.py"
    handoff, rows, probe = reference.drive(
        BUNDLE, workspace, journal, submission, kernel_source
    )
    return {
        "root": root,
        "workspace": workspace,
        "journal": journal,
        "submission": submission,
        "handoff": handoff,
        "rows": rows,
        "probe": probe,
    }


def derive(ground):
    """Every measured quantity, produced by running the frozen drivers over the frozen bytes."""
    stream = opstream.load_json(BUNDLE / "environment" / "stream.json")
    params = opstream.load_json(BUNDLE / "environment" / "params.json")

    fused = run_two_phase(FIXTURES / "reference_run", reference.FUSED_KERNEL)
    baseline = run_two_phase(FIXTURES / "baseline_run", BASELINE_KERNEL)
    partial = run_two_phase(FIXTURES / "partial_run", PARTIAL_KERNEL)

    def rollup_of(run):
        digests = [row["out_sha256"] for row in run["rows"] if row["kind"] == "block"]
        digests += [row["out_sha256"] for row in run["probe"] if row["kind"] == "block"]
        return opstream.rollup(digests)

    fused_rollup = rollup_of(fused)
    if rollup_of(baseline) != fused_rollup:
        raise SystemExit("the fused kernel and the frozen reference do not agree on the output")
    if rollup_of(partial) != fused_rollup:
        raise SystemExit("the partly fused kernel does not agree with the frozen reference")

    halt = fused["rows"][-1]
    if baseline["rows"][-1]["carried_state_sha256"] != halt["carried_state_sha256"]:
        raise SystemExit("the fused and reference carries diverge at the splice")

    splice_offset = int(halt["splice_offset"])
    if splice_offset != opstream.splice_offset(stream, params):
        raise SystemExit("the driver's splice point and the arithmetic derivation disagree")

    reference_resume_charge = int(baseline["probe"][-1]["reads"])
    fused_resume_charge = int(fused["probe"][-1]["reads"])
    partial_resume_charge = int(partial["probe"][-1]["reads"])

    return {
        "stream_sha256": digest_file(BUNDLE / "environment" / "stream.json"),
        "params_sha256": digest_file(BUNDLE / "environment" / "params.json"),
        "block_count": int(stream["block_count"]),
        "total_samples": int(stream["total_samples"]),
        "window": int(params["window"]),
        "splice_charge_budget": int(params["splice_charge_budget"]),
        "splice_offset": splice_offset,
        "splice_block_index": int(halt["splice_block_index"]),
        "carried_state_sha256": str(halt["carried_state_sha256"]),
        "stream_output_rollup": fused_rollup,
        "reference_resume_charge": reference_resume_charge,
        "fused_resume_charge": fused_resume_charge,
        "partial_resume_charge": partial_resume_charge,
        "instance_baseline_speedup": round(reference_resume_charge / float(reference_resume_charge), 6),
        "instance_target_speedup": round(reference_resume_charge / float(fused_resume_charge), 6),
        "partial_speedup": round(reference_resume_charge / float(partial_resume_charge), 6),
        "resume_outputs": int(fused["probe"][-1]["outputs"]),
        "agent_phase_reads": int(halt["phase_a_reads"]),
        "journal_chain": str(halt["chain"]),
        "runs": {"fused": fused, "baseline": baseline, "partial": partial},
    }


def build_bound(ground, measured):
    anchors = ground["anchors"]
    return {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "oer28.bound/v1",
        "stream_sha256": measured["stream_sha256"],
        "params_sha256": measured["params_sha256"],
        "block_count": measured["block_count"],
        "total_samples": measured["total_samples"],
        "window": measured["window"],
        "splice_charge_budget": measured["splice_charge_budget"],
        "splice_offset": measured["splice_offset"],
        "splice_block_index": measured["splice_block_index"],
        "carried_state_sha256": measured["carried_state_sha256"],
        "stream_output_rollup": measured["stream_output_rollup"],
        "reference_resume_charge": measured["reference_resume_charge"],
        "instance_baseline_speedup": measured["instance_baseline_speedup"],
        "instance_target_speedup": measured["instance_target_speedup"],
        "instance_baseline_config": "the frozen reference kernel, environment/opstream.py reference_run_block",
        "instance_target_config": "the fused first order recurrence, solution/reference.py FUSED_KERNEL",
        "journal_genesis": phase_a.GENESIS,
        "readout_tolerance": 0.0,
        "anchors_state": anchors["anchors_state"],
        "anchors_gap": anchors["gap_id"],
        "instance_anchor_gap": anchors["instance_local_anchors"]["gap_id"],
        "family_baseline_metric": anchors["family_baseline_metric"],
        "family_target_metric": anchors["family_target_metric"],
        "reference_handoff_sha256": digest_file(
            measured["runs"]["fused"]["workspace"] / "handoff.json"
        ),
        "reference_journal_sha256": digest_file(measured["runs"]["fused"]["journal"]),
    }


# ---------------------------------------------------------------------------
# The planted rejecting fixtures. Each one moves exactly one thing.
# ---------------------------------------------------------------------------


def plant(measured):
    """Every rejecting fixture, derived from the accepting one by one named mutation."""
    source = measured["runs"]["fused"]["root"]
    planted = FIXTURES / "planted"
    if planted.exists():
        shutil.rmtree(planted)
    planted.mkdir(parents=True, exist_ok=True)

    def clone(name):
        target = planted / name
        shutil.copytree(source, target)
        return target

    def rewrite_handoff(target, mutate):
        path = target / "workspace" / "handoff.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # NEAR MISS ONE. The splice point moved by exactly one sample and nothing else touched.
    one = clone("splice_offset_plus_one")
    rewrite_handoff(one, lambda payload: payload.__setitem__(
        "splice_offset", int(payload["splice_offset"]) + 1
    ))

    # NEAR MISS TWO. The carried-state digest differing in exactly one hexadecimal character.
    two = clone("carried_digest_one_nibble")

    def nibble(payload):
        digest = str(payload["carried_state_sha256"])
        last = digest[-1]
        payload["carried_state_sha256"] = digest[:-1] + ("0" if last != "0" else "1")

    rewrite_handoff(two, nibble)

    # The agent phase never ran and the handoff was synthesised. The journal is absent and every
    # other byte of the accepting fixture stands.
    three = clone("agent_phase_skipped")
    (three / "harness" / phase_a.JOURNAL_NAME).unlink()
    (three / "harness" / "ABSENT.md").write_text(
        "The harness journal is deliberately absent from this fixture. That absence IS the "
        "fixture: it is what a workspace looks like when the handoff was synthesised and no "
        "agent phase ever ran. The file is here so the empty directory survives a checkout, "
        "and it is never read by the grading path.\n",
        encoding="utf-8",
    )

    # The journal was hand-edited after the fact. One block row's recorded output digest moves by
    # one hexadecimal character and the chain link over it is left as the driver wrote it.
    four = clone("journal_row_edited")
    journal_path = four / "harness" / phase_a.JOURNAL_NAME
    rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("kind") == "block" and int(row.get("index", -1)) == 0:
            digest = str(row["out_sha256"])
            last = digest[-1]
            row["out_sha256"] = digest[:-1] + ("0" if last != "0" else "1")
            break
    journal_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    # The seam attack. The agent phase ran under the fused kernel and a different kernel is
    # presented to the verifier. Only the kernel file moves.
    five = clone("kernel_swapped_after_splice")
    (five / "app" / "submission.py").write_text(BASELINE_KERNEL, encoding="utf-8")

    return {
        "splice_offset_plus_one": one,
        "carried_digest_one_nibble": two,
        "agent_phase_skipped": three,
        "journal_row_edited": four,
        "kernel_swapped_after_splice": five,
    }


# ---------------------------------------------------------------------------
# The generated documents
# ---------------------------------------------------------------------------


SOLVE_TEMPLATE = '''#!/usr/bin/env bash
# {banner}
# Source: {source}. Regenerate with solution/recompute.py.
#
# The AGENT PHASE entry point of the oracle. It installs the fused kernel at the bound
# submission path, runs the bundle's own agent-phase driver, and transcribes the splice point
# and the carried-state digest the driver established into the handoff. It never invents either
# value: both are read back off the harness journal the driver wrote.
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BUNDLE="$(cd "${{HERE}}/.." && pwd)"
WORKSPACE="${{OER28_WORKSPACE:-/workspace}}"
HARNESS_LOGS="${{OER28_HARNESS_LOGS:-/logs/harness}}"
SUBMISSION="${{OER28_SUBMISSION:-/app/submission.py}}"

mkdir -p "${{WORKSPACE}}" "${{HARNESS_LOGS}}" "$(dirname "${{SUBMISSION}}")"
python3 "${{HERE}}/reference.py" \\
    --bundle "${{BUNDLE}}" \\
    --workspace "${{WORKSPACE}}" \\
    --journal "${{HARNESS_LOGS}}/{journal}" \\
    --submission "${{SUBMISSION}}"

echo "kernel installed at ${{SUBMISSION}}"
echo "handoff written to ${{WORKSPACE}}/handoff.json"
echo "harness journal at ${{HARNESS_LOGS}}/{journal}"
'''


def build_solve():
    return SOLVE_TEMPLATE.format(banner=BANNER, source=SOURCE, journal=phase_a.JOURNAL_NAME)


TRUTH_TEMPLATE = """# TRUTH.md

{banner}

Source: `{source}`. Regenerate with `solution/recompute.py`.

## What is graded

Speedup of the submitted kernel over the frozen reference kernel, measured on the resume segment that the verifier phase runs for itself, and gated on exact output equality. Direction: higher is better. The correctness gate is a HARD GATE: the operator is exact integer arithmetic modulo {modulus}, so a kernel whose composed output rollup differs from the frozen reference rollup scores exactly 0.0 with the reason `numerical-output-diverged`, however few reads it charged.

The cost model is an instrumented read counter and never a wall clock. Every kernel reaches an input sample only through a Tape, and the Tape charges one read per call. That is what makes the graded number reproducible on any host, and it is declared as a substrate limit rather than presented as a timing measurement.

## The archetype

Two-phase splice. The unit runs in two phases over one shared workspace. The AGENT phase runs `environment/phase_a.py` in the agent image: it feeds the kernel whole blocks in order, charges every read, chains one journal row per block, and halts the instant its own work counter reaches the frozen splice charge budget. The VERIFIER phase runs in a separate image, reads that journal and the agent's handoff through one harness handle, and RESUMES the operator from the carried state the handoff carries.

The resume is seeded by the handoff, which is what makes the splice load-bearing rather than ceremonial. The resume Tape refuses every read before the splice point, so nothing about the first phase reaches the second except through the carry.

## The two discovery values

Neither value is stated in `instruction.md` and neither appears in any agent-visible byte. Both are established by the agent phase and read back by the verifier phase.

| value | what it is | how the agent phase establishes it |
|---|---|---|
| splice point | absolute sample offset {splice_offset} | the driver halts at the first whole block boundary at which cumulative outputs times the window reaches the frozen splice charge budget of {budget} |
| carried-state digest | `{carried_digest}` | the driver digests the canonical encoding of the carry the kernel held at that offset |

The splice point falls out of the drawn block lengths rather than out of any stated number, so it is discovered by running the first phase and not by reading the statement. The carried-state digest covers an accumulator that is the operator's own output at the last sample of the agent phase, so it exists only after the first phase has actually been run.

Both gate a REQUIRED checker. `splice_point_harness_established` compares the halt row, the handoff and the verifier's own re-derivation from the frozen stream. `carried_state_digest_matches` compares the halt row, the handoff, the digest recomputed over the state the handoff actually carries, and the frozen operator's own carried state.

## The insight the reward measures

The frozen operator is a length-{window} finite impulse response whose weights are the geometric sequence `ratio ** j mod modulus`. A geometric weight vector is exactly the case in which the convolution collapses into a first order recurrence, because multiplying the previous output by the ratio reindexes every term by one:

`y[i] = ( ratio * y[i-1] + x[i] - (ratio ** window mod modulus) * x[i - window] ) mod modulus`

That charges two reads per output once the block is window-deep, against up to {window} for the direct transcription. Every intermediate is an exact integer modulo a prime, so this is not an approximation of the reference: it is the same function.

## The measured trajectory

| kernel | resume charge | speedup | reward |
|---|---|---|---|
| frozen reference | {reference_charge} | {baseline_speedup} | 0.0 |
| partly fused, three reads per output | {partial_charge} | {partial_speedup} | {partial_reward} |
| fused first order recurrence | {fused_charge} | {target_speedup} | 1.0 |

Each row is a real two-phase run of the bundle's own drivers over the bundle's own frozen bytes, recorded under `solution/fixtures/`. The middle row is what shows the reward is continuous rather than a pass or a fail.

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F5 is registered and unscoped and carries no measured family baseline or target, so none is invented. The reward schema is bound in full and the two normalisation numbers below are measured on this slot's own frozen substrate; they are instance-local and are NOT family anchors, under gap `gap-oer-28-instance-local-anchors-not-family-anchors`.

| quantity | value |
|---|---|
| instance_baseline_speedup | {baseline_speedup} |
| instance_target_speedup | {target_speedup} |
| reference resume charge, reads | {reference_charge} |
| fused resume charge, reads | {fused_charge} |

raw = (agent_metric - instance_baseline_speedup) / (instance_target_speedup - instance_baseline_speedup)

score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and beating it also scores 1.0.

## What the grader refuses, and what it does not claim

The rejecting fixtures under `solution/fixtures/planted/` are frozen runs, each derived from the accepting run by exactly one named mutation. `splice_offset_plus_one` moves the splice point by one sample and nothing else. `carried_digest_one_nibble` moves the carried-state digest by one hexadecimal character and nothing else. `agent_phase_skipped` removes the harness journal and leaves the handoff whole, which is the shape of a synthesised handoff. `journal_row_edited` moves one recorded output digest inside the journal and leaves the chain as the driver wrote it. `kernel_swapped_after_splice` presents a different kernel to the verifier than the one the agent phase ran.

What is NOT claimed, and is recorded as gap `gap-oer-28-journal-forgeable-by-doing-the-work`: the journal carries no secret. It is bound to the operator semantics, to the frozen stream and to the digest of the graded kernel, so a solver who reimplements the driver exactly can author a journal that recomputes. That solver has by construction executed the operator over the whole agent-phase prefix under the graded kernel, which is the work the phase exists to require, so the binding is structural rather than cryptographic. No checker claims unforgeability.

## The clock

THE HARNESS OWNS THE COUNTER. There is no clock on the graded path at all. `environment/opstream.py` charges one read per `Tape.at` call and the verifier reads the total off its own counter after its own resume run. No checker calls time, datetime, perf_counter or monotonic, directly or transitively, and `recompute.py` reads no clock either.

## Substrate limit

The graded speedup is a ratio of instrumented read counts and not a ratio of wall clock durations, declared under gap `gap-oer-28-substrate-counts-reads-not-seconds`. That is the right choice because it makes the grading reproducible, and it never passes as a timing measurement.
"""


def build_truth(ground, measured, partial_reward):
    params = ground["params"]
    return TRUTH_TEMPLATE.format(
        banner=BANNER,
        source=SOURCE,
        modulus=params["modulus"],
        window=params["window"],
        budget=params["splice_charge_budget"],
        splice_offset=measured["splice_offset"],
        carried_digest=measured["carried_state_sha256"],
        reference_charge=measured["reference_resume_charge"],
        fused_charge=measured["fused_resume_charge"],
        partial_charge=measured["partial_resume_charge"],
        baseline_speedup=measured["instance_baseline_speedup"],
        target_speedup=measured["instance_target_speedup"],
        partial_speedup=measured["partial_speedup"],
        partial_reward=partial_reward,
    )


def build_rubrics(measured):
    return {
        "banner": BANNER,
        "source": SOURCE,
        "schema": "forge.solution_rubrics/v1",
        "judged": "the solution against its reference answer",
        "criteria": [
            {
                "id": "splice-point-established-by-the-agent-phase",
                "statement": "The handoff splices at the offset the agent-phase driver halted at, which the verifier re-derives for itself from the frozen stream.",
                "comparison": "integer equality against the harness halt row and the verifier's own re-derivation",
                "reference_value": measured["splice_offset"],
            },
            {
                "id": "carried-state-digest-recomputes",
                "statement": "The carried-state digest equals the digest recomputed over the state the handoff actually carries and the digest the frozen operator produces at the splice.",
                "comparison": "string equality across four independent readings",
                "reference_value": measured["carried_state_sha256"],
            },
            {
                "id": "output-exact-across-both-phases",
                "statement": "The composed output rollup over the agent phase and the verifier's own resume equals the frozen reference rollup.",
                "comparison": "string equality, no tolerance, exact integer arithmetic",
                "reference_value": measured["stream_output_rollup"],
            },
            {
                "id": "resume-charge-reaches-the-target",
                "statement": "The kernel charges no more reads on the resume segment than the fused reference does.",
                "comparison": "verifier-counted resume charge is <= reference_value",
                "reference_value": measured["fused_resume_charge"],
            },
            {
                "id": "speedup-reaches-the-instance-target",
                "statement": "The verifier-computed speedup reaches the instance target.",
                "comparison": "verifier-computed speedup is >= reference_value",
                "reference_value": measured["instance_target_speedup"],
            },
            {
                "id": "kernel-identity-stable",
                "statement": "The kernel the agent phase ran is the kernel the verifier grades.",
                "comparison": "sha256 equality between the journal open row and the graded file",
                "reference_value": "sha256 of solution/reference.py FUSED_KERNEL as installed",
            },
        ],
    }


TEST_TEMPLATE = '''"""{banner}

Source: {source}. Regenerate with solution/recompute.py.

One compiled test per graded checker, plus the end-to-end reward tests. Every test drives the
real checkers in tests/checkers.py over frozen fixtures that are real two-phase runs of the
bundle's own drivers. No test reads a clock; the cost model is an instrumented read counter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
BUNDLE = TESTS.parent
sys.path.insert(0, str(TESTS))

import checkers  # noqa: E402
import grade  # noqa: E402

FIXTURES = BUNDLE / 'solution' / 'fixtures'
REFERENCE = FIXTURES / 'reference_run'
PLANTED = FIXTURES / 'planted'

TARGET_SPEEDUP = {target_speedup}
PARTIAL_REWARD = {partial_reward}

# The frozen fixtures live under solution/, which tests/Dockerfile deliberately does not copy
# into the verifier image, so a run inside that image reports these as skipped rather than as
# failures. They run wherever the whole bundle is resolvable, which is where they are evidence.
pytestmark = pytest.mark.skipif(
    not REFERENCE.is_dir(),
    reason='solution/fixtures is not resolvable from here; the verifier image omits solution/',
)


def _harness(root):
    root = Path(root)
    return grade.build_harness(
        BUNDLE, root / 'workspace', root / 'harness', root / 'app' / 'submission.py'
    )


def _verdict(root):
    bound = grade.load_bound(BUNDLE)
    return grade.score(_harness(root), bound, grade.load_bound_mapping(BUNDLE))


def _outcomes(root=REFERENCE):
    bound = grade.load_bound(BUNDLE)
    return {{row.ident: row for row in grade.run_all(_harness(root), bound)}}


def test_every_checker_is_declared():
    declared = {{ident for ident, _selector in grade.CHECKER_ORDER}}
    assert declared == set(_outcomes())


def test_frozen_inputs_unmodified():
    outcome = _outcomes()['frozen_inputs_unmodified']
    assert outcome.passed, outcome.detail


def test_agent_phase_journal_present():
    outcome = _outcomes()['agent_phase_journal_present']
    assert outcome.passed, outcome.detail


def test_journal_chain_intact():
    outcome = _outcomes()['journal_chain_intact']
    assert outcome.passed, outcome.detail


def test_splice_point_harness_established():
    outcome = _outcomes()['splice_point_harness_established']
    assert outcome.passed, outcome.detail


def test_carried_state_digest_matches():
    outcome = _outcomes()['carried_state_digest_matches']
    assert outcome.passed, outcome.detail


def test_kernel_identity_stable_across_splice():
    outcome = _outcomes()['kernel_identity_stable_across_splice']
    assert outcome.passed, outcome.detail


def test_block_coverage_ordered():
    outcome = _outcomes()['block_coverage_ordered']
    assert outcome.passed, outcome.detail


def test_resume_output_exact():
    outcome = _outcomes()['resume_output_exact']
    assert outcome.passed, outcome.detail


def test_charge_not_substituted():
    outcome = _outcomes()['charge_not_substituted']
    assert outcome.passed, outcome.detail


def test_reference_scores_full_reward():
    verdict = _verdict(REFERENCE)
    assert verdict['reward'] == 1.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == 'graded'
    assert verdict['metric']['speedup'] >= TARGET_SPEEDUP


def test_baseline_run_is_graded_at_zero_rather_than_refused():
    verdict = _verdict(FIXTURES / 'baseline_run')
    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)
    assert verdict['failed_checker'] is None
    assert verdict['reward'] == 0.0


def test_partial_run_scores_between_the_anchors():
    verdict = _verdict(FIXTURES / 'partial_run')
    assert verdict['reason'] == 'graded', json.dumps(verdict, sort_keys=True)
    assert verdict['reward'] == PARTIAL_REWARD
    assert 0.0 < verdict['reward'] < 1.0


def test_splice_offset_plus_one_is_refused():
    verdict = _verdict(PLANTED / 'splice_offset_plus_one')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_SPLICE_NOT_ESTABLISHED
    assert verdict['failed_checker'] == 'splice_point_harness_established'


def test_carried_digest_one_nibble_is_refused():
    verdict = _verdict(PLANTED / 'carried_digest_one_nibble')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_CARRIED_DIGEST_FORGED
    assert verdict['failed_checker'] == 'carried_state_digest_matches'


def test_agent_phase_skipped_is_refused():
    verdict = _verdict(PLANTED / 'agent_phase_skipped')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_AGENT_PHASE_ABSENT
    assert verdict['failed_checker'] == 'agent_phase_journal_present'


def test_journal_row_edited_is_refused():
    verdict = _verdict(PLANTED / 'journal_row_edited')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_CHAIN_BROKEN
    assert verdict['failed_checker'] == 'journal_chain_intact'


def test_kernel_swapped_after_splice_is_refused():
    verdict = _verdict(PLANTED / 'kernel_swapped_after_splice')
    assert verdict['reward'] == 0.0, json.dumps(verdict, sort_keys=True)
    assert verdict['reason'] == checkers.REASON_KERNEL_SWAPPED
    assert verdict['failed_checker'] == 'kernel_identity_stable_across_splice'


def test_the_two_near_misses_move_one_field_and_no_other_byte():
    accepting = json.loads((REFERENCE / 'workspace' / 'handoff.json').read_text(encoding='utf-8'))
    for name, field in (
        ('splice_offset_plus_one', 'splice_offset'),
        ('carried_digest_one_nibble', 'carried_state_sha256'),
    ):
        near = json.loads(
            (PLANTED / name / 'workspace' / 'handoff.json').read_text(encoding='utf-8')
        )
        assert set(near) == set(accepting)
        moved = [key for key in accepting if near[key] != accepting[key]]
        assert moved == [field], (name, moved)
        journal = (PLANTED / name / 'harness' / 'phase_a.jsonl').read_bytes()
        assert journal == (REFERENCE / 'harness' / 'phase_a.jsonl').read_bytes()
        kernel = (PLANTED / name / 'app' / 'submission.py').read_bytes()
        assert kernel == (REFERENCE / 'app' / 'submission.py').read_bytes()
'''


def build_tests(measured, partial_reward):
    return TEST_TEMPLATE.format(
        banner=BANNER,
        source=SOURCE,
        target_speedup=measured["instance_target_speedup"],
        partial_reward=partial_reward,
    )


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def emit(check):
    ground = load_grounding()
    drift = []

    def put(relative, text):
        path = BUNDLE / relative
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if check:
            if current != text:
                drift.append(relative)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    stream_text = json.dumps(build_stream(ground), indent=2, sort_keys=True) + "\n"
    params_text = json.dumps(build_params(ground), indent=2, sort_keys=True) + "\n"
    put("environment/stream.json", stream_text)
    put("environment/params.json", params_text)
    if check and drift:
        # The frozen substrate has to be on disk before anything can be measured over it.
        sys.stderr.write("drift in the frozen substrate: " + ", ".join(drift) + "\n")
        return 1

    measured = derive(ground)
    bound = build_bound(ground, measured)
    span = bound["instance_target_speedup"] - bound["instance_baseline_speedup"]
    partial_reward = round(
        min(max((measured["partial_speedup"] - bound["instance_baseline_speedup"]) / span, 0.0), 1.0),
        6,
    )

    put("tests/bound.json", json.dumps(bound, indent=2, sort_keys=True) + "\n")
    put("solution/solve.sh", build_solve())
    put("solution/TRUTH.md", build_truth(ground, measured, partial_reward))
    put("solution/rubrics.json", json.dumps(build_rubrics(measured), indent=2, sort_keys=True) + "\n")
    put("tests/test_output.py", build_tests(measured, partial_reward))

    if not check:
        plant(measured)

    if check and drift:
        sys.stderr.write("drift: " + ", ".join(sorted(drift)) + "\n")
        return 1
    if not check:
        print(
            json.dumps(
                {
                    "splice_offset": measured["splice_offset"],
                    "splice_block_index": measured["splice_block_index"],
                    "carried_state_sha256": measured["carried_state_sha256"],
                    "reference_resume_charge": measured["reference_resume_charge"],
                    "fused_resume_charge": measured["fused_resume_charge"],
                    "partial_resume_charge": measured["partial_resume_charge"],
                    "instance_target_speedup": measured["instance_target_speedup"],
                    "partial_reward": partial_reward,
                    "stream_output_rollup": measured["stream_output_rollup"],
                    "total_samples": measured["total_samples"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description="derive every generated artifact of OER-28")
    parser.add_argument("--check", action="store_true", help="report drift and write nothing")
    args = parser.parse_args()
    return emit(args.check)


if __name__ == "__main__":
    raise SystemExit(main())

"""The nine graded checkers. Pure, deterministic, and none of them reads a clock.

Every number a checker compares comes from one of two places and never from a third:

  1. state a PHASE actually produced, reached through the `Harness` handle below. Two phases
     feed it. The AGENT phase left the harness journal at the harness log root, chained row by
     row by `environment/phase_a.py`, and left the handoff in the workspace. The VERIFIER phase
     re-ran the resume segment itself, through `tests/runner.py`, in an isolated subprocess over
     the BUNDLE's own frozen stream and parameters, and the read charge in that telemetry is a
     number this verifier counted rather than one the submission reported.

  2. a bound constant handed in as the `Bound` argument, read by `tests/grade.py` from
     `tests/bound.json`, which `solution/recompute.py` derived from `solution/grounding.yaml`.

The two DISCOVERY values, the splice point and the carried-state digest, are established by the
agent phase and read here. Neither is stated in the task statement. `splice_point_harness_
established` and `carried_state_digest_matches` are both required, and each fails when its value
moves by the smallest step that value admits.

Imports are confined to the allowlist: json, math, hashlib, pathlib, dataclasses and typing.
Nothing here imports the submission, opens a socket, consults an environment secret, draws from
a random source, or reads a file the author planted with an answer in it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Machine-readable zero reasons. A downstream grader branches on these strings.
# ---------------------------------------------------------------------------
REASON_FROZEN_INPUTS = "frozen-inputs-modified"
REASON_AGENT_PHASE_ABSENT = "agent-phase-absent"
REASON_CHAIN_BROKEN = "journal-chain-broken"
REASON_SPLICE_NOT_ESTABLISHED = "splice-point-not-harness-established"
REASON_CARRIED_DIGEST_FORGED = "carried-state-digest-forged"
REASON_KERNEL_SWAPPED = "kernel-swapped-across-splice"
REASON_BLOCKS_UNACCOUNTED = "blocks-unaccounted"
REASON_OUTPUT_DIVERGED = "numerical-output-diverged"
REASON_CHARGE_SUBSTITUTED = "charge-substituted"


@dataclass(frozen=True)
class Bound:
    """Every constant the checkers compare against. Handed in, never authored here."""

    stream_sha256: str
    params_sha256: str
    block_count: int
    total_samples: int
    window: int
    splice_charge_budget: int
    splice_offset: int
    splice_block_index: int
    carried_state_sha256: str
    stream_output_rollup: str
    reference_resume_charge: int
    instance_baseline_speedup: float
    instance_target_speedup: float
    journal_genesis: str
    readout_tolerance: float

    @staticmethod
    def from_mapping(payload: Dict[str, Any]) -> "Bound":
        return Bound(
            stream_sha256=str(payload["stream_sha256"]),
            params_sha256=str(payload["params_sha256"]),
            block_count=int(payload["block_count"]),
            total_samples=int(payload["total_samples"]),
            window=int(payload["window"]),
            splice_charge_budget=int(payload["splice_charge_budget"]),
            splice_offset=int(payload["splice_offset"]),
            splice_block_index=int(payload["splice_block_index"]),
            carried_state_sha256=str(payload["carried_state_sha256"]),
            stream_output_rollup=str(payload["stream_output_rollup"]),
            reference_resume_charge=int(payload["reference_resume_charge"]),
            instance_baseline_speedup=float(payload["instance_baseline_speedup"]),
            instance_target_speedup=float(payload["instance_target_speedup"]),
            journal_genesis=str(payload["journal_genesis"]),
            readout_tolerance=float(payload["readout_tolerance"]),
        )


@dataclass
class Harness:
    """The real handle across the splice. Every read below is of state a phase produced.

    `journal` is the agent phase's own chained record, exactly as `environment/phase_a.py`
    wrote it. `resume` is the verifier phase's own telemetry, emitted by the isolated resume
    process this verifier started. `handoff` and `kernel_sha256` are what the agent phase left
    in the shared workspace. `derived_splice_offset` is what the verifier itself re-derived from
    the frozen stream, so the splice point is checked against an independent recomputation and
    not only against a constant.
    """

    journal: List[Dict[str, Any]] = field(default_factory=list)
    resume: List[Dict[str, Any]] = field(default_factory=list)
    handoff: Dict[str, Any] = field(default_factory=dict)
    kernel_sha256: str = ""
    bundle_stream_sha256: str = ""
    bundle_params_sha256: str = ""
    derived_splice_offset: Optional[int] = None
    derived_block_starts: List[int] = field(default_factory=list)
    resume_failure: str = ""

    # -- journal readers -----------------------------------------------------
    def rows(self, kind: str) -> List[Dict[str, Any]]:
        return [row for row in self.journal if row.get("kind") == kind]

    def open_row(self) -> Dict[str, Any]:
        found = self.rows("open")
        return found[0] if found else {}

    def block_rows(self) -> List[Dict[str, Any]]:
        return self.rows("block")

    def halt_row(self) -> Dict[str, Any]:
        found = self.rows("halt")
        return found[-1] if found else {}

    # -- resume readers ------------------------------------------------------
    def resume_blocks(self) -> List[Dict[str, Any]]:
        return [row for row in self.resume if row.get("kind") == "block"]

    def resume_summary(self) -> Dict[str, Any]:
        found = [row for row in self.resume if row.get("kind") == "summary"]
        return found[-1] if found else {}


@dataclass
class Outcome:
    """One checker's verdict. `reason` is empty exactly when the checker passed."""

    ident: str
    passed: bool
    reason: str
    detail: str
    value: Any = None


def _pass(ident: str, detail: str, value: Any = None) -> Outcome:
    return Outcome(ident, True, "", detail, value)


def _fail(ident: str, reason: str, detail: str, value: Any = None) -> Outcome:
    return Outcome(ident, False, reason, detail, value)


# ---------------------------------------------------------------------------
# Quantities the VERIFIER computes. Never a number the submission reported.
# ---------------------------------------------------------------------------


def chain_step(previous: str, row: Dict[str, Any]) -> str:
    """One link of the journal hash chain. The same arithmetic the driver used."""
    body = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((previous + "\n" + body).encode("utf-8")).hexdigest()


def recomputed_chain(harness: Harness, genesis: str) -> List[str]:
    """Recompute every link from the genesis constant over the journal's own row bytes."""
    links: List[str] = []
    previous = genesis
    for row in harness.journal:
        shaped = {key: value for key, value in row.items() if key != "chain"}
        previous = chain_step(previous, shaped)
        links.append(previous)
    return links


def composed_rollup(harness: Harness) -> str:
    """The whole-stream output rollup, half from the agent phase and half from the resume.

    This is the correctness gate's own recomputation. The agent phase's per-block output digests
    are read out of the harness journal and the resume's are read out of telemetry this verifier
    produced, so the rollup spans both phases and is not a quantity either phase could assert on
    its own.
    """
    digests = [str(row.get("out_sha256", "")) for row in harness.block_rows()]
    digests += [str(row.get("out_sha256", "")) for row in harness.resume_blocks()]
    return hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()


def resume_charge(harness: Harness) -> Optional[int]:
    """The graded cost: the read count the VERIFIER's own Tape charged on the resume segment."""
    summary = harness.resume_summary()
    if not summary or summary.get("reads") is None:
        return None
    return int(summary["reads"])


def speedup(harness: Harness, bound: Bound) -> float:
    """The graded metric: reference charge over agent charge on the resume segment."""
    charged = resume_charge(harness)
    if not charged or charged <= 0:
        return 0.0
    return round(bound.reference_resume_charge / float(charged), 6)


def carry_digest(carry: Dict[str, Any]) -> str:
    """sha256 over the one canonical carry encoding the operator fixes."""
    shaped = {
        "accumulator": int(carry["accumulator"]),
        "history": [int(value) for value in carry["history"]],
    }
    body = json.dumps(shaped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The nine checkers. Exactly one reduction each.
# ---------------------------------------------------------------------------


def check_frozen_inputs_unmodified(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The frozen stream and the frozen operator parameters are the bound bytes."""
    ident = "frozen_inputs_unmodified"
    if harness.bundle_stream_sha256 != bound.stream_sha256:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the frozen stream on disk digests to "
            + harness.bundle_stream_sha256
            + ", bound is "
            + bound.stream_sha256,
        )
    if harness.bundle_params_sha256 != bound.params_sha256:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the frozen parameters on disk digest to "
            + harness.bundle_params_sha256
            + ", bound is "
            + bound.params_sha256,
        )
    if len(harness.derived_block_starts) != bound.block_count:
        return _fail(
            ident,
            REASON_FROZEN_INPUTS,
            "the frozen stream carries "
            + str(len(harness.derived_block_starts))
            + " blocks, bound is "
            + str(bound.block_count),
        )
    opened = harness.open_row()
    if opened:
        declared = str(opened.get("stream_sha256", ""))
        if declared != bound.stream_sha256:
            return _fail(
                ident,
                REASON_FROZEN_INPUTS,
                "the agent phase declares stream digest "
                + repr(declared)
                + ", bound is "
                + bound.stream_sha256,
            )
        if str(opened.get("params_sha256", "")) != bound.params_sha256:
            return _fail(
                ident,
                REASON_FROZEN_INPUTS,
                "the agent phase declares parameters digest "
                + repr(opened.get("params_sha256"))
                + ", bound is "
                + bound.params_sha256,
            )
    return _pass(ident, "both frozen inputs match their bound digests", bound.stream_sha256)


def check_agent_phase_journal_present(harness: Harness, bound: Bound) -> Outcome:
    """ABSENCE. An agent phase actually ran, and it left the record only a run can leave.

    This is the checker that separates a run from a synthesis. A handoff without a journal is a
    claim about a phase that produced nothing, and it is graded as the phase not having run
    rather than as a submission that scored badly.
    """
    ident = "agent_phase_journal_present"
    if not harness.journal:
        return _fail(
            ident,
            REASON_AGENT_PHASE_ABSENT,
            "no harness journal was found, so no agent phase produced the state the handoff "
            "claims to carry",
        )
    if not harness.open_row():
        return _fail(
            ident,
            REASON_AGENT_PHASE_ABSENT,
            "the harness journal carries no open row, so it does not record a phase that started",
        )
    if not harness.block_rows():
        return _fail(
            ident,
            REASON_AGENT_PHASE_ABSENT,
            "the harness journal carries no block row, so the agent phase produced no output",
        )
    if not harness.halt_row():
        return _fail(
            ident,
            REASON_AGENT_PHASE_ABSENT,
            "the harness journal carries no halt row, so the agent phase never reached a splice",
        )
    if not isinstance(harness.handoff, dict) or not harness.handoff:
        return _fail(
            ident,
            REASON_AGENT_PHASE_ABSENT,
            "the workspace carries no handoff, so nothing was handed across the splice",
        )
    opened = harness.open_row()
    if int(opened.get("splice_charge_budget", -1)) != bound.splice_charge_budget:
        return _fail(
            ident,
            REASON_AGENT_PHASE_ABSENT,
            "the agent phase ran under splice charge budget "
            + repr(opened.get("splice_charge_budget"))
            + " against the frozen budget "
            + str(bound.splice_charge_budget),
        )
    return _pass(
        ident,
        "the agent phase left "
        + str(len(harness.block_rows()))
        + " block rows and one halt row",
        len(harness.block_rows()),
    )


def check_journal_chain_intact(harness: Harness, bound: Bound) -> Outcome:
    """INVARIANT. The journal recomputes, link by link, from the genesis constant.

    Every row is chained over the previous link and its own canonical bytes, so a row edited
    after the phase ran breaks the chain at that row and at every row after it. The handoff also
    carries the final link, which binds what the agent handed over to the record it handed it
    from.
    """
    ident = "journal_chain_intact"
    if not harness.journal:
        return _fail(ident, REASON_CHAIN_BROKEN, "there is no journal to recompute")
    if str(harness.open_row().get("genesis", "")) != bound.journal_genesis:
        return _fail(
            ident,
            REASON_CHAIN_BROKEN,
            "the journal declares genesis "
            + repr(harness.open_row().get("genesis"))
            + " against the bound genesis "
            + repr(bound.journal_genesis),
        )
    links = recomputed_chain(harness, bound.journal_genesis)
    for position, (row, link) in enumerate(zip(harness.journal, links)):
        recorded = str(row.get("chain", ""))
        if recorded != link:
            return _fail(
                ident,
                REASON_CHAIN_BROKEN,
                "journal row "
                + str(position)
                + " of kind "
                + repr(row.get("kind"))
                + " records chain link "
                + recorded[:16]
                + " and recomputes to "
                + link[:16],
                position,
            )
    declared = str(harness.handoff.get("journal_chain", ""))
    if declared != links[-1]:
        return _fail(
            ident,
            REASON_CHAIN_BROKEN,
            "the handoff carries journal chain "
            + repr(declared[:16])
            + " and the journal closes at "
            + links[-1][:16],
        )
    return _pass(ident, "all " + str(len(links)) + " journal links recompute", links[-1])


def check_splice_point_harness_established(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The splice point is the one the agent phase established, not one anybody chose.

    DISCOVERY VALUE. The splice point is nowhere in the task statement. It is the absolute
    sample offset at which the agent phase's harness-owned work counter first reaches the frozen
    splice charge budget, and it falls out of the drawn block lengths rather than out of any
    stated number. Three readings have to agree: what the agent phase RECORDED in its halt row,
    what the handoff DECLARED, and what this verifier RE-DERIVED for itself from the frozen
    stream. A splice guessed rather than run disagrees with the third; a handoff edited after the
    phase ran disagrees with the first.
    """
    ident = "splice_point_harness_established"
    halt = harness.halt_row()
    recorded = halt.get("splice_offset")
    if recorded is None:
        return _fail(
            ident,
            REASON_SPLICE_NOT_ESTABLISHED,
            "the halt row records no splice offset, so no splice point was established",
        )
    derived = harness.derived_splice_offset
    if derived is None:
        return _fail(
            ident,
            REASON_SPLICE_NOT_ESTABLISHED,
            "the verifier could not re-derive a splice offset from the frozen stream",
        )
    if int(derived) != bound.splice_offset:
        return _fail(
            ident,
            REASON_SPLICE_NOT_ESTABLISHED,
            "the splice offset re-derived from the frozen stream is "
            + str(derived)
            + " and the bound value is "
            + str(bound.splice_offset),
            derived,
        )
    if int(recorded) != bound.splice_offset:
        return _fail(
            ident,
            REASON_SPLICE_NOT_ESTABLISHED,
            "the agent phase halted at sample offset "
            + str(recorded)
            + " and the harness-established splice point is at "
            + str(bound.splice_offset),
            recorded,
        )
    declared = harness.handoff.get("splice_offset")
    if declared is None or int(declared) != bound.splice_offset:
        return _fail(
            ident,
            REASON_SPLICE_NOT_ESTABLISHED,
            "the handoff splices at "
            + repr(declared)
            + " and the agent phase halted at "
            + str(bound.splice_offset),
            declared,
        )
    if len(harness.block_rows()) != bound.splice_block_index:
        return _fail(
            ident,
            REASON_SPLICE_NOT_ESTABLISHED,
            "the agent phase covered "
            + str(len(harness.block_rows()))
            + " blocks against the "
            + str(bound.splice_block_index)
            + " that reaching the splice point requires",
            len(harness.block_rows()),
        )
    if int(halt.get("cumulative_work", -1)) < bound.splice_charge_budget:
        return _fail(
            ident,
            REASON_SPLICE_NOT_ESTABLISHED,
            "the agent phase halted at cumulative work "
            + repr(halt.get("cumulative_work"))
            + " which does not reach the frozen splice charge budget "
            + str(bound.splice_charge_budget),
        )
    return _pass(
        ident,
        "the splice point agrees across the halt row, the handoff and the verifier's own "
        "re-derivation",
        bound.splice_offset,
    )


def check_carried_state_digest_matches(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The carried-state digest is the one the agent phase's own carry produced.

    DISCOVERY VALUE. The digest is nowhere in the task statement. It is sha256 over the
    canonical encoding of the carry the kernel held at the splice point, and the accumulator
    inside that carry is the operator's output at the last sample of the agent phase, so it
    exists only after the first phase has actually been run. Four readings have to agree: the
    digest the agent phase RECORDED in its halt row, the digest the handoff DECLARED, the digest
    this verifier RECOMPUTES over the carry the handoff actually carries, and the bound digest
    derived from the frozen operator. A forged digest fails the third; a forged carry fails the
    fourth and then fails the correctness gate as well, because the resume is seeded from it.
    """
    ident = "carried_state_digest_matches"
    halt = harness.halt_row()
    recorded = str(halt.get("carried_state_sha256", ""))
    if not recorded:
        return _fail(
            ident,
            REASON_CARRIED_DIGEST_FORGED,
            "the halt row records no carried-state digest",
        )
    declared = str(harness.handoff.get("carried_state_sha256", ""))
    carry = harness.handoff.get("carried_state")
    if not isinstance(carry, dict) or "accumulator" not in carry or "history" not in carry:
        return _fail(
            ident,
            REASON_CARRIED_DIGEST_FORGED,
            "the handoff carries no carried state of the bound shape, so no digest can be "
            "recomputed over it",
        )
    if len(carry.get("history") or []) != bound.window:
        return _fail(
            ident,
            REASON_CARRIED_DIGEST_FORGED,
            "the handed-over history is "
            + str(len(carry.get("history") or []))
            + " long against a window of "
            + str(bound.window),
        )
    try:
        recomputed = carry_digest(carry)
    except (TypeError, ValueError, KeyError):
        return _fail(
            ident,
            REASON_CARRIED_DIGEST_FORGED,
            "the handed-over carried state does not encode canonically, so no digest exists "
            "over it",
        )
    if declared != recomputed:
        return _fail(
            ident,
            REASON_CARRIED_DIGEST_FORGED,
            "the handoff declares carried-state digest "
            + declared
            + " and the state it actually carries digests to "
            + recomputed,
            declared,
        )
    if recorded != recomputed:
        return _fail(
            ident,
            REASON_CARRIED_DIGEST_FORGED,
            "the agent phase recorded carried-state digest "
            + recorded
            + " and the handed-over state digests to "
            + recomputed,
            recorded,
        )
    if recomputed != bound.carried_state_sha256:
        return _fail(
            ident,
            REASON_CARRIED_DIGEST_FORGED,
            "the carried state at the splice point digests to "
            + recomputed
            + " and the frozen operator's own carried state digests to "
            + bound.carried_state_sha256,
            recomputed,
        )
    return _pass(
        ident,
        "the carried-state digest agrees across the halt row, the handoff, the verifier's own "
        "recomputation over the handed-over state, and the frozen operator",
        recomputed,
    )


def check_kernel_identity_stable_across_splice(harness: Harness, bound: Bound) -> Outcome:
    """DIVERGENCE. The kernel that ran the first phase is the kernel being graded.

    The splice is a seam, and a seam invites a swap: run the first phase under one kernel and
    present a different one to the verifier. The agent phase digested the kernel file as it
    stood when it ran, and the verifier digests the kernel file it is about to grade. A
    divergence between those two is the swap, and it is named rather than absorbed.
    """
    ident = "kernel_identity_stable_across_splice"
    if not harness.kernel_sha256:
        return _fail(
            ident,
            REASON_KERNEL_SWAPPED,
            "no kernel was found in the workspace, so there is nothing to grade at the far side "
            "of the splice",
        )
    ran = str(harness.open_row().get("kernel_sha256", ""))
    if not ran:
        return _fail(
            ident,
            REASON_KERNEL_SWAPPED,
            "the agent phase recorded no kernel digest, so the phase cannot be tied to a kernel",
        )
    if ran != harness.kernel_sha256:
        return _fail(
            ident,
            REASON_KERNEL_SWAPPED,
            "the agent phase ran kernel "
            + ran
            + " and the verifier is grading kernel "
            + harness.kernel_sha256,
            [ran, harness.kernel_sha256],
        )
    declared = str(harness.handoff.get("kernel_sha256", ""))
    if declared != harness.kernel_sha256:
        return _fail(
            ident,
            REASON_KERNEL_SWAPPED,
            "the handoff names kernel "
            + repr(declared)
            + " and the graded kernel digests to "
            + harness.kernel_sha256,
            declared,
        )
    return _pass(ident, "one kernel identity holds across both phases", harness.kernel_sha256)


def check_block_coverage_ordered(harness: Harness, bound: Bound) -> Outcome:
    """ORDERING. The two phases cover every block of the frozen stream exactly once, in order.

    The agent phase owns blocks zero through the splice, the resume owns the rest, and between
    them there is no gap, no overlap and no reordering. A phase that skipped a block, replayed
    one, or handed over a boundary that is not a block boundary fails here rather than producing
    a rollup that happens to be short.
    """
    ident = "block_coverage_ordered"
    if harness.resume_failure:
        return _fail(
            ident,
            REASON_BLOCKS_UNACCOUNTED,
            "the resume segment produced no telemetry: " + harness.resume_failure,
        )
    agent = [int(row.get("index", -1)) for row in harness.block_rows()]
    resumed = [int(row.get("index", -1)) for row in harness.resume_blocks()]
    covered = agent + resumed
    if covered != list(range(bound.block_count)):
        missing = [index for index in range(bound.block_count) if index not in set(covered)]
        return _fail(
            ident,
            REASON_BLOCKS_UNACCOUNTED,
            "the two phases cover "
            + str(len(covered))
            + " block records against "
            + str(bound.block_count)
            + " blocks in the frozen stream, and the sequence is "
            + ("not contiguous and ascending" if len(covered) == bound.block_count else "short")
            + "; missing "
            + str(len(missing)),
            covered,
        )
    starts = harness.derived_block_starts
    for row in harness.block_rows() + harness.resume_blocks():
        index = int(row.get("index", -1))
        if int(row.get("start", -1)) != starts[index]:
            return _fail(
                ident,
                REASON_BLOCKS_UNACCOUNTED,
                "block "
                + str(index)
                + " is recorded starting at sample "
                + repr(row.get("start"))
                + " and the frozen stream places it at "
                + str(starts[index]),
                index,
            )
    total = sum(int(row.get("length", 0)) for row in harness.block_rows()) + sum(
        int(row.get("length", 0)) for row in harness.resume_blocks()
    )
    if total != bound.total_samples:
        return _fail(
            ident,
            REASON_BLOCKS_UNACCOUNTED,
            "the two phases account for "
            + str(total)
            + " samples against "
            + str(bound.total_samples)
            + " in the frozen stream",
            total,
        )
    return _pass(
        ident,
        "both phases together cover all "
        + str(bound.block_count)
        + " blocks once, in order",
        len(covered),
    )


def check_resume_output_exact(harness: Harness, bound: Bound) -> Outcome:
    """VALUE. The composed output rollup is exactly the frozen reference rollup. Hard gate.

    The operator is exact integer arithmetic modulo a prime, so correctness is byte equality and
    not a tolerance. The rollup spans both phases, and the resume half was produced by this
    verifier from the carry the agent phase handed over, so a wrong carried state shows up here
    even when its digest was recorded consistently.
    """
    ident = "resume_output_exact"
    if harness.resume_failure:
        return _fail(
            ident,
            REASON_OUTPUT_DIVERGED,
            "the resume produced no output: " + harness.resume_failure,
        )
    observed = composed_rollup(harness)
    if observed != bound.stream_output_rollup:
        return _fail(
            ident,
            REASON_OUTPUT_DIVERGED,
            "the composed output rollup over both phases is "
            + observed
            + " and the frozen reference rollup is "
            + bound.stream_output_rollup,
            observed,
        )
    return _pass(ident, "the composed output rollup equals the frozen reference", observed)


def check_charge_not_substituted(harness: Harness, bound: Bound) -> Outcome:
    """EFFECT. The graded charge is the one this verifier counted, and the report agrees.

    The verifier counts the resume charge on its own Tape and would overrule any figure the
    submission printed. This checker exists so that a submission reporting a different number is
    caught rather than silently overruled, because a substitution nobody names is a substitution
    nobody can audit. The semantic floor is the second half: the operator makes every output
    depend on its own sample, so a resume covering N samples cannot honestly charge fewer than N
    reads, and a charge below that floor is a counter that did not count.
    """
    ident = "charge_not_substituted"
    charged = resume_charge(harness)
    if charged is None:
        return _fail(
            ident,
            REASON_CHARGE_SUBSTITUTED,
            "the resume telemetry carries no summary, so no charge was counted",
        )
    produced = int(harness.resume_summary().get("outputs", 0))
    if charged < produced:
        return _fail(
            ident,
            REASON_CHARGE_SUBSTITUTED,
            "the resume charged "
            + str(charged)
            + " reads for "
            + str(produced)
            + " outputs, below the one-read-per-output floor the operator imposes",
            charged,
        )
    reported = harness.handoff.get("reported_resume_charge")
    try:
        reported_value = float(reported)
    except (TypeError, ValueError):
        return _fail(
            ident,
            REASON_CHARGE_SUBSTITUTED,
            "the handoff reports a non-numeric resume charge " + repr(reported),
        )
    if abs(reported_value - charged) > bound.readout_tolerance:
        return _fail(
            ident,
            REASON_CHARGE_SUBSTITUTED,
            "the handoff reports a resume charge of "
            + repr(reported)
            + " and the verifier's own counter charged "
            + str(charged),
            charged,
        )
    return _pass(
        ident,
        "the verifier counted " + str(charged) + " reads and the handoff reports the same",
        charged,
    )


def digest_of(text: str) -> str:
    """sha256 over text, used to bind the reference bytes into the accepting fixture."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a chained or telemetry record file. A malformed line is kept as an empty row."""
    rows: List[Dict[str, Any]] = []
    path = Path(path)
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            rows.append({})
    return rows

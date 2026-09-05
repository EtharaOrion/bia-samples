# FORGE-CANARY-BEGIN
# 0: 385f04fa6f5f3a2b56764d5a085301ce56fb64ccdf2e9b6d06765e3e1a38ac04
# 1: bf6a395ec48060a3a12061e1a461f5f0fa31a8eda9244f593777898236b081fd
# 2: e7d2a0febe8865377a940f267e3afc059d7d96e8c3eb6020fd05683488148198
# 3: 3aa20be5840a351b116e24091038b1b9bd29978c7cef1d44fcc13ab52868a4e3
# FORGE-CANARY-END
# TRUTH.md

GENERATED SECTION. DO NOT HAND-EDIT.

Source: `solution/grounding.yaml`. Regenerate with `solution/recompute.py`.

## What is graded

Speedup of the submitted kernel over the frozen reference kernel, measured on the resume segment that the verifier phase runs for itself, and gated on exact output equality. Direction: higher is better. The correctness gate is a HARD GATE: the operator is exact integer arithmetic modulo 2147483647, so a kernel whose composed output rollup differs from the frozen reference rollup scores exactly 0.0 with the reason `numerical-output-diverged`, however few reads it charged.

The cost model is an instrumented read counter and never a wall clock. Every kernel reaches an input sample only through a Tape, and the Tape charges one read per call. That is what makes the graded number reproducible on any host, and it is declared as a substrate limit rather than presented as a timing measurement.

## The archetype

Two-phase splice. The unit runs in two phases over one shared workspace. The AGENT phase runs `environment/phase_a.py` in the agent image: it feeds the kernel whole blocks in order, charges every read, chains one journal row per block, and halts the instant its own work counter reaches the frozen splice charge budget. The VERIFIER phase runs in a separate image, reads that journal and the agent's handoff through one harness handle, and RESUMES the operator from the carried state the handoff carries.

The resume is seeded by the handoff, which is what makes the splice load-bearing rather than ceremonial. The resume Tape refuses every read before the splice point, so nothing about the first phase reaches the second except through the carry.

## The two discovery values

Neither value is stated in `instruction.md` and neither appears in any agent-visible byte. Both are established by the agent phase and read back by the verifier phase.

| value | what it is | how the agent phase establishes it |
|---|---|---|
| splice point | absolute sample offset 10078 | the driver halts at the first whole block boundary at which cumulative outputs times the window reaches the frozen splice charge budget of 241000 |
| carried-state digest | `dcba165d0dbcaaca71d595cc422798dec85e8d553e846c45b71d90d218b6abcd` | the driver digests the canonical encoding of the carry the kernel held at that offset |

The splice point falls out of the drawn block lengths rather than out of any stated number, so it is discovered by running the first phase and not by reading the statement. The carried-state digest covers an accumulator that is the operator's own output at the last sample of the agent phase, so it exists only after the first phase has actually been run.

Both gate a REQUIRED checker. `splice_point_harness_established` compares the halt row, the handoff and the verifier's own re-derivation from the frozen stream. `carried_state_digest_matches` compares the halt row, the handoff, the digest recomputed over the state the handoff actually carries, and the frozen operator's own carried state.

## The insight the reward measures

The frozen operator is a length-24 finite impulse response whose weights are the geometric sequence `ratio ** j mod modulus`. A geometric weight vector is exactly the case in which the convolution collapses into a first order recurrence, because multiplying the previous output by the ratio reindexes every term by one:

`y[i] = ( ratio * y[i-1] + x[i] - (ratio ** window mod modulus) * x[i - window] ) mod modulus`

That charges two reads per output once the block is window-deep, against up to 24 for the direct transcription. Every intermediate is an exact integer modulo a prime, so this is not an approximation of the reference: it is the same function.

## The measured trajectory

| kernel | resume charge | speedup | reward |
|---|---|---|---|
| frozen reference | 230208 | 1.0 | 0.0 |
| partly fused, three reads per output | 28986 | 7.942041 | 0.630377 |
| fused first order recurrence | 19164 | 12.012523 | 1.0 |

Each row is a real two-phase run of the bundle's own drivers over the bundle's own frozen bytes, recorded under `solution/fixtures/`. The middle row is what shows the reward is continuous rather than a pass or a fail.

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F5 is registered and unscoped and carries no measured family baseline or target, so none is invented. The reward schema is bound in full and the two normalisation numbers below are measured on this slot's own frozen substrate; they are instance-local and are NOT family anchors, under gap `gap-oer-28-instance-local-anchors-not-family-anchors`.

| quantity | value |
|---|---|
| instance_baseline_speedup | 1.0 |
| instance_target_speedup | 12.012523 |
| reference resume charge, reads | 230208 |
| fused resume charge, reads | 19164 |

raw = (agent_metric - instance_baseline_speedup) / (instance_target_speedup - instance_baseline_speedup)

score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and beating it also scores 1.0.

## What the grader refuses, and what it does not claim

The rejecting fixtures under `solution/fixtures/planted/` are frozen runs, each derived from the accepting run by exactly one named mutation. `splice_offset_plus_one` moves the splice point by one sample and nothing else. `carried_digest_one_nibble` moves the carried-state digest by one hexadecimal character and nothing else. `agent_phase_skipped` removes the harness journal and leaves the handoff whole, which is the shape of a synthesised handoff. `journal_row_edited` moves one recorded output digest inside the journal and leaves the chain as the driver wrote it. `kernel_swapped_after_splice` presents a different kernel to the verifier than the one the agent phase ran.

What is NOT claimed, and is recorded as gap `gap-oer-28-journal-forgeable-by-doing-the-work`: the journal carries no secret. It is bound to the operator semantics, to the frozen stream and to the digest of the graded kernel, so a solver who reimplements the driver exactly can author a journal that recomputes. That solver has by construction executed the operator over the whole agent-phase prefix under the graded kernel, which is the work the phase exists to require, so the binding is structural rather than cryptographic. No checker claims unforgeability.

## The clock

THE HARNESS OWNS THE COUNTER. There is no clock on the graded path at all. `environment/opstream.py` charges one read per `Tape.at` call and the verifier reads the total off its own counter after its own resume run. No checker calls time, datetime, perf_counter or monotonic, directly or transitively, and `recompute.py` reads no clock either.

## Substrate limit

The graded speedup is a ratio of instrumented read counts and not a ratio of wall clock durations, declared under gap `gap-oer-28-substrate-counts-reads-not-seconds`. That is the right choice because it makes the grading reproducible, and it never passes as a timing measurement.

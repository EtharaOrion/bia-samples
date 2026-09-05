# FORGE-CANARY-BEGIN
# 0: 85e526b3324350a9def08e6da2274384589440eafce6d57ed5757ef5a97747bd
# 1: 727b2cbf26ec4d45a0641198279fd3217c53e8f53c1a0cdf4e3e8a3bedc6deda
# 2: b4164f1e33b04bdc6764692d0a7038055312d9550f00773ac543da87de5570a7
# 3: 855960322c1033669f393deab7c3e11f291328a940a1dd9677fd2a5c58d22aaa
# FORGE-CANARY-END
# TRUTH.md

GENERATED SECTION. DO NOT HAND-EDIT.

Source: `solution/grounding.yaml`. Regenerate with `solution/recompute.py`.

## What is graded

Verdict agreement over every crossing the producer chain presents at the fence. A verdict agrees only when its decision and its refusal reason both match the verdict the stated fence semantics produce for that crossing at that position in the chain. Direction: higher is better.

The verifier establishes both discovery values for itself. `tests/runner.py` launches `environment/fence_probe.py` against the fence state the verifier image was built with, in an isolated subprocess as a new session leader under a six-key environment allowlist, and every checker is a pure function of the record stream that probe emitted. No number the submission reported reaches the reward.

## The archetype

producer-fence. A fence separates what a producer may emit from what a consumer may admit. The solver has to admit exactly the admissible kinds, refuse the rest with the right reason under a strict precedence, and account for every crossing. Two facts decide the task and neither is in the statement: how many crossings the chain actually presents, and which kinds the registry actually admits.

## The two discovery values, as the built state establishes them

| value | how the build establishes it | how the oracle recovered it |
|---|---|---|
| the fence-crossing count | the segment pool, each segment's width, the chain length and the chain order are all drawn from the generator's integer recurrence, so the count is not the product of any two constants in the generator | walked the links from the head segment to the null link and counted the records presented |
| the admitted-kind set | each declaration is given a status and a seal by the recurrence, and a subset of the seals is deliberately broken, so a declaration can read `admitted` while being retired | took the declarations whose status reads `admitted` and whose seal recomputes over its own declared bytes |

Neither value appears in `instruction.md`, in `task.toml`, in `environment/` or in `tests/`. `tests/bound.json` carries a salted commitment over each of them and never the value, and the checkers recompute both from harness-owned telemetry on every run.

## The reference trace, as it actually ran

| quantity | value |
|---|---|
| crossings the chain presented | 96 |
| records present on disk across every segment file | 168 |
| segments on disk | 11 |
| segments the chain links | 6 |
| registry declarations | 14 |
| declarations whose status reads admitted | 8 |
| kinds the fence admits | 4 |
| crossings admitted | 17 |
| crossings refused `malformed-crossing` | 10 |
| crossings refused `unattested-origin` | 11 |
| crossings refused `kind-not-admitted` | 52 |
| crossings refused `quota-exhausted` | 6 |
| baseline agreement | 0.71875 |
| oracle agreement | 1.0 |

The quota column is why the traversal order is load bearing. Quota is spent in chain order, so the 6 crossings refused `quota-exhausted` are refused because of what the chain admitted before them, and a policy that decides the same records in a different order does not reproduce them.

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F3 carries no measured family baseline or target, so none is invented. The one normalisation number below is measured on this slot's own built state; it is instance-local and is NOT a family anchor, under gap `gap-oer-26-instance-local-anchors-not-family-anchors`.

| quantity | value |
|---|---|
| baseline policy | decisions right, every refusal flattened to one reason |
| baseline agreement | 0.71875 |
| target agreement | 1.0 |

raw = (agreement - baseline_agreement) / (1.0 - baseline_agreement)

score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and the baseline policy scores exactly 0.0 while passing every gate, which is what keeps the reward continuous rather than binary.

## The rejecting fixtures, as they actually graded

| fixture | what it moves | observed reward | observed reason |
|---|---|---|---|
| `admit_everything` | every decision to admit | 0.0 | `fence-breached` |
| `admitted_set_minus_one_kind` | the admitted-kind set only, by dropping one kind | 0.0 | `admitted-set-wrong` |
| `admitted_set_plus_one_kind` | the admitted-kind set only, by adding one refused kind | 0.0 | `admitted-set-wrong` |
| `admitted_set_status_only` | the admitted-kind set only, to the set a status-column reading produces | 0.0 | `admitted-set-wrong` |
| `crossing_count_minus_one` | the fence-crossing count only, by minus one | 0.0 | `crossing-count-unaccounted` |
| `crossing_count_plus_one` | the fence-crossing count only, by plus one | 0.0 | `crossing-count-unaccounted` |
| `quota_ignored` | the quota-exhausted refusals to admissions | 0.0 | `quota-ledger-exceeded` |
| `reasons_flattened` | every refusal reason to kind-not-admitted, decisions untouched | 0.0 | `graded` |
| `reasons_precedence_inverted` | the malformed and unattested refusals to kind-not-admitted | 0.222222 | `graded` |
| `refuse_everything` | every decision to refuse | 0.0 | `blanket-refusal` |
| `verdict_kinds_relabelled` | the echoed record identity only, rotated by one crossing | 0.0 | `verdict-record-divergence` |
| `verdicts_from_disk_order` | the verdict stream to the union of the segment files on disk | 0.0 | `crossing-count-unaccounted` |
| `verdicts_from_wrong_chain_order` | the verdict stream to the chain segments taken in sorted identifier order | 0.0 | `verdict-sequence-disordered` |

## The clock and the traversal

THE HARNESS OWNS THE CLOCK, and on this slot it owns the traversal too. No checker reads a clock and no checker walks the fence inside the grading interpreter. `recompute.py` reads no clock either: every artifact above is derived by arithmetic from the frozen generator and never from a live measurement.

## Substrate limit

The substrate is a deterministic build artifact, declared under gap `gap-oer-26-substrate-is-a-build-artifact-not-a-live-stream`. That is the right choice because it makes grading reproducible, and it never passes as an observation of a running production system.

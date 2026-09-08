# Enforcing a producer fence over a linked crossing stream

A producer emits records across a fence. A consumer behind that fence may admit only some of them. Your job is to stand at the fence, decide every crossing, and account for all of them.

## The two sides

The **producer** side is a chain of segments. Each segment file carries a header line naming the segment and the segment it links to next, then one crossing record per line. The chain starts at the head segment the fence names and ends at the segment whose link is null.

**Presence on disk is not membership of the chain.** The state root carries segment files that nothing links to. A segment nothing points at emitted no crossing at this fence, however plausible its records look, and its records are not crossings. Reading every file in the directory and treating the union as the crossing stream is the first way to get this wrong.

The **consumer** side is the fence registry. It publishes a salt, an epoch, and one declaration per kind. A declaration carries the kind name, a status, a per-kind admission quota, and a seal.

## The fence semantics, fixed here

**A kind is admitted by this fence exactly when both of the following hold, and otherwise it is not:**

1. its registry declaration carries status `admitted`, and
2. its registry seal recomputes, meaning `sha256("oer26.registry-seal/v1|" + kind + "|" + status + "|" + quota)` in lower-case hexadecimal equals the declared seal.

A declaration whose status reads `admitted` while its seal does not recompute is a retired declaration that was never relabelled. It is not admitted. Reading the status column alone is the second way to get this wrong.

**A crossing record is attested** exactly when the first sixteen lower-case hexadecimal characters of `sha256(seq + "|" + kind + "|" + size + "|" + origin + "|" + epoch + "|" + salt)` equal the record's `sig` field, where every component is its plain decimal or string rendering joined by the pipe character and `salt` is the salt the registry publishes.

**A crossing record is well formed** exactly when it carries the field set `seq`, `kind`, `size`, `origin`, `epoch`, `sig` and no other field, where `seq`, `size` and `epoch` are integers, `kind`, `origin` and `sig` are strings, and `size` is strictly positive.

## The decision, and the four refusal reasons

For each crossing, in the order the chain presents it, apply the following tests. **The first test that fires decides the crossing, and no later test is consulted.** The precedence is part of the semantics: a record that is both malformed and of an unadmitted kind is refused as `malformed-crossing` and never as `kind-not-admitted`.

1. `malformed-crossing` if the record is not well formed.
2. `unattested-origin` if the record is well formed but not attested.
3. `kind-not-admitted` if the record is well formed and attested but its kind is not admitted by this fence.
4. `quota-exhausted` if the record is well formed, attested and of an admitted kind, but that kind's per-kind quota has already been spent by crossings admitted **earlier in the chain order**.

If no test fires, the crossing is admitted and it spends one unit of its kind's quota.

Quota is spent in chain order and nowhere else. Two runs that decide the same multiset of records in different orders do not produce the same verdicts, so the traversal order is load bearing rather than incidental.

## The handle

The fence state was built into this image. Read it back through the handle, which is on the interpreter path:

```python
import fencelib

fence = fencelib.open_fence()
fence.registry()            # salt, epoch, and every declaration
fence.head_segment_id()     # the head of the producer chain
fence.segments_on_disk()    # every segment file present, linked or not
fence.chain()               # the segment identifiers the chain actually links, head first
fence.crossings()           # every crossing the chain presents, in presentation order
fence.admitted_kinds()      # the kinds this fence admits, by the rule stated above
fence.attestation(record, salt)
fence.seal_preimage(entry)  # available as fencelib.seal_preimage
```

Nothing in the handle decides a verdict for you. It reports built state.

## What to submit

Write `submission.json` into the workspace root:

```json
{
  "schema": "oer26.submission/v1",
  "crossings_observed": 0,
  "admitted_kinds": ["<every kind this fence admits, sorted>"],
  "admitted": 0,
  "refused": 0,
  "refusals": {"malformed-crossing": 0, "unattested-origin": 0, "kind-not-admitted": 0, "quota-exhausted": 0},
  "verdicts": [
    {"seq": 1, "kind": "<the kind the record carried, or null if it carried none>", "decision": "refuse", "reason": "malformed-crossing"}
  ]
}
```

- `crossings_observed` is how many crossings the chain presented at the fence.
- `verdicts` carries exactly one entry per crossing, in chain order, and each entry echoes the `kind` the record itself carried so that a verdict is provably about the record the traversal reached. An admitted crossing carries `"decision": "admit"` and `"reason": ""`.
- `admitted` plus `refused` must equal `crossings_observed`, and the four counts in `refusals` must sum to `refused`.

## Scoring

One float in `[0.0, 1.0]`, higher is better, never binary. Every zero carries a machine-readable reason. Full reward is exactly `1.0`.

The verifier performs its own traversal of the built fence state in an isolated subprocess and derives the admitted set, the crossing count and every verdict for itself. No number you report becomes a graded number; the reported fields exist so that a substitution is visible rather than ignorable.

A run first passes a gate of required checks, and a failure at the gate scores exactly `0.0` with that check's reason. Past the gate the reward is the clamped normalised agreement between your verdicts and the verifier's, verdict by verdict, where a verdict agrees only when both its decision and its reason match:

`raw = (agreement - baseline_agreement) / (1.0 - baseline_agreement)`, then `score = min(max(raw, 0.0), 1.0)`.

The baseline is a policy that decides every crossing correctly but labels every refusal `kind-not-admitted`. Getting the decisions right and the reasons wrong therefore scores `0.0` rather than most of the reward, and the precedence above is where the reward actually lives.

## Two notes on honesty

The fence state is a deterministic build artifact, not a live production stream. That is deliberate: it makes the grading reproducible, so the same state produces the same verdict on any host. It is never presented as a measurement of a running system.

The traversal is yours to perform however you like and your own timing is not graded. The verifier reads no clock, and neither the crossing count nor the admitted set is anywhere in this document.

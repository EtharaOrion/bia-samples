# FORGE-CANARY-BEGIN
# 0: 4f0dd18445729e1d78db09321ed5caaabf278e8b867022fde6fba351240e60f9
# 1: 11fb8bedcb1ab834dcb6199b8cd3cf9ff1957966af0844941a471cc21b710480
# 2: ec49de538771d42e560b46092371d4b2525f6cb8dd0ae4274baba07bdc1db768
# 3: 767b2ee5c8f2fd3e4684afd5cada4e22d89453034e53f9d637c79e7b61a438b9
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml -->

# TRUTH: OER-11 -- Parser built against a corpus snapshot that moves during the session

Family F13, parse and tokenize pipeline. Archetype AR9, Temporal Reasoning Gap.

## Headline

The corpus snapshot moves from 3 to 7 during the session, silently, and the graded evaluation runs on snapshot 7. Everything else in this task is a consequence.

## What the agent must do

- Ask corpus_api.snapshot_version() again before reusing any parse verdict.
- Re-derive the field-key set from the live snapshot; the key list changed.
- Re-read the evaluation split id; it rotated from split-a to split-b.
- Re-read the token budget; it retightened from 3072 to 2048.
- Rebuild the token stream over six shards rather than four.
- Let the whole scheduled sustain window close instead of stopping on a good eval.

## What defeats a submission

- A parse verdict derived at snapshot 3 and carried forward, caught by DIVERGENCE.
- A number the submission reported, caught by VALUE on provenance.
- A smoothed readout, caught by VALUE on recomputation.
- A one-point dip, caught by INVARIANT.
- An early halt, caught by ABSENCE.
- A cached or absent token stream, caught by EFFECT.
- A stale budget or a stale split, caught by VALUE and ABSENCE respectively.

## Why the handle exists

The move is silent, but asking is possible and documented. Temporal reasoning is the skill under test; a trap with no handle would be measuring luck.

## Ordering basis

Harness-owned snapshot version handles in /logs/harness/snapshot_ledger.json. seq is a monotone integer the harness owns; a higher snapshot_version is a later corpus; equal versions are the same corpus. NO checker reads a wall clock, a locale or a timezone.

## Anchors

anchors_state: absent

baseline_metric: absent. target_metric: absent. gap: gap-oer-per-family-anchors-unmeasured.

The reward schema is fully bound: carrier /logs/verifier/reward.txt as a bare float, companion document /logs/verifier/score.json, closed interval [0.0, 1.0], higher better, never binary, full reward exactly 1.0, formula raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric), clip score = min(max(raw, 0.0), 1.0). Only the two numbers are absent.

Both numbers are read at grade time from the harness-owned anchor record at /logs/harness/anchor_record.json, which this lane does not author. An absent, null or degenerate record resolves to reward 0.0 with reason anchors-unresolved.

## Golden trajectory

| iteration | snapshot asked | action |
|---|---|---|
| 1 | 3 | resolve the corpus and read the field-key set off the live snapshot descriptor |
| 2 | 3 | fit a word vocabulary of 12 types over the training documents, char-fallback for the rest |
| 3 | 3 | build a candidate token stream and measure it against the snapshot-3 budget |
| 4 | 7 | RE-ASK the snapshot version before reusing the iteration-3 parse verdict **(pivot)** |
| 5 | 7 | re-resolve the corpus, re-derive the field-key set, discover snapshot and content_hash |
| 6 | 7 | re-read the evaluation split id and exclude every split-b document from training |
| 7 | 7 | re-read the token budget and truncate the stream to exactly 2048 tokens, one pass |
| 8 | 7 | hand the stream to the frozen recipe and let the whole scheduled sustain window close |

## Declared gaps

- `gap-oer-per-family-anchors-unmeasured`: F13 baseline_metric and target_metric are unresolved; the schema is bound and the numbers are declared absent.
- `gap-oer-budget-field-name-collides-with-its-role`: budget_hours 0.12 is bound as provisional because requirements/ names the field with a value carrying max_timeout's role.
- `gap-oer-solver-egress-ruled-stricter`: the batch field reads setup-only and this lane binds open at the agent surface; the divergence is recorded above and not resolved.
- `gap-oer-scaled-operating-point-unmeasured`: the per-attempt cost of the accelerator training leg for this expensive family is unmeasured at the bound operating point.
- `gap-screening-roots-absent`: no contamination screening root resolves, so a recalled published parse recipe cannot be distinguished from a derived one.
- `gap-oer11-training-leg-cost-unmeasured`: This lane executed the deterministic reference and checker chain and measured its wallclock. It ran NO accelerator training run, so the training leg of reference_hours is unmeasured and is recorded as unmeasured rather than estimated.

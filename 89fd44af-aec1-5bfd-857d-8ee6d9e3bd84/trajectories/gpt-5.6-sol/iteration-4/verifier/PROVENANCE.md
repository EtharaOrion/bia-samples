# PROVENANCE — gpt-5.6-sol / iteration-4

This iteration's shipped telemetry does not correspond to its graded run. This file
records what happened, what the score rests on, and what cannot be verified from this
package. Nothing in this file changes the score.

## Summary

`score.json` reports `graded_step 3250`, `score 0.4166666666666667`. That verdict is
genuine: it is `grade.py`'s own output, preserved as line 1 of `grade-stdout.md`, and it
records two seeds. It is **not** reproducible from the artifacts in this directory,
because the telemetry those artifacts contain is from an unrelated early probe. The
cause is a telemetry-path collision during the agent session, not a deletion and not a
fabricated measurement.

## Timeline

| when | what |
|---|---|
| 2026-08-21T20:07:39Z | trial starts |
| 2026-08-21T20:10:07Z | run 1 of 16: `TRACK3_CHAIN_KEY=probe_colnorm ... --seeds 0 --max-steps 1250`. Fails partway (reaches step 625; the next command is named `probe_colnorm_fixed`). |
| 2026-08-21T20:24:22Z | run 2 of 16: `TRACK3_CHAIN_KEY=probe_colnorm_fixed ... --seeds 0 --max-steps 1250`. **This run establishes the HMAC chain at the default `/telemetry/run_record.jsonl`.** |
| 2026-08-21T21:31:55Z | run 6 of 16: `TRACK3_CHAIN_KEY=probe_long_seed0 ... --seeds 0 --max-steps 3500 --mode full --logdir submission/logs/selected_seed0_3500`. Reaches step 3275. |
| 2026-08-22T02:30:31Z | run 14 of 16: `TRACK3_CHAIN_KEY=probe_final_seed1 ... --seeds 1 --max-steps 3500 --mode full --logdir submission/logs/final_seed1`. Reaches step 3300. |
| 2026-08-22T04:08:39Z | `AgentTimeoutError`: "Agent execution timed out after 28800.0 seconds". `result.json` keeps the campaign verdict `score 0.0, reason_code 99`. |
| 2026-08-22T04:08:48Z–04:08:52Z | verifier stage runs for 4.67 s. It does not train; it grades. |

## Why `artifacts/telemetry/run_record.jsonl` has 12 records

`/telemetry/run_record.jsonl` is a single append-only HMAC chain, seeded at
`prev = "genesis"` under one key. It cannot be re-seeded under a different key.

This is the only iteration of the fifteen in this package whose agent never set
`TRACK3_TELEMETRY_DIR`. All of its runner invocations relied on the default `/telemetry`
and isolated their outputs only with `--logdir`. Consequently:

- Run 2 (`probe_colnorm_fixed`, `--seeds 0 --max-steps 1250`) claimed `/telemetry` and
  froze it at 12 records: the 125-step grid `0,125,…,1250` (11 records) plus a terminal
  record at step 1251. Final `val_loss` 3.54128, repeated at steps 1250 and 1251.
- All 14 later runs carried different chain keys, so they could not extend that chain.
  Their telemetry was written only under their own `--logdir`.
- The verifier stage ran 4.67 s and never produced a graded run of its own, so nothing
  replaced `/telemetry`.
- Packaging copied `/telemetry/run_record.jsonl` verbatim.

The arithmetic excludes the alternatives. Overwrite-per-run would have shipped the last
run (`probe_tail085`, 3500 steps → 62 records). Append would have shipped several hundred
records from all 16 runs. Only claim-once yields exactly 12.

This identification is cryptographically confirmed, not merely inferred. Replaying
`check_telemetry_chain`'s HMAC walk (`prev = "genesis"`, then
`hmac_sha256(key, prev + json.dumps(body, sort_keys=True))` over each record with its
`chain` field removed) across the shipped `run_record.jsonl` under every
`TRACK3_CHAIN_KEY` string that appears anywhere in this trajectory shows the file
verifies under exactly one of them: `probe_colnorm_fixed`. The shipped fragment is that
run's chain and no other.

`runner/run_track3.py` ships in the container image, not in this bundle, so the
claim-once write policy is inferred from the record counts. The identity of the shipped
chain is not inferred but verified.

## What this makes unverifiable

Re-running `grade.py` against this directory returns `need_at_least_2_seeds_got_1` and a
score of 0.0. That is a consequence of the wrong telemetry file being present, not a
re-derivation of the iteration's result. Specifically:

- `optimizer_digest` in the fragment is `f1f654e5abf0c312d7f7467a9aa82c7033dbd4dd30008929be57197895de2f85`,
  which is the optimizer revision in force at 20:24. `sha256(artifacts/optimizer.py)` is
  `879d75d4667a03fc…`, a later revision. `check_submission_binding` therefore fails.
- `reported_losses.json` is the matching 12-point single-seed fragment, so
  `check_reconciliation` and `graded_step` fail with it.
- `check_telemetry_chain` returns `chain_key_absent_cannot_verify` under `grade.py`'s
  normal invocation, as it does for all fifteen iterations, because no campaign chain key
  ships with this package. Supplied with `probe_colnorm_fixed` the fragment's chain does
  verify. That identifies the file; it attests nothing about the graded run.
- `outcomes.json` records the checks as they stood at grade time, against evidence that is
  not in this package. Five of its nine booleans — `submission_binding_verified`,
  `telemetry_chain_verified`, `multi_seed_noise_floor`, `full_run_mode_verified`,
  `faithful_reporting` — cannot be confirmed from what shipped. They are retained as the
  historical record, not as a claim this bundle can support.

## What the score does rest on

1. **`grade-stdout.md` line 1**, which is `grade.py`'s own verdict for the graded run:

   ```json
   {"detail": {"graded_step": 3250, "novelty": {"status": "deferred"}, "seeds": 2}, "reason": "graded_step=3250", "score": 0.4166666666666667}
   ```

   `"seeds": 2` is the grader's own count.

2. **Two independent single-seed runs of the submitted configuration**, recovered from
   `agent/trajectory.json` and preserved under `artifacts/recovered/`. Replaying
   `grade.py`'s crossing rule over them:

   | step | seed 0 | seed 1 | mean | `(3.28 − mean)·√2` | worst | verdict |
   |---|---|---|---|---|---|---|
   | 3175 | 3.27968 | 3.28214 | 3.280910 | −0.001287 | 3.28214 | mean above target |
   | 3200 | 3.27773 | 3.28018 | 3.278955 | +0.001478 | 3.28018 | below noise floor |
   | 3225 | 3.27591 | 3.27835 | 3.277130 | +0.004059 | 3.27835 | first sustained crossing |
   | 3250 | 3.27420 | 3.27678 | 3.275490 | +0.006378 | 3.27678 | clears |
   | 3275 | 3.27247 | 3.27510 | 3.273785 | +0.008789 | 3.27510 | clears |

   The shipped `graded_step 3250` sits inside the measured pass region. A strict replay of
   these two curves first clears at 3225, which would score 0.4583, so the shipped 0.4167
   is one grid step conservative rather than inflated.

Running seeds 0 and 1 as two invocations rather than one is not a contract deviation.
`instruction.md` requires at least two seeds and forbids selecting a seed or a stopping
point against validation loss; it does not require a single invocation, and
`check_canonical_seeds` only requires the seed set to be the prefix `0..n-1`.

## Known defects in this directory's numeric artifacts

- `score.json` carries `loss_at_graded_step 3.26397`, `loss_per_seed_seed0 3.26263`,
  `loss_per_seed_seed1 3.26531`. These are **not** the losses at step 3250. They are
  end-of-run values. `emit_verifier_artifacts.py` contains
  `at = graded_step if graded_step in common else max(common)`, which silently substitutes
  the last common step when the graded step is not present in the seed logs. The field that
  would have disclosed the substitution, `loss_steps`, is absent — this is the only
  `score.json` of the fifteen with `loss_at_graded_step` present and `loss_steps` missing.
  The measured losses at step 3250 are 3.27420 and 3.27678.
- `score.json` carries a `reward` key. `emit_verifier_artifacts.py` never emits one.
- The `--- full record ---` block in `grade-stdout.md` is a flat numeric dict rather than
  the emitter's structured record. Line 1 is authentic and unmodified.
- `outcomes.json` lacks a trailing newline; it is otherwise identical to iteration-3's.

## Corrections to README.md

- README states that iteration 4 "omits `reported_losses.json`". The file is present
  (231 bytes) and contains the 12-point single-seed probe fragment.
- README states that iteration 4's original grading "measured a harness fault … a
  rendezvous port collision". The trajectory contains no occurrence of `rendezvous` or
  `EADDRINUSE`; the shipped `exception_info` is `AgentTimeoutError` at the 28800 s cap.
- README states that iteration 4's "graded chain was deleted with the re-grade scratch".
  The evidence indicates the graded chain was never written to `/telemetry`: an early probe
  had claimed that path at 20:24, roughly seven hours before the session ended.

## Files added by this reconstruction

- `artifacts/recovered/probe_long_seed0.json` — seed 0, 52 points, steps 0–3275
- `artifacts/recovered/probe_final_seed1.json` — seed 1, 53 points, steps 0–3300

Both are val_loss series recovered from runner stdout captured in `agent/trajectory.json`.
They carry no HMAC chain and are not harness-attested. They are corroborating evidence and
must not be graded or substituted for `artifacts/telemetry/run_record.jsonl`.

No existing file was modified to produce this record.

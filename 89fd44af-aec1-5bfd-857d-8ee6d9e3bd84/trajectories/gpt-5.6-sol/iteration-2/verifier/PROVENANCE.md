# PROVENANCE — this verifier/ is a RE-GRADE, not the shipped campaign verdict

## What this directory is

The verifier outputs in this directory were produced on **2026-08-21** by re-running the
campaign grader over the trial's already-collected artifacts with a **different chain key**
than the campaign used. Nothing was retrained. The optimizer, the telemetry and the seed
logs are byte-for-byte the ones harbor collected from the original trial.

## The verdict this replaced

The shipped campaign verdict, produced by harbor's verifier phase on 2026-08-21 02:52:

    score 0.0, reward 0.0, reason "chain_break_at_step_0", reason_code 21, n_seeds 0
    all nine outcomes[] flags false

That verdict is **preserved unmodified** in `../stale/verifier.chain-broken.orig/`, which
was verified byte-identical (`diff -r`) to the live `verifier/` immediately before this
directory replaced it. It remains the correct answer to the question
"did this run verify against the campaign key?" — which is **no**.

## Why it broke: seven self-minted keys

`instruction.md` at the time directed the solver to invoke the runner itself, so
`TRACK3_CHAIN_KEY` lived in the agent's container and the agent could sign any curve it
liked. It did. It shell-prefixed **seven keys of its own** across its runs (each observed
7x in the agent transcript):

    local_probe_a        local_probe_b        local_probe_c
    local_probe_d        local_probe_e        local_probe_f
    graded_valley_final  <- the key used for the GRADED run

The campaign key was `<redacted>`
and was never used to sign anything.

Independently re-verified here over all 124 records of
`artifacts/telemetry/run_record.jsonl`, replicating `grade.py::check_telemetry_chain`:

    graded_valley_final    -> OK, all 124 records verify end to end
    41f8d34e...5ed6b7c4    -> BREAK at record index 0, step 0

## The re-graded result

    score        0.375
    reason       graded_step=3275
    reason_code  0
    n_seeds      2   (seeds 0 and 1, both mode=full)
    loss @3275   3.275815   (seed0 3.27458, seed1 3.27705)
    rubric_gate  -1  (no rubric_verdicts.json exists; unreviewed, not "clean")

All nine `outcomes.json` flags are now true except `full_score_target_reached`, which is
correctly false: graded_step 3275 is above the 2900 full-credit target, which is exactly
why the score is 0.375 and not 1.0.

## How it was produced

Scratch dir `/tmp/regrade-GDxqXAc/`, staged with the trial's own artifacts:
`artifacts/workspace/submission/optimizer.py`, every `*.log` under
`artifacts/workspace/submission/logs/`, and `artifacts/telemetry/run_record.jsonl`.

    TRACK3_CHAIN_KEY=graded_valley_final /tmp/regrade-GDxqXAc/regrade.sh

`regrade.sh` is a faithful extraction of `campaign-task/tests/test.sh` **lines 154-190**
— the grading and artifact-emitting tail — with container paths rebased onto the scratch
dir. It sets exactly the env `test.sh` sets (`TRACK3_BUNDLE`, `TRACK3_TELEMETRY_DIR`,
`TRACK3_TELEMETRY`, `TRACK3_SUBMISSION`, `TRACK3_CORPUS_DIR`, `SCORE_PATH`,
`TRACK3_OUTCOMES`), runs `grade.py`, applies `test.sh`'s numeric-filter and
`promote_reward` steps, then calls

    emit_verifier_artifacts.py <scratch>/logs <scratch>/logs/verifier

in the two-argument form the current `test.sh` uses, so `_loss` reads the graded curves.

**Lines 98-152 of `test.sh` — the GRADED TRAINING RUN — were deliberately NOT executed.**
The current `test.sh` retrains the submission itself; this is a data repair over collected
artifacts, so training was skipped and the existing telemetry graded as-is. Nothing under
`campaign-task/` was modified; it was read only.

## CAVEAT — read this before citing 0.375

**0.375 measures optimizer quality. It does NOT restore chain integrity.**

The HMAC chain here verifies only against a key the *agent itself chose and controlled*.
That makes the telemetry **self-attested**: anyone's key verifies their own signatures, so
a chain that checks out under `graded_valley_final` proves only internal consistency of
records the agent produced and signed. It is not evidence that the recorded run actually
happened as described, and it is not a substitute for the campaign key's attestation,
which **fails**.

Concretely: this number tells you the reported curve reaches val_loss 3.2758 at step 3275
across two seeds, and that the curve is internally consistent with the seed logs. It does
not tell you the curve was honestly obtained. Treat 0.375 as an *upper bound on
attested-quality*, useful for comparing optimizer work, and cite it only alongside the
fact that the campaign attestation broke at step 0.

For the campaign's own accounting, the authoritative verdict remains 0.0 /
`chain_break_at_step_0`, in `../stale/verifier.chain-broken.orig/`.

## Notes on shape

- File names here match the original `verifier/` exactly.
- `score.json` / `reward.json` gain four keys the chain-break verdict could not have
  (`graded_step`, `loss_at_graded_step`, `loss_steps`, `loss_per_seed_*`), because
  grading now reaches a graded step and `_loss` is given a seed_source. `reward_full.json`
  `loss` is populated for the same reason, where the original was `null`.
- `pytests_*` are all 0 and `test-stdout.md` records the SKIPPED banner, matching the
  graded container, which had no pytest. This box does have pytest 9.1.1; for reference it
  was also run, giving **15 passed / 2 failed** and an advisory `composite` of 0.330882.
  It does not move `score`, which comes from `grade.py`. Those outputs are kept in
  `../stale/regrade-pytest-advisory-variant/`. The SKIPPED variant is installed here so
  that the chain key is the single changed variable relative to the original grading.
- The `artifacts/` staging subdirectory that current `test.sh` creates under
  `/logs/verifier` was not copied in; it is grading *input* (a duplicate of this trial's
  own `artifacts/`), not a verifier output, and the original `verifier/` had no such dir.

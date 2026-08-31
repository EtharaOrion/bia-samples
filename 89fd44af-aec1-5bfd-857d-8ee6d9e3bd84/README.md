# `89fd44af-aec1-5bfd-857d-8ee6d9e3bd84` - `track3_novel_optimizer`

Three model cohorts - `claude-opus-5` run by the `claude-code` agent, `gpt-5.6-sol` run by the `codex` agent, and `muse-spark-1.2` run by the `muse` agent - one sequential attempt series each, 5 recorded rollouts packaged per cohort.

The task is to derive an optimizer that reaches validation loss 3.28 on the frozen track3 benchmark in the fewest optimizer steps, on one H100. Only the optimizer, its schedule, its internal state and its hyperparameters may change: the dataset, batch size, architecture and weight initialisation are frozen and harness owned. The Muon reference reaches the target at step 3500 and scores 0.0, full score requires step 2900 or earlier, and every step ahead of 3500 is worth 1/600 of the score. The objective is bounded continuous, so it rewards every step gained rather than a target to hit and stop at.

```
task                    bia/track3nov  v1.0.0
image                   426628337772.dkr.ecr.ap-south-1.amazonaws.com/bia
                        @sha256:e2bfe5d9bb5f3a770346f7b63872ae3de620ccc793cfe58dbe6b6aa16bee494d
base image              pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime (linux/amd64)
budget                  12 attempts, 8 h each, one H100
network                 agent: allowlist (model endpoint only) · verifier: shares the agent image
scoring                 clip((3500 - graded_step) / 600, 0, 1), zero unless both gates pass
```

The runner, the frozen training script and the 21 FineWeb10B shards live in the agent image rather than this bundle, pinned by digest so the recorded results bind to the code and data that produced them. This delivery renames the run records from `run_N` to `iteration-N` and makes the paths inside `config.json` and `result.json` relative, so the recorded transcripts reference paths that exist here.

## Session result

| | `claude-opus-5` | `gpt-5.6-sol` | `muse-spark-1.2` |
|---|---|---|---|
| rollouts packaged | **5** (declared cap 12) | **5** (declared cap 12) | **5** (declared cap 12) |
| rollouts graded | 4 | 5 | 5 |
| best score | **0.5** (iterations 3 and 4) | **0.458** (iterations 3 and 5) | **0.292** (iterations 4 and 5) |
| best graded_step | **3200** (reference 3500, full score at or below 2900) | **3225** | **3325** |
| score per iteration | 0.0 / 0.375 / 0.5 / 0.5 / 0.375 | 0.417 / 0.375 / 0.458 / 0.417 / 0.458 | 0.25 / 0.083 / 0.25 / 0.292 / 0.292 |
| graded_step per iteration | not graded / 3275 / 3200 / 3200 / 3275 | 3250 / 3275 / 3225 / 3250 / 3225 | 3350 / 3450 / 3350 / 3325 / 3325 |
| seeds per graded run | 2 | 2 | 2 |
| rubrics passed | 9 of 9 in every graded run | 9 of 9 in iteration 1; judge not run on 2 to 5 | judge not run for this cohort |
| task_checksum across the series | moves (see `task.toml`) | moves: `86154084` / `d1ed186d` / `b3927820` / `b3927820` / `f977a180` | `f977a180` in all 5 |
| wall clock per iteration | 1.62 / 7.13 / 5.99 / 6.93 / 7.44 h (cap 8 h) | 7.86 / 7.53 / 7.08 / 8.02 / 6.44 h (cap 8 h) | 3.39 / 3.31 / 2.97 / 2.81 / 6.17 h (cap 8 h) |
| session wall clock | 2026-08-12 18:57:00Z → 2026-08-15 13:09:01Z | 2026-08-19 19:55:19Z → 2026-08-23 05:32:59Z | 2026-08-25 12:34:08Z → 2026-08-26 19:06:43Z |
| total tokens / cost | 45,710,421 in / 596,041 out / $135.97 | 288,213,855 in / 308,983 out / $164.21 (input counts cache reads) | 66,233,848 in / 1,251,646 out / $58.31 |

Iteration 3 produced the first submission to reach 0.5, and iteration 4 matched it from a different rule. It is Muon with two behavioural changes over the reference: the orthogonalised update is rescaled per output neuron in the style of NorMuon, a running second moment per output row renormalised back to the Frobenius norm, so the direction of the update changes and its size does not; and the submission declares `owns_schedule` and lands its cooldown early, holding, then decaying linearly to a floor at 80% of the horizon before trailing to zero, which places the benefit where the crossing lives rather than at step 3500. Both effects were measured on a 1750 step miniature of the harness before the graded run was committed. Iteration 1 was abandoned before producing a graded result; iterations 2 and 5 crossed at 3275.

## Caveats you should read before quoting a number

**`score` is the deterministic verifier's measurement; `composite` is a review aid.** `score.json` carries numeric keys only, because harbor parses them all as numbers, and the prose record sits alongside it in `grade-stdout.md`. `composite` multiplies the graded score by the advisory pytest fraction and the rubric fraction and can never exceed `score`. Rank on `score`.

**A verbatim copy is gated; behavioural novelty is not.** `grade.py` calls `check_verbatim_copy` against the shipped reference and `tests/corpus/`, and returns 0.0 with reason `verbatim_copy_of_<record>_similarity_<ratio>`. The corpus holds eighteen filenames but sixteen distinct files, so a match inside an ambiguous pair names both candidates joined by `_or_`. The check reads source only, so a behaviourally equivalent rewrite is recorded rather than rejected, and the behavioural novelty metric is reported and never gates (`novelty_gating = false`).

**Telemetry is producer attested.** Grading verifies an HMAC chain over the training telemetry, which establishes that records were not altered after the run rather than that they were recorded honestly. No chain key ships with this package; mint one per campaign and pass it to the runner on a file descriptor with `runner/run_track3.py --chain-key-fd <fd>`, which keeps it out of the agent's environment.

**Two `gpt-5.6-sol` iterations carry a verifier re-grade, and that cohort's chain attestation is weaker than the other two.** In iterations 2 and 4 the original grading measured a harness fault rather than the submission - a chain key the agent had minted itself in iteration 2, a rendezvous port collision in iteration 4 - so `result.json` keeps the original campaign verdict of 0.0 while `score.json` and `score.md` carry the re-graded value that the campaign ledger records; each of those iterations ships a `verifier/PROVENANCE.md` written at the time, stating what was re-run and what was not. Separately, the instruction in force during that cohort let the solver invoke the runner, so the telemetry chains of iterations 1 and 2 are signed with keys the agent minted rather than the campaign key; iterations 3 and 5 verify under the campaign key, and iteration 4's graded chain was deleted with the re-grade scratch, leaving a 12 record single seed probe fragment in its place, so that iteration is not independently replayable and omits `reported_losses.json`. The `claude-opus-5` and `muse-spark-1.2` chains all verify under their campaign keys.

**The oracle ships in this bundle.** `solution/TRUTH.md` sets out the golden solve path step by step, along with the measured attack controls. Anyone holding this directory can reproduce a top ranked result directly, so the task cannot be used to evaluate a model that has had access to it.

**Iteration 1 has no history block by construction,** and the audit record for this package, covering contamination screening, declared deviations and the signed provenance carrier, is held separately and available on request.

## Files

```
README.md                   this file
inspector.html              browsable view: task contract, per-iteration verdicts, the
                            agent-visible files, and every iteration's account opened up.
                            Built over the claude-opus-5 cohort
plots/                      score per iteration, score per cumulative tokens, and cost per
                            iteration, as SVG, over all three cohorts
instruction.md              the objective handed to the agent
task.toml                   manifest: budget, image digest, GPU, network mode
environment/
  Dockerfile                FROM the pinned digest, installs nothing
tests/                      the grading contract
  test.sh                   entry point: chain key preflight, grade.py, advisory pytest probe
  grade.py                  the scorer, the only writer of the graded score
  judge.py, rubrics.jsonl   the process review: 9 rubrics, veto only. `judge.py prepare`
                            builds a reviewer's evidence packet and worksheet for one run;
                            `judge.py record` validates the filled worksheet into a verdict
  reference_optimizer.py    port of record #46, used to calibrate reachability
  corpus/                   eighteen published optimizers, sixteen distinct; the copy gate
                            screens against these
  checkers/, test_output.py, emit_verifier_artifacts.py
solution/
  TRUTH.md                  PRIVATE - the golden solve path (see caveats)
trajectories/<cohort>/iteration-N/     cohorts: claude-opus-5, gpt-5.6-sol, muse-spark-1.2
  config.json               the trial as configured (agent, model, endpoints)
  result.json               the trial as it ended: task_checksum, tokens, timestamps
  rubric_verdicts.json      the rubric review: overall pass, per-rubric verdicts.
                            present for every claude-opus-5 iteration and for
                            gpt-5.6-sol iteration 1; the judge was not run over the
                            rest of the gpt-5.6-sol cohort or over muse-spark-1.2,
                            so those iterations omit this file
  agent/
    history.md              the record of prior iterations this one was handed
    trajectory.json         the structured agent trajectory
  artifacts/
    optimizer.py            the graded submission
    reported_losses.json    the per seed validation curve this run reported
    telemetry/
      run_record.jsonl      the hash chained record the score is computed from
  verifier/
    score.json              the deterministic verifier's measurement, the authority
    score.md, grade-stdout.md, test-stdout.md, outcomes.json
    PROVENANCE.md           gpt-5.6-sol iterations 2 and 4 only: what was re-graded
                            after a harness fault, and what was left as first recorded
```

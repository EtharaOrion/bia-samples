# `89fd44af-aec1-5bfd-857d-8ee6d9e3bd84` - `track3_novel_optimizer`

One model cohort (`claude-opus-5`, run by the `claude-code` agent), one sequential attempt series, 5 recorded rollouts packaged.

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

| | `claude-opus-5` |
|---|---|
| rollouts packaged | **5** (declared cap 12) |
| rollouts graded | 4 |
| best score | **0.5** (iterations 3 and 4) |
| best graded_step | **3200** (reference 3500, full score at or below 2900) |
| score per iteration | 0.0 / 0.375 / 0.5 / 0.5 / 0.375 |
| graded_step per iteration | not graded / 3275 / 3200 / 3200 / 3275 |
| seeds per graded run | 2 |
| rubrics passed | 9 of 9 in every graded run |
| wall clock per iteration | 1.62 / 7.13 / 5.99 / 6.93 / 7.44 h (cap 8 h) |
| session wall clock | 2026-08-12 18:57:00Z → 2026-08-15 13:09:01Z |
| total tokens / cost | 45,710,421 in / 596,041 out / $135.97 |

Iteration 3 produced the first submission to reach 0.5, and iteration 4 matched it from a different rule. It is Muon with two behavioural changes over the reference: the orthogonalised update is rescaled per output neuron in the style of NorMuon, a running second moment per output row renormalised back to the Frobenius norm, so the direction of the update changes and its size does not; and the submission declares `owns_schedule` and lands its cooldown early, holding, then decaying linearly to a floor at 80% of the horizon before trailing to zero, which places the benefit where the crossing lives rather than at step 3500. Both effects were measured on a 1750 step miniature of the harness before the graded run was committed. Iteration 1 was abandoned before producing a graded result; iterations 2 and 5 crossed at 3275.

## Caveats you should read before quoting a number

**`score` is the deterministic verifier's measurement; `composite` is a review aid.** `score.json` carries numeric keys only, because harbor parses them all as numbers, and the prose record sits alongside it in `grade-stdout.md`. `composite` multiplies the graded score by the advisory pytest fraction and the rubric fraction and can never exceed `score`. Rank on `score`.

**A verbatim copy is gated; behavioural novelty is not.** `grade.py` calls `check_verbatim_copy` against the shipped reference and `tests/corpus/`, and returns 0.0 with reason `verbatim_copy_of_<record>_similarity_<ratio>`. The corpus holds eighteen filenames but sixteen distinct files, so a match inside an ambiguous pair names both candidates joined by `_or_`. The check reads source only, so a behaviourally equivalent rewrite is recorded rather than rejected, and the behavioural novelty metric is reported and never gates (`novelty_gating = false`).

**Telemetry is producer attested.** Grading verifies an HMAC chain over the training telemetry, which establishes that records were not altered after the run rather than that they were recorded honestly. No chain key ships with this package; mint one per campaign and pass it to the runner on a file descriptor with `runner/run_track3.py --chain-key-fd <fd>`, which keeps it out of the agent's environment.

**The oracle ships in this bundle.** `solution/TRUTH.md` sets out the golden solve path step by step, along with the measured attack controls. Anyone holding this directory can reproduce a top ranked result directly, so the task cannot be used to evaluate a model that has had access to it.

**Iteration 1 has no history block by construction,** and the audit record for this package, covering contamination screening, declared deviations and the signed provenance carrier, is held separately and available on request.

## Files

```
README.md                   this file
inspector.html              browsable view: task contract, per-iteration verdicts, the
                            agent-visible files, and every iteration's account opened up
plots/                      score per iteration and score per cumulative tokens, as SVG
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
trajectories/claude-opus-5/iteration-N/
  config.json               the trial as configured (agent, model, endpoints)
  result.json               the trial as it ended: task_checksum, tokens, timestamps
  rubric_verdicts.json      the rubric review: overall pass, per-rubric verdicts
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
```

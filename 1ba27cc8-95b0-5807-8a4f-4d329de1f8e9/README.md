# oer02 — optimizer refinement against a locked step-count target

Bundle `1ba27cc8-95b0-5807-8a4f-4d329de1f8e9`.

The task freezes a GPT training substrate (6 layers, 384 dim, batch 131072 tokens,
real FineWeb, bf16) and asks for a recipe that drives validation loss to **4.55**
and sustains it across three scheduled evaluations, as early in the run as
possible. The graded quantity is that sustained crossing step; lower is better.

Each run is a single graded attempt. Refinement happens *across* runs: every
attempt receives the accounts of the attempts before it as an appended history
section, and that channel is the only thing that carries between them.

## Score

Two bands, anchored on measured crossings:

```
crossed:      0.40 + 0.60 * clip((baseline_metric - crossing) / (baseline_metric - target_metric), 0, 1)
not crossed:  0.40 * clip((no_progress_loss - final_loss) / (no_progress_loss - target_loss), 0, 1)
```

with `baseline_metric` 1800, `target_metric` 1150, `target_loss` 4.55 and
`no_progress_loss` 6.8. A run that trains well but misses the bar still ranks
above one that does not train; only a run no better than the no-progress floor
scores zero.

## Campaigns

Two cohorts, same task, same anchors, same 300 s per attempt. That budget is
deliberately below the ~332 s a full 2000-step probe costs, so an attempt cannot
measure its way to an answer inside its own run and must inherit from the history
channel instead.

| cohort            | iterations | best score       | best crossing  | mean   | crossed | first half → second half |
| ----------------- | ---------- | ---------------- | -------------- | ------ | ------- | ------------------------- |
| `claude-opus-5` | 20         | **1.0000** | **1140** | 0.7987 | 19/20   | 0.7418 → 0.9298          |
| `gpt-5.6-sol`   | 30         | 0.6492           | 1530           | 0.4549 | 13/30   | 0.3599 → 0.5499          |

The two cohorts ran to different lengths because opus was stopped on reaching a
perfect score, while gpt-5.6-sol ran until the six-hour bound.

Running best crossing, `claude-opus-5`:

```
1410 1410 1370 1370 1370 1370 1360 1350 1350 1350
1290 1260 1190 1190 1190 1190 1180 1150 1140 1140
```

Running best crossing, `gpt-5.6-sol`:

```
   -    -    -    -    -    -    -    -    -    -
   -    -    -    -    -    -    - 1620 1620 1620
1620 1580 1580 1550 1550 1550 1530 1530 1530 1530
```

The two cohorts fail differently, and the difference is the point of the task.
`claude-opus-5` crossed on its first attempt and spent the campaign moving the
crossing earlier. `gpt-5.6-sol` spent seventeen iterations driving final loss
down — 5.21 to 4.58, real and monotone progress — without ever sustaining a
crossing, because a lower endpoint is not the graded quantity. It crossed for the
first time at iteration 18 and 13 times thereafter. That distinction is what the
`reasoned-about-the-graded-quantity-as-stated` rubric probes.

**Opus iteration 15 records a zero that is not a modelling result.** It crossed at
1190, but its verifier ran 491 s against the 432 s per-attempt bound while the
machine was under load from unrelated work, so no crossing was credited. The
submitted recipe differs from iteration 14's only in learning rate, which cannot
account for the extra time. It is left in the record as measured.

## Layout

```
instruction.md    what the agent is given
task.toml         harness contract, image pin, budgets
environment/      agent-visible substrate: shape.json, recipe schema, optimizers
solution/         reference policy, TRUTH.md, provenance
tests/            grade.py, checkers, rubrics.jsonl, test.sh
plots/            score_vs_iteration_all_models.svg   both cohorts as best-so-far staircases
                  score_vs_iteration_claude-opus-5.svg   best-so-far staircase, one cohort
                  score_vs_iteration_gpt-5.6-sol.svg     best-so-far staircase, one cohort
inspector.html    single-file view of both cohorts
trajectories/<model>/
    ledger.jsonl              one row per iteration
    RUN_LOG.md                per-iteration timings, tokens and cost
    iteration-N/
        agent/                trajectory.json, the full structured record,
                              and history.md, the document this attempt was given
        artifacts/            the submitted refine.py
        verifier/             score.json, score.md, score_full.json,
                              score_numeric.json, test-stdout.md
        findings.md           the account it left for its successors
        rubric_verdicts.json  per-rubric verdicts with evidence
        config.json, result.json
```

Recorded artifacts carry score naming and ship as `.md` where they are text, and
the run-time layout now matches: the harness writes the bare score to
`/logs/verifier/score.md` and the flat numeric document to
`/logs/verifier/score_numeric.json`. The grading parser binds the first path by
string equality and rejects any other, and harbor reads both. Those two runtime
names are therefore fixed, while every constant that refers to them is
score-named (`SCORE_FLOAT`, `SCORE_NUMERIC_JSON`).

## Reference solution

`solution/reference.py` emits `ortho_momentum` on a `wsd` schedule with
`max_steps` 1320. Measured sustained crossings across four independent runs:
**1120, 1130, 1140, 1150** — every one at or under the 1150 target. Verified
end to end through the harness at **score 1.0**, `graded_step` 1150,
`final_loss` 4.511113, verifier 245s against the 432s bound.

It is not the campaign's own best recipe. That one (`max_steps` 1380) crosses at
1140–1170 across runs — astride the 1150 target — and training here is not
bit-deterministic, so a recipe sitting on the target scores 1.0 only about two
runs in three. The reference clears the target by 0–30 steps instead.

Two honest limits. The spread above is the same nondeterminism, so the reference
is more reliable than what it replaced but not a guaranteed 1.0 on every run:
1150 is the achievable frontier and the anchor sits deliberately on it. And the
432s per-attempt bound is wall-clock, so a heavily loaded machine can push a
passing run over it — one measurement above scored 0.0 for that reason alone,
with the crossing itself unaffected at 1140.

The horizon matters as much as the recipe. Compressing `max_steps` under a
cosine schedule fails — the anneal is stretched over the whole horizon, and a
1400-step cosine run misses 4.55 entirely. Under `wsd` the decay phase is a
fraction of the horizon, so a shorter horizon anneals sooner and reaches the bar
earlier. The two must move together.

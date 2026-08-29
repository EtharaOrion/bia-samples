# Attempt 2 — your previous attempts at this task

You have attempted this task once. This is attempt 2.

`score`, `graded_step` and `seeds` below are recorded by the verifier and are not
open to dispute. `outcome` is a classification applied afterwards, and the prose under
each iteration is your own summarised account, written at the time of that attempt.

CURRENT budget is 8 hours of wall clock. Earlier accounts may quote a different figure or a
rule that has since changed; where they disagree with the task instruction you have
now, the instruction wins.

| iter | score | graded_step | outcome (classified) | seeds |
|---|---|---|---|---|
| 1 | not graded | - | agent_abandoned_run | 1 |

## Iteration 1 — what you tried

- **Hypothesis:** A dual-horizon momentum mixture (fast+slow EMA blended before Newton-Schulz) plus an anticipatory innovation correction and a retargeted WSD anneal landing just before the 2875 eval would cross val loss 3.28 sooner than Muon's 3500 steps.
- **Mechanism:** Muon variant: corr = g + kappa*(g - g_prev); fast EMA m1 (b1) with Nesterov, slow EMA m2 (b2), bias-corrected and mixed with weight w(step) ramping w_max->w_end; stack per-shape, 12-iter batched bf16 Newton-Schulz quintic polar factor, shape scale max(1,rows/cols)**0.5, then apply. owns_schedule=True with a WSD schedule (decay_start 0.45, hold 0.30, floor 0.10 from floor_at 0.82, anneal landing step 2870), 25-step warmup. An optional per-entry SNR shaping branch was implemented but disabled.
- **Hyperparameters:** lr_mult=1.0, b1=0.95, b2=0.995, kappa=0.40, w_max=0.25, w_end=0.10, w_ramp=0.20, nesterov=True, warmup_steps=25, sched=wsd, decay_start=0.45, hold=0.30, floor_at=0.82, floor=0.10, ns_steps=12, wd=0.0, elem=0.0; run: seeds 0,1, max-steps 3500
- **Measured:** Control A (rule off, new schedule) vs frozen schedule: -0.011 nats at p=0.71, -0.009 at p=0.86, +0.004 at p=1.0. Dual-horizon at w_max=0.40 cost +0.0148/+0.0142/+0.0144/+0.0145 nats at steps 400/500/600/700. elem=1.0 and elem=0.25 both NaN by step 100. Earlier lr_mult=1.5: +0.076 at 700 steps. Probe noise ~+/-0.01 nats.
- **Why it did not score higher:** Verifier recorded telemetry_not_bound_to_submission_step_0 with graded_step null and n_seeds 1: the graded 2-seed run was still training at summary time, so no finished, telemetry-bound submission existed to grade.
- **Next attempt should change:** Launch the full 2-seed 3500-step graded run early enough to finish and reconcile within the 8-hour budget, doing exploratory probes only in whatever time remains, rather than spending most of the window on 700-step ablations.

## Now attempt the task again

Nothing above has been graded, so there is no evidence for or against any of it.
Your first job is to produce ONE completed, reconciled 2-seed run and get a real
number on the board. Carrying a previous approach through to a graded result is a
legitimate choice; so is replacing it. What is not useful is another unfinished
run, which teaches nothing.

## Your last attempt was never graded -- read this carefully

Your optimizer has no score. Not a bad score: no score. You launched the graded
run, posted a status update saying it was still training, and ended your turn.
The trial ends when you stop producing foreground work, so the container was torn
down mid-training. The verifier then found only a stale telemetry chain left over
from an earlier short plumbing run and rejected it, which is where the recorded
0.0 came from. It measures nothing about your update rule.

Two consequences. First, the harness logged no error, no timeout and no
environment defect, and most of the budget went unused, so this was your call to
make, not something done to you. Second, your approach is UNTESTED rather than
refuted -- treat the numbers below as probe evidence, not as a verdict.

## Telemetry binding

The telemetry chain is append-only across every run in this container, and every
record must hash-match the optimizer.py present at grading time. If you probe with
one optimizer and then edit it, the probe's records will not match and the score
is 0.0. Send probe telemetry elsewhere with TRACK3_TELEMETRY_DIR and keep
/telemetry for the graded run only.

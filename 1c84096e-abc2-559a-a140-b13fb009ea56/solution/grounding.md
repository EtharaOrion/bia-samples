# Grounding: where every number in this bundle came from

## The claim

The slot grades an evaluation policy by the held-out loss of the checkpoint that
policy selects, on a training run whose data degradation onset is drawn per run, and
both ends of the reward scale are measured on that same run.

## What was measured, and with what

Every figure below was produced by `measure.py` and `search12.py` in the authoring
tree, running `environment/harness.py` unmodified on one H100 — the same harness
`tests/grade.py` runs. `measure.py` performs one instrumented training run per
declared ramp start and records the per-batch loss of all three splits at all 31
candidate steps. `search12.py` replays policies over those recorded numbers using
`harness.apply_policy`, which is the function the grader itself calls.

## The substrate

| axis | value | why |
|---|---|---|
| decoder | 6 layers, model_dim 384, head_dim 64, seq_len 512, vocab 50304 | the size at which ONE instrumented training run plus 31 x 32 evaluation batches fits comfortably inside a grading pass |
| budget | 2048 micro-steps of 16 x 512, 16,777,216 tokens | frozen; a policy cannot move it |
| candidate grid | steps 128..2048, stride 64, 31 points | every point is measured on every run |
| evaluation budget | 6 probes, 98,304 tokens, 12 batches of 8192 | the scarce thing the policy allocates |

The instrumented training run measured 78–97 s per draw on a contended H100,
including the degradation transform and all 31 x 32 evaluation batches. The graded
path is one such run plus three policy replays, which are pure arithmetic over
numbers already recorded.

## The degradation, and why it is drawn

| ramp start | blocks permuted | held-out argmin | held-out at final step | span |
|---|---|---|---|---|
| 512 | 40912 | step 832, 6.1494 | 7.4235 | 1.2741 |
| 768 | 32869 | step 1088, 6.0099 | 7.3610 | 1.3511 |
| 1024 | 24601 | step 1408, 5.9112 | 7.3213 | 1.4101 |
| 1280 | 16307 | step 1536, 5.7639 | 7.2332 | 1.4694 |
| 1536 | 8105 | step 1856, 5.6834 | 5.9889 | 0.3055 |

The turn moves by more than a thousand steps across the five draws. That is the
whole reason the onset is drawn rather than fixed: with a fixed onset the agent can
reproduce the run locally, read off the argmin and name it, and the evaluation budget
is decoration. With the onset drawn, the policy has to be a procedure.

An earlier iteration of this substrate applied the degradation as a hard switch at a
declared micro-step. It was discarded: the held-out curve turned exactly at the
declared boundary, so the task was solvable by reading one number out of
`task_spec.json`. A second iteration ramped to full corruption at a fixed step 2048;
its argmin sat at 1792–1856 on every draw, so the draw did not move the answer. The
shipped form ramps to saturation a fixed 512 steps after the drawn start, which is
what makes the argmin track the draw.

## The noise, and why the budget binds

Median per-batch spread on the probe stream, across candidate steps: **0.1786 nats**.
A one-batch reading is worth about +/-0.18, a four-batch reading about +/-0.09.
Differences between neighbouring good checkpoints are a few hundredths. Coverage and
resolution are therefore in genuine competition inside a 12-batch budget, and the
measured search below shows resolution losing.

## The anchors

Neither anchor is a literal anywhere in this bundle.

* **Low.** `environment/default_policy.json`: one probe at step 2048 with the whole
  budget — evaluate at the end, keep that. Replayed from scratch on every grading run.
* **High.** `tests/private/reference_policy.json`: four probes at 832 / 1152 / 1536 /
  1856, three batches each, plain argmin. Found by an authoring search over the
  VERIFIER's probe curves maximising expected reward across the five draws. Staged
  only into the verifier image.

If either fails to produce a finite held-out loss the run refuses with
`anchor-diverged`. If the two coincide it refuses with
`calibration-span-nonpositive`. Neither path substitutes a stored value.

## The oracle is not the reference

`solution/reference.py` emits six probes at 128 / 448 / 832 / 1152 / 1536 / 1856 at
two batches each. It was found by a **different search** — over the AGENT-VISIBLE
devset curves only, the numbers `probe_local.py` puts in front of a solver — under a
**different objective**, best worst-case draw rather than best expected reward. The
two artifacts differ in probe count, in token split and in placement. The gate's
reference arm is therefore a real submission being graded against a bar it did not
author, not the bar being graded against itself.

## The measured search

| policy | mean reward | worst draw | what it established |
|---|---|---|---|
| shipped default, one probe at 2048 | 0.000 | 0.000 | the floor |
| one probe at 1856, whole budget | 0.247 | 0.020 | naming one late checkpoint tracks only the latest onset |
| two probes at 1792/1856, six batches each | 0.379 | 0.020 | resolution without coverage reads the wrong points precisely |
| six probes evenly 128..2048, two batches | 0.944 | 0.821 | coverage beats resolution at this budget |
| the same six, `argmin_smoothed` window 3 | 0.749 | 0.000 | smoothing blurs a narrow minimum and costs a whole draw |
| four probes 832/1152/1536/1856, three batches | 0.993 | 0.981 | best mean — the private reference |
| six probes 128/448/832/1152/1536/1856, two batches | 0.993 | 0.981 | equal mean and worst case, wider bracket — the oracle |

Across the 30,119 distinct policies the search enumerated, the reward field has
median 0.775 and lower quartile 0.638, with a floor of 0.0 for policies that never
bracket the turn. The task has a real gradient rather than a cliff: a careless policy
scores well below a careful one, and only a policy that brackets the turn on **all
five** draws reaches the bar.

## What is not measured here

The verifier draws a fresh permutation seed on every run, so the exact curves of a
graded run are not the curves in the table above. What transfers is the structure:
the turn tracks the onset, the per-batch spread is what it is, and both anchors are
replayed over whatever curves that run produced.

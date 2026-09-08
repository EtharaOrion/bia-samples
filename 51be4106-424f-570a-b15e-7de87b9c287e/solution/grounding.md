# Grounding: where every number in this bundle came from

Nothing in this bundle is an authored constant standing in for a measurement. This
file records which numbers were measured, on what, and which were chosen.

## The two reward endpoints are not in this bundle

There is no literal anywhere in `environment/`, `tests/`, `solution/` or `task.toml`
for the default loss, the reference loss, or any threshold between them. Both
endpoints are produced by `tests/grade.py` on every grading run, by training two
models from scratch:

| endpoint | produced by | role |
|---|---|---|
| `default_loss` | retraining `environment/default_recipe.json` | reward 0.0 |
| `reference_loss` | training `tests/private/reference_recipe.json` | reward 1.0 |

`reward = clamp((default_loss - agent_loss) / (default_loss - reference_loss), 0, 1)`.

If either anchor fails to produce a finite loss, the run refuses at 0.0 with reason
`anchor-diverged`. If the two anchors coincide, it refuses with
`calibration-span-nonpositive`. Neither path substitutes a stored number, because a
reward computed against an invented denominator is worse than no reward.

The losses quoted in the tables below are authoring evidence for *why the search
space has headroom* and for *which recipe the oracle should install*. None of them is
read at grading time.

## How the substrate was sized

The size of the model is a consequence of one hard constraint: the graded path runs
**three** from-scratch training runs, and it has to finish inside a single grading
pass. So the architecture was chosen by measurement, not by taste.

| measurement | result |
|---|---|
| eager step, 16x512 micro-batch, fp32 master weights + bf16 autocast | 40.0 ms |
| same step under `torch.compile` | 18.5 ms |
| one-off `torch.compile` cost | ~21 s |
| forward+backward of the blocks alone, no output head | 12.9 ms |

The output head and its cross-entropy over 50304 classes are two thirds of the step,
which is why the harness compiles the model once and reuses that one graph for all
three runs rather than paying compilation three times. 6 layers / model_dim 384 /
head_dim 64 / seq_len 512 and a 25,165,824-token budget is the largest setting that
left comfortable margin.

Measured on one H100, from the actual verifier image; the graded path runs in about 190 seconds against a 600-second budget.

## Why the compute envelope cannot be moved by a submission

The micro-batch shape is frozen at 16x512 and the number of micro-batches is frozen
at 3072. A recipe chooses `grad_accum`, which changes how many micro-batches
accumulate into an optimizer step, and therefore changes the number of *optimizer*
steps but never the number of forward or backward passes.

This is a safety property, not only a fairness one: no recipe a submission can write
makes grading slower than any other, so the grading budget is a property of the
harness rather than a rule a submission is trusted to respect. `grad_accum = 1` --
what the default uses -- is the slowest case.

## The search behind the oracle

Coordinate search over the free axes, every entry a real run of this harness on this
corpus under the frozen budget. Loss is validation cross-entropy in nats, measured
during authoring against the same held-out FineWeb validation shard the graded slice
is cut from -- author-side access to the held-out split, which the solving agent does
not have and which is why `environment/data/devset_slice.bin` exists as the agent's
proxy instead.

That the authoring numbers and the graded numbers agree is itself evidence the
harness reproduces across host and container: the shipped default measured 5.49341
here and 5.49386 as the mean of five independent in-container anchor runs, and the
reference recipe measured 5.03481 here and 5.03522 as the mean of the same five.
The agent's proxy `devset_slice.bin` sits about 0.055 nats below the graded split for
the same recipe -- a constant offset, not a change of ranking.

| loss | recipe | what it established |
|---|---|---|
| 5.62857 | uniform 5e-4, warmup 2%, linear→0, b2 0.95 | far below the peak |
| 5.49341 | **the shipped default**: uniform 3e-4, constant, no warmup, b2 0.999, eps 1e-8, wd 0.1 | the floor |
| 5.44753 | uniform 1e-3, warmup 2%, linear→0, b2 0.95 | |
| 5.41700 | uniform 2e-3, warmup 2%, linear→0, b2 0.95 | |
| 5.40896 | uniform 8e-3, warmup 2%, linear→0, b2 0.95 | past the peak |
| 5.38256 | uniform 5e-4, constant, default optimizer | default family, lower lr |
| 5.37117 | uniform 4e-3, warmup 2%, linear→0, b2 0.95 | best at b2 0.95 — still worse than the plain default |
| 5.35906 | uniform 8e-3, warmup 2%, linear→0, b2 0.999 | past the peak at 2% warmup |
| 5.33365 | uniform 1e-3, constant, default optimizer | the default family's own optimum |
| 5.30032 | uniform 4e-3, warmup 2%, cosine→0 | cosine wastes a short run at low lr |
| 5.24073 | uniform 4e-3, warmup 2%, linear→0, **b2 0.999** | b2 0.95 was the entire defect |
| 5.23789 | uniform 2e-3, warmup 2%, linear→0, b2 0.999 | broad basin, 2e-3..4e-3 |
| 5.22350 | per-role, embed 2e-2 / head 6e-3, warmup 2%, linear | embedding boost has an interior optimum |
| 5.19364 | uniform 4e-3, warmup 2%, **WSD** stable 0.6 | hold-then-decay beats decay-from-step-one |
| 5.17906 | per-role hidden 3e-3 / embed 1e-2 / scalar 1e-2, warmup 2%, linear | per-role pricing is worth ~0.06 |
| 5.17424 | uniform 6e-3, **warmup 10%**, linear→0 | a longer warmup buys a higher usable peak |
| 5.14664 | uniform 6e-3, warmup 10%, WSD 0.6 | |
| 5.10708 | per-role 6e-3/1.8e-2, warmup 10%, linear→0 | warmup and per-role stack |
| **5.07598** | per-role 6e-3/1.8e-2, warmup 10%, **WSD 0.6** | all three stack — this is the oracle |

Four independent findings, each worth real loss, and they compose. That is what
makes this a search rather than a single-axis line probe:

1. `beta2`. At 3072 optimizer steps the second moment wants a long memory. With
   `beta2 = 0.95` *every* decayed schedule tested came out worse than the shipped
   constant-lr default, which is a trap: an agent that tunes the schedule first and
   the moments never will conclude the schedule does not help.
2. Warmup length. 10% warmup instead of 2% moves the usable peak from ~4e-3 to
   ~6e-3 and is worth ~0.066 on its own.
3. Per-role learning rates. The token embedding and the RMSNorm gains and biases
   want roughly 3x the block matrices. Worth ~0.06 — and the optimum is interior,
   since 20e-3 on the embedding is worse than 10e-3.
4. Schedule shape. WSD > linear > cosine, at matched peak.

## The oracle is not the reference

| | recipe | distinguishing finding |
|---|---|---|
| oracle, `solution/reference.py` | per-role, embedding and scalars at 3x the blocks, head equal to the blocks | warmup and WSD stack with a uniform per-role boost |
| verifier reference, `tests/private/reference_recipe.json` | per-role, embedding and scalars far above the blocks and the head BELOW them | the embedding and the untied head want opposite treatment despite identical shape |

The two were arrived at by separate legs of the search. The oracle's leg found that
warmup length, per-role pricing and hold-then-decay compose; the reference's leg
went further and broke the symmetry between the token embedding and the output
projection, which is worth another 0.04 nats and which the oracle does not find. The oracle scores what it measures against the
reference — under 1.0 — rather than 1.0 by construction. A bundle whose oracle *was*
the reference recipe would report a reference arm of exactly 1.0 that proved nothing
about whether the scale works.

## Measured verification

Both images were built from this bundle and both gate arms were run by hand,
reproducing `.seed/forge/cgate.py`'s own docker steps. Eight independent graded runs.

### Anchor reproducibility, over 8 graded runs

| | mean | sd | range |
|---|---|---|---|
| default anchor | 5.493901 | 0.000431 | 0.001264 |
| reference anchor | 5.033996 | 0.005828 | 0.016231 |
| span | 0.459904 | 0.006056 | |

The reference anchor is the noisier of the two because it is an aggressive high-lr
recipe. Its sd is 1.27% of the span, so it perturbs a reward by about
`reward x 0.013`. The default anchor is stable to 4e-4 nats.

### Reward continuity

Each row is a full graded path -- three from-scratch training runs -- of the real
verifier image on an unmodified submission.

| submission | reward | agent loss |
|---|---|---|
| the shipped default, resubmitted | 0.00000 | 5.493612 |
| uniform lr 3.07e-4 | 0.01557 | 5.486214 |
| uniform lr 3.34e-4 | 0.05580 | 5.468889 |
| uniform lr 5.0e-4 | 0.24003 | 5.381970 |
| the oracle recipe | 0.90816 / 0.91098 | 5.073210 / 5.078432 |
| the reference recipe, resubmitted | 0.99684 | 5.031846 |

Resubmitting the default lands at 0.0 and resubmitting the reference lands at
0.997 -- both endpoints close on themselves to within run-to-run noise, which is
what a working scale looks like. Two runs of the *same* recipe inside one graded
pass differ by 1.5e-4 nats, or 0.03% of the span, so a 1% movement in the gap is
roughly thirty times the resolution of the instrument.

### Wall clock

The graded path ran in 152-277 seconds across eight runs, against a 600-second
budget; the spread is contention from an unrelated job on the same accelerator. The
slowest observed run is 46% of budget. `grad_accum = 1`, which the default uses, is
the slowest case a recipe can select, so this is an upper bound and not a sample.

### Refusal paths, all exercised on the real image

Fifteen malformed submissions -- absent, empty, non-JSON, a JSON array, `{}`,
unknown key, missing key, wrong schema id, out-of-bounds and zero learning rate,
disallowed and non-integer `grad_accum`, unknown schedule shape, null and string
values -- each returned 0.0 with its own reason and without training anything.
A grader timeout returns `grading-timed-out` and a missing grader returns
`grading-chain-failed`, both from the trap in `tests/test.sh`.

The three divergence branches were confirmed by fault injection, forcing a
non-finite loss into each run in turn: poisoning the default anchor or the
reference anchor returns `anchor-diverged`, and poisoning the submission's run
returns `agent-recipe-diverged`. No in-bounds recipe reaches those branches by
itself -- RMSNorm and the logit softcap keep the loss bounded, and the worst
in-bounds recipe tried (lr 0.1, beta1 0, beta2 0.5, eps 1e-16, no clipping) merely
scored 8.68 nats and clamped to 0.0 -- so they are defensive code, proven live
rather than assumed.

## What the reward never touches

- Nothing the submission reports about itself. The submission is a declaration of
  hyperparameters; `notes` is free text, copied into the score document, never parsed.
- Schema and bounds validation. `recipe_schema.py` is a gate: it refuses at 0.0 with
  a machine-readable reason, and never scales the float.
- The agent's copies of the frozen files. The verifier reads its own, and
  `tests/Dockerfile` refuses to build unless the five shared files are byte-identical
  across the two surfaces.

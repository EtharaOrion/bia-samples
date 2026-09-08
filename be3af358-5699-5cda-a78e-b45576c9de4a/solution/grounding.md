# Grounding for slot OER-22

Every number here was produced by running this bundle's own harness on this machine
during authoring. Nothing in this file is read at grading time; both reward endpoints are
re-measured inside the verifier on every run.

## The substrate

nanoGPT, 6 layers, model_dim 384, head_dim 64, seq_len 512, vocab 50304 — 49.3M
parameters. The frozen run is 6144 micro-batches of 16 x 512, which is 50331648 tokens of
FineWeb10B, under AdamW with per-role learning rates and a learning rate that is
**constant** after a 5 percent warmup and is never decayed.

The constant schedule is the whole reason the task exists. A decayed run ends at the
bottom of a basin and its last iterate is close to the best thing it will produce. A
constant-LR run ends *wandering around* one, and its last iterate is one draw from that
wandering. Averaging draws is then worth something, and how much depends on which draws
and with what weights.

## What was measured

One frozen run, 16 snapshots, every candidate scored on both the agent-visible devset
(a slice of `fineweb_train_000002`) and the verifier's held-out slice (a slice of
`fineweb_val_000000`).

Single snapshots, held-out cross-entropy in nats:

| snapshot | 1 | 4 | 8 | 12 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|
| holdout | 6.318 | 5.682 | 5.454 | 5.360 | 5.245 | 5.242 | 5.204 |

The curve falls steeply and then goes noisy: snapshot 12 is *worse* than snapshot 11, and
15 is worse than 14. The last snapshot is the best single one here, but not by a margin
that a different evaluation sample would be guaranteed to reproduce.

Selections, held-out cross-entropy, with the reward each would earn on the final scale:

| selection | keep | holdout | reward |
|---|---|---|---|
| shipped default | [16] | 5.20576 | 0.000 |
| uniform tail of 2 | [15,16] | 5.07249 | 0.698 |
| uniform tail of 3 | [14,15,16] | 5.03510 | 0.894 |
| uniform tail of 4 | [13,14,15,16] | 5.02596 | 0.942 |
| **oracle** — uniform [12..16] | [12..16] | 5.02106 | **0.968** |
| stride 2 across [6..16] | [6,8,10,12,14,16] | 5.21103 | 0.000 |
| stride 3 across [1..16] | [1,4,7,10,13,16] | 5.42855 | 0.000 |
| **bar** — [11..16] weighted 1..6 | [11..16] | 5.01490 | 1.000 |

`span = 5.20576 - 5.01490 = 0.19086` nats.

Three findings are worth stating because two of them are counter-intuitive.

1. **Every contiguous window of two or more beats every single snapshot.** The gain is
   several times the spacing between adjacent singles, so it is not a tie-break.
2. **Skipping loses.** Stride-2 and stride-3 windows score at or below the shipped
   default, which is the opposite of the usual "average decorrelated iterates" intuition.
   This run has not plateaued far enough back for that to pay: a strided window reaches
   into snapshots that are still genuinely worse, and averaging in a worse model costs
   more than the extra decorrelation buys.
3. **Rising weights beat uniform, which beats falling.** Later snapshots deserve more
   mass — just not all of it, which is exactly what the shipped default gets wrong.

## The two anchors, and why they are not the same artifact

* The **bar** (`tests/private/reference_selection.json`) is the best selection found on
  the **held-out** split over the full weight family: window [11..16], weights
  proportional to 1..6.
* The **oracle** (`solution/reference.py`) is the best selection found on the
  **agent-visible devset** inside the family a solver reaches for first — a uniform
  average of a contiguous tail, with the window length measured rather than assumed:
  window [12..16], uniform.

They were derived from different evidence and they differ in both window and weighting.
The oracle is expected to land close to but below the bar, and the exact distance is a
measurement the verifier makes on each run, not a number written anywhere.

## Cost

| stage | measured |
|---|---|
| frozen training run, 6144 micro-batches, compiled | ~100 s including one compile |
| one evaluation of one averaged state, 2.1M tokens | ~1-2 s |
| whole graded path: setup + one run + three evaluations | ~150 s |

The grading budget is 600 s and the harness timeout is 1320 s.

## Reward noise

The frozen run is seeded and the token stream is a deterministic function of position, so
the snapshots are reproducible on the same kernels. The residual noise is bf16 autocast
reduction order, which moves the third decimal place of a cross-entropy. Against a span of
0.19 nats that is well under one percent of the scale.

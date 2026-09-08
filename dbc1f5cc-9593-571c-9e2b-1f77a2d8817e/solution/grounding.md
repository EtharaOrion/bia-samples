# Grounding — where every number in this bundle came from

Nothing in this bundle is a declared anchor. The two endpoints of the reward scale are
measured by the verifier on every grading run. What is recorded here is the AUTHORING
search that chose the shipped default, the private reference and the oracle, so a reader
can see that those three artifacts are points of a real measured table rather than
assertions.

## The measurement

Every row below is a full training run of `environment/harness.py` over the frozen
8388608-token budget on `environment/data/train_slice.bin`, evaluated at the frozen
`16 x 512` geometry on `tests/holdout/val_slice.bin` — the split the verifier grades
against. Losses are validation cross entropy in nats; lower is better.

| plan | geometry as run | val loss |
|---|---|---|
| `ramp_3` | `64x128x1:348st -> 32x256x1:338st -> 16x512x1:337st` | 5.77967 |
| `ramp_2` | `64x128x1:512st -> 16x512x1:512st` | 5.77972 |
| `ramp_early` | `64x128x1:256st -> 16x512x1:768st` | 5.77983 |
| `ramp_late` | `64x128x1:768st -> 16x512x1:256st` | 5.78544 |
| `flat_32x256x1` | `32x256x1:1024st` | 5.80632 |
| `flat_8x1024x1` | `8x1024x1:1024st` | 5.81044 |
| `flat_16x512x1` | `16x512x1:1024st` | 5.81523 |
| `anti_ramp_2` | `16x512x1:512st -> 64x128x1:512st` | 5.83977 |
| `flat_64x128x1` | `64x128x1:1024st` | 5.85422 |
| `flat_32x512x1` | `32x512x1:512st` | 6.00955 |

## What the table says

1. **The optimizer-step count dominates every other axis.** Every geometry whose
   micro-batch is the smallest admissible 8192 tokens buys the full 1024 optimizer
   steps and lands between 5.780 and 5.854. `flat_32x512x1`, which doubles the
   optimizer-step token count and therefore halves the step count, lands at
   6.00955 -- 0.155 nats WORSE than the weakest full-step-count plan. The frozen
   square-root batch rule gives the larger step a larger learning rate and does not
   give back the updates.
2. **Context length has an INTERIOR optimum, and it is not the longest row.** At the
   same step count, 256-token rows (5.80632) beat 1024 (5.81044),
   512 (5.81523) and 128 (5.85422). "Use the longest context you can
   afford" is wrong here, and so is "use the context the evaluation uses"; a
   submission that reasons instead of measuring will land on one of them.
3. **A curriculum beats every single geometry, and its DIRECTION is not free.**
   Short-rows-then-long reaches 5.77967, better than the best flat plan by
   0.02665 nats. Running the same two phases in the opposite order --
   `anti_ramp_2`, long rows first -- gives 5.83977, which is 0.06004 nats
   worse than the same two phases the other way round and nearly the whole span. The
   ordering is the finding, not the mixture.

## The three artifacts

* **`environment/default_packing.json`** — `flat_64x128x1`, measured 5.85422. The
  cheapest shape on the grid and a plausible untuned choice. It is the LOW anchor and
  the verifier retrains it on every grading run.
* **`tests/private/reference_packing.json`** — `ramp_3`, measured 5.77967. The best
  point the search found. It is the HIGH anchor, staged only into the verifier image.
* **`solution/reference.py`** — `ramp_late`, measured 5.78544. A DIFFERENT point of
  the same search. Against the two anchors above it closes
  (5.85422 - 5.78544) / (5.85422 - 5.77967) = **0.923** of the gap.

## Measured span and what it means for reward resolution

The span between the anchors is 0.07455 nats. Repeat runs of this harness on this
accelerator differ only by the accelerator's own reduction order; the two nearly tied
curriculum points in the table above sit 0.00006 nats apart, which bounds the
resolution a submission can expect to be graded at.

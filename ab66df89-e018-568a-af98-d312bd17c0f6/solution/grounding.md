# Grounding: where every number in this bundle came from

## The claim

The slot grades a compute allocation by the validation loss of the model it trains,
inside a FLOP budget counted by declared arithmetic, and both ends of the reward scale
are measured on the same grading run.

## What was measured, and with what

Every figure below was produced by `frontier.py` and `measure16.py` in the authoring
tree, running `environment/harness.py` unmodified on one H100 — the same harness
`tests/grade.py` runs, and the same `micro_batch_flops` that bounds a submission on
the agent surface. Losses are on `tests/holdout/val_slice.bin`, the graded split.

## The menu, priced

FLOP budget 2e15. Corpus 19,521,536 tokens.

| entry | layers | dim | params | max micro-steps | tokens | epochs |
|---|---|---|---|---|---|---|
| m02x128 | 2 | 128 | 13.3M | 5735 | 47.0M | 2.41 |
| m03x192 | 3 | 192 | 20.7M | 3515 | 28.8M | 1.48 |
| m04x256 | 4 | 256 | 29.0M | 2383 | 19.5M | 1.00 |
| m06x384 | 6 | 384 | 49.3M | 1260 | 10.3M | 0.53 |
| m12x384 | 12 | 384 | 60.0M | 898 | 7.4M | 0.38 |
| m08x512 | 8 | 512 | 76.8M | 738 | 6.0M | 0.31 |
| m06x640 | 6 | 640 | 94.0M | 620 | 5.1M | 0.26 |

## The frontier, at full budget with grad_accum 1

| entry | micro-steps | epochs | val loss | wall |
|---|---|---|---|---|
| m02x128 | 5728 | 2.40 | **5.2740** | 119 s |
| m03x192 | 3512 | 1.47 | **5.2742** | 99 s |
| m04x256 | 2376 | 1.00 | 5.3706 | 72 s |
| m06x384 | 1256 | 0.53 | 5.6692 | 68 s |
| m12x384 | 896 | 0.38 | 5.8853 | 101 s |
| m08x512 | 736 | 0.31 | 5.9658 | 65 s |
| m06x640 | 616 | 0.26 | 6.0271 | 40 s |

Monotone across 0.75 nats, and it does **not** turn at one epoch of the corpus: the
two entries that must repeat data are the two best. The frontier is flat between them
— 0.0002 nats, well inside run-to-run noise on this substrate — which is why the
private reference and the oracle can name different entries without either being
wrong.

## The other two axes

| control | val loss | against |
|---|---|---|
| m06x384, half budget (624 steps), accum 1 | 6.1039 | 5.6692 at full budget |
| m06x384, full budget, accum 4 | 6.1055 | 5.6692 at accum 1 |
| m04x256, half budget, accum 1 | 5.7636 | 5.3703 at full budget |
| m04x256, full budget, accum 4 | 5.6931 | 5.3703 at accum 1 |
| m06x640, full budget, accum 4 | 6.4742 | 6.0271 at accum 1 |

Spending half the budget costs 0.39–0.43 nats. A four-fold optimizer batch costs
0.32–0.45. Both are comparable to, or larger than, the entire model-size effect
available on the menu, so an allocation has three ways to be wrong and all three are
expensive.

## The anchors

Neither anchor is a literal anywhere in this bundle.

* **Low.** `environment/default_allocation.json`: `m06x640`, 616 micro-steps,
  `grad_accum` 8 — the largest affordable model with a large optimizer batch, which
  is where both common instincts point. Retrained from scratch on every grading run.
* **High.** `tests/private/reference_allocation.json`: `m02x128`, 5728 micro-steps,
  `grad_accum` 1. Staged only into the verifier image.

If either fails to produce a finite loss the run refuses with `anchor-diverged`. If
the two coincide it refuses with `calibration-span-nonpositive`. Neither path
substitutes a stored value.

## The oracle is not the reference

`solution/reference.py` emits `m03x192` at 3512 micro-steps, `grad_accum` 1 — a
different menu entry from the reference, reaching the same place at 1.47 epochs rather
than 2.40. The gate's reference arm is therefore a real submission graded against a
bar it did not author.

## Cost of the graded path

Three runs: the default anchor at ~40 s, the reference anchor at ~119 s, and the
submission at whatever entry it names, 40–120 s. Plus one `torch.compile` per distinct
menu entry named, roughly 2 s each after the first model is built. Measured end to end
at approximately 300 s on a contended H100, against a 900 s budget. Because every
admissible allocation spends the same FLOP budget, no submission can make the graded
path materially more expensive than any other.

## What is not measured here

Run-to-run variation on this substrate is a few thousandths of a nat, which is smaller
than every gap in the frontier table except the one between the two smallest entries.
Both anchors are retrained on the same pass as the submission, so any drift common to a
run cancels out of the ratio.

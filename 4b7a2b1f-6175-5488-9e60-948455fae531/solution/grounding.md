# Grounding: where every number in this bundle came from

## The claim

The slot grades a training-data plan by the validation loss of the model it trains,
under a token budget fixed in forward and backward passes, and both ends of the reward
scale are retrained from scratch on every grading run.

## What was measured, and with what

Every figure below was produced by `measure08.py` and `measure08b.py` in the authoring
tree, running `environment/harness.py` unmodified on one H100 — the same harness
`tests/grade.py` runs. Held-out losses are on `tests/holdout/val_slice.bin`, the graded
split; devset losses are on `environment/data/devset_slice.bin`, which is what a solver
sees. Every row spends the identical 8,388,608-token budget over 1024 micro-steps.

## The substrate

| axis | value |
|---|---|
| decoder | 6 layers, model_dim 384, head_dim 64, seq_len 512, vocab 50304, 49.3M parameters |
| budget | 1024 micro-steps of 16 x 512, 8,388,608 tokens |
| pool | 8 sources of 4,194,304 tokens, 33,554,432 in all — the budget buys a quarter |
| run cost | 34–56 s per training run plus evaluation on a contended H100 |

The graded path is three such runs from one built model and one compiled graph:
measured at roughly 160 s against a 600 s budget.

## What each source is actually worth

The whole budget drawn from one source, that source cycled as needed:

| source | declared transform | held-out | devset |
|---|---|---|---|
| clean-beta | none | 5.8912 | 5.8335 |
| clean-gamma | none | 5.8916 | 5.8409 |
| clean-alpha | none | 5.9145 | 5.8605 |
| spliced-theta | alternating clean / permuted 512-token windows | 6.3784 | 6.3253 |
| scrambled-delta | 256-token windows permuted | 7.6163 | 7.5866 |
| scrambled-epsilon | 256-token windows permuted | 7.6213 | 7.5930 |
| looped-zeta | a 16384-token passage repeated | 11.9536 | 11.9325 |
| looped-eta | a 65536-token passage repeated | 12.8742 | 12.8324 |

The manifest declares the transform on each source. It does not, and could not,
declare this ordering: that block-permutation costs 1.7 nats while a 65536-token loop
costs 7.0 is a property of what the model can learn, not of the corpus description.

## The plans

| plan | held-out | devset |
|---|---|---|
| **the shipped default** — equal eighths, manifest order | **7.3881** | 7.3488 |
| clean thirds, order beta, gamma, alpha | **5.8038** | 5.7523 |
| clean thirds, order alpha, gamma, beta | 5.8363 | 5.7843 |
| clean thirds, order alpha, beta, gamma | 5.8382 | 5.7844 |
| clean-beta alone, cycled twice | 5.8899 | 5.8320 |
| clean trio interleaved in six draws | 5.8966 | 5.8361 |
| clean 3/4 then spliced-theta 1/4 | 5.9311 | 5.8791 |
| spliced-theta 1/4 then clean 3/4 | 6.0138 | 5.9615 |
| spliced-theta 1/2 then clean 1/2 | 6.1129 | 6.0607 |
| scrambled-delta 1/4 then clean 3/4 | 6.4965 | 6.4415 |
| clean 3/4 then scrambled-delta 1/4 | 7.4137 | 7.3901 |

Three things this establishes.

**Omission is the big win.** Dropping the five damaged sources takes the default from
7.3881 to 5.84 or better — about 1.55 of the 1.58 nats available. The handed template
cannot express omission, which is what makes this an option-expansion task rather than
a weighting one.

**Spreading beats cycling.** Three clean sources at a third each (5.8382) beats one
clean source cycled twice (5.8899). The budget is twice one source's length, so a
single-source plan sees everything twice; the third pass over fresh text is worth more
than a second pass over the same text.

**Order is a real axis and it is not subtle.** `clean 3/4 then scrambled 1/4` and
`scrambled 1/4 then clean 3/4` are the same tokens in the same proportion and differ by
**0.92 nats**. Among the clean-only orderings the spread is 0.034, which is the same
size as the gap between the clean sources measured alone.

The devset column ranks every plan in the same order as the held-out column, with an
offset of about 0.05. The proxy is faithful.

## The anchors

Neither anchor is a literal anywhere in this bundle.

* **Low.** `environment/default_plan.json`: equal eighths of all eight sources in
  manifest order. Retrained from scratch on every grading run.
* **High.** `tests/private/reference_plan.json`: clean-beta, clean-gamma, clean-alpha
  at 342 / 341 / 341 draws of 8192. Staged only into the verifier image.

If either fails to produce a finite loss the run refuses with `anchor-diverged`. If the
two coincide it refuses with `calibration-span-nonpositive`. Neither path substitutes a
stored value.

## The oracle is not the reference

`solution/reference.py` emits clean-alpha, clean-beta, clean-gamma at 341 / 341 / 342 —
a different order and a different placement of the odd draw. It measured 5.8382 against
the reference's 5.8038, so on the authoring measurements the oracle closes

    (7.3881 - 5.8382) / (7.3881 - 5.8038) = 0.978

of the gap rather than 1.0. The gate's reference arm is a real submission graded
against a bar it did not author, and its reward is a measurement rather than an
identity.

## What is not measured here

Run-to-run variation on this substrate is a few thousandths of a nat. It is smaller
than every gap in the plan table except the one between the top three clean orderings,
where a different seed could plausibly reorder them. Both anchors are retrained on the
same pass as the submission, so drift common to a run cancels out of the ratio.

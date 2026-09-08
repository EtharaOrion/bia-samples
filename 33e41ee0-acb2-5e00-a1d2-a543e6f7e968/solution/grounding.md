# OER-04 grounding

Every number in this file was produced by running this slot's own harness on this
slot's own substrate during authoring. Nothing here is a target the verifier reads:
the verifier measures both endpoints of its reward scale from scratch on every
grading run, and no literal for either appears anywhere in the bundle.

## The space

The parameter band [48,000,000, 50,600,000] admits **42 shapes** over
`num_layers` in [2, 16], `model_dim` in {256, 320, 384, 448, 512} restricted to
multiples of 64, `head_dim` in {32, 64, 128} and `mlp_ratio` in {2, 3, 4, 6}.

The band is what makes the choice a trade. At vocabulary 50304 the token embedding
plus the untied output projection are 2*50304*d + 50304 parameters — 78% of a
384-wide model — so a wider residual stream is paid for almost entirely in depth.
448 wide buys two blocks; 384 wide buys six at ratio 4; 320 wide buys fourteen.

## Measured

Every row trained by this harness on the identical token stream, under the identical
frozen optimizer and schedule, from the frozen seed, and evaluated on the held-out
slice:

| shape | params | val loss | train |
|---|---|---|---|
| 2L x 448, head_dim 32, mlp 4 | 49,950,336 | **5.244879** | 42s |
| 2L x 448, head_dim 64, mlp 4 | 49,950,336 | 5.101248 | 78s |
| 9L x 384, head_dim 64, mlp 2 | 49,332,480 | 5.063790 | 101s |
| 11L x 320, head_dim 64, mlp 6 | 50,313,664 | 5.057733 | 82s |
| 7L x 384, head_dim 64, mlp 3 | 49,033,344 | 5.054880 | 98s |
| 14L x 320, head_dim 64, mlp 4 | 49,497,984 | 5.051740 | 154s |
| 6L x 384, head_dim 64, mlp 4 | 49,326,720 | 5.030424 | 81s |
| 6L x 384, head_dim 128, mlp 4 | 49,326,720 | **5.018505** | 92s |

Three findings, all of them measurements rather than opinions.

**The frontier has an interior optimum.** Both ends lose. Two blocks at the widest
legal stream cannot compose enough, and fourteen thin blocks are worse than six
wider ones *and* cost nearly twice the wall clock for the same parameters.

**Trading MLP width for depth loses here.** At 384 wide, ratio 4 with 6 blocks beats
ratio 3 with 7 and ratio 2 with 9, even though all three sit in the band.

**Head width is a lever of its own, and the biggest one in the table.** `head_dim`
does not change the parameter count at all — it only decides how the same residual
width is split. Going from 6 heads of 64 to 3 heads of 128 is worth 0.0119 nats;
going the other way, to 14 heads of 32 at 448 wide, costs 0.1436 nats, which is more
than the entire depth decision. This is the axis a submission is most likely to
leave at its default.

## The anchors and the span

* **control** — `2L x 448, head_dim 32, mlp 4`, measured **5.244879**
* **reference** — `6L x 384, head_dim 128, mlp 4`, measured **5.018505**
* **span** — **0.226374 nats**

The control is not a straw man: it trains to a finite, ordinary loss in 42 seconds
and it is what the band most obviously invites — maximise width, then maximise head
count. It is wrong on both counts, and the task is finding that out.

## The oracle

`solution/reference.py` writes `6L x 384, head_dim 64, mlp 4`, measured 5.030424,
which closes **(5.244879 - 5.030424) / 0.226374 = 94.7%** of the gap.

It is a different document from the private reference and it reaches a different
loss. The reason it does not reach the bar is recorded rather than hidden: the
search behind the oracle explored depth, width and MLP ratio while holding head_dim
at 64, the conventional split for this architecture and the one the shipped default
would suggest. The private reference relaxed that. The reference arm of the gate is
therefore never the reference shape being graded against itself.

## The repeatability floor

The verifier trains the control shape **twice** on every grading run. Two runs of the
same shape start from identical tensors under the frozen seed and consume the
identical token stream; what separates them is only the order the accelerator reduced
in, which is non-zero on this hardware because the embedding and cross-entropy
backward passes accumulate with atomics.

That spread is this slot's ambiguous-intermediate-state threshold and it is measured
on every run rather than stored. A submission whose advantage over the control is
smaller than the floor is reported with
`separation-unproven-at-the-repeatability-floor`: neither a demonstrated improvement
nor a demonstrated failure. The reward is still exactly what the measurement says.

## Cost

Four from-scratch training runs plus four evaluations. The control is the cheapest
shape in the table at 42s and it is trained twice; the reference costs 92s; the
worst legal shape a submission could send costs about 175s. A grading pass measured
**about 560s** end to end on one H100 including three compilations, against a
declared budget of 1200s.

## A note on torch.compile

Every arm compiles a different shape from the same source function, and
`torch.compile` counts recompiles per code object rather than per module instance.
The stock limit is 8 and a grading pass needs one entry per shape per grad mode.
`harness.configure_backends()` raises that limit explicitly, because hitting it makes
dynamo fall back to eager silently — which would change both the wall clock and the
arithmetic in the middle of a graded run. This was observed during authoring before
it was fixed.

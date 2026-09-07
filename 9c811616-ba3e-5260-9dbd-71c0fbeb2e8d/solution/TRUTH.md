# FORGE-CANARY-BEGIN
# 0: c6e9a72e791a0e62fe8a1add61c2919dddd85f70e202c103f720ae7958721c02
# 1: 59c7fcb8906764fb9e61a693b1a638c1108ebb86cd564ec976267bf66b9fd2f0
# 2: 674df7c03d849146dff89096c43bd576c98ee6307beaded7629a45748f571251
# 3: 323ca789f906cc7ed67e7138d5fcc8d123058a8596d74d1a10ca3220ad1c80dc
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. source: solution/grounding.yaml -->

# TRUTH, slot OER-22

## What is graded

Perplexity degradation at a fixed bit budget, being `ppl(quantized) - ppl(unquantized reference)` over the held-out FineWeb folds, averaged over every scheduled point. Lower is better. Both perplexities are produced by `tests/evaluate.py` inside the verifier process, from the verifier's own pristine checkpoint, in the same run through the same code path. No number the submission reported or wrote ever enters the graded path.

## The thing the slot is about

The calibration set the allocation is fitted against **moves** between allocation time and evaluation time, and nothing announces the move. A calibration version is a pin into the frozen FineWeb10B train shards rather than a table of numbers, and the per-tensor cost curve is measured by real forward passes on whichever slice is in force, so a slice that moved is genuinely different activations and genuinely a different answer about which tensor deserves the next bit. An allocation tuned to the slice in force at iteration three is mis-tuned against the slice in force at the graded evaluation. `environment/calibration_probe.py` is the first-class handle for asking; it is a skill under test, not a trap with no handle.

## The discovery value the graded path depends on

The **content digest of the calibration slice in force**, being sha256 over the raw staged little-endian uint16 token bytes of the pinned slice. It is established only in built environment state, because it is a digest of FineWeb bytes the image put on disk rather than of anything written down in this bundle, and it appears on no agent-visible byte. A run must read it back out of that state through `environment/model.py slice_content_sha256` or the probe, and carry it as `calibration_slice_witness`. The checker `calibration_slice_witness_matches_live_state` refuses any other reading with `calibration-slice-content-not-established`. No value for that digest is published in this bundle, by construction, and none is invented here.

## The frozen substrate

| quantity | value |
|---|---|
| architecture | vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024 |
| quantizable tensors | 48, being 4 matmul weights per layer |
| quantizable parameters | 84934656 |
| excluded from allocation and budget | embeddings, position table, output head, norm gains |
| bit choices | 2, 3, 4, 5, 6, 7, 8 |
| group size, scale width | 128, 16 |
| bit budget, bits | 350355456 |
| budget accounting includes scale tensors | True |
| unquantized reference perplexity | measured in run, never declared |
| scheduled evaluation points | shard-a, shard-b, shard-c, shard-d, being four disjoint folds of the held-out validation slice |
| calibration version in force | 2 |

The budget is exactly what the uniform 4-bit allocation costs: `84934656 * 4 + 663552 * 16 = 350355456`. The uniform allocation therefore saturates it exactly, which is what makes it the baseline scale point. The same arithmetic decides which floors a fit may consider: a uniform-w allocation fits the budget for w in 2, 3, 4 and busts it for every wider w.

## The reference allocation procedure

`environment/model.py fit_allocation` is the procedure, expressed once and called by `solution/reference.py` on the agent surface and by `tests/evaluate.py` on the verifier surface. It cuts the pinned calibration slice into one window per scheduled evaluation point, measures the per-tensor cost curve at every admissible width on the FIRST window, reads that curve as its tightest non-increasing non-negative envelope, builds a candidate family over two derived knobs, and then selects on the windows it did not fit on.

The family is one candidate per admissible floor width, plus one per demotion cap, the cap being how many tensors may go below the uniform reference width and walked over the powers of two up to the tensor count, plus the uniform reference allocation itself. A candidate qualifies only if it is at least as good as the uniform allocation on EVERY selection window, which is the calibration-side reading of the per-fold invariant `check_degradation_sustained_across_scheduled_points` enforces, and among the qualifying candidates the lowest mean wins.

Three properties follow, and they are the three the predecessor procedure did not have. The fit is anchored to a candidate the budget admits for free, so it can never ship a fit its own measurement ranks below the baseline. The curve is measured at the widths the fit spends at rather than at one width and extrapolated, so the ordering that decides which tensor is starved comes from a reading with signal in it. And selection runs on tokens the curve never saw, because a candidate scored on the tokens it was fitted to is scored on the tokens it overfits.

## The two scale points

No allocation table and no degradation number is published here, and that is a consequence of the re-base rather than an omission. Under a real checkpoint they are the output of forward passes on a GPU over a corpus this bundle does not carry, so publishing a table of them would be publishing fabricated measurements. The verifier measures all three in run through one code path:

- baseline scale point: the uniform allocation at 4 bits, which saturates the budget exactly
- target scale point: the allocation `environment/model.py fit_allocation` returns over the calibration slice in force
- per-point bar: the baseline allocation's degradation at each fold

`raw = (baseline - agent) / (baseline - target)`, then `score = min(max(raw, 0.0), 1.0)`. The reference procedure reaches the target scale point by construction, because the target IS what that procedure measures, so it scores exactly 1.0. Beating it also scores 1.0; the target is a bar.

Every required checker gates the score. A failing required checker yields 0.0 with its machine-readable reason in `/logs/verifier/score.json`.

## The simulator test

First, weights in the loop. The graded artifact is a real parameter snapshot of the frozen architecture, and `check_graded_artifact_binds_to_declared_architecture` compares its state dict shape for shape against the map `environment/nanogpt_substrate.json` implies before anything is evaluated. A weight table, a cost model or a statistics blob fails that comparison. PASSES.

Second, the verifier recomputes. The graded scalar is produced by `tests/evaluate.py` inside the verifier process on a held-out FineWeb validation slice that `environment/Dockerfile` explicitly refuses to stage into the agent image and that is resolved only from `tests/bound.json`, which no solver-reachable call touches. Everything the in-container harness prints is telemetry. PASSES.

Third, no forward pass means no score. Delete `GPT.forward` in `environment/model.py` and every perplexity in this bundle raises. There is no closed-form degradation, no shipped sensitivity vector, no count table and no tick counter left anywhere that could keep emitting a number. PASSES.

## The regeneration chain

`solution/recompute.py` derives every artifact above from `solution/grounding.yaml` and from the architecture declaration that file names, and it VERIFIES rather than writes the one vendored carrier this bundle holds. `tests/eval_corpus.json` carries 1048576 real FineWeb10B token ids and the vendored record corpus; it is absent from the artifact map, so no emit path can overwrite it, and a length or sha256 that does not match the declaration stops the recompute rather than rebuilding over the top of real bytes.

## Declared gaps

- `gap-oer-per-family-anchors-unmeasured`: baseline_metric and target_metric are unbound for this slot. The reward schema is bound in full and instantiated against two scale points the verifier measures in run.
- `gap-oer-22-reward-scale-points-are-measured-in-run`: The two numbers the bound reward formula is instantiated against are in-run measurements over this bundle's own frozen checkpoint and the verifier's own held-out folds, not published evidence. They make the formula computable and they do not stand in for family anchors. This supersedes gap-oer-22-reward-scale-points-are-substrate-internal, which described the retired surrogate.
- `gap-oer-22-attempt-budget-unmeasured-against-real-forward-passes`: budget_hours is bound to 0.3 and was first set against a closed-form surrogate that did no forward pass. This lane has measured the verifier side of the real path and records that measurement in solution/TRUTH.md, and it has NOT measured a full solving attempt on the bound one-H100 envelope, so the per-attempt bound stays provisional.
- `gap-oer-22-calibration-transfer-unquantified`: The reference selects its allocation on windows of the calibration slice in force, and the graded reading is on held-out validation folds. This lane measured that the selected candidate beats the uniform baseline at every scheduled fold at 1500 checkpoint steps, and measured that it does not at 600 or at 150. It did NOT measure how far a calibration-slice ranking generalises in general, nor where between 600 and 1500 steps the crossing lies. The reference's margin over the baseline is a measured fact about this checkpoint and these folds, not a bound.
- `gap-oer-22-checkpoint-build-cost-exceeds-declared-attempt-budget`: The checkpoint step count is 1500, which is the smallest of the three counts this lane measured at which the reference reaches its own bar at every scheduled fold. It costs 1500 x 3.25s, being 81 minutes, on one H100 on each surface. environment/provision.sh materialises it on entry to the agent surface, so a solving attempt that pays that build exceeds the declared per-attempt budget_hours of 0.3, being 18 minutes. This lane did NOT move budget_hours to fit, because lowering a declared bound to admit a measurement is the one move it is not allowed to make. Which of the two values is wrong is not resolvable inside this slot: the step count is fixed by what makes the metric measurable at all, and the attempt budget is a batch-level grant.
- `gap-oer-22-contamination-screen-absent`: No contamination screen is authored here. probe-recall-a-published-recipe in shortcut_probes.yaml is recorded as NOT closed.
- `gap-oer-budget-field-name-collides-with-its-role`: budget_hours is bound to 0.3 and marked provisional in task.toml.
- `gap-oer-solver-egress-ruled-stricter`: see solver_egress_divergence above.

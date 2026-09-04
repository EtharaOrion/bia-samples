# FORGE-CANARY-BEGIN
# 0: 2a1062652f4a508600d65e99317da43589702a34350d3ca2a9b495495f1d66c3
# 1: d9c27dee0bb175a6f9c53a2303a772bb11c1c8684282c3c53d1c7575f2b696b2
# 2: eaf2283c82a161bb43dd5cb4361691f51b275ed93b96c9f12fc79ac9318c6cea
# 3: 1de63e9a1835dad0cc717351ac68638bd3d414989b114ebd36893cbca0ba38bf
# FORGE-CANARY-END
# TRUTH, slot OER-22

This file was hand-corrected by the OER-2026-08-20 re-base lane. It is no longer derived from `solution/grounding.yaml`, because that grounding file and `solution/recompute.py` still describe the retired statistics surrogate. Both must be re-derived against the bytes staged here before this slot is frozen, and until they are, `python3 solution/recompute.py --check` will report drift on every artifact this re-base touched. That is the residual work item and it is stated rather than hidden.

## What the re-base changed

This was a replacement of the model and the evaluation path, not a declaration-level repin. The prior revision graded a closed-form function, `degradation(alloc, s) = K * sum_i s_i * numel_i * 2 ** (-2 * alloc_i) / sum_i numel_i`, over twelve invented tensors totalling 108544 parameters and a shipped per-tensor sensitivity vector, against a stated reference perplexity of 12.75. There was no model, no checkpoint, no token and no forward pass anywhere on the graded path, and the whole graded quantity could be computed on paper from bundle bytes. A repin of a declaration would not have reached that; the arithmetic had to be replaced by a model.

What stands now: the artifact quantized IS a real nanoGPT checkpoint of the canonical 12-layer, 768-dim decoder declared at `environment/nanogpt_substrate.json`, carrying 84934656 quantizable parameters across 48 real matmul weights, and the graded scalar is the verifier's own forward pass of those parameters over a held-out slice of the FineWeb10B validation shards that is absent from the agent container.

## What is graded

Perplexity degradation at a fixed bit budget, being `ppl(quantized) - ppl(unquantized reference)` over the held-out FineWeb folds, averaged over every scheduled point. Lower is better. Both perplexities are produced by `tests/evaluate.py` inside the verifier process, from the verifier's own pristine checkpoint, in the same run through the same code path. No number the submission reported or wrote ever enters the graded path.

## The thing the slot is about

The calibration set the allocation is fitted against **moves** between allocation time and evaluation time, and nothing announces the move. The statement survives the re-base intact; what changed is that a calibration version is now a pin into the frozen FineWeb10B train shards rather than a table of numbers. The per-tensor sensitivity is measured, by `tests/evaluate.py measured_sensitivity` on the verifier side and by the solver's own forward passes on the agent side, so a slice that moved is genuinely different activations and genuinely a different answer about which tensor deserves the next bit. An allocation tuned to the slice in force at iteration three is mis-tuned against the slice in force at the graded evaluation. `environment/calibration_probe.py` is the first-class handle for asking; it is a skill under test, not a trap with no handle.

## The discovery value the graded path depends on

The retired surrogate scale constant is gone with the surrogate that carried it. Its successor is the **content digest of the calibration slice in force**, being sha256 over the raw staged little-endian uint16 token bytes of the pinned slice. It is established only in built environment state, because it is a digest of FineWeb bytes the image put on disk rather than of anything written down in this bundle, and it appears on no agent-visible byte. A run must read it back out of that state through `environment/model.py slice_content_sha256` or the probe, and carry it as `calibration_slice_witness`. The checker `calibration_slice_witness_matches_live_state` refuses any other reading with `calibration-slice-content-not-established`, so the environment stays load-bearing for a discovery value rather than decorative. No value for that digest is published in this bundle, by construction, and none is invented here.

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

The budget is exactly what the uniform 4-bit allocation costs: `84934656 * 4 + 663552 * 16 = 350355456`. The uniform allocation therefore saturates it exactly, which is what makes it the baseline scale point.

## The reference allocation, and the two scale points

No allocation table and no degradation number is published here, and that is a consequence of the re-base rather than an omission. Under the surrogate every one of those numbers was computable from bundle bytes with pure Python arithmetic. Under a real checkpoint they are the output of forward passes on a GPU over a corpus this bundle does not carry, so publishing a table of them would be publishing fabricated measurements. The verifier measures all three in run through one code path:

- baseline scale point: the uniform allocation at 4 bits, which saturates the budget exactly
- target scale point: the greedy marginal-gain allocation fitted over sensitivity measured on the calibration slice in force
- per-point bar: the baseline allocation's degradation at each fold

`raw = (baseline - agent) / (baseline - target)`, then `score = min(max(raw, 0.0), 1.0)`. The reference procedure reaches the target scale point by construction, because the target IS what that procedure measures, so it scores exactly 1.0. Beating it also scores 1.0; the target is a bar.

Every required checker gates the score. A failing required checker yields 0.0 with its machine-readable reason in `/logs/verifier/score.json`.

## The simulator test

First, weights in the loop. The graded artifact is a real parameter snapshot of the frozen architecture, and `check_graded_artifact_binds_to_declared_architecture` compares its state dict shape for shape against the map `environment/nanogpt_substrate.json` implies before anything is evaluated. A weight table, a cost model or a statistics blob fails that comparison. PASSES.

Second, the verifier recomputes. The graded scalar is produced by `tests/evaluate.py` inside the verifier process on a held-out FineWeb validation slice that `environment/Dockerfile` explicitly refuses to stage into the agent image and that is resolved only from `tests/bound.json`, which no solver-reachable call touches. Everything the in-container harness prints is telemetry. PASSES.

Third, no forward pass means no score. Delete `GPT.forward` in `environment/model.py` and every perplexity in this bundle raises. There is no closed-form degradation, no shipped sensitivity vector, no count table and no tick counter left anywhere that could keep emitting a number. PASSES.

## Declared gaps

- `gap-oer-per-family-anchors-unmeasured`: baseline_metric and target_metric are unbound for this slot. The reward schema is bound in full and instantiated against two scale points the verifier measures in run.
- `gap-oer-22-reward-scale-points-are-measured-in-run`: the two numbers the bound reward formula is instantiated against are in-run measurements over this bundle's own frozen checkpoint and the verifier's own held-out folds, not published evidence. They make the formula computable and they do not stand in for family anchors. This supersedes `gap-oer-22-reward-scale-points-are-substrate-internal`, which described the retired surrogate.
- `gap-oer-22-attempt-budget-unmeasured-against-real-forward-passes`: budget_hours is bound to 0.12 and was set against a closed-form surrogate that did no forward pass. Nothing in this bundle establishes what a calibration sweep and an allocation over the real checkpoint cost on the bound one-H100 envelope. This supersedes `gap-oer-22-substrate-is-a-deterministic-surrogate`, which is closed by the re-base: the surrogate is gone.
- `gap-oer-22-derivation-source-not-yet-re-derived`: `solution/grounding.yaml` and `solution/recompute.py` still describe the retired surrogate, so the generated-artifact chain does not currently recompute to these bytes.
- `gap-oer-22-contamination-screen-absent`: no contamination screen is authored here. probe-recall-a-published-recipe in shortcut_probes.yaml is recorded as NOT closed.
- `gap-oer-budget-field-name-collides-with-its-role`: budget_hours is bound to 0.12 and marked provisional in task.toml.
- `gap-oer-solver-egress-ruled-stricter`: see solver_egress_divergence in the slot's grounding.

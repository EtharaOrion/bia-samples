# FORGE-CANARY-BEGIN
# 0: 6f13a46ae33a78158a9263377b704acc7505bfac07e3c00bffa171e87ffd637b
# 1: b3765566900b51401938cec5d3c9ba24d02ed1d65120f729c70c1592e49b678e
# 2: bd9c3352489338c2d6c3b7507f57f8cf91a94d7390665017425ebea0d26813b0
# 3: d8ed3190881c0e9446602dfdccc521807ce37d820ec558639b5ef2fc790a822d
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. source: solution/grounding.yaml -->

# TRUTH, slot OER-22

## What is graded

Perplexity degradation at a fixed bit budget, measured against the unquantized
reference over the frozen held-out evaluation corpus. Lower is better. The
number is computed **inside the verifier process** from its own pristine copy of
the substrate under the submitted allocation. No number the submission reported
or wrote ever enters the graded path.

## The thing the slot is about

The calibration set the allocation is fitted against **moves** between allocation
time and evaluation time, and nothing announces the move. An allocation tuned to
the calibration statistics as of iteration three is mis-tuned against the set in
force at the graded evaluation. The agent must reason about WHEN each calibration
statistic was true. `environment/calibration_probe.py` is the first-class handle
for asking; it is a skill under test, not a trap with no handle.

## The discovery value the graded path depends on

The surrogate scale constant is established only in built environment state, at
`environment/substrate.json` `degradation_constant_K`, and appears on no
agent-visible byte. A run must read it back out of that state, replay the probe the
substrate's `degradation_form` defines over the uniform reference allocation against
the in-force calibration sensitivity vector, and carry the digest as
`surrogate_scale_witness`. The checker
`surrogate_scale_witness_matches_live_substrate` refuses any other reading with
`surrogate-scale-constant-not-established`, so the environment is load-bearing for
the constant rather than decorative.

| quantity | value |
|---|---|
| probe digest the verifier expects | `e1e72240deb6299474577024a7e9fd53eafc4c5e01f3311504fb0f10b40d56a9` |
| same probe under the constant displaced by 0.1 | `b5747bb8e136167f1291b5843d255aa0e9eb90521c555fa98fc9f76d0d5cdc8f` |
| same probe under a unit constant | `d9a5fd2baf274521a1ca9deccf787668e1dd3d38f09282f244743b1a759ae855` |

## The frozen substrate

| quantity | value |
|---|---|
| tensors | 12 |
| total parameters | 108544 |
| bit choices | 2, 3, 4, 5, 6, 7, 8 |
| bit budget, bits | 447744 |
| budget accounting includes scale tensors | True |
| unquantized reference perplexity | 12.75 |
| scheduled evaluation points | shard-a, shard-b, shard-c, shard-d |
| calibration version in force | 2 |

## The reference allocation

Fitted by greedy marginal gain against the calibration version in force.

| tensor | bits |
|---|---|
| blk00.attn.qkv | 4 |
| blk00.attn.out | 3 |
| blk00.mlp.up | 4 |
| blk00.mlp.down | 4 |
| blk01.attn.qkv | 5 |
| blk01.attn.out | 3 |
| blk01.mlp.up | 4 |
| blk01.mlp.down | 4 |
| blk02.attn.qkv | 4 |
| blk02.attn.out | 4 |
| blk02.mlp.up | 5 |
| blk02.mlp.down | 3 |

Allocated bits: 447744 of 447744.

## Graded degradation, per evaluation point

| point | bar | reference | sustained |
|---|---|---|---|
| shard-a | 0.277196344 | 0.254201061 | True |
| shard-b | 0.282694575 | 0.267239092 | True |
| shard-c | 0.298621757 | 0.265941922 | True |
| shard-d | 0.312724794 | 0.290016952 | True |

Mean graded degradation: 0.269349757.

## Reward

Family anchors are ABSENT under `gap-oer-per-family-anchors-unmeasured`, so no
`baseline_metric` and no `target_metric` number is bound. The bound reward SCHEMA
is instantiated against substrate-internal scale points recomputed from frozen
bytes, declared under `gap-oer-22-reward-scale-points-are-substrate-internal`:

- baseline scale point: 0.292809367
- target scale point: 0.269349757

`raw = (baseline - agent) / (baseline - target)`, then
`score = min(max(raw, 0.0), 1.0)`. The reference reaches the target scale point,
so it scores exactly 1.0. Beating it also scores 1.0; the target is a bar.

Every required checker gates the score. A failing required checker yields 0.0
with its machine-readable reason in `/logs/verifier/score.json`.

## Declared gaps

- `gap-oer-per-family-anchors-unmeasured`: baseline_metric and target_metric are unbound for this slot. The reward schema is bound in full and instantiated against substrate-internal scale points recomputed from frozen bytes.
- `gap-oer-22-reward-scale-points-are-substrate-internal`: The two numbers the bound reward formula is instantiated against are derived from this bundle's own frozen substrate, not from published evidence. They make the formula computable and they do not stand in for family anchors.
- `gap-oer-22-substrate-is-a-deterministic-surrogate`: reference_hours measures the cost of deriving, submitting and grading an allocation over the closed-form surrogate. It does NOT cover a live forward pass, a real quantization kernel, GPU residency, or the wall time of a real perplexity evaluation over a token corpus. Nothing in this bundle establishes what those would cost on the bound one-H100 envelope.
- `gap-oer-22-contamination-screen-absent`: No contamination screen is authored here. probe-recall-a-published-recipe in shortcut_probes.yaml is recorded as NOT closed.
- `gap-oer-budget-field-name-collides-with-its-role`: budget_hours is bound to 0.12 and marked provisional in task.toml.
- `gap-oer-solver-egress-ruled-stricter`: see solver_egress_divergence above.

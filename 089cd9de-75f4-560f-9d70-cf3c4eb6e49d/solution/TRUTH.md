# FORGE-CANARY-BEGIN
# 0: 5843b8dae079fc36a7a6b39e74381df547d77d4942f5d7deda3b09c248dbe1e4
# 1: e085c3c8822a1f407d72d7730738f9f72748382149b65a215575ea40660391a9
# 2: 20cc0c51ff508bea79c441b8a3e29ff1b2ee52481c165ae347f6bcc944b3ca0c
# 3: ddd3b794bebb1a783771413f2edcc9ba04f35dcd259dcea67d8e589a737b4e87
# FORGE-CANARY-END
<!-- GENERATED SECTION. DO NOT HAND-EDIT. -->
<!-- Source: solution/grounding.yaml -->
<!-- Generator: solution/recompute.py -->

# OER-15 TRUTH

`GENERATED SECTION. DO NOT HAND-EDIT.` Every number below is transcribed from `solution/grounding.yaml`.

## What is graded

Metric: **bits per byte at fixed compute**, direction **lower is better**.

The verifier computes the number itself, from the model state its own harness
held at the bound evaluation point, over the frozen evaluation corpus measured
in **bytes**. It is never a number the submission reported, never a smoothed or
averaged series, and never a checkpoint the submission selected.

## Frozen axes

| axis | value |
|---|---|
| `compute_budget_updates` | `2200` |
| `encoder` | `greedy longest match, maximum token length 16 bytes` |
| `eval_corpus_bytes` | `5739` |
| `eval_corpus_digest` | `08b8c7f2c2c27609015850751fb832d9ebf8034ef38907aae02a7ee50fa978ce` |
| `model` | `bigram with a fixed two-level interpolation, alpha 1.0 toward the unigram and beta 1.0 toward the uniform floor` |
| `model_spec_digest` | `394e561b626899f39c68badf5444624e30c9b8cdee2cae10c2a2ba9259645ae6` |
| `optimizer` | `streaming count update, one update per token position visited` |
| `train_corpus_bytes` | `21599` |
| `train_corpus_digest` | `7059a0e5b11dfa95fc9c16c4d60317aaa6c5d3c57f8331bcd2c105bf6d5635ac` |
| `vocab_budget` | `1280` |

Free axis: **the construction of the vocabulary**.

## The archetype, measured rather than asserted

Archetype **AR5**, adversarial option expansion. The default
construction handed to the agent is `environment/default_tokenizer.py, whole-word top-k frequency list`.
Its complete option grid carries **60 rows**, and this lane swept every one of
them on this substrate.

| quantity | measured |
|---|---|
| default construction optimum | `0.33735183879296543` bits per byte |
| default construction worst row | `1.9746683855542342` bits per byte |
| reference construction optimum | `0.2818765313056418` bits per byte |
| **measured plateau gap** | `0.055475307487323655` bits per byte |
| acceptance bar | `0.3173518387929654` bits per byte |

The optimum row of the handed grid is `{"attach_leading_space": true, "lowercase": false, "min_frequency": 1, "top_k": 256}`.

The plateau is a property of the **construction**, not of the knobs. The corpus
carries 201 distinct word types, so every unit of vocabulary budget past
that count is unusable by a word-list construction no matter which option values
are chosen. Raising `top_k` further changes nothing.

Stability of the gap across compute budgets, each row measured:

| compute budget updates | default optimum | reference optimum | gap |
|---|---|---|---|
| 800 | 0.3724 | 0.3145 | 0.0579 |
| 1200 | 0.3536 | 0.2971 | 0.0565 |
| 2200 | 0.3374 | 0.2819 | 0.0555 |
| 4000 | 0.3281 | 0.2718 | 0.0563 |

## The derivation the agent is expected to perform

1. Read instruction.md and environment/manifest.json; record that the denominator is 5739 frozen bytes and the compute budget is 2200 token updates.
   - establishes: what is graded and what is frozen
2. Read environment/default_tokenizer.py and enumerate DEFAULT_OPTION_GRID; note it has 60 rows and that every row produces whole whitespace-delimited words only.
   - establishes: the handed option set, and its structural limit
3. Sweep all 60 rows through harness.bits_per_byte and take the minimum.
   - establishes: the plateau at 0.33735183879296543 bits per byte, reproduced rather than accepted
4. Observe that top_k above 256 changes nothing because the corpus carries 201 distinct word types, so the plateau is a property of the construction and not of the knobs.
   - establishes: the option set is exhausted and the objective is not
5. Re-read the objective. Compute is counted in token positions, the denominator is a frozen byte count, so a vocabulary that expresses the same bytes in fewer tokens covers more corpus per update and costs fewer predictions at evaluation.
   - establishes: the derivation of the wider construction
6. Build the wider construction: greedy merges of the most frequent adjacent symbol pair over the training corpus, ties broken lexicographically, stopping when no pair occurs twice or the budget is spent.
   - establishes: 623 merges, a vocabulary of 879 entries
7. Run it through the harness and read the bound point at 2200 updates.
   - establishes: 0.2818765313056418 bits per byte, a gap of 0.055475307487323655 below the plateau
8. Check the two verifier-scheduled sustain points at 2310 and 2420 updates before claiming anything.
   - establishes: the reading is sustained, not a transient dip
9. Write submission/tokenizer.py exporting build_vocab, declare no HALT_AT_UPDATES, and let the verifier compute the number.
   - establishes: full reward, 1.0

## The reference answer

`solution/reference.py, greedy corpus-derived pair merges` produces **623 merges**, a vocabulary of **879 entries**,
encoding the evaluation corpus in **970 tokens** and the training corpus in **3641 tokens**.

## Anchors

`anchors_state: absent` under gap `gap-oer-per-family-anchors-unmeasured`.
`baseline_metric` is `None` and `target_metric` is `None`.

seed/contract.yaml families.entries.F10 carries scope_state unscoped and scope null, so no published F10 anchor exists. The reward SCHEMA is bound in full and the two numeric anchors are declared absent rather than invented. The carriers the scoring path actually uses are the substrate-measured operating points above, named as such wherever they appear, and they are not presented as family anchors.

## Reward

Carrier `/logs/verifier/reward.txt`, one bare float. Reason and
metric block at `/logs/verifier/score.json`. Interval `[0.0, 1.0]`, 
higher is better, never binary.

Formula: `raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric); score = min(max(raw, 0.0), 1.0)`

The two carriers standing in for the absent anchors are `measured.default_construction_optimum_bpb`
and `measured.reference_construction_optimum_bpb`, both substrate-measured operating points
recorded by this lane and named as such wherever they appear. The reference
construction scores `1.0`; the default
construction's own optimum scores `0.0`.

## Statement ambiguity

`outcomes_admitted` is `1`. The readings that reduce to it:

| reading | source | numerator | denominator bytes | point | direction |
|---|---|---|---|---|---|
| `reading-from-the-fraction` | instruction.md, the displayed fraction under 'What you are optimizing' | raw bits the model assigns to environment/corpus/eval.txt | `5739` | the bound evaluation point, 2200 updates | lower |
| `reading-from-the-graded-list` | instruction.md, item 1 of 'What is graded, in one list' | raw bits the model assigns to environment/corpus/eval.txt | `5739` | the bound evaluation point, 2200 updates | lower |

## Solver egress divergence

This bundle binds solver_egress open at the agent surface and verifier egress denied, as the lane brief requires, and records the divergence rather than amending the batch-level field. This lane does not amend seed/contract.yaml.

Gap `gap-oer-solver-egress-ruled-stricter`.

## Bound grants

| field | value | role |
|---|---|---|
| `max_timeout_hours` | `6.0` | refinement-loop terminator across attempts |
| `budget_hours` | `0.12` | per-attempt completion bound, 7.2 minutes |
| `max_attempts` | `50` | attempts per session |
| `final_selection` | `best` | best of k |
| `budget_envelope` | `one H100, single accelerator, no multi-GPU scaling` | |
| `budget_margin` | `10 percent` | carried by name |

`budget_hours` is `provisional` under gap `gap-oer-budget-field-name-collides-with-its-role`. `max_timeout_hours` and `budget_hours` are
never aliases: the first bounds the session across attempts, the second bounds
one attempt.

## Tier

Target tier `Frontier-defeat`, `tier_exemption_granted: none`.
Frontier-defeat floor `0.5`, separation margin `0.05`.
Anchorable tier capped below Hard.

This task family is text and code only, so the Perception axis carries no category. The exemption is batch-level, recorded once, closes by nothing, and no perception lever is authored to fill it. It caps the anchorable tier below Hard while leaving the authoring target at Frontier-defeat.

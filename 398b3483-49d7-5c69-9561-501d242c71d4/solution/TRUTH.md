# FORGE-CANARY-BEGIN
# 0: 84fbbcc23270485ce7d52aac1a833e30c8e185feddead16fa1c6e210b7ef06dd
# 1: 713b339700cfc64946bf478941773b0fa3fd7746c53002cafc5750339dce3a56
# 2: 4ef9ab8098eeb89b7f85ab62c75254483281b7c5fae1c33060388c7eede20136
# 3: d80b9ef439f727947c62ec860417d054af11d176ac16dc00d0feb553cc53d8c6
# FORGE-CANARY-END
# TRUTH.md

GENERATED SECTION. DO NOT HAND-EDIT.

source: solution/grounding.yaml

## What is graded

Accuracy of the frozen model on the frozen held-out benchmark, evaluated by the verifier from harness-owned model state at the bound evaluation point.

Direction: higher-is-better.
Reward: `raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)`, then `score = min(max(raw, 0.0), 1.0)`.

## Anchors

anchors_state: absent

baseline_metric: absent. target_metric: absent. Gap `gap-oer-per-family-anchors-unmeasured`.

F14 anchors are unmeasured. This slot binds the reward schema in full and declares the two anchor values absent rather than inventing them. No baseline_metric or target_metric number appears anywhere in this bundle.

With the anchors absent the normalization has no denominator, so grade.py resolves the reward to 0.0 carrying reason anchors-absent-unnormalizable and records the verifier-computed agent_metric in the score document for the batch to normalize once anchors land. The arithmetic itself is exercised in adequacy.py against a fixture-supplied anchor pair that is declared as fixture material and is not a slot binding.

## The archetype this slot carries

AR6, silent execution failure. A generator can collapse to a low-diversity
mode partway through a run, keep emitting, keep the sample count and the
format exactly right, and exit zero without announcing anything. The corpus
is degenerate and every shape check passes. This slot grades that corpus as a
failure with its own reason, `generator-collapsed-silently`, which is
deliberately distinct from any low-score reason: a degenerate corpus that
happens to score above the floor is still a collapse and is named as one.

## The reference

`solution/reference.py`, sha256 `5a035c79faa72c5072e51279bd089a8d7c9566e64917845a702c9228d4dd9aa0`.

Measured profile of the reference corpus, recomputed by the verifier:

| statistic | reading | bound |
|---|---|---|
| corpus distinct-trigram ratio | 0.238273 | reported, not gated |
| weakest segment distinct-trigram ratio | 0.219144 | floor 0.34 |
| max mode share | 0.008333 | ceiling 0.08 |
| benchmark near-duplicate hits | 0 | must be zero at threshold 0.55 |
| optimizer updates fed | 1440 | bound 1440 |
| graded accuracy at update 1440 | 0.875 | agent_metric, unnormalizable while anchors are absent |

## How the graded quantity resists a shaped readout

- The score is the verifier's own evaluation, recomputed from the confusion
  counts its evaluator recorded. It is never a number the submission printed.
- The graded readout is raw. A smoothed readout on the graded path scores zero
  with `readout-smoothing-on-graded-path`.
- A reading must hold across every point the verifier schedules, [960, 1200, 1440], within 0.08.
- A run halted at a favourable evaluation is graded as not having established
  the score, with `early-stop-score-not-established`, never as an absent result.
- The evaluated weights are the harness trainer's own state at the bound point.

## Perception-axis exemption

This task family is text and code only, so the Perception axis carries no category. The exemption is batch-level, recorded once, closes by nothing, and no perception lever is authored to fill it. It caps the anchorable tier below Hard while leaving the authoring target at Frontier-defeat.

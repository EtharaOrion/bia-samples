# TRUTH.md

GENERATED SECTION. DO NOT HAND-EDIT.

Source: `solution/grounding.yaml`. Regenerate with `solution/recompute.py`.

## What is graded

Query plan speedup over the frozen baseline plan, gated on result-set equality. Direction: higher is better. The gate is HARD: a plan whose result set differs from the required result set by one row, one column or one ordering position scores exactly 0.0 with the reason `result-set-divergence`, however cheap it is.

Cost is the database engine's own virtual-machine step count for the whole result, read off the engine progress callback inside `environment/planrun.py`. It is not a wall-clock time and it is never presented as one, under the declared gap `gap-oer-29-cost-is-a-step-count-not-a-wall-clock-time`. The step count is only portable while the engine build is fixed, which is why both Dockerfiles pin the same base image by digest and why `tests/grade.py` refuses with `engine-cost-model-drift` if the frozen baseline plan does not cost what it cost when these constants were derived.

## The archetype

Exclusion closure. Exactly one lineage row fails the frozen integrity rule, and excluding it invalidates every row that transitively depends on it. The dependency relation is the union of two edge sets, the parent edges in `lineage` and the edges in `derives`, so a node can be reached by more than one path and the closure has to be run to fixpoint rather than read off one join.

The failure mode the slot targets is stopping at the first hop. The excluded row has 5 immediate successors, so a first-hop answer names 6 nodes; the closure actually reaches 30 nodes at depth 10. A first-hop answer therefore looks like an answer, produces a plan that runs, and is refused with `closure-incomplete`.

## The two discovery values

Neither value appears in `instruction.md`, in `task.toml`, or in any authored file under `environment/`. Both are established by the first Docker build stage, which composes the warehouse from the frozen spec and perturbs exactly one declared checksum, and both are read back by querying the built database through `environment/probe.py run`.

| quantity | value |
|---|---|
| excluded row identifier | nd-1522 |
| closure cardinality | 30 |
| closure depth | 10 |
| closure layer profile | 1, 5, 1, 4, 4, 4, 3, 2, 3, 2, 1 |
| readings inside the closure | 23 |
| required result rows | 1577 |
| baseline plan cost, harness steps | 57926 |
| reference plan cost, harness steps | 24043 |
| instance_baseline_speedup | 1.0 |
| instance_target_speedup | 2.409267 |

## How the oracle found them

The oracle does not know either value in advance and does not read them from a file. It runs one read-only query for the row whose declared checksum disagrees with its own payload, then walks the union of the two edge relations breadth first from that row until the frontier is empty, recording the layers as it goes. The cardinality is the length of what the walk returned. Only then does it fold the result into a plan.

`solution/oracle_run.jsonl` is the step trace that run actually emitted inside the built environment image, and the `plan.sql` and `submission.json` it wrote there are byte-identical to the frozen fixture under `solution/fixtures/reference_run/`. That recording is not a generated artifact and `recompute.py --check` does not cover it.

## The reference plan

The closure is computed once, off the graded path, and folded into the plan as a literal exclusion list. The recursive term of the baseline plan, the two joins it drives and the whole scan of the `derives` table then disappear, and what is left is one scan of `reading` against a small set membership test in `reading_id` order.

## Anchors

`anchors_state: absent`, gap `gap-oer-per-family-anchors-unmeasured`. F6 carries no measured family baseline or target, so none is invented. The reward schema is bound in full and the two normalisation numbers above are measured on this slot's own frozen substrate inside the pinned image; they are instance-local and are NOT family anchors, under gap `gap-oer-29-instance-local-anchors-not-family-anchors`.

raw = (agent_metric - instance_baseline_speedup) / (instance_target_speedup - instance_baseline_speedup)

score = min(max(raw, 0.0), 1.0). Reaching the target scores exactly 1.0 and beating it also scores 1.0.

## The clock

No clock is read anywhere on the graded path. The cost is a step count the engine reports, every checker reads recorded state through the `tests/checkers.py` Harness handle, and `recompute.py` reads no clock either.

## What is not established

No signed pilot has been run against this bundle, under `gap-oer-29-no-signed-pilot`, so the target tier is an authoring target and not a measured one. The bundle carries no detached provenance signature, under `gap-oer-29-no-detached-signature`, because the authoring lane holds no signing key; the absence is recorded rather than filled in.

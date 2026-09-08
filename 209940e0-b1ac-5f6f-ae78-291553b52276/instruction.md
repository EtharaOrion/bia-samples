# Admitted-set query plan under a quarantine closure

You are handed a frozen provenance warehouse and one query to make fast. Your job is to return the **required result set** using **fewer harness-counted steps** than the frozen baseline plan does.

## What is graded, stated plainly

The graded quantity is **speedup over the frozen baseline plan**, higher is better, **subject to exact result-set equality**.

Read the next three points carefully, because they decide what is worth doing.

1. **The verifier runs both plans itself.** After you submit, the verifier executes the frozen baseline plan and your plan on the harness's own executor, over the frozen warehouse, and computes the speedup from its own step counts. No number you print, report or write into a file becomes the graded number. `reported_plan_cost` and `reported_baseline_cost` exist in the submission so a mismatch is visible, not so they can be used.

2. **Result-set equality is a HARD GATE.** It is not a penalty term you can trade against speed. A plan whose result set differs from the required result set by one row, one column or one ordering position scores exactly `0.0`, however cheap it is. The cheapest plan in the space returns nothing at all, and it is not the answer.

3. **The plan is one read-only statement.** Exactly one statement, opening with `SELECT`, `WITH` or `VALUES`. The warehouse is opened read-only and an authorizer refuses attach, pragma, and every write. A plan that is not a single read-only statement scores `0.0` with the reason `plan-not-read-only-single-statement`.

Cost is the database engine's own virtual-machine step count for the whole result, not a wall-clock time. Your own timing is not graded and cannot move the reward.

## The warehouse

`/task/environment/warehouse.db`, read-only, three tables:

| table | columns |
|---|---|
| `lineage` | `node_id`, `parent_id`, `p_alpha`, `p_beta`, `p_gamma`, `declared_check` |
| `derives` | `edge_id`, `source_node`, `target_node` |
| `reading` | `reading_id`, `node_id`, `value` |

## The exclusion rule

A `lineage` row is **intact** when the checksum it declares agrees with its own payload, and **quarantined** when it does not:

```
intact  <=>  declared_check = (p_alpha * 31 + p_beta * 17 + p_gamma * 7) % 1000003
```

**Exactly one row in the warehouse is quarantined.** Which one is a property of the built warehouse. It is not written down anywhere and you are expected to find it.

## The closure operation

Quarantine propagates forward along the **dependency relation**, which is the union of two edge sets and not either one alone:

- a `lineage` row depends on the row its `parent_id` names, so there is an edge from the parent to the child;
- a `derives` row records that `target_node` depends on `source_node`, so there is an edge from the source to the target.

The **closure** of the quarantined row is that row together with every node reachable from it along those edges, transitively, to fixpoint. A node reachable only through several hops, or only through a chain that alternates between the two edge sets, is inside the closure exactly as much as an immediate successor is. The immediate successors of the quarantined row are the first hop of that traversal and they are not the traversal.

## The required result set

Every `reading` row whose `node_id` is **outside** the closure, projected as `reading_id`, `node_id`, `value`, in ascending `reading_id` order. Nothing else, in no other order, under no other column names.

## The frozen baseline plan

This is the denominator of your speedup. It is correct, and it is the plan to beat.

```sql
WITH RECURSIVE quarantined(node_id) AS (
    SELECT node_id
      FROM lineage
     WHERE declared_check <> (p_alpha * 31 + p_beta * 17 + p_gamma * 7) % 1000003
    UNION
    SELECT l.node_id FROM lineage l JOIN quarantined q ON l.parent_id = q.node_id
    UNION
    SELECT d.target_node FROM derives d JOIN quarantined q ON d.source_node = q.node_id
)
SELECT reading_id, node_id, value
  FROM reading
 WHERE node_id NOT IN (SELECT node_id FROM quarantined)
 ORDER BY reading_id
```

## The handle

`environment/probe.py` is how you read the built warehouse back. There is no subcommand that hands you a quarantine verdict or a closure; `run` will execute any read-only query you can write against the frozen tables, and everything you need is reachable from there.

```sh
python3 /task/environment/probe.py summary
python3 /task/environment/probe.py baseline
python3 /task/environment/probe.py run --plan myplan.sql --preview 5
```

## What to submit

Write `plan.sql` and `submission.json` in the workspace root. `plan.sql` holds your plan, as one read-only statement and nothing else.

```json
{
  "schema": "oer29.submission/v1",
  "excluded_node_id": "<the node_id the exclusion rule selects>",
  "closure_nodes": ["<every node in the closure, sorted, the excluded row included>"],
  "closure_layers": [["<layer 0: the excluded row alone>"], ["<layer 1>"], ["<layer 2>"]],
  "closure_cardinality": 0,
  "admitted_row_count": 0,
  "reported_plan_cost": 0,
  "reported_baseline_cost": 0
}
```

- `closure_layers` is the breadth-first witness of your traversal: layer zero is the excluded row on its own, and layer `k` holds the nodes first reached at distance `k`, sorted. It is checked layer for layer, so it records the walk you actually did rather than the set you ended with.
- `closure_cardinality` is checked three ways over: against the closure the verifier walks for itself, against your own `closure_nodes`, and against your own `closure_layers`. A figure that was reported rather than derived fails at least one of the three.
- `admitted_row_count` is the number of rows your plan returns.

## Scoring

Stated once more, independently of the section above, so there is exactly one reading: the graded metric is **speedup over the frozen baseline plan**, **higher is better**, and exact result-set equality is a **hard gate** on it rather than a term traded against it.

One float in `[0.0, 1.0]`, higher is better, never binary. Every zero carries a machine-readable reason. Full reward is exactly `1.0`, and beating the target also scores `1.0`; the target is a bar, not a point on a gradient.

`raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)`, then `score = min(max(raw, 0.0), 1.0)`. The baseline metric is the frozen baseline plan's speedup against itself, which is `1.0`.

## Two notes on honesty

The cost is a **step count**, not a wall-clock measurement, and it is never presented as one. That is deliberate: a step count over frozen bytes is exactly reproducible, so the same plan earns the same verdict on any host running the pinned image.

Nothing announces that a traversal stopped early. A closure that took only the first hop terminates, produces a list, and produces a plan that runs and returns rows. The verifier is the thing that notices, and by then the run is over.

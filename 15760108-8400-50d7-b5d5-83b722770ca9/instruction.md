# Multi-objective frontier: build a Pareto set, not a point

## Objective

Two objectives are frozen. You are scored on the hypervolume of the frontier you achieve across both of them, not on either one alone. The optimizer is yours. Everything else is not.

Objective one is validation loss, in nats, on a frozen held-out shard of a frozen synthetic source. Lower is better. Objective two is weight-matrix density, the fraction of weight entries that survive a scale-invariant magnitude threshold. Lower is better. The tradeoff between them is real and is declared below: capacity you remove from the weight matrices is capacity you cannot spend on predicting the source.

A graded attempt evaluates exactly **six** frontier points. Each point is one short training run of the frozen model under an optimizer you configure. You choose all six configurations before any of them runs.

## What you submit

Write `submission/frontier.py`. It must expose exactly two callables.

```python
FRONTIER_POINTS = 6

def propose_frontier() -> list:
    """Return exactly six entries. Each entry is opaque to the harness and is
    handed straight back to build_optimizer as its `point` argument. Use it to
    carry whatever your update rule needs to place that point on the surface."""

def build_optimizer(named_params, point, lr, **kwargs) -> torch.optim.Optimizer:
    """named_params is a list of (name, torch.nn.Parameter) pairs for the whole
    live model, so you can treat embeddings, block matrices and vectors
    differently. point is one entry from propose_frontier. lr is the frozen base
    learning rate; you may rescale it inside your rule, and you may not ignore it
    without saying so through `point`."""
```

Returning a count other than six is a hard failure and scores zero. Returning anything that is not a `torch.optim.Optimizer` is a hard failure and scores zero.

Produce your runs by invoking `harness/run_frontier.py`. That runner is the only writer of the telemetry record. A log you write by hand is not evidence, and a telemetry directory containing a file the runner did not write scores zero.

## How the two objectives are measured

**Validation loss.** Mean next-token cross entropy in nats over the frozen validation shard, measured after your final optimizer step, with the model in eval mode. The harness owns the measurement and the shard.

**Density.** For every two-dimensional parameter tensor in the model, that is both embeddings and the four block matrices of every layer, an entry is retained when its absolute value is at or above `density_tau` times the root mean square of its own tensor. Density is the retained fraction over all such entries, weighted by entry count. The threshold is relative to each tensor's own root mean square, so uniformly rescaling your weights does not move this number. It is not a route.

`density_tau` and every other frozen constant are in `harness/frontier_spec.json`. Read it. It is the contract.

## How you are graded

The score is a single float on the closed interval from zero to one, higher is better. It is the hypervolume of your achieved set, clamped.

Each objective is mapped to a normalized coordinate on the closed unit interval by the fraction of the reference-to-ideal distance it covers.

| Axis | Reference coordinate, normalized to 0 | Ideal coordinate, normalized to 1 |
|---|---|---|
| validation loss | `ln(vocab_size)`, the cross entropy of the uniform next-token predictor | `entropy_rate + loss_offset_ideal`, the information-theoretic floor of the frozen source plus a declared margin |
| density | `erfc(density_tau / sqrt(2))`, the retained fraction of a dense unstructured Gaussian matrix | `density_ideal`, the declared sparse ideal |

Both coordinates of both points are closed-form functions of the frozen bytes, so the normalization needs no measured reference run and does not move with the machine. The runner prints all four resolved numbers before it trains anything.

In that normalized space, higher is better on both axes, the **hypervolume reference point is `(0.0, 0.0)`** and the **ideal point is `(1.0, 1.0)`**. Your score is the area of the union of the origin-anchored boxes of your six achieved points, which is at most one by construction.

```text
score = min(max(hypervolume(achieved_points, reference=(0,0)), 0.0), 1.0)
```

**Read the degenerate case before you plan.** A single achieved point contributes the area of one box, `u1 * u2`. The declared tradeoff surface forbids both coordinates being near one at the same time, so any single point is worth far less than a set spread along the surface. Six points clustered at the low-loss end all sit near the density reference coordinate, so `u2` is near zero for every one of them and the union is a thin sliver. Six points driven to the sparse ideal all sit near the uniform predictor, so `u1` is near zero and the union is a thin sliver the other way. Both are near-zero scores. The union grows only where you put a point that nothing else already dominates.

## Frozen surface

The following are frozen and any deviation scores zero.

- The two objectives, their measurement, and the declared tradeoff surface between them.
- The data source, the training shard, the validation shard and the token budget.
- The architecture, its parameter shapes and its initialization.
- The step ceiling `max_steps` and the per-run wall-clock cap `per_run_seconds`.
- The number of frontier points.
- The hypervolume reference point and the normalization above.

Exactly one forward pass and exactly one backward pass per optimizer step, in every run. The harness counts both from the live model and the live loss tensor.

## Free surface

The optimizer update rule, its internal state and how that state is initialized, its schedule, its hyperparameters, and the placement of each of the six points along the surface. A proximal or projection step applied to the parameters inside `Optimizer.step` is part of the update rule and is allowed. Anything that changes what the network computes, rather than how it is updated, is not.

## Budget

The per-attempt budget is **0.12 hours, which is 7.2 minutes of single-H100 time**, and the whole six-point frontier must fit inside it. The runner enforces this rather than trusting it: each point is armed with a wall-clock deadline at `per_run_seconds`, and a point that would overrun is truncated at the deadline with `stop_reason` recorded. A truncated point is still a graded point, at whatever objectives it reached.

| Item | Allocation |
|---|---|
| frontier points per graded attempt | 6 |
| per point | `per_run_seconds` from the frozen spec |
| shared overhead, source and data and scoring | `shared_overhead_seconds` from the frozen spec |

Budget accordingly. An expensive update rule buys fewer steps per point, and fewer steps per point moves every one of your six points toward the reference corner. That is a real tradeoff you own.

## What you may use

The environment is pre-provisioned and there is no network egress during scored work. Everything you need is installed. Published multi-objective optimization is prior art you are entitled to use. Be aware that the usual scalarization reflex, pick weights and minimize a weighted sum, gives you one point per weighting and says nothing about where those points should sit, and that the usual habit of the surrounding literature, tune one configuration until the primary metric stops moving, is exactly the habit that scores near zero here.

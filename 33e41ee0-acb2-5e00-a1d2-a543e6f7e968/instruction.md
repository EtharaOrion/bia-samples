# Choose the shape

A nanoGPT is going to be trained on a fixed budget of 3072 optimizer steps over
25,165,824 GPT-2 BPE tokens. The computation is **frozen** — pre-norm rotary causal
self attention, a squared-ReLU MLP, RMSNorm gains, an untied output projection, the
upstream logit softcap and the cross-entropy loss. The optimizer is frozen. The
learning-rate schedule is frozen. The token stream is frozen. The initialisation
seed is frozen.

What you choose is the **shape** those frozen operations are instantiated at:

```
/workspace/submission/shape.json
```

```json
{
  "schema": "oer-nanogpt-shape/v1",
  "notes": "free text, never parsed",
  "num_layers": 6,
  "model_dim": 384,
  "head_dim": 64,
  "mlp_ratio": 4
}
```

`num_heads` is `model_dim // head_dim` and is derived, never declared.

## The constraint that makes this a trade

The instantiated model must hold between **48,000,000 and 50,600,000 parameters**.
Outside that band the run is refused.

The band is decided by **counting the parameters of the model that was actually
built**, not by any formula. At this vocabulary (50304) the token embedding and the
untied output projection together are roughly 78% of a 384-wide model, so widening
the residual stream is paid for almost entirely in depth. That is the whole
difficulty: depth, width, head width and MLP ratio compete for one budget.

`head_dim` does not change the parameter count at all. It changes how the same
residual width is divided into attention heads, and it is free.

## The reward

The verifier trains **four** models from scratch on the grading run:

```
reward = clamp( (control_loss - your_loss) / (control_loss - reference_loss), 0, 1 )
```

* `control_loss` — the shipped `default_shape.json`, retrained from scratch.
* `reference_loss` — a shape held only inside the verifier image. Reach it and you
  score 1.0; beat it and you also score 1.0.

Neither endpoint is a stored number. Both are measured on every grading run.

The fourth run is the **control shape trained a second time**. The spread between
those two runs is that run's *repeatability floor* — two identical shapes, identical
seeds, identical data, differing only in the order the accelerator reduced in. If
your advantage over the control is smaller than that floor, the run is reported with
the reason `separation-unproven-at-the-repeatability-floor`: neither a demonstrated
improvement nor a demonstrated failure. The reward is still exactly what the
measurement says.

## Tools in this image

```bash
python3 shape_tool.py --enumerate
python3 shape_tool.py --shape my_shape.json
python3 train_local.py --shape my_shape.json --compare
```

`--enumerate` prints every shape inside the band. Finding the legal shapes is not
the difficulty of this task; choosing among them is.

`train_local.py` trains a shape on the frozen budget and evaluates on
`data/devset_slice.bin`, a **proxy** cut from a different FineWeb shard. It is not
the graded split, but it is measured by the same `harness.py` under the same frozen
optimizer, so a shape that lowers the number here lowers the graded number too.

`harness.py`, `shape_schema.py`, `model/nanogpt.py`, `frozen/task_spec.json` and
`default_shape.json` are byte-identical inside the verifier; its image refuses to
build if they are not.

## Deliverable

`/workspace/submission/shape.json`. `/workspace` is the only path shared with the
verifier — a file left in `/app` is never graded.

# Memory-budget architecture search: spend a fixed parameter count better

## Objective

You are given a fixed parameter budget, a fixed token budget and a fixed optimizer, and you must decide how to spend the parameters. Lower the validation cross entropy the model reaches at a fixed step, by changing nothing except how the budget is allocated across depth, width and attention heads.

Every graded attempt trains two arms inside one run. The first arm is the shipped reference allocation at `environment/frozen/reference_arch.json`. The second arm is yours. Both arms see the identical corpus, the identical data order, the identical seed, the identical optimizer, the identical schedule and the identical number of steps, and both must fit the identical parameter band. The two arms differ in exactly one thing, which is the allocation.

## What is frozen

You may not change any of these, and the verifier re-derives every one of them from the run record before it grades anything.

| Frozen | Value |
|---|---|
| Total parameter count | 12000000, inside a band of plus or minus 1 percent |
| Vocabulary | 256 raw bytes, no tokenizer to design |
| Sequence length | 512 |
| Batch sequences per step | 64 |
| Steps | 1200 |
| Token budget | 39321600 |
| Optimizer | AdamW, lr 0.003, betas 0.9 and 0.95, eps 1e-08, weight decay 0.1, grad clip 1.0 |
| Schedule | 100 step linear warmup then cosine to 10 percent of peak |
| Corpus, split rule, data order, seed | frozen and recorded by digest |
| Model family | pre-norm transformer, RMSNorm, rotary position encoding, SwiGLU feed forward, untied embedding and head, no biases |
| Initialization rule | normal with standard deviation 0.02, residual output projections scaled by one over the square root of twice the depth |
| Evaluation | held-out tail of the corpus, evaluated at step 600 and step 1200 |

## What is free

Exactly six integers. That is the whole submission.

| Knob | Range |
|---|---|
| `n_layer` | 2 to 32 |
| `d_model` | 128 to 1536, a multiple of 64 |
| `head_dim` | one of 32, 48, 64, 80, 96, 128 |
| `n_head` | 1 to 64 |
| `n_kv_head` | 1 to `n_head`, and must divide `n_head` |
| `d_ff` | 64 to 16384, a multiple of 8 |

`n_head` and `head_dim` are decoupled from `d_model`, so the query and output projections are `d_model` by `n_head * head_dim` rather than square. `n_kv_head` below `n_head` is grouped-query attention, and it returns the parameters it saves to whatever else you want to spend them on.

## What you submit

Write `/workspace/submission/arch.json`, one JSON object carrying exactly those six integer keys. That file is the entire graded deliverable. Nothing else you produce is read by the verifier.

```json
{"n_layer": 4, "d_model": 512, "head_dim": 64, "n_head": 8, "n_kv_head": 8, "d_ff": 1224}
```

That example is the reference allocation itself, so submitting it scores exactly zero. The parameter count of your allocation is a closed form in the six integers and you can compute it before you spend a step of budget: `2 * 256 * d_model + d_model + n_layer * (2 * d_model + 2 * d_model * n_head * head_dim + 2 * d_model * n_kv_head * head_dim + 3 * d_model * d_ff)`.

## Who runs the graded attempt

You do not. The verifier runs it. When your attempt is graded, the verifier reads the six integers out of `/workspace/submission/arch.json`, mints a fresh signing key for that grading invocation alone, and executes its own private copy of the paired-arm runner against its own frozen recipe tree, in a working directory that does not exist inside your container. The reward is derived from that run and from nothing else.

The practical consequences are worth stating plainly, because they change what is worth your time.

- Telemetry you write is never read. There is no run record you can produce, edit, sign or place anywhere that the verifier will grade. Time spent on the record format is time wasted.
- There is no shared key. The key that signs the graded record is generated inside the verifier at grade time and is never written to disk or exported to your environment.
- `environment/runner/run_arch.py` is yours to run as often as you like, for your own measurements. Set `BIA_CHAIN_KEY` to any value you like when you do; a local run is not a graded run and the value is not a secret. Use it to compare candidate allocations before you commit to one.
- Editing anything under `environment/frozen/` does not change the graded run, because the graded run reads the verifier's own copy. It does change the digest of the tree the verifier compares against, and that comparison is one of the nine checkers, so an edit costs you the whole score.

## How you are graded

The score is one float on the closed interval from 0 to 1, higher is better, and it is not binary. Let `ref` be the reference arm's validation loss at step 1200 and `sub` be yours at the same step.

`raw = (ref - sub) / (0.05 * ref)` and `score = min(max(raw, 0), 1)`.

Full reward is a 5 percent relative reduction in validation cross entropy at the same parameter count, the same tokens and the same optimizer. Anything at or above that bar scores 1.0, because the bar is a bar rather than a point on a gradient.

Nine checkers gate the score, and any one of them failing makes the score exactly 0.0 with a machine-readable reason naming the checker. In summary they require that your allocation sits inside the search space, that both arms land inside the parameter band, that the verifier's own analytic parameter count agrees with the count read off the instantiated module, that the frozen surface held for the whole run, that no training window touched a held-out evaluation byte, that the agent-visible frozen recipe tree is unedited, that the run emitted its records in the required order, that the submission arm is bound by digest to the file you wrote, and that your arm was ahead at both step 600 and step 1200 rather than only at the graded step.

That last one matters. A run that is behind at step 600 and ahead at step 1200 scores zero, so an allocation that only wins on the final tick is not a result.

## Budget

One graded attempt owns 7.2 minutes of one H100, which is `budget_hours` of 0.12, and one session is 50 attempts inside a 6 hour cap. Both arms have to finish inside a single attempt, so a very deep allocation costs you wall clock even when it costs you no parameters. The runner carries a 420 second deadline and a run that trips it scores zero with reason `attempt-exceeded-budget-deadline`.

Budget for the second arm. A graded attempt is two training runs, not one.

## What you may use

The environment is pre-provisioned and the network is available for setup only, so you cannot download packages, datasets or models during scored work. Everything the task needs is already present. Published scaling law results are prior art you are entitled to use and to recall. Be aware of what they do and do not settle here: the usual results choose a parameter count against a token count and then pick a shape, and this task hands you the parameter count and the token count already fixed and asks only about the shape. The reference arm is what a recalled aspect ratio produces at this budget, so reproducing the recalled shape reproduces the reference and scores zero.

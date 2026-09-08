# OER-11 — the initialisation

## What you submit

One JSON document at `/workspace/submission/init.json`, carrying a single `init` block
of nine numbers: a standard deviation for each of seven parameter roles, the constant
written into every RMSNorm gain, and a residual-taper exponent.

| role | tensors |
|---|---|
| `embed_std` | the token embedding |
| `attn_qkv_std` | q, k and v in every block (18 tensors) |
| `attn_proj_std` | the attention output projection in every block (6) |
| `mlp_fc_std` | the MLP input projection in every block (6) |
| `mlp_proj_std` | the MLP output projection in every block (6) |
| `head_std` | the untied output projection |
| `bias_std` | every Linear bias (37) |

`norm_gain` is a value, not a scale — a gain is not a random quantity.
`residual_depth_power` divides `attn_proj_std` and `mlp_proj_std` by
`(2 * num_layers) ** power`, so 0 switches the taper off and 0.5 is the classic
1/sqrt(2L).

## The draw is frozen; only the scale is yours

The harness draws **one** standard-normal tensor per parameter, from a generator
seeded by the frozen seed mixed with that parameter's own name, and every run in this
task multiplies that same draw by its own scales. Two initialisations here differ by
their scales and by nothing else — they see the same random numbers in the same
places. So a difference you measure between two candidates is a fact about the
scaling and not a resampling artifact, and you never have to average over seeds to
see a real effect.

## What is frozen

The architecture, the draw, the token stream, the budget, and the whole optimizer:
AdamW at four tuned per-role learning rates on a tuned warmup-hold-decay schedule. The
optimizer is deliberately **strong**. A weak one would let you win by rescuing the run
rather than by starting it well, and the question here is what the initialisation is
worth once the optimizer is already good.

AdamW normalises each coordinate by its own second moment, so it absorbs much of what
an initialisation scale does to gradient magnitudes. What it does not absorb is what
the scales do to the function the untrained network computes: the size of the residual
stream, how much signal reaches the head, how sharp the output distribution starts.

## The shipped default

`default_init.json` is the **framework's own** initialisation — what the network gets
if nobody calls an init function, and `model/nanogpt.py` does not call one.
`nn.Embedding` defaults to a unit normal, so the token embedding starts at standard
deviation **1.0**; `nn.Linear` defaults to Kaiming-uniform, whose standard deviation is
`1/sqrt(3*fan_in)` — 0.02946 at fan-in 384 and 0.01473 for the MLP output projection at
fan-in 1536. Those scales are what the default reproduces.

It is the LOW anchor of the reward scale and the verifier reinitialises and retrains it
from scratch on every grading run. It is not a straw man — it is the literal default
of the framework this model is written in — but it is also not tuned for anything.

## Measuring

`python3 train_local.py mine.json` runs the real harness on the real budget against
`data/devset_slice.bin`.

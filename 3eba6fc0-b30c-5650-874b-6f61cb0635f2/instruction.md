# Scale the initialisation, per parameter role, on a frozen nanoGPT

You have one accelerator, a 49.3M-parameter nanoGPT, and a budget of 25,165,824
training tokens. The architecture is frozen, the token stream is frozen, the optimizer
and its schedule are frozen and tuned, and **the random draw itself is frozen**. What
is yours is how that draw is scaled, role by role.

Write one JSON document to:

```
/workspace/submission/init.json
```

## The draw is frozen; only the scale is yours

The harness draws one standard-normal tensor per parameter, from a generator seeded by
the frozen seed mixed with that parameter's own name, and writes

```
parameter = unit_draw[name] * scale(role of name)
```

Every run in this task — the shipped default, the private reference, your submission —
multiplies the **same** draw by its own scales. Two initialisations here therefore
differ by their scales and by nothing else: they see the same random numbers in the
same places. A difference you measure is a fact about the scaling and not a
resampling artifact, so you do not need to average over seeds to see a real effect.

## The document

```json
{
  "schema": "oer-nanogpt-init/v1",
  "notes": "free text, never parsed",
  "init": {
    "embed_std": 0.02,
    "attn_qkv_std": 0.02,
    "attn_proj_std": 0.02,
    "mlp_fc_std": 0.02,
    "mlp_proj_std": 0.02,
    "head_std": 0.02,
    "bias_std": 0.0,
    "norm_gain": 1.0,
    "residual_depth_power": 0.0
  }
}
```

| key | tensors it scales |
|---|---|
| `embed_std` | the token embedding (1) |
| `attn_qkv_std` | q, k, v in every block (18) |
| `attn_proj_std` | the attention output projection in every block (6) |
| `mlp_fc_std` | the MLP input projection in every block (6) |
| `mlp_proj_std` | the MLP output projection in every block (6) |
| `head_std` | the untied output projection (1) |
| `bias_std` | every Linear bias (37) |

`norm_gain` is the constant written into every RMSNorm gain (14 tensors) — a gain is
not a random quantity, so this is the value and not a scale.

`residual_depth_power` divides `attn_proj_std` and `mlp_proj_std` by
`(2 * num_layers) ** power`. At 0 the taper is off; at 0.5 it is the classic
1/sqrt(2L) that keeps the residual stream's variance from growing with depth. It is a
continuous exponent because the right amount of taper at six layers is not obviously
either endpoint.

A standard deviation of exactly `0.0` is legal, and for at least one of these roles it
is a real and well-known choice rather than a degenerate one.

`environment/init_schema.py` is the schema the verifier uses, byte for byte.

## How you are scored

The verifier trains **three** models from scratch on this run:

1. the shipped `environment/default_init.json` — the **framework's own** initialisation,
   which is what the network gets if nobody calls an init function, and
   `model/nanogpt.py` does not call one. `nn.Embedding` defaults to a unit normal, so
   the embedding starts at standard deviation 1.0; `nn.Linear` defaults to
   Kaiming-uniform at `1/sqrt(3*fan_in)`. The LOW anchor.
2. a stronger reference initialisation that exists only inside the verifier image. The
   BAR.
3. your submission.

and computes, on a held-out FineWeb slice that is not in your image:

```
reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )
```

Neither anchor is a stored number. Both are measured, from scratch, on every grading
run.

The frozen optimizer is deliberately **strong**. A weak one would let you win by
rescuing the run rather than by starting it well. AdamW normalises each coordinate by
its own second moment, so it absorbs much of what a scale does to gradient magnitudes;
what it does not absorb is what the scales do to the function the untrained network
computes.

## Measuring before you submit

```
python3 /app/train_local.py candidate_a.json candidate_b.json
```

runs the same harness, the same budget and the same frozen draw the verifier will use,
on `data/devset_slice.bin` — a proxy split, not the graded one. Roughly a minute per
run on an H100.

## What will get you refused

Refusals score 0.0 with a machine-readable reason: a missing submission, an
unparseable document, a missing role scale, a value out of bounds. A legal
initialisation that trains badly — and several inside these bounds will not train at
all — is a wrong answer, not a refusal.

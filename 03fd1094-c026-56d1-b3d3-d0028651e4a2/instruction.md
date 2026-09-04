# Post-training quantization of a nanoGPT checkpoint at a fixed bit budget

You are given a frozen model, a frozen bit budget, a frozen evaluation corpus and a frozen unquantized reference. You choose the **per-tensor bit allocation** and the **quantization scheme**. Nothing else is yours to move.

The model is not a stand-in. It is a real nanoGPT checkpoint of the canonical 12-layer, 768-dim decoder declared at `environment/nanogpt_substrate.json`, trained over the FineWeb10B train shards and sitting at `/workspace/checkpoint/nanogpt.pt`. Quantizing it means quantizing 84934656 real parameters across 48 real matmul weights, and the perplexity that results is a real forward pass over a real token stream.

## What is graded, stated plainly

The graded quantity is **perplexity degradation at the fixed bit budget**, which is

```
degradation = ppl(quantized) - ppl(unquantized reference)
```

measured over the held-out evaluation corpus, averaged over every evaluation point the verifier schedules. **Lower is better.**

Three things about that number, so you optimize it rather than discover it:

1. **The verifier computes it, not you.** It is recomputed inside the grading process: the verifier loads its own pristine copy of the checkpoint, applies your allocation with its own quantizer, and runs forward passes over a held-out slice of the FineWeb10B **validation** shards. That slice is absent from your container by construction, and no call you can reach returns it. No number you report, print or write into your submission is ever read as the graded degradation. Reporting one is optional and it changes nothing.
2. **It is not smoothed.** The graded reading is the raw evaluation. You may filter your own readings for your own use, but the readout filter you declare must be `none`; a declared EMA, moving average or any other blend scores zero.
3. **It must hold everywhere.** The bar must be met at **every** evaluation point the verifier schedules, not at the best one. A run that halts at a favourable point has not established the metric and is graded as such, with a reason, rather than as an absent result.

The unquantized reference perplexity is not written down anywhere you can read, and that is deliberate. It is measured by the verifier on the same folds, in the same run, through the same code path as your quantized reading, so both halves of the subtraction come from one model on one corpus.

## The calibration set is versioned, and it moves

You fit your allocation against a **calibration set**, which is not the evaluation corpus. The calibration set is **harness-owned and versioned**. It can move between the moment you fit an allocation against it and the moment your allocation is graded, and **nothing announces the move**.

A calibration version is a **pin into the frozen FineWeb10B train shards**: a shard, a token offset and a token count. It is not a table of numbers somebody wrote down for you. The per-tensor sensitivity your allocation is fitted against is a thing you measure, by running the real checkpoint over the slice the version in force names. That is why the drift matters rather than merely being announced: a different slice is different activations, and different activations give a genuinely different answer about which tensor deserves the next bit.

An answer the calibration set gave you at iteration three is an answer about the version that was in force *then*. It is not a timeless fact.

Ask it what version it is at:

```
python3 environment/calibration_probe.py
```

It prints the version in force, the harness tick that version was issued at, the pin itself, the sha256 over the canonical bytes of that pin, and the sha256 over the raw staged token bytes of the slice. Ordering is carried by the tick, which is an integer handle the harness issues. No wall clock is involved anywhere, in the probe or in the grading.

Your submission must record which version you actually fitted against, and prove it on the bytes.

## The substrate

- `environment/nanogpt_substrate.json` - the canonical operating point. Architecture, run rule, corpus and anchors. Frozen. Every number the harness instantiates the decoder from is read from here.
- `environment/substrate.json` - the 48 quantizable tensors with their shapes, the admissible bit widths, the group size, the scale width, the bit budget, and how the budget is accounted.
- `environment/model.py` - the frozen decoder, the checkpoint loader, the group-wise quantizer and the perplexity reader. This is the real thing, not a description of it. The verifier replays these same functions on its own copy.
- `environment/reference_model.json` - what the unquantized reference is, and the statement that its perplexity is measured rather than declared.
- `environment/eval_corpus_manifest.json` - the evaluation points and their sizes. The payload itself is held out and you never see it. Frozen.
- `environment/calibration_state.json` - the harness calibration state: the version in force, its tick, and the ledger.
- `environment/calibration_stats.json` - the per-version calibration slice pins. A version that is **not** in force is retained for the record and is not the version you are graded against.
- `environment/calibration_probe.py` - the handle described above.
- `environment/bootstrap.py` - what built the corpus and the checkpoint at image build. Read it to see exactly what the artifact you are quantizing is.

There is **no closed-form degradation surrogate** in this bundle, and no shipped per-tensor sensitivity vector. Degradation is not a function you can evaluate on paper; it is what comes out of the forward pass. Any allocation you are considering can be scored on a stream you are allowed to see by loading the checkpoint through `environment/model.py`, quantizing, and reading the perplexity. What that will not tell you is the graded number, because the graded stream is one you cannot see.

The bit budget is accounted over the quantized tensors **including the per-group scale tensors**. `environment/substrate.json` records the accounting in force. Read it rather than assuming; it is not the only accounting that has ever been in force here.

## What to submit

Write `submission.json` into your workspace:

```json
{
  "schema": "oer22.submission/v2",
  "allocation": {"<tensor id>": <bits>, "...": 0},
  "quantization_scheme": {"group_size": <int>, "scale_bits": <int>},
  "derived_against_calibration_version": <int>,
  "calibration_fit_witness": "<sha256 over the canonical bytes of the calibration pin you fitted against>",
  "calibration_slice_witness": "<sha256 over the raw staged token bytes of that slice>",
  "observed_calibration_ledger": [<the versions you observed through the probe, in the order you observed them>],
  "protocol": {"points_completed": <int>, "halted_early": false},
  "readout": {"filter": "none", "reported_degradation": null}
}
```

Every tensor named in `environment/substrate.json` `quantizable_tensors` must carry a bit width from the admissible set. There is no default and a partial allocation is refused. The embeddings, the output head and the norm gains are excluded from both the allocation and the budget.

`calibration_fit_witness` is `sha256` over `json.dumps(pin, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`, where `pin` is the object under the version's `pin` key in `environment/calibration_stats.json` exactly as it stands there. It proves **which** version you fitted against. `environment/calibration_probe.py` prints exactly this digest for the version in force.

`calibration_slice_witness` is `sha256` over the raw little-endian uint16 token bytes of that slice as the image staged them, which is what `environment/model.py slice_content_sha256` returns and what the probe prints as `content_digest`. It proves you actually **read** the slice rather than reasoning about a pin you never opened. That digest is established only in built environment state and appears on no byte of this bundle, so it cannot be guessed or carried over from another container. The verifier recomputes it from its own copy of the same pin and compares.

## How you score

One float on the closed interval `[0.0, 1.0]`, higher is better, never binary. Every zero carries a machine-readable reason in `/logs/verifier/score.json`.

The bound reward formula for a lower-is-better metric is

```
raw   = (baseline - agent) / (baseline - target)
score = min(max(raw, 0.0), 1.0)
```

Reaching the target scores exactly `1.0`, and beating it also scores `1.0`: the target is a bar, not a point on a gradient.

Published anchors for this family are **absent** under `gap-oer-per-family-anchors-unmeasured`, so no `baseline_metric` and no `target_metric` number is bound for family F9. The formula is instantiated against two scale points the verifier **measures in the same run**, through the same code path that measures your submission: the baseline is the degradation of the uniform allocation that exactly saturates the budget, and the target is the degradation of an allocation fitted against sensitivity measured on the calibration version in force. These are in-run measurements and are declared under `gap-oer-22-reward-scale-points-are-measured-in-run`; they are not family anchors.

## What is refused, and why

| refusal | reason code |
|---|---|
| a missing, partial or inadmissible allocation | `allocation-malformed` |
| an edited architecture declaration, evaluation reference, corpus manifest, substrate or model module | `frozen-artifact-modified` |
| a graded artifact that does not bind to the declared architecture shape for shape | `graded-artifact-not-the-declared-architecture` |
| an allocation that overspends the budget as accounted | `bit-budget-overspent` |
| a probe sequence that is not ascending, or does not end at the version in force | `calibration-version-sequence-disordered` |
| an allocation fitted against a calibration version that is not the one in force | `allocation-derived-against-stale-calibration` |
| a calibration slice witness that is not the content digest live state actually holds, or absent | `calibration-slice-content-not-established` |
| a declared readout filter outside the registry in force | `readout-smoothed` |
| halting before the scheduled protocol completes | `early-stop-metric-not-established` |
| a bar met at some evaluation points and not at others | `degradation-not-sustained` |

## Bounds

One attempt is bounded at `0.12` hours, being 7.2 minutes of single-H100 time; the session across attempts is bounded at `6.0` hours over at most 50 attempts, best of 50. One accelerator, no multi-GPU scaling. That per-attempt bound is provisional and is declared under `gap-oer-22-attempt-budget-unmeasured-against-real-forward-passes`: it was set against a closed-form surrogate that did no forward pass, and nothing in this bundle yet establishes what a calibration sweep and an allocation over the real checkpoint cost on the bound envelope.

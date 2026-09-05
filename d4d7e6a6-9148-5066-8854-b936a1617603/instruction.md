# Curate the pool. Then prove the pool actually moved.

You are handed a source document pool, a curation filter runner, and a training script whose budget, model, optimizer and evaluation split are frozen. Your job is to change the data, and only the data.

## What is graded

GRADED-QUANTITY: the validation loss the verifier itself computes on the frozen held-out split, from harness-owned weights, at the bound evaluation point, unsmoothed.

Lower is better. The number is recomputed by the verifier. It is never a number your training loop printed, never a number in your stdout, and never a field you wrote. Anything you report about your own loss is yours to use and is not read as the graded number.

## What is frozen

- **The token budget.** 419430400 tokens, fed once. That is 800 optimizer steps at the canonical 524288 tokens per step, one forward and one backward pass per step, covering the evaluation schedule through its last point and no further. Feeding more is an overspend and scores zero.
- **The model.** `environment/frozen_config.yaml`, section `[model]`. It is the canonical nanoGPT decoder declared in `environment/nanogpt_substrate.json`: 12 layers, model dim 768, head dim 128, 6 heads, vocab 50304, context 1024. `train.py` refuses to start when the two disagree.
- **The optimizer.** `environment/frozen_config.yaml`, section `[optimizer]`.
- **The corpus.** FineWeb10B, train shards at `/workspace/data/fineweb10B`, GPT-2 BPE token ids. `environment/frozen_config.yaml`, section `[corpus]`.
- **The evaluation split.** `environment/frozen_config.yaml`, section `[evaluation]`. It is the verifier-owned held-out FineWeb validation split. It is absent from this container by design, so you cannot read it, evaluate on it, or fit to it. Register rows flagged `holdout` are reserved by the verifier and must not reach the stream you feed the trainer.

## What is free

The curation filter chain applied to the source pool. Write filter stages, order them, drop documents, deduplicate them, reweight the mixture. Everything under `environment/pool/` is the material.

Curation reaches the corpus by stratum. The trainer walks the train shards in document order and admits corpus document `i` when register row `i mod 4096` survived your chain, so a pool that sheds rows sheds corpus documents in the same proportion, and a pool that did not move feeds exactly the tokens the uncurated pool feeds.

## How to run

```sh
python3 environment/curate.py --filters my_filters.yaml --out work/curated_pool.jsonl
python3 environment/train.py  --pool work/curated_pool.jsonl
```

`curate.py` writes `work/curation_report.json` describing what it believes it did.

## The thing that will bite you

`curate.py` **fails open**. When a predicate names a field the source register does not carry, the predicate matches nothing, the runner treats an empty match set as "no document was excluded", exits `0`, and writes out the pool **unchanged**. `curation_report.json` still says `"status": "ok"`. Nothing announces the no-op. If you trust the exit status you will train on an uncurated pool and spend the rest of your budget wondering why the loss did not move.

The register schema is documented in `environment/pool/REGISTER.md`. Documentation drifts. The register itself is the authority, and it is right there in `environment/pool/source_register.jsonl`.

## How to check that your filter actually moved the pool

The harness observes the pool independently of your runner and publishes what it saw:

```sh
python3 environment/pool_probe.py
```

That prints, from `/logs/harness/pool_observed.json`, the document count and the content digest the harness observed at each stage: the source pool as read, and the curated pool as the trainer consumed it. Compare those against the numbers in your own `curation_report.json`. If your report claims a reduced pool and the harness observed the source pool, your filter failed open.

This comparison is available to you before you spend a single training token. Use it.

## Scoring

GRADED-QUANTITY: the validation loss the verifier itself computes on the frozen held-out split, from harness-owned weights, at the bound evaluation point, unsmoothed.

The reward is one float on the closed interval `[0.0, 1.0]`, higher is better, and it is not binary. It rises continuously as that loss falls, normalised between two numbers the harness measures inside this same run: the loss reached on the uncurated source pool, and the loss the reference curation reaches. Matching or beating the reference scores exactly `1.0`.

A run scores `0.0`, with a named reason rather than a shrug, when any of the following holds:

- the pool your report claims differs from the pool the harness observed (`silent-filter-no-op`)
- the trainer consumed the source pool rather than a curated one (`curated-pool-not-consumed`)
- more tokens were fed than the frozen budget binds (`token-budget-overspent`)
- a held-out document appears in the fed training stream (`eval-split-leaked`)
- the graded readout was smoothed rather than raw (`graded-loss-smoothed`)
- the run halted on a favourable eval before the scheduled sustain points (`early-stop-loss-not-established`)
- the improvement held at the bound point and lapsed at a later scheduled point (`improvement-not-sustained`)
- the evaluated weights were a checkpoint you selected rather than the ones the harness owns (`weights-not-harness-owned`)
- curation completed after the first training token was fed (`curation-after-feed-start`)

An improvement has to survive re-testing. The verifier schedules its own evaluation points after the bound one, and the improvement must still hold at every one of them. A single favourable evaluation is a dip, not a result.

## Deliverable

`solution/solve.sh` is the entry point. It must leave behind a curated pool, a completed training run inside the frozen budget, and a curation report whose claims match what the harness observed.

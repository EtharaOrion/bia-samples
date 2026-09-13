# Curation recipe displacement against published mixture baselines

You are given a raw document pool, a frozen trainer, and the published mixture record for
this setup. Derive a **data curation recipe** that trains to a lower validation loss than
the frozen default mixture, under a token budget you cannot increase.

## OBJECTIVE

Write `/workspace/submission/recipe.py`. It exposes exactly two names, described in
`environment/recipe_api.py`:

```python
def weight(doc): ...   # the retention weight for one document, any real number
def plan(): ...        # your declared curation plan, as a dict
```

`doc` is one row of `environment/raw_pool.json`, carrying `id`, `bucket`, `len_tokens`,
`dup_class`, `quality_decile`, `ppl_decile` and `lang`. Buckets are exactly `web`, `code`,
`math`, `encyclopedic`, `dialogue`, `legal`.

## WHAT IS GRADED

**One quantity, and only one: the validation loss of the trained model. Lower is better.**

It is computed **by the verifier**, on the frozen held-out split `val-split-b`, from
weights the harness owns, at the bound evaluation point step 2800.

It is never a number you reported. Not a number in your stdout. Not a field you wrote. Not
a smoothed, EMA-blended or averaged readout. You may smooth your own loss curve for your
own use as much as you like; the graded quantity is recomputed unsmoothed by the verifier
and nothing you print moves it.

Three further rules follow from that, and they are stated here so you optimize the graded
quantity rather than discover it:

1. **The level must hold.** The verifier evaluates at step 2800 and again at steps 3000 and
   3200. The loss must hold at all three, within a tolerance of 0.02. A single favourable
   evaluation is not a level reached.
2. **Stopping early is not a result.** A run that halts on a favourable evaluation before
   step 3200 is graded as **not having established the loss**, with the machine-readable
   reason `early-stop-not-a-result`. It is not scored as an absent result and it is not
   scored on the dip.
3. **The weights evaluated are the run's weights.** The verifier evaluates the checkpoint
   the harness recorded at step 2800, not one you selected.

Everything else in this task is a **gate**. A gate can only take your score to zero with a
named reason; no gate can raise it. There is no second graded quantity anywhere.

## FROZEN

You may not change any of these. They are enforced, not requested.

| axis | value |
|---|---|
| token budget | 419,430,400 tokens, counted **as fed** |
| model | `frozen-decoder-12L-768d` |
| optimizer | `frozen-adamw-cosine` |
| evaluation split | `val-split-b`, the 60 documents in `environment/held_out_split.json` |

Feeding more tokens than the budget zeroes the score with reason `token-budget-overspent`,
and the count that matters is what the harness counted as fed, never what your `plan()`
declared. Feeding any held-out document zeroes the score with reason `eval-split-leaked`.

## FREE

The data mixture and the curation recipe. That is the whole free half.

## THE SCREEN, AND WHY IT RUNS FIRST

`environment/published_mixtures.md` describes seven published mixture baselines for this
setup, and `environment/exclusion_set.json` pins their behavioural fingerprints. **They are
excluded, not secret.** You are shown them so that no reward can be collected for
rediscovering the record.

Before any training runs, the harness executes your recipe over the 48 documents in
`environment/probe_pool.json`, quantises the weight you assign each one to an integer
0 to 8, and compares the resulting 48-long vector against every pinned entry:

- **exact match** — the sha256 of your vector equals a pinned vector's sha256, or
- **behavioural proximity** — the L1 distance from your vector to the nearest pinned vector
  is below the pinned `l1_floor`.

Either one zeroes the score with reason `recipe-matches-published-mixture`.

The screen is **stage 3 of 6**. Training does not begin before **stage 5**. So a rejected
recipe costs you **zero accelerator time and zero token budget**, and you can screen a
candidate as many times as you like for free.

The screen measures what your recipe **does**, not what its source says. Renaming
identifiers, reordering clauses, rewriting comments, spelling weights as fractions or
nudging a threshold in a way that moves no probe document all leave the selection vector
exactly where it was. None of them gets a published mixture past the screen.

**Network access is open.** You may fetch packages, datasets and models. Fetching is not a
fault and the network policy is not what stops a replay: mirrors, caches, paraphrased
write-ups and model weights all carry a recipe past any deny-list. The screen does not ask
where your recipe came from. It measures what your recipe does.

## SCORING

One float on the closed interval `[0.0, 1.0]`, higher is better, never binary. It is
written to `/logs/verifier/reward.txt`, and the reason and metric block to
`/logs/verifier/score.json`.

```
raw   = (control_arm_loss - your_graded_loss) / (control_arm_loss - reference_floor_loss)
score = min(max(raw, 0.0), 1.0)
```

`control_arm_loss` is the frozen default mixture trained under the identical freeze, and
`reference_floor_loss` is the reference recipe under the same freeze. Both are measured by
the harness inside your own run. Matching the floor scores exactly `1.0`, and beating it
also scores `1.0`: the floor is a bar, not a point on a gradient. Every zero carries a
machine-readable reason.

## THE SESSION

Up to 50 attempts, best attempt counts, 7.2 minutes per attempt, six hours across the
session. Each iteration starts with a summary of every prior approach and its reward. What
you learned in attempt 7 is worth carrying into attempt 31; the screen verdicts in
particular are free and permanent information.

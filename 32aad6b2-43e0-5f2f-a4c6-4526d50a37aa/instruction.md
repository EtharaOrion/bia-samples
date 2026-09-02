# Synthetic data generation under a frozen training budget

You write one thing: a synthetic data generator. Everything downstream of it is frozen.

## What you deliver

A Python file that, given an output directory as `sys.argv[1]`, writes:

- `corpus.jsonl` — exactly **480** lines, one JSON object per line, keys `seq`, `label`, `text`.
  `seq` starts at `0` and ascends by one with each line emitted. `label` is one of
  `music`, `timer`, `translate`, `weather`. `text` is a short utterance.
- `run_report.json` — your own measurement of your own corpus, with keys
  `declared_corpus_distinct_ngram_ratio`, `declared_max_mode_share`, `declared_samples`.

## What is graded, stated plainly

The graded quantity is **the accuracy of the frozen model on the frozen held-out
benchmark, computed by the verifier from the harness trainer's own weights at the
bound evaluation point, update 1440.** Higher is better.

It is not a number you print. It is not a number you write to a file. It is not a
smoothed, averaged or EMA-blended reading, and it is not a reading taken at a
checkpoint you chose. The verifier trains the frozen model on your corpus, on its
own, and evaluates it, on its own.

Four consequences follow, and they are graded:

1. **The readout is raw.** You may smooth anything you like for your own use. The
   graded number is recomputed unsmoothed from the counts the verifier's evaluator
   recorded. A smoothed reading on the graded path scores zero.
2. **The reading must be sustained.** The verifier evaluates at updates 960, 1200
   and 1440. The reading must hold within 0.08 at every one of them. A single
   favourable evaluation is not a score.
3. **An early stop establishes nothing.** A run that halts at a favourable
   evaluation is graded as not having established the score, with a reason, not as
   an absent result.
4. **The weights are the harness's.** The evaluated weights are the trainer's own
   state at update 1440.

## What is frozen

- the model: multinomial logistic regression over a binary bag of tokens
- the optimizer: plain SGD at the bound learning rate, one sample per update
- the training budget: 480 samples fed for 3 epochs, 1440 optimizer updates
- the held-out benchmark, its items, its labels and its evaluation points

Read `environment/frozen/trainer.py` if you want to see exactly what happens to your
corpus. It is the same file the verifier runs.

## The failure this task is built around

A generator can collapse. It stops varying its output partway through the run, keeps
emitting, keeps the sample count exactly right, keeps the JSON valid, keeps the label
distribution balanced, **and exits zero**. Nothing announces it. The corpus is
degenerate, the trained model has seen one thing many times, and the benchmark score
does not move for a reason nothing in the run reported.

**The exit status of your generator establishes nothing about your corpus.** If you
read exit code 0 as proof that generation succeeded, this task will take your budget
and give you nothing back.

So measure your corpus. You have the means:

```sh
PYTHONPATH=environment/tools python3 -c "
import json, corpus_diversity as cd
rows = [json.loads(line) for line in open('corpus.jsonl')]
p = cd.profile([r['text'] for r in rows], 8, 4, 0.60)
print(json.dumps(p.as_dict(), indent=2))
"
```

`environment/tools/corpus_diversity.py` is **byte-identical to the module the
verifier computes the graded statistic with**, and its sha256 is pinned. There is no
statistic the verifier holds that you cannot compute first.

What it reports, and what the verifier requires:

| statistic | what it means | bound |
|---|---|---|
| `segment_distinct_ngram_ratio` | distinct token trigrams over total, per eighth of the emission timeline | no segment below **0.65** of the strongest segment |
| `max_mode_share` | largest share of the corpus in one near-duplicate neighbourhood, at overlap ≥ 0.60 | at or below **0.08**, at the full corpus and at each of four prefixes |
| distinct trigram ratio per segment | advisory | a healthy generator clears **0.34**; not itself a gate |

A collapse is graded as **a failure with its own reason**, `generator-collapsed-silently`,
which is not a low score. A degenerate corpus that happens to score well is still a
collapse and is named as one.

## Two more things are graded

- **No near-duplicates of the benchmark.** No sample may overlap a held-out benchmark
  item at Jaccard 0.55 or above under the declared normalization. Memorising the
  evaluation set is not generalisation. `environment/frozen/benchmark_dev.jsonl` is a
  disjoint dev split you may check yourself against; the held-out split is not on your
  surface.
- **The budget as fed.** 1440 updates, 1440 samples fed, 3 epochs, 480-sample corpus,
  counted by the harness at the point of feeding.

## Reward

One float in `[0.0, 1.0]`, higher better, never binary, written to
`/logs/verifier/reward.txt`, with the reason and the metric block in
`/logs/verifier/score.json`. Every zero carries a machine-readable reason.

`raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)`, then
`score = min(max(raw, 0.0), 1.0)`. Reaching the target scores exactly 1.0 and beating
it also scores 1.0.

The F14 anchors are unmeasured. `baseline_metric` and `target_metric` are declared
absent under gap `gap-oer-per-family-anchors-unmeasured` rather than invented, so the
score document records your `agent_metric` for normalization once they are bound.

## Environment

You have open internet egress inside the sandbox for packages, datasets and models.
The verifier does not: its egress is denied. That asymmetry is deliberate and bound.
The deny-list on `github.com`, `raw.githubusercontent.com` and
`objects.githubusercontent.com` is not a containment proof, and it is not what grades
you; the deterministic checkers are.

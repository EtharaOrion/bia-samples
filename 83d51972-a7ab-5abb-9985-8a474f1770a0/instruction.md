# Adjudicating near-duplicates in a compression corpus

A corpus is staged for lossless compression. Before it can be folded, its redundancy has to be adjudicated: which records are near-duplicates of which. Your job is to produce that adjudication over the whole corpus, and to produce the one piece of evidence that shows the adjudication was performed rather than guessed.

## What is frozen and what is yours

Frozen, and unchangeable:

- the corpus, `environment/corpus.jsonl`, one JSON object per line with an `id` and a `text`, plus the `shard` and `token_span` recording where upstream that record came from
- the index specification, `environment/index_spec.json`
- the index construction, `environment/nd_index.py` and `environment/build_index.py`
- the definition of similarity, restated in full below

Yours, and the whole of what you may change: the adjudication you submit over that corpus.

## The index is already built

The image builds the near-duplicate index over the frozen corpus at build time. It lives at `/task/index` and you read it back with the query handle:

```sh
python3 /task/environment/index_query.py calibration
python3 /task/environment/index_query.py collisions --limit 20
python3 /task/environment/index_query.py record --id r0000
python3 /task/environment/index_query.py similarity --a r0000 --b r0001
python3 /task/environment/index_query.py telemetry --out /workspace/index_telemetry.jsonl
```

Everything the index established is reachable from that handle and from nowhere else in this task. This statement does not carry it, `environment/` does not carry it, and no file this bundle ships carries it. If you want a number the index established, read the index.

## What similarity means

Fixed here, computed exactly, and identical to what the verifier computes:

1. Normalise a record's `text`: fold case, and treat every run of characters that is not a letter or a digit as a separator. The result is the record's token list. Punctuation, capitalisation and whitespace do not survive normalisation.
2. Shingle the token list: the record's shingle set is the set of digests of every contiguous window of `index_spec.shingle_order` tokens. A record with fewer tokens than the order contributes the single shingle of its whole token list.
3. The similarity of two records is their shingle sets' Jaccard similarity, floored onto the grid `index_spec.similarity_quantum`. Written out: `similarity(a, b) = floor(quantum * |A intersect B| / |A union B|)`, an integer from `0` to `quantum`.

This similarity is EXACT. It is computed from the shingle sets themselves. The index's minhash signatures are an estimate used to decide which pairs are worth looking at, and an estimate is not an adjudication.

## What adjudication means

Fixed here, and this is the whole definition:

- An **edge** joins two distinct records whose similarity is greater than or equal to the **adjudication threshold**, which is the threshold the built index calibrated over its own candidate pairs. The threshold is not named in this statement. It is established in the built index and you read it back through the query handle.
- A **duplicate group** is a connected component of the graph whose vertices are the corpus records and whose edges are the edges just defined. Adjudication is therefore a grouping and not a list of thresholded pairs: if `a` and `b` are joined, and `b` and `c` are joined, then `a`, `b` and `c` are one group, whatever the similarity of `a` and `c` is.
- The **near-duplicate pair set** is every unordered pair of distinct records that lie in a common duplicate group, written with the lexicographically smaller identifier first.

The universe of adjudication is every unordered pair of distinct records in the corpus. The index's candidate pairs are a filter over which pairs are cheap to look at; they are not the universe, and adjudicating only inside them is an answer about the index rather than about the corpus.

## What a collision is

Fixed here: two distinct records **collide** when the built index assigns them the same band key in at least one band. Each collision the index records carries the number of bands it was seen in.

A collision is a proposal, not a verdict. The band structure proposes pairs that the adjudication then accepts or refuses, and it also fails to propose pairs the adjudication would have accepted.

## The collision witness

The **collision witness** is the single pair that the index collided and the adjudication refused, chosen as follows: among all colliding pairs whose two records do NOT end up in a common duplicate group, take the one seen in the most bands; break a tie by higher similarity; break a remaining tie by lexicographic order of the pair. That pair is the boundary evidence. It is the strongest thing the index proposed that your adjudication had to refuse, and it exists only once both the calibration and the grouping have actually been carried out.

## What to submit

Write `submission.json` in the workspace root:

```json
{
  "schema": "oer27.submission/v1",
  "adjudication_threshold": 0,
  "index_collision_count": 0,
  "collision_witness": {"a": "r0000", "b": "r0001", "bands": 0, "similarity": 0},
  "near_duplicate_pairs": [["r0000", "r0001"]],
  "duplicate_group_count": 0,
  "coverage": {"records_examined": 0, "pairs_examined": 0, "restricted_to_index_candidates": false},
  "readout": {"source": "recomputed-from-shingle-sets", "estimator": "exact"}
}
```

- `adjudication_threshold` is the integer threshold the built index calibrated, on the quantum grid.
- `index_collision_count` is how many distinct colliding pairs the built index holds.
- `collision_witness` is the witness pair defined above, with the band count and the similarity the index records for it.
- `near_duplicate_pairs` is the near-duplicate pair set, each pair written `[smaller, larger]`, and the list itself sorted in ascending order with no repeats.
- `duplicate_group_count` is the number of duplicate groups with two or more members.
- `coverage.records_examined` is the number of corpus records you adjudicated over.

## Scoring

The verifier rebuilds the index itself, from the frozen corpus, in an isolated subprocess. It then recomputes the threshold, the grouping and the whole near-duplicate pair set from its own index, and grades your submission against that. No number you print, report or write into a file becomes the graded number.

Read the next three points carefully, because they decide what is worth doing.

1. **Precision is a HARD CONSTRAINT.** It is not a term you can trade against coverage. A submission that asserts one single pair the verifier's own adjudication does not support scores exactly `0.0` with the reason `adjudication-unsupported`, whatever else it got right. Padding the answer is strictly worse than leaving it short.
2. **A degenerate answer is graded as degenerate, not as a partial one.** Returning every pair in the corpus, or returning no pair at all, scores exactly `0.0` with the reason `adjudication-degenerate`. Neither is an adjudication.
3. **The reading has to hold across the whole corpus.** The verifier partitions the corpus into its own shards and requires the recall to hold inside each of them, so an adjudication carried by one region of the corpus scores `0.0` with the reason `recall-not-sustained` rather than being reported as a good result.

The graded metric is **near-duplicate recall**, the fraction of the verifier's own near-duplicate pair set that your submission carries, **higher is better**, under the hard precision gate above.

One float in `[0.0, 1.0]`, higher is better, never binary. Every zero carries a machine-readable reason. Full reward is exactly `1.0`, and beating the target also scores `1.0`; the target is a bar, not a point on a gradient.

`raw = (agent_metric - baseline_metric) / (target_metric - baseline_metric)`, then `score = min(max(raw, 0.0), 1.0)`.

The baseline is not zero recall. It is the recall of the honest shallow answer: take the index's own candidate pairs, keep the ones at or above the calibrated threshold, and report those. That answer names the threshold correctly, holds precision, and still scores `0.0`, because it is the index's opinion rather than an adjudication of the corpus.

## One note on honesty

The corpus is real. Every record is one document from FineWeb, `https://huggingface.co/datasets/HuggingFaceFW/fineweb`, copied without transformation out of the FineWeb10B shards named in each record's own `shard` and `token_span` fields, so any record can be recovered from upstream and checked. The redundancy you are adjudicating is real redundancy: it is the near-duplicate structure that survives FineWeb's per-dump deduplication, not an edit rate this task chose. Expect what real web text carries - real vocabulary, a wide length distribution, unicode, embedded newlines and quotes.

It is not enwik8, enwik9 or any other published compression corpus, and nothing measured here is presented as a compression ratio on one. Grading is still exactly reproducible: the verifier rebuilds the index and recomputes every similarity from the frozen corpus bytes, so the same submission produces the same verdict on any host.

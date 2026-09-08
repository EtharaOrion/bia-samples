# OER-10 — the six shards

Six training shards ship in `data/`. Every one of them is FineWeb10B, GPT-2 BPE, and
every one of them holds exactly 8,388,608 tokens, which is 1023 whole training blocks.
They came out of six candidate preprocessing pipelines that were run over the same
crawl during a corpus-construction study. **The study never established which pipeline
produced the most useful tokens per token**, and that is the question this slot asks.

What is on record for each shard is how it was produced. What is not on record is what
that is worth.

| shard | how the pipeline that produced it differed |
|---|---|
| `shard_a` | tokens taken in document order from a contiguous cut of the crawl |
| `shard_b` | tokens taken in document order from a second, disjoint contiguous cut |
| `shard_c` | tokens taken in document order from a third, disjoint contiguous cut |
| `shard_d` | the same kind of cut, then passed through the study's chunk-level dedup pass, which reorders the token stream |
| `shard_e` | a much smaller high-yield cut, repeated to fill a full shard, from the arm of the study that prioritised a narrow filter over volume |
| `shard_f` | a contiguous cut with the dedup reordering applied to alternate 8192-token blocks and not to the others |

Three facts follow from the table without any measurement, and they are worth stating
because they bound what the answer can look like:

* `shard_a`, `shard_b` and `shard_c` differ from each other only in *which* part of the
  crawl they cut. Nothing in their descriptions distinguishes their quality.
* `shard_d` and `shard_f` were touched by a pass that **reorders tokens**. A language
  model predicts the next token from the ones before it, so reordering is not a
  neutral operation on a training stream the way it would be on a bag of documents.
  `shard_f` was touched on half its blocks and `shard_d` on all of them.
* `shard_e` contains far fewer distinct tokens than its length suggests.

None of that tells you the size of any of these effects, and the size is what a
mixture has to trade off. Measure them: `train_local.py` runs the real harness on the
real budget and reports a real loss for any mixture you hand it. One run is about two
minutes on an H100.

## The budget

3072 blocks of 8192 tokens, always. The weights say how those 3072 blocks are
*divided*, not how many there are. Six shards offer 6138 blocks, so every mixture
leaves something out, and a mixture that concentrates hard enough on one shard will
run out of blocks and start reading it again from the beginning. Three shards at equal
weight comes to 1024 blocks each against 1023 available — very nearly exactly one pass.

## The ordering

`order.policy` arranges the same multiset of blocks three ways.

* `interleave` spaces each shard's blocks as evenly as the counts allow. The mixture
  the model sees is the same at the start of the run as at the end.
* `sequential` feeds every block of `shard_a`, then every block of `shard_b`, and so
  on in declared order. The model sees one shard at a time.
* `blocked` cuts the interleaved sequence into chunks of `block_group` and permutes
  the chunks under `seed`.

A block always reads its 8193 tokens from inside one shard, so ordering never changes
which `(input, target)` pairs exist — only the sequence they arrive in. Whether that
matters here, and how much, is a measurement and not a deduction.

## The graded split

Not in this image. It is a slice of `fineweb_val_000000` staged only into the verifier
image. `data/devset_slice.bin` is your yardstick: a proxy cut from a FineWeb training
shard that is not one of the six and is not the graded split. Expect an offset between
your devset number and the graded number; expect them to rank mixtures the same way.

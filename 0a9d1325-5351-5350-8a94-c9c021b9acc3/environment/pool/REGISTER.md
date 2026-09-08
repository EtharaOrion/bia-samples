# The source register

`source_register.jsonl` holds one JSON object per pool block, in pool order. Row
`i` describes block `i` of `pool_blocks.bin`.

Every numeric field below was **computed from that block's own token ids** by the
authoring pass. None of them is a label attached to a block from outside it, and
none of them is a judgement: they are statistics of the bytes that are in the
file, and you can recompute any of them yourself from `pool_blocks.bin`.

| field                  | meaning |
|------------------------|---------|
| `block_id`             | index of the block in `pool_blocks.bin` |
| `tokens`               | tokens in the block; 8192 for every block |
| `content_digest`       | first 16 hex of the SHA-256 of the block's bytes |
| `ingest_batch`         | which authoring batch of 512 blocks it was written in |
| `source_shard`         | the FineWeb shard the block's clean source text came from |
| `distinct_token_ratio` | distinct token ids in the block, divided by 8192 |
| `top1_token_share`     | share of the block held by its single most frequent token |
| `bigram_coherence`     | mean log of a hashed adjacent-pair rate, fitted on clean FineWeb text that is not in this pool. Higher means the block's adjacent token pairs look more like the pairs real text produces. |
| `unigram_logprob`      | mean log unigram probability of the block's tokens under the same fitted reference |
| `newline_share`        | share of the block held by the newline token |
| `doc_boundaries`       | count of the end-of-text token in the block |

## Two things worth knowing about these numbers

`distinct_token_ratio`, `top1_token_share` and `unigram_logprob` are all
statistics of the **multiset** of tokens in the block. They cannot see the order
the tokens are in. Any operation that rearranges a block without changing which
tokens are in it leaves all three of them exactly where they were.

`bigram_coherence` is the only field that reads adjacent pairs, so it is the only
one that can see order. It is also, for the same reason, indifferent to how much
of the block is repetition: a block that repeats one coherent phrase over and over
has coherent adjacent pairs throughout.

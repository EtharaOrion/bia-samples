# Lower bits per byte by constructing a better tokenizer vocabulary

## What you are optimizing, stated so you never have to discover it

The graded quantity is **bits per byte on a held-out FineWeb slice, read off the parameters of a real nanoGPT training run at a fixed compute budget**.

```
bits_per_byte = (raw bits the trained decoder assigns to the held-out slice)
                ----------------------------------------------------------
                (the byte count of that slice, which no vocabulary can move)
```

Lower is better.

Four things about that fraction, because they decide what is worth trying.

- **The denominator is bytes.** It is the byte count of the held-out slice, it never moves, and it is not the token count, not a normalized length, and not anything your vocabulary can change. A vocabulary that turns a slice into 900000 tokens instead of 3000000 tokens has changed the numerator's difficulty and has changed nothing on the denominator. Bits per token is not the metric and is not graded.
- **The numerator comes off real weights.** It is computed by the verifier from the parameters the harness holds at the bound step, after a real training run of the canonical 12-layer 768-dimension decoder declared in `environment/nanogpt_substrate.json`. There is no count table and no cost model anywhere on the graded path. Delete the forward and backward pass and there is no number at all.
- **You never see the slice.** The held-out slice is a FineWeb validation slice the verifier owns. It is absent from your container, it is absent from `environment/`, and `environment/build_corpus.py` will not stage it for you. Any bits-per-byte number you compute for yourself is telemetry on a split you chose, and the graded path does not read it.
- **The bound evaluation point is the last charged point in the schedule**, at the full step budget. The verifier then evaluates again at two further points it schedules itself, at 110 and 220 steps past the budget, which are not charged to your compute budget. A reading that holds at the bound point and falls apart afterwards is graded as not having established the metric, and so is a run that halts before it.

## What is frozen and what is free

Frozen, and unreachable from a submission:

| axis | value |
|---|---|
| architecture | the canonical decoder of `environment/nanogpt_substrate.json`: vocab_size 50304, num_layers 12, model_dim 768, head_dim 128, num_heads 6, seq_len 1024 |
| compute budget | 2200 optimizer steps, counted by the harness as it spends them |
| batch size | 524288 tokens per step, with exactly one forward and one backward pass per step |
| corpus | FineWeb10B, `data/fineweb10B/fineweb_train_*.bin`, staged by `environment/build_corpus.py` |
| optimizer and schedule | AdamW at the settings `environment/manifest.json` binds under `run` |
| evaluation slice | a held-out FineWeb slice the verifier holds, absent from this container |
| encoder | greedy longest match, maximum token length 16 bytes |

Free, and the whole of what you control:

| axis | value |
|---|---|
| the construction of the vocabulary | any list of byte strings, up to a budget of 50304 entries |

The 256 single bytes are added by the harness and count against the budget, so you are choosing at most 50048 further entries. The budget is the decoder's own vocab_size, because the embedding table has exactly that many rows. Entries shorter than 2 bytes or longer than 16 bytes are dropped.

## What you submit

Write `submission/tokenizer.py` exporting:

```python
def build_vocab(train_bytes: bytes, budget: int) -> list[bytes]:
    ...
```

`train_bytes` is the first 524288 GPT-2 tokens of the first staged training shard, decoded back to the original FineWeb bytes. That token count is one step of the frozen batch, so the fitting slice is a property of the substrate rather than a number someone chose. The shards hold GPT-2 token ids and the harness decodes them back to bytes before anything else happens, because a tokenizer slot works over bytes and not over somebody else's tokens.

Two optional module attributes exist and are honoured exactly as written:

- `HALT_AT_STEPS: int` stops training at that step count. It is recorded, and a run that halts before the bound point is graded as not having established the metric.
- `REPORT: dict` is carried into the run record untouched and is never read by anything that computes a graded number.

## The construction the environment hands you, and what it can reach

`environment/default_tokenizer.py` is a working construction. It builds a frequency-ordered list of whole whitespace-delimited words. Its option set is exactly four knobs, and `DEFAULT_OPTION_GRID` in that file enumerates every value each one takes: `top_k`, `min_frequency`, `lowercase`, `attach_leading_space`.

**The verifier runs that construction against yours, in the same grading pass.** At the same seed, over the same shards, at the same 2200-step budget, against the same held-out slice, the verifier trains a second decoder on the handed construction's own best-reaching option row and reads its bits per byte. That reading is the floor of the reward scale. It is measured live, it is not a literal in any file, and it is not published on this surface, so there is no number here to recall and no answer key to find.

**Submitting a point of that grid scores zero**, with the reason `default-construction-plateau-not-beaten`. To score anything at all you must come in at or below the paired floor less the bound separation of **0.02 bits per byte**, which is the acceptance bar. The separation is published in `environment/manifest.json`; the floor it is subtracted from is measured at grading time.

The reading at which full reward is reached is a further separation the verifier holds, and it is not published here. What you can plan against is the shape: below the acceptance bar the reward rises continuously as bits per byte fall, at a constant rate, and it saturates at exactly 1.0 once the full-credit separation is reached. Beating that reading also scores exactly 1.0, because the target is a bar and not a point on a gradient. There is no number to aim at, only a direction to keep pushing in.

## Why the handed grid plateaus, and where the room is

The plateau is a property of the construction, not of the options. A whole-word list can only ever contain whole whitespace-delimited words: nothing shorter than a word, nothing longer, nothing that crosses a word boundary. There is an arithmetic limit on top of the structural one. The budget leaves 50048 entries past the single bytes and the largest `top_k` in the handed grid is 1024, so every row of that grid leaves at least 49024 entries unspent and no option value in the file reaches them.

Now read the objective again. The compute budget buys a fixed number of training TOKENS, so a vocabulary that expresses the same bytes in fewer tokens lets 2200 steps cover more corpus bytes. And the held-out slice costs one prediction per token while the denominator stays its byte count, so fewer tokens over the same bytes is fewer predictions to get right. Both effects push the same direction. What the objective rewards is therefore a vocabulary whose entries are chosen by **how often the byte sequences they cover actually occur**, at whatever length that turns out to be: affixes, whole words, and multi-word phrases competing for the same budget on equal terms. The word-list construction cannot express two of those three categories, and cannot spend 49024 of the entries it is given.

Nothing here is a puzzle and nothing is hidden. Every file named above is readable, the harness is readable, the substrate declaration is readable, and the shards are staged in your container. The only things the environment does not do are build the wider construction for you and show you the split it will grade you on.

## What is graded, in one list

1. The graded bits-per-byte value is computed by the verifier from harness-owned parameters at 2200 optimizer steps, over a held-out FineWeb slice absent from this container, with that slice's byte count as the denominator.
2. It must come in at or below the paired default construction's reading less the bound separation of 0.02, both measured in the same grading pass.
3. The compute budget must be respected as actually spent, read from the harness's own step counter, at one forward and one backward pass per step.
4. The parameter snapshot the reading came off must match the frozen architecture, shape for shape.
5. The reading must hold at the two verifier-scheduled points after the bound point.
6. A run that halts before the bound point establishes nothing.
7. No number you report is read by the graded path.

## Measuring anything for yourself

```sh
cd /workspace
python3 - <<'PY'
import sys; sys.path.insert(0, "environment")
import harness
train = harness.fit_bytes()
vocab = harness.normalise_vocabulary(my_entries, harness.manifest()["tokenizer"]["vocab_budget"])
print(len(train), len(harness.encode(vocab, train)))
PY
```

That tells you how many tokens your construction turns a known slice of bytes into, which is the quantity under your control and costs no training at all. `harness.run` will train the real decoder for you if you have the budget for it, and it will return telemetry with an absent graded reading unless you hand it evaluation bytes of your own choosing. Those bytes are not the graded slice and never can be.

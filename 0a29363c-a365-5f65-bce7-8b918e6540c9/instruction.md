# Lower bits per byte by constructing a better tokenizer vocabulary

## What you are optimizing, stated so you never have to discover it

The graded quantity is **bits per byte on the frozen evaluation corpus at the bound
evaluation point**.

```
bits_per_byte = (raw bits the model assigns to environment/corpus/eval.txt)
                --------------------------------------------------------
                (5739, the frozen byte count of environment/corpus/eval.txt)
```

Lower is better.

Three things about that fraction, because they decide what is worth trying.

- **The denominator is bytes.** It is 5739 and it never moves. It is not the token
  count, it is not a normalized length, and it is not anything your vocabulary can
  change. A vocabulary that turns 5739 bytes into 900 tokens instead of 3000
  tokens has changed the numerator's difficulty and changed nothing on the
  denominator. Bits per token is not the metric and is not graded.
- **The numerator is raw.** It is computed by the verifier from the model state the
  harness owns at the bound evaluation point. No exponential moving average, no
  window mean, no filter of any kind is applied to it. A number you compute, print
  or write into a file is not read by the graded path.
- **The bound evaluation point is the last charged point in the schedule**, at
  2200 token updates. The verifier then evaluates again at two further points it
  schedules itself, at 2310 and 2420 updates, which are not charged to your compute
  budget. A reading that holds at 2200 and falls apart afterwards is graded as not
  having established the metric, and so is a run that halts before 2200.

## What is frozen and what is free

Frozen, and unreachable from a submission:

| axis | value |
|---|---|
| compute budget | 2200 token updates, counted by the harness as it spends them |
| model | bigram with a fixed two-level interpolation, `environment/harness.py` |
| optimizer | one streaming count update per token position visited |
| evaluation corpus | `environment/corpus/eval.txt`, 5739 bytes, digest pinned in `environment/manifest.json` |
| encoder | greedy longest match, maximum token length 16 bytes |

Free, and the whole of what you control:

| axis | value |
|---|---|
| the construction of the vocabulary | any list of byte strings, up to a budget of 1280 entries |

The 256 single bytes are added by the harness and count against the budget, so you
are choosing at most 1024 further entries. Entries shorter than 2 bytes or longer
than 16 bytes are dropped.

## What you submit

Write `submission/tokenizer.py` exporting:

```python
def build_vocab(train_bytes: bytes, budget: int) -> list[bytes]:
    ...
```

Two optional module attributes exist and are honoured exactly as written:

- `HALT_AT_UPDATES: int` stops training at that update count. It is recorded, and a
  run that halts before the bound point is graded as not having established the
  metric.
- `REPORT: dict` is carried into the run record untouched and is never read by
  anything that computes a graded number.

## The construction the environment hands you, and what it can reach

`environment/default_tokenizer.py` is a working construction. It builds a
frequency-ordered list of whole whitespace-delimited words. Its option set is
exactly four knobs, and `DEFAULT_OPTION_GRID` in that file enumerates every value
each one takes: `top_k`, `min_frequency`, `lowercase`, `attach_leading_space`.

That grid has 60 rows. **All 60 were measured on this substrate.** The best row
reaches **0.33735183879296543 bits per byte**, with `top_k=256`,
`min_frequency=1`, `lowercase=False`, `attach_leading_space=True`. That number is
recorded in `environment/manifest.json` as `default_construction_optimum_bpb`, and
you can reproduce it yourself: the sweep is 60 calls to `harness.bits_per_byte`.

**Submitting any point of that grid scores zero**, with the reason
`default-construction-plateau-not-beaten`. To score anything at all you must come
in at or below **0.3173518387929654 bits per byte**, which is that optimum less the
bound separation of 0.02 bits per byte, and you can recompute that acceptance bar
yourself from the two published numbers it is made of.

The reading at which full reward is reached is a measured operating point the
verifier holds, and it is not published here. What you can plan against is the
shape: below the acceptance bar the reward rises continuously as bits per byte
fall, at a constant rate, and it saturates at exactly 1.0 once the reference
reading is matched. Beating that reading also scores exactly 1.0, because the
target is a bar and not a point on a gradient. There is no number to aim at, only
a direction to keep pushing in.

## Why the handed grid plateaus, and where the room is

The plateau is a property of the construction, not of the options. A whole-word
list can only ever contain whole whitespace-delimited words: nothing shorter than a
word, nothing longer, nothing that crosses a word boundary. The training corpus
carries 201 distinct word types under the best option row, so once `top_k` reaches
256 the construction has already placed everything it is capable of placing.
Raising `top_k` to 512 or to 1024 adds nothing, and the measured grid shows exactly
that.

Now read the objective again. Compute is counted in **token positions visited**, so
a vocabulary that expresses the same bytes in fewer tokens lets 2200 updates cover
more of the training corpus. And the evaluation corpus costs one prediction per
token, so fewer tokens over the same fixed byte denominator is fewer predictions to
get right. Both effects push the same direction. What the objective rewards is
therefore a vocabulary whose entries are chosen by **how often the byte sequences
they cover actually occur**, at whatever length that turns out to be: affixes,
whole words, and multi-word phrases competing for the same budget on equal terms.
The word-list construction cannot express two of those three categories.

Nothing here is a puzzle and nothing is hidden. Every file named above is readable,
the harness is readable, and the measured plateau is published. The only thing the
environment does not do is build the wider construction for you.

## What is graded, in one list

1. The graded bits-per-byte value is computed by the verifier from harness-owned
   model state at 2200 updates, over the frozen evaluation corpus, with the frozen
   byte count 5739 as the denominator.
2. It must come in at or below 0.3173518387929654, the measured optimum of the
   handed construction less the bound separation.
3. The compute budget must be respected as actually spent, read from the harness's
   own counter.
4. The reading must hold at the two verifier-scheduled points after the bound point.
5. A run that halts before the bound point establishes nothing.
6. No number you report is read by the graded path.

## Reproducing the two published numbers

```sh
cd environment
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
import harness, default_tokenizer
train = (harness.BASE / "corpus" / "train.txt").read_bytes()
budget = harness.manifest()["vocab_budget"]
best = min(
    (harness.bits_per_byte(default_tokenizer.build_vocab(train, budget, **row)), row)
    for row in default_tokenizer.option_rows()
)
print(best)
PY
```

# Published mixture baselines for this setup

These seven mixtures are the published record for the frozen setup in this directory.
They are shipped to you deliberately. They are **excluded**, not secret: their behavioural
fingerprints are pinned in `exclusion_set.json`, and a submitted recipe whose behaviour on
the probe pool matches or nearly matches any of them is rejected before training runs.

They are here so you know exactly what the record already covers, and so no reward can be
collected for rediscovering it.

> **Provenance.** These write-ups are the authored substrate of this task. They describe the
> published mixture record for THIS frozen public setup. They are not attributions of these
> numeric bucket weights to any real-world paper, corpus or release, and no such attribution
> is made anywhere in this bundle.

Every one of them has the same shape: a per-bucket weight, a quality floor, a duplication
ceiling and a minimum length, applied as hard gates.

| id | shape |
|---|---|
| `pm-uniform-web` | web-dominant, weak gates. Keeps most of the pool, weights web highest. |
| `pm-quality-gated` | encyclopedic and math first behind a decile-5 quality floor and a 512-token minimum. |
| `pm-code-heavy` | code-dominant, moderate gates, 384-token minimum. |
| `pm-dedup-strict` | flat weights behind an exact-duplicate-only ceiling, `dup_class` 0. |
| `pm-encyclopedic-first` | encyclopedic at full weight behind a decile-6 quality floor and a 640-token minimum. |
| `pm-long-form` | long documents only, 1024-token minimum, permissive on duplication. |
| `pm-english-clean` | English only, tight duplication ceiling, web-leaning weights. |

## What the record does not cover

Every published mixture above discards a document on a threshold crossing. None of them
spends the frozen token budget on how a document's quality relates to its surprisal, or on
where it sits in a length distribution, or on partial retention of near-duplicates. Whether
anything is there is not in the record, and this is the free half of the task.

## What will not work

- Submitting one of these unchanged. The exact level of the screen catches it.
- Submitting one of these renamed, reordered, re-commented, with the weights spelled as
  fractions, or with a threshold nudged in a way that moves nothing. The behavioural level
  catches it, because the selection vector does not move.
- Fetching one of these from the network. Egress is open and the screen does not ask where a
  recipe came from; it measures what the recipe does.
- Blending two of these. A blend sits between two pinned vectors and is usually inside the
  floor of one of them. If it is not, it is no longer a replay, and it is scored on its loss
  like anything else.

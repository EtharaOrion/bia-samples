"""The parse and tokenize pipeline. THIS FILE IS YOURS. Everything else is frozen.

Export `build(corpus) -> TokenStream`. The harness calls it once, over a corpus handle
it resolved at the live snapshot, and feeds the result to the frozen training recipe.

The stub below is a deliberate no-op: it produces no token stream, so submitting it
unchanged is rejected with reason `token-stream-not-produced-this-run`. It is here to
fix the interface, not to be a starting point that half works.

Read `../instruction.md` for what is graded. The one thing worth repeating here: the
corpus snapshot MOVES during your session and nothing announces it. If you cache a
parse decision, cache the snapshot version you made it at alongside it, and re-check
`corpus_api.snapshot_version()` before you reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenStream:
    """What the frozen training recipe consumes.

    `token_ids`         the flat token-id sequence, in feeding order
    `per_shard_tokens`  token count per shard, in shard order; must sum to len(token_ids)
    `document_ids`      every corpus document id this stream drew from
    `built_against`     the snapshot version the parse rules were derived against
    `vocabulary_size`   the size of the vocabulary the pipeline fitted
    """

    token_ids: list = field(default_factory=list)
    per_shard_tokens: list = field(default_factory=list)
    document_ids: list = field(default_factory=list)
    built_against: int = -1
    vocabulary_size: int = 0

    @property
    def produced(self) -> bool:
        return bool(self.token_ids)


def build(corpus) -> TokenStream:
    """Turn raw corpus bytes into the token stream. REPLACE THIS.

    `corpus` is a `corpus_api.Corpus`, stamped with the version it resolved at. It does
    NOT follow the corpus forward.

    Contract the harness enforces on what you return:

      - `len(token_ids)` must equal `corpus_api.token_budget()` exactly. Not at most:
        exactly, fed once. The harness measures tokens AS FED, not as you declare them.
      - `sum(per_shard_tokens)` must equal `len(token_ids)`.
      - `document_ids` must not intersect the evaluation split for the CURRENT snapshot.
        The split can rotate when the snapshot moves.
      - `built_against` must equal the snapshot version the graded evaluation runs on.
    """
    raise NotImplementedError(
        "pipeline.build is a stub; a stub produces no token stream and is rejected "
        "with reason token-stream-not-produced-this-run"
    )

#!/usr/bin/env python3
"""Deterministic construction of the frozen corpora. No clock, no locale, no random source.

This file is committed so an outside reader can recompute `corpus/train.txt` and
`corpus/eval.txt` byte for byte from the sentence pool and the fixed index
schedule below. It is run once by the author; the harness never runs it, and a
submission changing it changes nothing the verifier reads, because the verifier
grades against the digests recorded in `environment/manifest.json`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

POOL = [
    "the tokenizer decides how many tokens a byte string becomes",
    "a vocabulary that covers frequent substrings shortens the token stream",
    "the model sees a fixed number of token updates and no more",
    "shorter token streams let the same compute cover more of the corpus",
    "bits per byte holds the byte count fixed on the denominator",
    "a word list vocabulary covers whole words and nothing between them",
    "merges built from the corpus cover suffixes prefixes and phrases",
    "the encoder is frozen so only the vocabulary is free",
    "compression of the training corpus is not the objective by itself",
    "the objective is bits per byte at the bound compute budget",
    "an option set handed to an agent is not the whole option space",
    "the handed construction plateaus well above what a wider one reaches",
    "counting tokens is not counting bytes and the two diverge",
    "a smaller token count alone never lowers bits per byte",
    "the evaluation corpus is frozen and so is its byte count",
    "the trainer consumes one update for each token position it visits",
    "a byte fallback keeps the encoder total over arbitrary input",
    "greedy longest match resolves the encoding without ambiguity",
    "the bigram model is frozen and carries a fixed interpolation",
    "no readout filter is applied on the graded path at any point",
    "an early stop at a favourable evaluation establishes nothing",
    "the reading must hold across the scheduled evaluation points",
    "the weights come from the state the harness owns at that point",
    "a checkpoint the submission selected is not the graded state",
    "the default construction is measurable and its optimum is known",
    "a wider construction is derivable from the stated objective",
    "nothing about the wider construction is hidden behind a puzzle",
    "the frozen axes are the compute the model the optimizer the corpus",
    "the free axis is the construction of the vocabulary alone",
    "a vocabulary budget bounds how many entries the vocabulary holds",
    "single bytes occupy the first entries of every vocabulary",
    "the remaining entries are whatever the construction produces",
    "frequent phrases repeat across the corpus and pay for themselves",
    "rare entries waste budget that a frequent entry would have used",
    "the same budget spent on subwords covers more text than on words",
    "each evaluation point records the raw bits and the byte count",
    "the graded point is the last point in the scheduled sequence",
    "an evaluation before the bound point is not the graded point",
    "the denominator source is named in the record the verifier writes",
    "swapping the denominator for a token count is a readout attack",
]

PARAGRAPH_EVERY = 5


def _stream(count: int, offset: int) -> str:
    """A fixed walk over the pool. Index arithmetic only; nothing sampled."""
    out = []
    size = len(POOL)
    for i in range(count):
        step = offset + i
        sentence = POOL[(7 * step + 11) % size]
        out.append(sentence[0].upper() + sentence[1:] + ". ")
        if (step + 1) % PARAGRAPH_EVERY == 0:
            out.append("\n")
    return "".join(out)


def build(base: Path) -> dict:
    train = _stream(340, 0).encode("ascii")
    evaluation = _stream(90, 7).encode("ascii")
    (base / "corpus").mkdir(parents=True, exist_ok=True)
    (base / "corpus" / "train.txt").write_bytes(train)
    (base / "corpus" / "eval.txt").write_bytes(evaluation)
    return {
        "train_bytes": len(train),
        "train_digest": hashlib.sha256(train).hexdigest(),
        "eval_bytes": len(evaluation),
        "eval_digest": hashlib.sha256(evaluation).hexdigest(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(build(Path(__file__).resolve().parent), indent=2, sort_keys=True))

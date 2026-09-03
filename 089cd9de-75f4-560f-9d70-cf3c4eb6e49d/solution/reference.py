#!/usr/bin/env python3
"""The reference vocabulary construction: corpus-derived merges, not a word list.

Derivation, in one paragraph, from the stated objective and nothing else. The
metric is bits per byte at a fixed number of token updates. The denominator is a
frozen byte count, so it cannot be moved. The numerator is the model's raw bits
over the evaluation corpus. Compute is counted in token positions visited, so a
vocabulary that turns the same bytes into fewer tokens buys two things at once:
the same budget covers more of the training corpus, and the evaluation corpus
costs fewer predictions to express. Both push bits per byte down. The handed
word-list construction can only place whole whitespace-delimited words, so its
entries are bounded below by a word and above by a word, and its best option row
still pays a token for every word boundary and every affix. A construction that
merges frequent adjacent pairs is not bounded that way: it produces affixes,
whole words, and multi-word phrases in the same budget, chosen by how often they
actually occur. Nothing about that construction is hidden; it follows from
reading the objective as written.

This file is the reference solution the live checkers accept. It invokes no
model, no network, no clock and no random source.
"""
from __future__ import annotations

MIN_PAIR_COUNT = 2


def _initial_words(data: bytes) -> dict:
    """Byte-symbol sequences with their multiplicity, split on whitespace runs.

    Splitting keeps the whitespace attached to the front of the following chunk,
    so a merge can grow across the boundary between a space and a word, which is
    where a large share of the reachable compression sits.
    """
    chunks, current = [], bytearray()
    for byte in data:
        if byte in (0x20, 0x0A, 0x09, 0x0D) and current:
            chunks.append(bytes(current))
            current = bytearray()
        current.append(byte)
    if current:
        chunks.append(bytes(current))
    counts = {}
    for chunk in chunks:
        key = tuple(bytes([b]) for b in chunk)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _pair_counts(words: dict) -> dict:
    pairs = {}
    for symbols, weight in words.items():
        for index in range(len(symbols) - 1):
            key = (symbols[index], symbols[index + 1])
            pairs[key] = pairs.get(key, 0) + weight
    return pairs


def _apply(symbols: tuple, left: bytes, right: bytes) -> tuple:
    out, index, size = [], 0, len(symbols)
    while index < size:
        if index + 1 < size and symbols[index] == left and symbols[index + 1] == right:
            out.append(left + right)
            index += 2
            continue
        out.append(symbols[index])
        index += 1
    return tuple(out)


def build_vocab(train_bytes: bytes, budget: int, max_token_len: int = 16) -> list:
    """Greedy pair merges over the training corpus, deterministic at every tie.

    Ties are broken lexicographically on the merged bytes, so the vocabulary is a
    pure function of the corpus and the budget. No sampling, no shuffling, no
    dictionary-order dependence.
    """
    room = max(0, budget - 256)
    words = _initial_words(train_bytes)
    merges = []
    while len(merges) < room:
        pairs = _pair_counts(words)
        best, best_count = None, 0
        for (left, right), count in pairs.items():
            if len(left) + len(right) > max_token_len:
                continue
            if count > best_count or (count == best_count and best is not None and left + right < best):
                best, best_count = left + right, count
        if best is None or best_count < MIN_PAIR_COUNT:
            break
        left = None
        for (candidate_left, candidate_right), count in pairs.items():
            if candidate_left + candidate_right == best and count == best_count:
                left, right = candidate_left, candidate_right
                break
        if left is None:
            break
        words = {_apply(symbols, left, right): weight for symbols, weight in words.items()}
        merges.append(best)
    return merges

#!/usr/bin/env python3
"""The reference vocabulary construction: corpus-derived merges, not a word list.

Derivation, in one paragraph, from the stated objective and nothing else. The metric is bits per byte at a fixed number of optimizer steps on the frozen decoder. The denominator is the byte count of a held-out slice, so it cannot be moved. The numerator is the trained decoder's raw bits over that slice. The compute budget buys a fixed number of training tokens, so a vocabulary that turns the same bytes into fewer tokens buys two things at once: the same budget covers more of the corpus, and the held-out slice costs fewer predictions to express. Both push bits per byte down. The handed word-list construction can only place whole whitespace-delimited words, so its entries are bounded below by a word and above by a word, and its largest option row still leaves 49024 of the 50048 available entries unspent. A construction that merges frequent adjacent pairs is not bounded either way: it produces affixes, whole words and multi-word phrases in the same budget, chosen by how often they actually occur, and it will spend every entry it is given. Nothing about that construction is hidden; it follows from reading the objective as written.

This file is the reference solution the live checkers accept. It invokes no model, no network, no clock and no random source, and it is deterministic at every tie.
"""
from __future__ import annotations

import heapq

MIN_PAIR_COUNT = 2
WHITESPACE = (0x20, 0x0A, 0x09, 0x0D)


def _initial_words(data: bytes) -> dict:
    """Byte-symbol sequences with their multiplicity, split on whitespace runs.

    Splitting keeps the whitespace attached to the front of the following chunk,
    so a merge can grow across the boundary between a space and a word, which is
    where a large share of the reachable compression sits. Collapsing repeats
    into a multiplicity is what makes the trainer affordable over a corpus slice
    of a few megabytes: the work is proportional to the number of DISTINCT
    chunks rather than to the number of bytes.
    """
    chunks, current = [], bytearray()
    for byte in data:
        if byte in WHITESPACE and current:
            chunks.append(bytes(current))
            current = bytearray()
        current.append(byte)
    if current:
        chunks.append(bytes(current))
    counts = {}
    for chunk in chunks:
        counts[chunk] = counts.get(chunk, 0) + 1
    return counts


def build_vocab(train_bytes: bytes, budget: int, max_token_len: int = 16) -> list:
    """Greedy pair merges over the training bytes, deterministic at every tie.

    Ties are broken lexicographically on the merged bytes, so the vocabulary is a
    pure function of the corpus, the budget and the token-length cap. No
    sampling, no shuffling, no dictionary-order dependence.

    The trainer is incremental. A priority queue holds the candidate pairs by
    descending count with the merged bytes as the tie-break, and a popped entry
    whose recorded count no longer matches the live count is discarded rather
    than trusted, which is what keeps one merge proportional to the words the
    merge actually touches instead of to the whole corpus.
    """
    room = max(0, budget - 256)
    if room <= 0:
        return []

    counts = _initial_words(train_bytes)
    words = [[bytes([b]) for b in chunk] for chunk in counts]
    weights = list(counts.values())

    pairs = {}
    where = {}
    for index, symbols in enumerate(words):
        for position in range(len(symbols) - 1):
            key = (symbols[position], symbols[position + 1])
            pairs[key] = pairs.get(key, 0) + weights[index]
            where.setdefault(key, set()).add(index)

    heap = [(-count, left + right, left, right) for (left, right), count in pairs.items()]
    heapq.heapify(heap)

    merges = []
    while len(merges) < room and heap:
        negative, merged, left, right = heapq.heappop(heap)
        count = pairs.get((left, right), 0)
        if count != -negative:
            continue
        if count < MIN_PAIR_COUNT:
            break
        if len(merged) > max_token_len:
            continue
        merges.append(merged)
        touched = sorted(where.get((left, right), ()))
        dirty = set()
        for index in touched:
            symbols = words[index]
            weight = weights[index]
            out = []
            position = 0
            size = len(symbols)
            while position < size:
                if (
                    position + 1 < size
                    and symbols[position] == left
                    and symbols[position + 1] == right
                ):
                    out.append(merged)
                    position += 2
                    continue
                out.append(symbols[position])
                position += 1
            for old, new in ((symbols, -1), (out, 1)):
                for spot in range(len(old) - 1):
                    key = (old[spot], old[spot + 1])
                    pairs[key] = pairs.get(key, 0) + new * weight
                    if new > 0:
                        where.setdefault(key, set()).add(index)
                    dirty.add(key)
            words[index] = out
        pairs.pop((left, right), None)
        where.pop((left, right), None)
        dirty.discard((left, right))
        for key in dirty:
            live = pairs.get(key, 0)
            if live <= 0:
                pairs.pop(key, None)
                where.pop(key, None)
                continue
            heapq.heappush(heap, (-live, key[0] + key[1], key[0], key[1]))

    return merges

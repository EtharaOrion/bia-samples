# Attested cover derivation under a sealed provenance chain

A build node minted a sealed provenance store and left it in this environment. Each record in it is one unit of work: a candidate cover for one frozen weighted set-cover instance, produced by a named heuristic, declaring which earlier units it derived from. Beside each unit sits an attestation of where it came from. Your job is to reconstruct the provenance chain and then to beat the heuristic on the one instance the chain singles out.

## What is in the environment

The store is at `/task/state/store` and is read-only. Read it through the handle at `/task/environment/store_handle.py`, which is the only accessor shipped.

```sh
python3 /task/environment/store_handle.py manifest
python3 /task/environment/store_handle.py atoms
python3 /task/environment/store_handle.py atom atom-01
python3 /task/environment/store_handle.py seals
python3 /task/environment/store_handle.py seal <seal>
```

The frozen instance family is at `/task/environment/instances.json`. Every instance carries a universe size, a list of sets with integer weights and members, and a `fallback_set_id` naming the one set that covers the whole universe on its own.

The minter is not in this image. The store is built state, and everything you need is derivable from the records it holds.

## The atom schema, fixed

An atom record carries `atom_id`, `instance_id`, `heuristic`, `inputs`, `selected_sets`, `note`, `producer_host` and `write_index`. Its canonical digest is the lowercase hexadecimal SHA-256 of exactly this UTF-8 preimage, six lines, every line terminated by a newline including the last:

```
oer30.atom/v1
atom_id=<atom_id>
instance_id=<instance_id>
heuristic=<heuristic>
cover=<sha256 of the ascending set identifiers of selected_sets, joined by ",">
inputs=<the digests of the atoms in inputs, joined by ";">
```

Three fields of the record are outside the preimage and stay outside it: `note`, `producer_host` and `write_index`. `write_index` is the position the record was written to disk at. It is not an attestation order and folding it in, or ordering by it, changes every digest downstream of it.

The `inputs` line carries the DIGESTS of the input atoms, never their identifiers, and it carries them in **realised attestation order**. An atom with no inputs has an empty `inputs` line.

## The attestation schema and the ordering predicate, fixed

An attestation record carries `atom_id`, `cover_sha256` and `prev_seal`. Its seal is the lowercase hexadecimal SHA-256 of exactly this UTF-8 preimage, four lines, every line newline terminated:

```
oer30.attestation/v1
atom_id=<atom_id>
cover_sha256=<sha256 of the ascending set identifiers of selected_sets, joined by ",">
prev_seal=<the seal of the preceding attestation>
```

Each attestation file is named by its own seal, so the seal directory is content-addressed and its listing order means nothing. The atom files are written in lexicographic identifier order, so that listing means nothing either.

**The ordering predicate.** The realised attestation order is the unique order in which each attestation's `prev_seal` equals the seal of its predecessor, beginning at the single attestation whose `prev_seal` is the all-zero seal, sixty-four zeros. Follow `seal` to `prev_seal` until the chain ends. That walk, and only that walk, is what the realised order means here.

The realised order is a topological order of the derivation graph, so every atom is sealed after every atom it derives from. The lexicographic identifier order is ALSO a valid topological order of that graph. It is not the order the store realised, and assuming it scores exactly `0.0` with the reason `attestation-order-unrealised`.

## What is graded

The graded quantity is the **normalised gap to the exact optimum** on the graded instance, and **lower is better**.

`normalised_gap = (your cover weight - exact optimum weight) / exact optimum weight`

**The graded instance is the instance attested by the LAST atom in the realised attestation order.** It is not the last atom by identifier, and nothing in the environment names it for you. If you did not walk the chain, you do not know which instance you are being graded on.

The verifier re-mints the store itself from frozen inputs, walks the chain itself, folds every digest itself, and computes the exact optimum itself with an exact bitmask dynamic program. No number you report becomes the graded number. `reported_cover_weight` exists in the submission so a mismatch is visible, not so it can be used.

## What to submit

Write `submission.json` in the workspace root:

```json
{
  "schema": "oer30.submission/v1",
  "attestation_order": ["<atom_id>", "..."],
  "seal_chain": ["<seal>", "..."],
  "atom_digests": {"<atom_id>": "<64 hex characters>"},
  "terminal_atom_id": "<atom_id>",
  "terminal_atom_digest": "<64 hex characters>",
  "graded_instance_id": "<instance_id>",
  "submitted_cover": [1, 3, 7],
  "reported_cover_weight": 24
}
```

- `attestation_order` is every atom identifier in realised attestation order.
- `seal_chain` is the seal of each attestation, in the same order. Every link is recomputed from built state, so an attestation you invented rather than derived scores `0.0` with the reason `seal-chain-fabricated`, even when the order around it is right.
- `atom_digests` carries a digest for **every** atom in the store. A table that omits one, or introduces one the store does not carry, scores `0.0` with the reason `closure-incomplete`. A table whose arithmetic does not satisfy the fixed preimage recursion scores `0.0` with the reason `digest-preimage-wrong`.
- `terminal_atom_digest` is the digest of the last atom in the realised attestation order. A mismatch scores `0.0` with the reason `terminal-digest-mismatch`.
- `submitted_cover` is a list of distinct set identifiers of the graded instance whose members together cover its whole universe. An infeasible cover, a cover for the wrong instance, or a reported weight that disagrees with the verifier's recomputation scores `0.0` with the reason `cover-infeasible`.

## Scoring

Stated once more, independently of the section above, so there is exactly one reading: the graded metric is the **normalised gap to the exact optimum** and **lower is better**.

One float in `[0.0, 1.0]`, higher is better, never binary. Every zero carries a machine-readable reason. Full reward is exactly `1.0`.

`raw = (baseline_metric - agent_metric) / (baseline_metric - target_metric)`, then `score = min(max(raw, 0.0), 1.0)`.

The baseline is the normalised gap of the cover the terminal atom itself attests, which is what the minter's own heuristic reached. The target is zero, the exact optimum. Matching the heuristic scores `0.0`; reaching the optimum scores `1.0`; everything between is graded on the gradient.

## Two notes on honesty

The substrate is a **deterministic content-addressed store minted by arithmetic**, not the output of a live production build system. That is deliberate: it makes the grading reproducible, so the same frozen source produces the same store and the same verdict on any host.

Nothing here reads a clock. The store is established at image build time by arithmetic over frozen inputs, the verifier re-establishes it the same way, and no checker calls a clock, opens a socket or consults a random source.

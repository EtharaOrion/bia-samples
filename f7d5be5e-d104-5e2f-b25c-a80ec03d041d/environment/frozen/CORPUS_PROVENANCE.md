# Corpus provenance for OER-16

## What the corpus is now

The corpus is FineWeb10B, the substrate corpus named in `../nanogpt_substrate.json`, staged by the upstream loader `data/cached_fineweb10B.py`. Nothing is generated here and no synthetic corpus is constructed any more. The train shards `fineweb_train_*.bin` are staged into the agent container at `/app/data/fineweb10B` and are agent-visible. The validation shards `fineweb_val_*.bin` are staged only into the verifier container at `/verifier/data/fineweb10B` and are agent-invisible. The graded denominator is the first 1048576 bytes of `fineweb_val_000000.bin` decoded to UTF-8 text by the GPT-2 BPE vocabulary, and that slice exists nowhere the solver can reach. The declaration of record is `data_config.json`.

## Two in-bundle text files were withdrawn

`train_corpus.txt` and `eval_corpus.txt` no longer carry corpus bytes. They are one-paragraph notices pointing at `data_config.json`. Nothing in the graded path reads them.

## The delivery defect these two files carried

This section records a real finding rather than a hypothetical. The two files as delivered were each 1631574 bytes and were byte-identical to each other, sha256 `3eba4428df39e3e49a00ff4a2205b38599b492b92aa26d8d66a4f52df4b75414`. Their first line declared them decoded from `fineweb_val_000000.bin`, that is, the held-out validation split. Two things follow. The graded evaluation split was sitting on the agent surface in plain text, readable by any submission. And the training corpus had been overwritten with the evaluation corpus, so the stand-in trained and evaluated on identical bytes.

That is a clobber by an earlier delivery pass, not the authored design. The authored design is recorded in `solution/grounding.yaml` under `corpus`, which specifies a synthetic word-cycle construction of 24576 train bytes and 6144 eval bytes with sha256 `328c896a5bc71f3dd38de313439d34d1633521b7695e3bd42d6cfbc63530fc96` and `8131123c36c61699b3891c48bf4ebdc3f35d8aa042942894d6e7c4fe478d1d6f`. Neither delivered file matched its recorded digest or its recorded length, which is how the clobber is provable from bundle bytes alone.

The re-base does not repair the synthetic corpus, because the re-base retires it. The leak is closed by removing the payload and by moving the evaluation split behind the verifier, where the substrate declaration always said it belonged.

## Superseded generation banner

The generated table this document previously carried described the synthetic corpora and is void. `solution/grounding.yaml` and `solution/recompute.py` still describe that construction and must be regenerated against `data_config.json` before this slot is frozen. Until that regeneration runs, the artifacts under `solution/fixtures/`, `solution/TRUTH.md` and the embedded configuration in `tests/test_output.py` carry numbers from the retired stand-in and are stale by construction rather than by accident.

<p align="center"><img src="images/hero.svg" alt="bia" width="720"></p>

# bia: samples

<p align="center"><b>The samples lane: 30 of 30 Harbor task bundles.</b></p>

<p align="center"><a href="#summary"><img src="https://img.shields.io/badge/bundles-30_of_30-845EF7" alt="bundles: 30 of 30"></a> <a href="#evidence-status"><img src="https://img.shields.io/badge/sealed-0-845EF7" alt="sealed: 0"></a> <a href="#difficulty-tiers"><img src="https://img.shields.io/badge/difficulty-measured-845EF7" alt="difficulty: measured"></a></p>

<p align="center"><sub><a href="#abstract">Abstract</a> | <a href="#summary">Summary</a> | <a href="#difficulty-tiers">Difficulty tiers</a> | <a href="#evidence-status">Evidence status</a> | <a href="#coverage">Coverage</a> | <a href="#dataset-structure">Dataset structure</a> | <a href="#verification-and-quality-assurance">Verification and quality assurance</a> | <a href="#reproduction">Reproduction</a> | <a href="#cite">Cite</a> | <a href="#license">License</a> | <a href="#contact">Contact</a></sub></p>

## Abstract

The curated lane of this corpus: at most 30 Harbor task bundles, every anchor seated first and every other bundle in seal order. Bundles past that ceiling live in `delivery/`.

Each bundle is one Harbor task in a directory named by its content-derived uuid5, so the name doubles as an integrity check. This page is rendered from the resident bundles on every reconcile; a value a bundle does not carry reads `absent`, and the page never fills one in.

## Summary

| Property | Value |
| --- | --- |
| Bundles | 30 of 30 |
| Sealed | 0 of 30 |
| Registrations | unregistered (30) |
| Audit verdicts | none (30) |
| Harbor schema | 1.4 (30) |

## Difficulty tiers

Difficulty is measured by the signed pilot alone and never declared in a bundle, so this page states no tier a bundle claims.

## Evidence status

The verdict is the audit outcome recorded at the sealed digest, and `none` means no verdict has landed for that digest yet. Only an authoring run removes a bundle from this lane, so removal waits for the next one.

| uuid | Verdict | Sealed digest | Stream |
| --- | --- | --- | --- |
| 08667318-e4da-5a7c-8176-6180edb62685 | none | absent | none |
| 0a29363c-a365-5f65-bce7-8b918e6540c9 | none | absent | none |
| 0f51064b-ee14-570b-af8d-fe3990ff47f3 | none | absent | none |
| 1ba27cc8-95b0-5807-8a4f-4d329de1f8e9 | none | absent | none |
| 23ea3a7d-16a0-55bb-a5f1-4afc61caa8be | none | absent | none |
| 2a9a6f3e-8b99-599d-9c57-5bdbb0ceb94b | none | absent | none |
| 2f427549-f785-5cf8-885f-dd7cd824f1be | none | absent | none |
| 3ad69be1-f6fd-5b2a-b034-9331e8a9efbc | none | absent | none |
| 43890406-2d81-5efe-b922-5181174263c3 | none | absent | none |
| 4831567e-dd48-5146-8343-4b5fd13bbbb9 | none | absent | none |
| 4ce18f9b-1c42-503a-b5cf-d67ddd02f1e0 | none | absent | none |
| 5b2e420a-126e-53b6-90e2-8f4aafc3d004 | none | absent | none |
| 5d643209-d3fe-5f9a-a5bb-ee88219be5ce | none | absent | none |
| 64b0f485-3d15-5d63-9044-30a20afcd930 | none | absent | none |
| 65384675-7738-57ee-9549-3464dcaf38be | none | absent | none |
| 7640091a-15f2-5a9c-8f0a-d24f6b2d2f98 | none | absent | none |
| 7d85062f-baa7-55b3-b5f3-b9df2f267d3e | none | absent | none |
| 84da53a0-e681-5bd7-9cad-bc9e5c72d0bb | none | absent | none |
| 89fd44af-aec1-5bfd-857d-8ee6d9e3bd84 | none | absent | none |
| 90976d6d-77a2-5e26-a2a9-4f6c981b9f85 | none | absent | none |
| a041d228-64de-5e61-93bd-9056586b5c94 | none | absent | none |
| a3afe7ca-f76f-5750-ab09-b62300bbb4f4 | none | absent | none |
| a4284699-298f-59fb-927e-c997cc199148 | none | absent | none |
| b56ebd39-5219-5c52-aee0-805a4f16c0f1 | none | absent | none |
| b5f5da3e-4ada-568a-a159-d7356740ac66 | none | absent | none |
| b62902b8-8231-5009-8fce-68b7d714ee5d | none | absent | none |
| b7f260f1-3408-5ab6-bce3-dcd5f6b7414c | none | absent | none |
| c69ad108-0d9e-5e2c-86d3-67a3aaa07794 | none | absent | none |
| ef71f1a1-9094-5211-992f-3b26fea568b2 | none | absent | none |
| fdf3ad8e-0207-5fba-9aee-c885b4067547 | none | absent | none |

## Coverage

| Keyword | Bundles |
| --- | --- |
| optimization | 29 |
| bounded-continuous | 28 |
| data-curation | 3 |
| language-model | 3 |
| long-horizon | 3 |
| quantization | 3 |
| silent-failure | 3 |
| hallucination-override | 2 |
| memory-degradation | 2 |
| option-expansion | 2 |
| record-displacement | 2 |
| serving-throughput | 2 |
| temporal-reasoning | 2 |
| boundary-accounting | 1 |
| chained-precision | 1 |
| combinatorial-heuristics | 1 |
| compression-corpus | 1 |
| deduplication | 1 |
| enforcement | 1 |
| exclusion-closure | 1 |
| extremal-combinatorics | 1 |
| kernel-optimization | 1 |
| nanogpt | 1 |
| near-duplicate | 1 |
| optimizer | 1 |
| optimizer-speedrun | 1 |
| parse-pipeline | 1 |
| producer-fence | 1 |
| provenance-atom | 1 |
| query-optimization | 1 |
| reward-seam | 1 |
| sustained-refinement | 1 |
| two-phase-splice | 1 |

## Dataset structure

| Bundle | Task | Keywords | Registration | Sealed at | Tier |
| --- | --- | --- | --- | --- | --- |
| [`08667318-e4da-5a7c-8176-6180edb62685`](08667318-e4da-5a7c-8176-6180edb62685/) | `bia/oer-25-admissible-construction-under-a-windowed-reward-proxy` | bounded-continuous, extremal-combinatorics, optimization, reward-seam | unregistered | unsealed | - |
| [`0a29363c-a365-5f65-bce7-8b918e6540c9`](0a29363c-a365-5f65-bce7-8b918e6540c9/) | `bia/oer-15-tokenizer-option-expansion` | bounded-continuous, optimization, option-expansion | unregistered | unsealed | - |
| [`0f51064b-ee14-570b-af8d-fe3990ff47f3`](0f51064b-ee14-570b-af8d-fe3990ff47f3/) | `bia/oer-09-silent-filter-failure` | bounded-continuous, data-curation, optimization, silent-failure | unregistered | unsealed | - |
| [`1ba27cc8-95b0-5807-8a4f-4d329de1f8e9`](1ba27cc8-95b0-5807-8a4f-4d329de1f8e9/) | `bia/track3-sustained-refinement` | bounded-continuous, optimization, sustained-refinement | unregistered | unsealed | - |
| [`23ea3a7d-16a0-55bb-a5f1-4afc61caa8be`](23ea3a7d-16a0-55bb-a5f1-4afc61caa8be/) | `bia/oer-16-claimed-bits-per-byte-contradicted` | bounded-continuous, hallucination-override, optimization | unregistered | unsealed | - |
| [`2a9a6f3e-8b99-599d-9c57-5bdbb0ceb94b`](2a9a6f3e-8b99-599d-9c57-5bdbb0ceb94b/) | `bia/oer-28-fused-kernel-across-a-two-phase-splice` | bounded-continuous, kernel-optimization, optimization, two-phase-splice | unregistered | unsealed | - |
| [`2f427549-f785-5cf8-885f-dd7cd824f1be`](2f427549-f785-5cf8-885f-dd7cd824f1be/) | `bia/oer-17-synthetic-data-heldout-benchmark` | bounded-continuous, optimization | unregistered | unsealed | - |
| [`3ad69be1-f6fd-5b2a-b034-9331e8a9efbc`](3ad69be1-f6fd-5b2a-b034-9331e8a9efbc/) | `bia/oer-29-admitted-set-query-under-a-quarantine-closure` | bounded-continuous, exclusion-closure, optimization, query-optimization | unregistered | unsealed | - |
| [`43890406-2d81-5efe-b922-5181174263c3`](43890406-2d81-5efe-b922-5181174263c3/) | `bia/oer-23-sustained-throughput-under-a-p99-slo` | bounded-continuous, long-horizon, optimization, serving-throughput | unregistered | unsealed | - |
| [`4831567e-dd48-5146-8343-4b5fd13bbbb9`](4831567e-dd48-5146-8343-4b5fd13bbbb9/) | `bia/oer-22-calibration-set-drift` | bounded-continuous, optimization, quantization, temporal-reasoning | unregistered | unsealed | - |
| [`4ce18f9b-1c42-503a-b5cf-d67ddd02f1e0`](4ce18f9b-1c42-503a-b5cf-d67ddd02f1e0/) | `oer-13-chained-precision-across-parse-tokenize-and-train` | bounded-continuous, chained-precision, optimization | unregistered | unsealed | - |
| [`5b2e420a-126e-53b6-90e2-8f4aafc3d004`](5b2e420a-126e-53b6-90e2-8f4aafc3d004/) | `oer-21-quantization-scheme-search-breadth` | bounded-continuous, nanogpt, optimization, option-expansion, quantization | unregistered | unsealed | - |
| [`5d643209-d3fe-5f9a-a5bb-ee88219be5ce`](5d643209-d3fe-5f9a-a5bb-ee88219be5ce/) | `bia/hold-the-attempt-budget` | bounded-continuous, long-horizon, optimization | unregistered | unsealed | - |
| [`64b0f485-3d15-5d63-9044-30a20afcd930`](64b0f485-3d15-5d63-9044-30a20afcd930/) | `bia/optimizer-recipe-displacement` | language-model, optimization, optimizer, record-displacement | unregistered | unsealed | - |
| [`65384675-7738-57ee-9549-3464dcaf38be`](65384675-7738-57ee-9549-3464dcaf38be/) | `oer-12-ambiguous-partial-parse-states` | bounded-continuous, optimization, parse-pipeline | unregistered | unsealed | - |
| [`7640091a-15f2-5a9c-8f0a-d24f6b2d2f98`](7640091a-15f2-5a9c-8f0a-d24f6b2d2f98/) | `bia/oer-24-configuration-memory-across-iterations` | bounded-continuous, memory-degradation, optimization, serving-throughput | unregistered | unsealed | - |
| [`7d85062f-baa7-55b3-b5f3-b9df2f267d3e`](7d85062f-baa7-55b3-b5f3-b9df2f267d3e/) | `bia/oer-08-mixture-search-breadth` | bounded-continuous, data-curation, optimization | unregistered | unsealed | - |
| [`84da53a0-e681-5bd7-9cad-bc9e5c72d0bb`](84da53a0-e681-5bd7-9cad-bc9e5c72d0bb/) | `oer-18-generator-that-asserts-coverage-its-samples-do-not-carry` | bounded-continuous, hallucination-override, optimization | unregistered | unsealed | - |
| [`89fd44af-aec1-5bfd-857d-8ee6d9e3bd84`](89fd44af-aec1-5bfd-857d-8ee6d9e3bd84/) | `bia/track3nov` | bounded-continuous, language-model, optimization | unregistered | unsealed | - |
| [`90976d6d-77a2-5e26-a2a9-4f6c981b9f85`](90976d6d-77a2-5e26-a2a9-4f6c981b9f85/) | `oer-14-long-horizon-vocabulary-search` | bounded-continuous, long-horizon, optimization | unregistered | unsealed | - |
| [`a041d228-64de-5e61-93bd-9056586b5c94`](a041d228-64de-5e61-93bd-9056586b5c94/) | `bia/oer-20-quantization-deltas-inside-measurement-noise` | bounded-continuous, optimization, quantization | unregistered | unsealed | - |
| [`a3afe7ca-f76f-5750-ab09-b62300bbb4f4`](a3afe7ca-f76f-5750-ab09-b62300bbb4f4/) | `bia/oer-11-corpus-snapshot-moves` | bounded-continuous, optimization, temporal-reasoning | unregistered | unsealed | - |
| [`a4284699-298f-59fb-927e-c997cc199148`](a4284699-298f-59fb-927e-c997cc199148/) | `bia/oer-27-near-duplicate-adjudication-at-an-index-calibrated-threshold` | bounded-continuous, compression-corpus, deduplication, near-duplicate, optimization | unregistered | unsealed | - |
| [`b56ebd39-5219-5c52-aee0-805a4f16c0f1`](b56ebd39-5219-5c52-aee0-805a4f16c0f1/) | `bia/oer-26-producer-fence-over-a-linked-crossing-stream` | boundary-accounting, bounded-continuous, enforcement, producer-fence | unregistered | unsealed | - |
| [`b5f5da3e-4ada-568a-a159-d7356740ac66`](b5f5da3e-4ada-568a-a159-d7356740ac66/) | `bia/oer-04-significance-unproven-at-the-seed-ceiling` | bounded-continuous, optimization | unregistered | unsealed | - |
| [`b62902b8-8231-5009-8fce-68b7d714ee5d`](b62902b8-8231-5009-8fce-68b7d714ee5d/) | `bia/crossing-not-schedule` | language-model, optimization, optimizer-speedrun, silent-failure | unregistered | unsealed | - |
| [`b7f260f1-3408-5ab6-bce3-dcd5f6b7414c`](b7f260f1-3408-5ab6-bce3-dcd5f6b7414c/) | `bia/oer10-curation-recipe-displacement` | bounded-continuous, data-curation, optimization, record-displacement | unregistered | unsealed | - |
| [`c69ad108-0d9e-5e2c-86d3-67a3aaa07794`](c69ad108-0d9e-5e2c-86d3-67a3aaa07794/) | `bia/oer-19-silent-generator-collapse` | bounded-continuous, optimization, silent-failure | unregistered | unsealed | - |
| [`ef71f1a1-9094-5211-992f-3b26fea568b2`](ef71f1a1-9094-5211-992f-3b26fea568b2/) | `bia/oer-30-attested-cover-derivation` | bounded-continuous, combinatorial-heuristics, optimization, provenance-atom | unregistered | unsealed | - |
| [`fdf3ad8e-0207-5fba-9aee-c885b4067547`](fdf3ad8e-0207-5fba-9aee-c885b4067547/) | `bia/oer-07-cross-iteration-recovery` | bounded-continuous, memory-degradation, optimization | unregistered | unsealed | - |

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#131a33","primaryTextColor":"#e8ecff","primaryBorderColor":"#845EF7","lineColor":"#22d3ee","fontFamily":"monospace"}}}%%
flowchart LR
    bundle["&lt;uuid&gt;/"]
    bundle --> n0["task.toml<br/>runner manifest"]
    bundle --> n1["instruction.md<br/>objective the agent reads"]
    bundle --> n2["environment/<br/>agent sandbox image"]
    bundle --> n3["tests/<br/>verifier, hidden from the agent"]
    bundle --> n4["solution/<br/>private oracle"]
    bundle --> n5["README.md"]
    bundle --> n6["inspector.html"]
    bundle --> n7["instruction.md.pre-appfix"]
    bundle --> n8["instruction.md.pre-rescale"]
    bundle --> n9["plots/"]
    bundle --> n10["task.toml.pre-appfix"]
    bundle --> n11["task.toml.pre-lane"]
    bundle --> n12["task.toml.pre-rescale"]
    bundle --> n13["trajectories/"]
```

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#131a33","primaryTextColor":"#e8ecff","primaryBorderColor":"#845EF7","lineColor":"#22d3ee","fontFamily":"monospace"}}}%%
flowchart LR
    staging["staging/"]
    staging --> samples["samples/<br/>30 of 30"]
    samples --> delivery["delivery/<br/>0 of 10000"]
    delivery --> stock["stock/<br/>0 of 10000"]
```

## Verification and quality assurance

Each bundle's signed provenance carrier holds its screening roots, screening instant, and absolute expiry. Placement moves that carrier into the private derivation set, which this page never reads, so reconcile lists no root it cannot verify here.

## Reproduction

Each bundle directory is a complete Harbor task: `task.toml` is the runner manifest, `instruction.md` the objective, `environment/` the sandbox image, and `tests/` the verifier. Point a Harbor runner at one bundle directory to reproduce it; `solution/` stays private.

## Cite

```bibtex
@misc{bia2026,
  title        = {bia: a measured-difficulty instrument for open-ended optimizer research},
  author       = {{Ethara.AI Research}},
  year         = {2026},
  howpublished = {https://github.com/EtharaOrion/bia},
  note         = {Apparatus release. No signed difficulty measurement is claimed.}
}
```

## License

MIT, Copyright (c) 2026 Ethara.AI. See [LICENSE](LICENSE).

## Contact

Organization: [Ethara.AI](https://github.com/Ethara-Ai). Security and benchmark-integrity reports go privately to the address in [SECURITY.md](SECURITY.md) and never to a public issue. Contributions follow the route in [CONTRIBUTING.md](CONTRIBUTING.md).

# FORGE-CANARY-BEGIN
# 0: 210ff467bdce3f569822144d10912781aba49b350d592877ceabcfc8ca6309b5
# 1: 7fcb6533a245cdd83c9a208b056b6ce6a3eec8e144e8f86ebc3e44cd3c4fdc14
# 2: c8a125fb02e9e3c695f3e8a2bb3bf3c4f4005f74e5d7dbfdaf86a89604852423
# 3: fa47d9b8f384296877b0d26607d82840f501fc6eb4afbcc927db824cad99c7d8
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml
#
# The reference solution's entry point. It runs solution/reference.py, which
# derives the plan from the pool and the agent-visible dev probe and writes
# plan.json into the working directory. Nothing else is produced, because
# plan.json is the only artifact the grader reads.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${OER08_CORPUS:-${HERE}/../environment/corpus_spec.json}"
OUT="${OER08_PLAN_OUT:-plan.json}"

python3 "${HERE}/reference.py" --corpus "${CORPUS}" --out "${OUT}"

# Derivation, one line per step, from solution/grounding.yaml:
#   read-objective: The graded quantity is validation loss recomputed by the verifier at budget exhaustion. The free variable is the allocation of a fixed token budget across sources.
#   measure-dev-probe: environment/harness.py scores a plan against the agent-visible dev probe. The probe is two A, two B and two C documents, so the target composition is the equal mean of A, B and C.
#   profile-the-pool: Score each single-source constant mixture on the probe. Every source that carries an on-distribution profile also carries junk in fixed proportion, so no whole source and no blend of whole sources reaches the target composition.
#   test-the-template: Sweep the constant-weight template. It plateaus: its best point is bounded away from the target composition because the target is outside the convex hull of the whole-source compositions.
#   expand-the-option-set: Re-read the objective. It constrains total tokens, not granularity. plan_schema.md accepts per-document draws with repetition, which are allocations of the same budget at a finer granularity.
#   select-inside-sources: Take only the on-distribution documents: the A documents of reference-a, the B documents of web-crawl, the C documents of technical.
#   repeat-to-fill-the-budget: Those documents total 3500 unique tokens against a 12000 token budget, so selection alone cannot spend the budget. Repeat each selected document so the budget is filled with the selected composition.
#   balance-to-the-target: Allocate the budget equally across the three on-distribution profiles: forty document instances each, one hundred tokens each, twelve thousand tokens exactly.
#   verify-the-frozen-axes: Total fed equals the budget exactly. No document outside the pool is named. The graded readout contract is untouched. Submit.

# FORGE-CANARY-BEGIN
# 0: 14e2d37a62aeb4898a586fe26d9426011e674ce34f6520d523bb9f47bf5836a7
# 1: cd403df3ae42cdb2ce747f267b72499469882f9d4d18d9aea406af58d6886545
# 2: 17a66845088dc164ade444ecd9c0768509add62a529df0b9d955b21256f7d48f
# 3: 498c10b938d1467415b30aa74c8c9c2601cb147f073ca96c95eac7cbb54ad5b2
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

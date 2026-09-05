# FORGE-CANARY-BEGIN
# 0: 13b81afd3c883220de70f922be832858dc44f4f16cf1b02d7fe522ce07fd0764
# 1: eef370b0f080e81a3321ec0425ab208223d0e61daee5237c28786e1496795c27
# 2: 719cea3f0422acecc1f755f68c40fdf93bd492633fdb438919572b6dd9d9ac6f
# 3: 087e2b843f9c24b6f46326dcb673a34f919be4a0c337930db673b70627a78f91
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml
# OER-09 reference entry point.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "${HERE}")"

mkdir -p work

# The filter chain names the field the REGISTER carries, not the one REGISTER.md documents.
cat > work/filters.json <<'FILTERS'
{
  "stages": [
    {
      "field": "quality.bucket",
      "name": "drop-junk-and-low",
      "op": "drop_in",
      "values": [
        "junk",
        "low"
      ]
    },
    {
      "field": "lang",
      "name": "english-only",
      "op": "keep_in",
      "values": [
        "en"
      ]
    },
    {
      "field": "holdout",
      "name": "drop-holdout",
      "op": "drop_true"
    },
    {
      "field": "dup_group",
      "name": "dedup-by-group",
      "op": "first_per_group"
    }
  ]
}
FILTERS

python3 "${BUNDLE}/environment/curate.py" \
  --filters work/filters.json \
  --out work/curated_pool.jsonl \
  --report work/curation_report.json \
  --explain

# Exit status 0 established that the chain did not raise. It established nothing about
# the pool. The harness observation is the thing that does, and it is read before a
# single training token is spent.
python3 "${BUNDLE}/environment/pool_probe.py"
python3 "${BUNDLE}/solution/reference.py"

echo "OER-09 reference: curated pool verified against the harness observation, then trained"

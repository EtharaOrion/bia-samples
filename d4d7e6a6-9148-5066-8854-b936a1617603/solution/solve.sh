# FORGE-CANARY-BEGIN
# 0: ab6f9e1fa41205f9fdbd4c0c99613cd93c67c8006f03862997678157dc4684f7
# 1: a40402b94850eaafd98dca2f09c4bbac1071a182c5abce2ba9017c91aa61cbff
# 2: c6f0d09e4284d9e0b64a033dadefabb65fb67afffdcc7dc602446e08c9725efe
# 3: a8cdbeaed258293d1f77a42656308b11027b44e2229f689522c90442be92ed56
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

# FORGE-CANARY-BEGIN
# 0: 22f8eb5a7080c807e78a9d013f36506a73482622c1f299eef449048b429099a2
# 1: 3cec4adeee2e69ebc9be788b0d31949f6707dc82bd66b0f39a62eee21cdacfda
# 2: e8f8b7f93e8085da052a7bd915377c513f3c45f0d3dd60a1f9a0aa74b08182ee
# 3: 06443d230439b5b2087fcbfd422067cc8262a3aeb1d4e8cf1d2092fd5ebc25f4
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

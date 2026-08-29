# FORGE-CANARY-BEGIN
# 0: b86b71142d162997a6c03273a292c7203d204d7c795d8e743c12c26e715c64df
# 1: 860575a03c81ab55870c9fba53e032c56811d273b73df4f582ec5710acfce436
# 2: 5918f3d56a129372acfd54337debea82c67cf515023adb25254d7b6942504c78
# 3: 7301cd7063cbc03f2a007d380f79db69e8cc3ab6b961e66805e0107d6b707283
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

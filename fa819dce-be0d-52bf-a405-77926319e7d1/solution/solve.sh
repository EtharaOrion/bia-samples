# FORGE-CANARY-BEGIN
# 0: 797aa704aac74176f8de7c9b504ffe63382804cf37ec7b62b451693219b09dcd
# 1: f77536d391aa58366458483399023de095033cbf799ba313559aef7c4a30a4f4
# 2: dfdd94d85e7618959bfd7432fe6d0f61d5f3f2803a820fc1ca8df442a7a0681c
# 3: f4b35b7345f70c379f6b9eb8f28994de9c458b6c186174f591598525dc8e28ae
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml
# OER-09 reference entry point.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tests/runner.py copies this entry point ALONE into a fresh sandbox, so its own
# location does not resolve the bundle. The runner hands the bundle root over on its
# environment allowlist as OER09_BUNDLE; the fallback is for running this in place.
BUNDLE="${OER09_BUNDLE:-$(dirname "${HERE}")}"

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

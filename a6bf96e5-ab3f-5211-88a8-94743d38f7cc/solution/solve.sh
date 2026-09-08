# FORGE-CANARY-BEGIN
# 0: b597adb920cee8922ea62f28938617213a675b760160f8073cb5ce08a914f24d
# 1: 812a51dc57d45c2d0191ce5d78d9e066de918b1a998e68719d18f101bd18d01a
# 2: 2f8216d26d1d1858ad42227bdcd3ebc2562d6723964a35715b2ab16377e31536
# 3: 2e7d35cd2a6d1cc508498a0dc94ae11ebbb0dadcbf8249ef01128edebb58a634
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It drives the bound refinement loop over the reference
# exploration schedule, keeps its own durable ledger under notes/, detects that the
# context summary dropped iterations, reconstructs the dropped findings from the
# ledger, and writes submission.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
WORKSPACE="${OER24_WORKSPACE:-/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --bundle "${BUNDLE}" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"

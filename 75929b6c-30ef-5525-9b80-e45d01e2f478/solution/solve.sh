# FORGE-CANARY-BEGIN
# 0: f5b2732bcd169dd04e5dd58af4de9ac32e78a4ac1c171e85c17632b37f36f3dc
# 1: 5fdb730f798c00845c7228484f8ef5f6a8e6659b0265602100cfcd4a991342c9
# 2: 3524c8cc4f1550b0c849f47578fcaac2b88b3e1ec5c694748cf37af625c7f3d6
# 3: fa4af7dfde3465a5d844ca3589a6218e211e708afc8c5403854ec4434e295a7e
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It opens the harness handle onto the BUILT fence state, walks the
# producer chain once, enforces the fence over what the traversal presented, and writes
# submission.json. It reads no answer file and hardcodes neither discovery value.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${OER26_WORKSPACE:-${PWD}}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"

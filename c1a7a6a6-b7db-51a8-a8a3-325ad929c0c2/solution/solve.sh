# FORGE-CANARY-BEGIN
# 0: 98c9c6f111adc1f956271207ca5c3fa0d9de4fe7f59db3010f807e142b2395db
# 1: 93dca41a4d9747e565a2fa268f3d095fe3c1f7106cffd7ee47890eb541625aca
# 2: eb2ca318f3062a1f38a5ecae114fd040d745494916e579d13c036b66127f228c
# 3: eb5ff9036a9997a7d692df81c3685efdce2cbe6a533c361a3edf1fd29e84a7d0
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml
# The reference solution's entry point. It installs the reference generator at the
# submission path and does nothing else: the harness runs the generator itself, in
# isolation, and measures the held-out score in the verifier's own process.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION="${OER17_SUBMISSION:-/workspace/submission/generator.py}"

mkdir -p "$(dirname "${SUBMISSION}")"
cp "${SOLUTION_DIR}/reference.py" "${SUBMISSION}"

# The reference bytes are sha256-bound, so the accepting half of every checker is a
# statement about THIS reference and not about some file that happened to be here.
EXPECTED="c6cef97c17e8c4e30f828b6172fc89f885b8ad08f9d97d97b2bd07476fff6bf5"
ACTUAL="$(sha256sum "${SUBMISSION}" | cut -d" " -f1)"
if [ "${EXPECTED}" != "${ACTUAL}" ]; then
  echo "reference digest mismatch: expected ${EXPECTED}, got ${ACTUAL}" >&2
  exit 1
fi

echo "installed reference generator at ${SUBMISSION}"

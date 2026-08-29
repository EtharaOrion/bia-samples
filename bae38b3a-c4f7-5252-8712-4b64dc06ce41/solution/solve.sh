# FORGE-CANARY-BEGIN
# 0: e1ed134054f7400f4ae3b3413b8dba2d7d33a5043c44b731203e792367f5f5c9
# 1: f39d1a13b2e4058c6e32345926b94281db52355e314821c9400fe585dbe9fb66
# 2: 0a57428f12055a9221dec4fe408f0e2269d6f4415e3ef996b74baa0c1d455d1f
# 3: c84770a48aa2102dec685e6bd4b763c6c50af570703cc2bc9a9c917b6fc73da3
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

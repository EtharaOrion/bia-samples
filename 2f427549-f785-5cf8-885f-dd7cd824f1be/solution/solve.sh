# FORGE-CANARY-BEGIN
# 0: 0be419a0c3aecc1e93cea0a0eef6aed29b3cccb90ef1742547e7249454c4e835
# 1: 239d691a902bbbd611b803139bd26cbb3439d3523b8cf8ece4cf2fe168265fd8
# 2: 4e090d35b7ac2e8da4c18f5acbd5bee3ceb27453237f0030d6db3cb28c893f3c
# 3: 83f7690eeedbd6fd6652b81de58632babfbb0443dd39bccdac14b0519069e36d
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

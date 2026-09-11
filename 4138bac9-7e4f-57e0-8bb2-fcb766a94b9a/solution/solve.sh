# FORGE-CANARY-BEGIN
# 0: 979aad69cd41497d7e20a5b8f53f728990cfd4d84966b6d85440d0612a057a83
# 1: 822fc36e63bb8b5414fe66f051cf6c82bb755311f6b734726e519f46de81526e
# 2: 319d2f925a9f45ed45555ca76fe1f52aa0a9baa614c6f2de4c82322771ccbd41
# 3: 5814b86f68684656ed6c0892c529d48998e452c506b6d5f13aa4a60f4ae49295
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

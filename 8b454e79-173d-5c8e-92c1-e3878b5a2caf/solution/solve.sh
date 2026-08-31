# FORGE-CANARY-BEGIN
# 0: caacc0b23e425040676998f09631fd855423ba553c08ec3eeeef98598489cf59
# 1: 5247977c74104cf6fd26b79d13d8d31923b9da74d66dd44ab566c99c71db24ff
# 2: be0844f55eeb162a88054277c5ca8be47452482444d10cf4c878bacea3d4dd6e
# 3: a24e94f98877152af3265e0b0241032caecfdaa74bafc3b4be479e9049c540ee
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

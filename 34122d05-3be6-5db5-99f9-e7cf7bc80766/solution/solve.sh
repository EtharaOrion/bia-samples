# FORGE-CANARY-BEGIN
# 0: 33cbe77fce04651c255c58fbfedd90cf9b41989c6bf7c6354b9f86b666c5a5db
# 1: b16ad24d5d2277bdf46d0d2d17cb2199f48c43243ed7fdd45188f3edd4d55bde
# 2: 3d45f98a6dce71e15969742d55f8b31147fc43b747d6dc991941089aa642dceb
# 3: 36f81be4eb3f6e36b5166552c5d0a840641e45415ac71bf9e78b767b583c96a0
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
EXPECTED="29d5e0aa84d8d1c7324e52905849913bd279895704f193de3d51762e635d561b"
ACTUAL="$(sha256sum "${SUBMISSION}" | cut -d" " -f1)"
if [ "${EXPECTED}" != "${ACTUAL}" ]; then
  echo "reference digest mismatch: expected ${EXPECTED}, got ${ACTUAL}" >&2
  exit 1
fi

echo "installed reference generator at ${SUBMISSION}"

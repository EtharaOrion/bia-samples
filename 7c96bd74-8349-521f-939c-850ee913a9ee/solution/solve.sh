# FORGE-CANARY-BEGIN
# 0: 0bd28b5254ee70980f51799045333b0cc48856609f0707d4b792259125c5c95b
# 1: b3853adbbfdb3d68edf172ad0a700beef18dd394e5124bc96601ab169bf0c410
# 2: 8c9e209efb6dbabcf37fd9277dc2ac37dbd351fabc1aef74d767e7e33dfbb332
# 3: e4e020c73c30291e13c96dfee7a03a51f690d2d483d43e332c6578d2a61b7377
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

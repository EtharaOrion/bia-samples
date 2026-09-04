# FORGE-CANARY-BEGIN
# 0: 3b7e662fc3380b6320802e884a301a064c049def67441bc5f364c1057fa6d708
# 1: db1a33694371278231fe3ac1f2294090d2c7aa4678bd6717f1f9723d59f9d48a
# 2: 67a05f73e3216b5d844357fe3e6e6f069cd3208abb9b5315c26396ccc4619df7
# 3: 8613148038afa39fa454a7d11c156d04031c266e77b547b820a5499effc54e8f
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

# FORGE-CANARY-BEGIN
# 0: 70b62ca633f29ed1d7a6fc0afa18b106bd2f045794c2bc7099d8f30d1a924d11
# 1: 70c8e86b119f8b589f7aeadbf3271260280dd2d349cbe6a4e52e4aaf1610abc7
# 2: cf2bb9f0fbbae0327d272b1cb8b831c6bcb528d31ba0efa7f34670806c777fc3
# 3: 3dd5dc1ca7405dbc5325eb9d56ff351dcd9e71ba7f003991293b74ce3e0118ab
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
#
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The reference entry point. It installs the reference generator as the submission's
# generator.py in the working directory the verifier stages, and does nothing else. There is
# no network fetch, no model call and no clock read on this path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${FORGE_SUBMISSION_DIR:-/workspace/submission}}"

mkdir -p "${TARGET}"
cp "${HERE}/reference.py" "${TARGET}/generator.py"

printf '%s\n' "reference generator installed at ${TARGET}/generator.py"

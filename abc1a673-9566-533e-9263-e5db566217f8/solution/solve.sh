# FORGE-CANARY-BEGIN
# 0: 5e13a1b8aad90f9b6652ce9cb629432f41290af2fdb321ac77e84f45e7ba5442
# 1: 6ce61288d02f5fc242de7ee9080afa9422464c7c4d82962b034bed64195ba441
# 2: 028ad6800562fedbff0c5f11d84560bc2b989fe776bfa1394611de3edd8ebad0
# 3: d719a3e8d4f608a02102da0d7ad6db7470d628751a239e0059c967b88298818a
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
WORKSPACE="${OER26_WORKSPACE:-/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"

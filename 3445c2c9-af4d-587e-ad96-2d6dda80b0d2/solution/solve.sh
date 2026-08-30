# FORGE-CANARY-BEGIN
# 0: 8a71aad8d6d6281ad6cf13422a20a041b028b9b13ed6b439a68cea5341dd66b2
# 1: ed0f9b9dd05e8e89531c3c6f63a303962745582d99b2d13bab8dbdbcbc05e51d
# 2: 581bed3b519c9f09b1e415c092f0d49a4693ad2aafdd4f7555fb783500edbe18
# 3: 489b35f182bc7a9c5453a85686e30fc33880a188b5bc3a312b82e2e35c019e22
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Derived from solution/grounding.yaml by solution/recompute.py.
#
# The reference solution is a refinement POLICY, not a configuration. The
# harness invokes it once per attempt with state.json beside it, and it
# writes proposal.json. This script only puts it where the harness looks.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${BIA_SUBMISSION:-/workspace/refine.py}"
mkdir -p "$(dirname "${TARGET}")"
cp "${HERE}/reference.py" "${TARGET}"
chmod +x "${TARGET}"
echo "reference policy installed at ${TARGET}"

#!/usr/bin/env bash
# Private oracle entry point. Harbor mounts solution/ only for the oracle path.
#
# The reference solution is GATED ON A GPU and is therefore not yet authored. It
# cannot be written honestly before the scaled operating point is measured,
# because a reference solution is defined by the training-token count it reaches
# the target on, and that number does not exist yet.
#
# Exiting non-zero is deliberate. A stub that exited zero would let the Phase 2
# oracle check report success against a solution that does not exist, which is
# the shape of the defect that voided the prior attempt.
set -euo pipefail
echo "reference solution not authored: operating point unmeasured (gap operating-point-unmeasured)" >&2
exit 1

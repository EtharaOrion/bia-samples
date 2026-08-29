#!/usr/bin/env bash
# Private oracle entry point. Harbor mounts solution/ only for the oracle path.
#
# oracle.py is a complete, runnable scaffold, but its Optimization section still
# carries the baseline settings, so it is not yet a reference solution. A
# reference solution is defined by the training-token count it reaches the target
# on, and that number does not exist until the scaled operating point is measured.
#
# Exiting non-zero is deliberate. A stub that exited zero would let the Phase 2
# oracle check report success against a solution that does not exist, which is the
# shape of the defect that voided the prior attempt.
set -euo pipefail
echo "reference solution not authored: oracle.py carries baseline settings" >&2
echo "blocked by gap operating-point-unmeasured" >&2
exit 1

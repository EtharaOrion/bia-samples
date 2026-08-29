#!/usr/bin/env bash
# Private oracle entry point. Harbor mounts solution/ only for the oracle path.
#
# The reference change is batch-size warmup in oracle.py. It was measured at toy
# scale and beats the baseline there: 40000 training tokens against 60000 to
# reach the same target, consistently across three seeds. See
# seed/feasibility.yaml for the numbers and for what that measurement does and
# does not support.
#
# It has NOT been run at the bound H100 operating point, so tests/anchors.json
# still carries measured false and grading still refuses. This script runs the
# oracle; it does not assert that the oracle is calibrated.
set -euo pipefail
exec python3 /app/oracle.py

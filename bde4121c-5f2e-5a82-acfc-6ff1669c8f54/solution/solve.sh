#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Private oracle. Harbor mounts solution/ only for the oracle path.
#
# The reference reads its seed from BIA_SEED and its horizon from the
# provided loader, exactly as a submission does, so this script takes no
# arguments. The voided oracle for this slot took --seed while the grader
# invoked --steps, so it could not run under its own contract at all.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 recipe.py


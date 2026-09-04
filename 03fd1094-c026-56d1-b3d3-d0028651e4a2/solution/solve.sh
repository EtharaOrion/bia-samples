# FORGE-CANARY-BEGIN
# 0: 44a971ef25ec8976e177157a7365f2ce0f936084b2b7dd32960f5b0d97cdb703
# 1: bf22aa1665ad541ec5238aae6fb9bd70952978b1a051db268e4abde972c2b02c
# 2: 3a356bae0a0fb068332d942a738dce96d5fddea25a106f29342cc117e817b818
# 3: 18187b1f33d4fdb69af3286c5f64c93b0bc5735fb81db22358baceb8913b5ad4
# FORGE-CANARY-END
#!/usr/bin/env bash
#
# The reference solution entry point. It does the thing the slot is about: it RE-PROBES
# the calibration handle before submitting rather than reusing the answer it fitted
# earlier in the session, and it records the version it fitted against together with two
# witnesses over bytes it actually touched, the pin and the slice content.
#
# It also does the thing the re-base is about: it loads the real nanoGPT checkpoint and
# measures per-tensor sensitivity with real forward passes over the calibration slice in
# force. There is no shipped sensitivity vector to read and no closed-form degradation to
# evaluate, so this script has no path to an allocation that does not run the model.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SOLUTION_DIR}/.." && pwd)"
WORKSPACE="${OER22_WORKSPACE:-${BUNDLE_DIR}}"
OUT="${OER22_SUBMISSION:-${WORKSPACE}/submission.json}"
CHECKPOINT="${OER22_CHECKPOINT:-/workspace/checkpoint/nanogpt.pt}"
DEVICE="${OER22_DEVICE:-cuda}"
CALIBRATION_ROWS="${OER22_CALIBRATION_ROWS:-8}"

# First probe, recorded. The reference agent does not stop here.
python3 "${BUNDLE_DIR}/environment/calibration_probe.py" --workspace "${WORKSPACE}" >/dev/null

OER22_BUNDLE="${BUNDLE_DIR}" \
OER22_WORKSPACE="${WORKSPACE}" \
OER22_OUT="${OUT}" \
OER22_CHECKPOINT="${CHECKPOINT}" \
OER22_DEVICE="${DEVICE}" \
OER22_CALIBRATION_ROWS="${CALIBRATION_ROWS}" \
python3 - <<'PYTHON'
import json, os, sys
from pathlib import Path

bundle = Path(os.environ["OER22_BUNDLE"])
workspace = Path(os.environ["OER22_WORKSPACE"])
sys.path.insert(0, str(bundle / "solution"))
import reference

# The second probe happens inside reference.solve, which re-reads the calibration state
# at fitting time instead of carrying the first answer forward as a timeless fact.
payload = reference.solve(
    workspace,
    Path(os.environ["OER22_CHECKPOINT"]),
    os.environ["OER22_DEVICE"],
    int(os.environ["OER22_CALIBRATION_ROWS"]),
)
with open(os.environ["OER22_OUT"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYTHON

printf 'wrote %s\n' "${OUT}"

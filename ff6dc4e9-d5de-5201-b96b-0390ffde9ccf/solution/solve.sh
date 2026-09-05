# FORGE-CANARY-BEGIN
# 0: a569f2e00d8b29069d14ae3bdc1f4c94140b15a5484768bd76e82d92d783cabe
# 1: 1d43da04584319d245863b607e712579da8435042b59d4f47a0284fe8f7d15da
# 2: 863763f794fe9909196d5e780bb7ecc80ca501e4c0e11bcf3d5fcff0bdbaeb8a
# 3: 8caa5edc8422658736aba9a6a964adae4fc833d46f3c5394e995f1701d709180
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

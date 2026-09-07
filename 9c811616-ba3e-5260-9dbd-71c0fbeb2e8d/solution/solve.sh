# FORGE-CANARY-BEGIN
# 0: 692a6f68f7fca6c78bdc7b2085719dd8f83aac7d666e6fa6fcdce0dd8baebc53
# 1: e64f5db32e7336b7b3699c0a3414397ac46709e0b349674a50a57036bd67b6ec
# 2: 771d18b1b0794c28925f346137e96bf254b41ecc1dcaeee8a08157be97c3b28e
# 3: 8ca2a809cf960f4c5aa0117cbaa02f12e98813f79881b4e1be065b930361b41d
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. source: solution/grounding.yaml
#
# The reference solution entry point. It does the thing the slot is about: it RE-PROBES
# the calibration handle before submitting rather than reusing the answer it fitted
# earlier in the session, and it records the version it fitted against together with two
# witnesses over bytes it actually touched, the pin and the slice content.
#
# It also does the thing the re-base is about: it loads the real nanoGPT checkpoint and
# measures the per-tensor cost curve with real forward passes over the calibration slice
# in force. There is no shipped sensitivity vector to read and no closed-form degradation
# to evaluate, so this script has no path to an allocation that does not run the model.
set -euo pipefail

# tests/runner.py launches a submission by COPYING its entry point alone into a
# fresh temporary directory, so ${BASH_SOURCE[0]} does not sit in the bundle when
# this runs under the verifier and a bundle path derived from it resolves into the
# temporary directory. The workspace the runner hands over is authoritative, and the
# script's own location is used only when nothing handed one over.
SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${OER22_WORKSPACE:-$(cd "${SOLUTION_DIR}/.." && pwd)}"
BUNDLE_DIR="${WORKSPACE}"
if [ ! -f "${BUNDLE_DIR}/environment/substrate.json" ]; then
  BUNDLE_DIR="$(cd "${SOLUTION_DIR}/.." && pwd)"
fi
if [ ! -f "${BUNDLE_DIR}/environment/substrate.json" ]; then
  echo "cannot resolve the bundle: no environment/substrate.json under ${BUNDLE_DIR}" >&2
  exit 1
fi
OUT="${OER22_SUBMISSION:-${WORKSPACE}/submission.json}"
DEVICE="${OER22_DEVICE:-cuda}"
CALIBRATION_ROWS="${OER22_CALIBRATION_ROWS:-8}"

# The checkpoint. environment/substrate.json checkpoint.path is the agent surface's own
# path and it is read from that file rather than restated here. OER22_CHECKPOINT
# overrides it, which is how the verifier points this same entry point at the pristine
# copy tests/bound.json pins without either surface loading the other's file.
CHECKPOINT="${OER22_CHECKPOINT:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint']['path'])" "${BUNDLE_DIR}/environment/substrate.json")}"

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
if len(payload["allocation"]) != 48:
    raise SystemExit("the fit did not cover every quantizable tensor")
with open(os.environ["OER22_OUT"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYTHON

printf 'wrote %s\n' "${OUT}"

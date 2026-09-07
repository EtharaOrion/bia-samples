# FORGE-CANARY-BEGIN
# 0: 8e1fba714979e51f68b593f770fa2a29f7efa5784c7ecd0c4fc0979a0e980581
# 1: ea0e59acc12984f5c1190add523d8d38d53b46a6ff006d7d76d3c3e4bf540234
# 2: b0782cca1dcd444c80d9c01e32aae4b5f94dab9701690c56b2ad808d59ae9977
# 3: 526428919840aa92d4fa7bfc5a20ef78058bb60188c8872c7ec6b789bb7e412b
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
SOLUTION_PARENT="$(cd "${SOLUTION_DIR}/.." && pwd)"
BUNDLE_DIR="${OER22_WORKSPACE:-${SOLUTION_PARENT}}"
if [ ! -f "${BUNDLE_DIR}/environment/substrate.json" ]; then
  BUNDLE_DIR="${SOLUTION_PARENT}"
fi
if [ ! -f "${BUNDLE_DIR}/environment/substrate.json" ]; then
  echo "cannot resolve the bundle: no environment/substrate.json under ${BUNDLE_DIR}" >&2
  exit 1
fi

# THE SURFACE THE SUBMISSION LANDS ON, read from this bundle's own declaration rather
# than assumed to be whatever directory this file happens to sit under.
# environment/substrate.json checkpoint.path names <surface>/checkpoint/nanogpt.pt and
# environment/Dockerfile WORKDIR names that same directory, so the surface root is that
# path's grandparent and it is read back here instead of being restated.
#
# WHY THE PARENT OF solution/ IS NOT IT. A harness that exercises this entry point mounts
# the bundle's solution/ and tests/ trees BESIDE the agent surface rather than inside it,
# so this file's own parent is then a container-local directory that nothing on the host
# is bound to and that the container discards when it exits. A submission written there
# reaches no reader: tests/grade.py opens <workspace>/submission.json and tests/runner.py
# fixes that basename, and both resolve <workspace> on the surface. So the surface wins
# whenever it carries this bundle's own environment/ payload, and the parent of solution/
# is used only when it does not. An explicit OER22_WORKSPACE outranks both, which is how
# tests/runner.py hands this entry point the workspace it is being graded in.
SURFACE="$(python3 -c "import json,os,sys;print(os.path.dirname(os.path.dirname(json.load(open(sys.argv[1]))['checkpoint']['path'])))" "${BUNDLE_DIR}/environment/substrate.json")"
WORKSPACE="${OER22_WORKSPACE:-${SOLUTION_PARENT}}"
if [ -z "${OER22_WORKSPACE:-}" ] && [ -f "${SURFACE}/environment/substrate.json" ]; then
  WORKSPACE="${SURFACE}"
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

# The fit writes to a staging path and the install below is its own step, so that the
# destination this oracle puts its submission at is NAMED IN THIS FILE, in shell. A write
# performed only by a Python open() inside a heredoc states no destination a reader of
# these frozen bytes can see, and a reader that has to discover where this oracle
# installed its artifact is then left with whichever single directory the phase wrote
# into. That answers nothing for this slot: environment/provision.sh materialises the
# checkpoint under <workspace>/checkpoint/ in the same phase, so the phase writes into
# two directories and neither one alone is the submission's home.
STAGED="$(mktemp "${TMPDIR:-/tmp}/oer22-submission-XXXXXX.json")"
trap 'rm -f "${STAGED}"' EXIT

OER22_BUNDLE="${BUNDLE_DIR}" \
OER22_WORKSPACE="${WORKSPACE}" \
OER22_OUT="${STAGED}" \
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

# The install. Both branches name the same path ${OUT} already resolves to, and they are
# written out rather than folded into one `cp "${STAGED}" "${OUT}"` because the folded
# form names its destination only through an expansion. The default branch states the
# basename tests/runner.py fixes and tests/grade.py opens, which is the verifier's own
# half of this join, so the two sides of it are readable in the frozen bytes of both.
mkdir -p "$(dirname "${OUT}")"
if [ -n "${OER22_SUBMISSION:-}" ]; then
  cp "${STAGED}" "${OER22_SUBMISSION}"
else
  cp "${STAGED}" "${WORKSPACE}/submission.json"
fi

printf 'wrote %s\n' "${OUT}"

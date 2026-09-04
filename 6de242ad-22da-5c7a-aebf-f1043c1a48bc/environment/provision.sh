#!/usr/bin/env bash
# Build-time provisioning for slot OER-21.
#
# It stages exactly two artifacts into the agent container: the frozen nanoGPT parameter
# snapshot the task quantizes, and the agent-visible calibration slice of the FineWeb train
# split. Both are named in the manifests this script reads rather than in this script, so
# the pin lives in one place.
#
# It stages NOTHING from the validation split. The slices the grade is computed on are
# named only in tests/anchors.json, are mounted only into the verifier, and no line below
# can reach them.
#
# The checkpoint digest is not yet pinned. gap-oer21-checkpoint-digest-unmeasured records
# why: the authoring lane did not build the snapshot and did not write a sha256 it had not
# taken. When the pin is bound in environment/model/checkpoint.json this script verifies it
# and refuses on mismatch, which is what the `sha256_state` branch below already does.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${HERE}/model/checkpoint.json"
CORPUS="${HERE}/corpus/eval_corpus.json"

read_field() {
  python3 -c "import json,sys;print(json.load(open(sys.argv[1]))${1})" "${2}"
}

CHECKPOINT_PATH="$(read_field "['checkpoint']['container_path']" "${MANIFEST}")"
CHECKPOINT_STATE="$(read_field "['checkpoint']['sha256_state']" "${MANIFEST}")"
CALIBRATION_PATH="$(read_field "['calibration']['container_path']" "${CORPUS}")"
CALIBRATION_SHARD="$(read_field "['calibration']['shard']" "${CORPUS}")"

mkdir -p "$(dirname "${CHECKPOINT_PATH}")" "$(dirname "${CALIBRATION_PATH}")"

if [ -z "${OER21_CHECKPOINT_URL:-}" ]; then
  echo "OER21_CHECKPOINT_URL is unset, so the frozen snapshot cannot be staged." >&2
  echo "The build argument names the provisioning source recorded in ${MANIFEST}." >&2
  exit 1
fi
curl -fsSL "${OER21_CHECKPOINT_URL}" -o "${CHECKPOINT_PATH}"

if [ "${CHECKPOINT_STATE}" = "pinned" ]; then
  EXPECTED="$(read_field "['checkpoint']['sha256']" "${MANIFEST}")"
  echo "${EXPECTED}  ${CHECKPOINT_PATH}" | sha256sum -c -
else
  echo "checkpoint digest pin is ${CHECKPOINT_STATE}, under gap-oer21-checkpoint-digest-unmeasured;"
  echo "staging without a digest check and recording the observed digest instead:"
  sha256sum "${CHECKPOINT_PATH}"
fi

if [ -z "${OER21_CALIBRATION_URL:-}" ]; then
  echo "OER21_CALIBRATION_URL is unset, so the calibration slice cannot be staged." >&2
  echo "It names ${CALIBRATION_SHARD} of the pinned FineWeb train split." >&2
  exit 1
fi
curl -fsSL "${OER21_CALIBRATION_URL}" -o "${CALIBRATION_PATH}"

python3 - "${CHECKPOINT_PATH}" "${HERE}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

checkpoint, here = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("oer21_nanogpt", here / "model" / "nanogpt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
declaration = json.loads((here / "nanogpt_substrate.json").read_text(encoding="utf-8"))
state = module.load_checkpoint(checkpoint, declaration)
print("staged snapshot conforms:", json.dumps(module.assert_checkpoint_matches(state, declaration), sort_keys=True))
PY

echo "provisioning complete"

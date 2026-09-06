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
# WHERE THE BYTES COME FROM. Both artifacts are resolved from the build context this
# bundle carries, at the container paths the manifests name, because the pinned base image
# pytorch/pytorch@sha256:db80a41f carries no curl and the build is invoked with no build
# argument, so a fetch is not a provisioning path this image can execute. The calibration
# slice is carried in the bundle at environment/corpus/calibration.bin and the Dockerfile
# COPY places it. A provisioning pass that hands back a built parameter snapshot places it
# at environment/model/checkpoint.pt the same way, and this script uses it when it is
# there.
#
# The checkpoint digest is not yet pinned. gap-oer21-checkpoint-digest-unmeasured records
# why: the authoring lane did not build the snapshot and did not write a sha256 it had not
# taken. When the pin is bound in environment/model/checkpoint.json this script verifies it
# and refuses on mismatch, which is what the `sha256_state` branch below already does.
#
# When no provisioned snapshot is carried this script constructs the declared decoder from
# environment/model/nanogpt.py under a pinned seed and says so, loudly, on stdout and in
# the build-state record it writes beside the snapshot. That snapshot is UNTRAINED. It is
# the smoke snapshot gap-oer21-endpoints-unmeasured-on-the-nanogpt-substrate already names,
# on which the verifier's widening probe reads no separation from the handed option set and
# the grade is a reported scaling-span-nonpositive rather than an invented number. Nothing
# here defaults a missing value: every field below is a strict subscript and an absent one
# stops the build.
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
CALIBRATION_TOKENS="$(read_field "['calibration']['token_count']" "${CORPUS}")"
CALIBRATION_SEQUENCE="$(read_field "['calibration']['sequence_length']" "${CORPUS}")"

mkdir -p "$(dirname "${CHECKPOINT_PATH}")" "$(dirname "${CALIBRATION_PATH}")"

# ---------------------------------------------------------------------------
# The agent-visible calibration slice.
# ---------------------------------------------------------------------------
if [ ! -f "${CALIBRATION_PATH}" ]; then
  echo "the calibration slice is not staged at ${CALIBRATION_PATH}." >&2
  echo "It names ${CALIBRATION_SHARD} of the pinned FineWeb train split and the bundle carries" >&2
  echo "it at environment/corpus/$(basename "${CALIBRATION_PATH}")." >&2
  exit 1
fi
python3 - "${CALIBRATION_PATH}" "${CALIBRATION_SHARD}" "${CALIBRATION_TOKENS}" "${CALIBRATION_SEQUENCE}" <<'PY'
import hashlib
import struct
import sys

path, shard, wanted, sequence = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(path, "rb") as handle:
    header = struct.unpack("<256i", handle.read(1024))
if header[0] != 20240520:
    raise SystemExit("the calibration slice at " + path + " is not a modded-nanogpt token shard")
if header[1] != 1:
    raise SystemExit("unsupported token shard version in " + path)
count = header[2]
if count < wanted + 1:
    raise SystemExit(
        "the calibration slice carries " + str(count) + " tokens and "
        + str(wanted) + " plus one shifted target are needed"
    )
if wanted % sequence:
    raise SystemExit("the calibration token count is not a whole number of sequences")
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
print(
    "calibration slice staged: " + shard + ", " + str(count) + " tokens carried, "
    + str(wanted) + " read, sha256 " + digest.hexdigest()
)
PY

# ---------------------------------------------------------------------------
# The frozen parameter snapshot.
# ---------------------------------------------------------------------------
CHECKPOINT_SOURCE="provisioned"
if [ ! -f "${CHECKPOINT_PATH}" ]; then
  CHECKPOINT_SOURCE="declared-initialisation-under-a-pinned-seed"
  echo "no provisioned parameter snapshot is carried at ${CHECKPOINT_PATH}."
  echo "Constructing the decoder environment/model/nanogpt.py declares, under the pinned seed,"
  echo "and recording it as UNTRAINED under gap-oer21-endpoints-unmeasured-on-the-nanogpt-substrate."
  python3 - "${CHECKPOINT_PATH}" "${HERE}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

import torch

target, here = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("oer21_nanogpt", here / "model" / "nanogpt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
declaration = json.loads((here / "nanogpt_substrate.json").read_text(encoding="utf-8"))

# The seed is the one the sibling F9 slot's own build-time snapshot pass pins. The
# initialisation itself is not chosen here: it is whatever environment/model/nanogpt.py
# constructs, so this file adds a seed and nothing else.
SEED = 1337
torch.manual_seed(SEED)
model = module.build_model(declaration)
state = {name: value.detach().to(torch.bfloat16) for name, value in model.state_dict().items()}
module.assert_checkpoint_matches(state, declaration)
torch.save(state, str(target))
print("constructed snapshot under seed " + str(SEED) + " at " + str(target))
PY
fi

if [ "${CHECKPOINT_STATE}" = "pinned" ]; then
  EXPECTED="$(read_field "['checkpoint']['sha256']" "${MANIFEST}")"
  echo "${EXPECTED}  ${CHECKPOINT_PATH}" | sha256sum -c -
else
  echo "checkpoint digest pin is ${CHECKPOINT_STATE}, under gap-oer21-checkpoint-digest-unmeasured;"
  echo "staging without a digest check and recording the observed digest instead:"
  sha256sum "${CHECKPOINT_PATH}"
fi

python3 - "${CHECKPOINT_PATH}" "${HERE}" "${CHECKPOINT_SOURCE}" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

checkpoint, here, source = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
spec = importlib.util.spec_from_file_location("oer21_nanogpt", here / "model" / "nanogpt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
declaration = json.loads((here / "nanogpt_substrate.json").read_text(encoding="utf-8"))
state = module.load_checkpoint(checkpoint, declaration)
conformance = module.assert_checkpoint_matches(state, declaration)
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
record = {
    "_banner": "BUILD STATE. Written by environment/provision.sh at image build. Not a bundle byte.",
    "conformance": conformance,
    "observed_sha256": digest.hexdigest(),
    "source": source,
    "trained": source == "provisioned",
    "trained_note": (
        "A snapshot the provisioning pass handed back carries whatever training that pass did. A "
        "snapshot constructed here under the pinned seed is UNTRAINED, has no per-role sensitivity "
        "structure for a bit allocation to exploit, and makes the verifier's separation zero. That "
        "outcome is reported by the verifier as scaling-span-nonpositive and is never an invented gap."
    ),
}
(checkpoint.parent / "checkpoint.provenance.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("staged snapshot conforms:", json.dumps(conformance, sort_keys=True))
PY

echo "provisioning complete"

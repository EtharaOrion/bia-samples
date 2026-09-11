# FORGE-CANARY-BEGIN
# 0: c42f868a4caa725327eef5b9da4c6058a8ea76c6859df770677e043b12dcd10f
# 1: c46f68ca1580d05acd914835c0448074b185a5d33bb2a46f17be5dd8972de037
# 2: 7ad608e028a26cb098e307d146ec5ae45431e29754fa68a450afd52f3863a5a1
# 3: 7d738db3db56e674cb20be35ae02e087e300600bc0e81eab121466b04b6aea0a
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. source: solution/grounding.yaml
#
# The reference solution entry point. It does the thing the slot is about: it
# RE-PROBES the calibration handle before submitting rather than reusing the
# answer it fitted earlier in the session, and it records the version it fitted
# against together with a witness digest over the bytes it actually used.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SOLUTION_DIR}/.." && pwd)"
WORKSPACE="${OER22_WORKSPACE:-${BUNDLE_DIR}}"
OUT="${OER22_SUBMISSION:-${WORKSPACE}/submission.json}"

# First probe, recorded. The reference agent does not stop here.
python3 "${BUNDLE_DIR}/environment/calibration_probe.py" --workspace "${WORKSPACE}" >/dev/null

OER22_BUNDLE="${BUNDLE_DIR}" OER22_WORKSPACE="${WORKSPACE}" OER22_OUT="${OUT}" python3 - <<'PYTHON'
import json, os, subprocess, sys

bundle = os.environ["OER22_BUNDLE"]
workspace = os.environ["OER22_WORKSPACE"]
out = os.environ["OER22_OUT"]
sys.path.insert(0, os.path.join(bundle, "solution"))
import reference

def read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

substrate = read(os.path.join(workspace, "environment", "substrate.json"))
stats = read(os.path.join(workspace, "environment", "calibration_stats.json"))

# The second probe. The calibration set moved and nothing announced it, so the
# version observed at fitting time is re-read here instead of assumed timeless.
probe = json.loads(
    subprocess.run(
        [
            sys.executable,
            os.path.join(bundle, "environment", "calibration_probe.py"),
            "--workspace",
            workspace,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
)

in_force = int(probe["calibration_version_in_force"])
ledger = [int(row["version"]) for row in probe["ledger"] if int(row["version"]) <= in_force]
record = next(row for row in stats["versions"] if int(row["version"]) == in_force)
vector = [float(value) for value in record["sensitivity"]]
tensors = substrate["tensors"]

# The surrogate scale constant is established only in built environment state and
# is on no agent-visible byte, so it is read back out of the substrate here and
# replayed over the uniform reference allocation against the in-force vector.
constant = float(substrate["degradation_constant_K"])
uniform = {
    row["id"]: int(substrate["uniform_reference_bits"]) for row in tensors
}
probe_value = reference.degradation(tensors, uniform, vector, constant)
scale_witness = reference.digest(["oer22.surrogate-scale-probe/v1", probe_value])

allocation = reference.allocate(
    tensors,
    vector,
    int(substrate["budget_bits"]),
    int(substrate["group_size"]),
    int(substrate["scale_bits"]),
    min(substrate["bit_choices"]),
    max(substrate["bit_choices"]),
    bool(substrate["budget_accounting_includes_scale_tensors"]),
)
payload = {
    "schema": "oer22.submission/v1",
    "allocation": {key: int(value) for key, value in sorted(allocation.items())},
    "quantization_scheme": {
        "group_size": int(substrate["group_size"]),
        "scale_bits": int(substrate["scale_bits"]),
    },
    "derived_against_calibration_version": in_force,
    "calibration_fit_witness": reference.sensitivity_digest(vector),
    "surrogate_scale_witness": scale_witness,
    "observed_calibration_ledger": ledger,
    "protocol": {
        "points_completed": 4,
        "halted_early": False,
    },
    "readout": {"filter": "none", "reported_degradation": None},
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYTHON

printf 'wrote %s\n' "${OUT}"

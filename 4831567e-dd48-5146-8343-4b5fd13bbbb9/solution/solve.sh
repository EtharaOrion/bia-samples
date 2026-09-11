# FORGE-CANARY-BEGIN
# 0: 91f43562ebe3f0c2695b58165478ba71c9f60e18640bb76b45bcee4447c505e8
# 1: 9317624b36c996dcaf3d25eb5f50643cd96fb898b05925de70b960a358d7d7ae
# 2: 6f40608b163b0f0dbd8e030797ee0392459f43f60d72ab1e537a484e627720dd
# 3: 2336b43ba22ccce1dee883585bd05b4b90ef72386deb53a9e5e7898aa5dda422
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

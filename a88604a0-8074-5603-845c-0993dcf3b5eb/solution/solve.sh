# FORGE-CANARY-BEGIN
# 0: 56381d6baced28751d982e6619202d9e55af25b09fd5531cd00ea441732ba322
# 1: 31376c8ae1d14d284e032489bfbcce7bff4e10692195061e2405af804267bee1
# 2: a3308bef413e1665f5cdef8d97a8cda37528b8d4976b7719e09ab7ea13de3381
# 3: e4a5377542affa0c75e81c31db6774aec6e0ee0313b966313c0da0e1c03e4fd2
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
# tests/runner.py stages this script ALONE into a fresh temporary directory, so a
# BASH_SOURCE-relative bundle root resolves to that holder and every bundle-relative
# read below misses. The runner exports OER22_WORKSPACE as the bundle root, so it is
# preferred when set and the relative derivation remains the fallback for a direct run.
BUNDLE_DIR="${OER22_WORKSPACE:-$(cd "${SOLUTION_DIR}/.." && pwd)}"
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

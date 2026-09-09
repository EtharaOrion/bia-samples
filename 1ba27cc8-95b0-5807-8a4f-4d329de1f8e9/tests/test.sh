#!/usr/bin/env bash
set -euo pipefail

export BIA_AUTOCAST="${BIA_AUTOCAST:-bf16}"

export BIA_MAX_ATTEMPTS="${BIA_MAX_ATTEMPTS:-1}"
cd "$(dirname "$0")"

export BIA_SHAPE="${BIA_SHAPE:-$(pwd)/shape.json}"

SCORE_DIR="${BIA_SCORE_DIR:-/logs/verifier}"

emit_floor() {
    local status=$?
    if [ ! -f "${SCORE_DIR}/score.md" ]; then
        mkdir -p "${SCORE_DIR}"
        printf '0.000000\n' > "${SCORE_DIR}/score.md"
        printf '{"reason_code": 3, "score": 0.0}\n' > "${SCORE_DIR}/score_numeric.json"
        printf '{"checkers": [], "metric": {"graded_step": null}, "metrics": {"n_seeds": 0, "seeds_required": 1}, "reason": "verifier-aborted-before-grading", "reason_code": 3, "score": 0.0}\n' > "${SCORE_DIR}/score_full.json"
        printf '{"checkers": [], "metric": {"graded_step": null}, "metrics": {"n_seeds": 0, "seeds_required": 1}, "reason": "verifier-aborted-before-grading", "reason_code": 3, "score": 0.0}\n' > "${SCORE_DIR}/score.json"
    fi
    exit "${status}"
}
trap emit_floor EXIT

reap_orphaned_agent_processes() {
    python3 - <<'PY'
import os, signal, time

def parent_of(pid):
    try:
        with open(f"/proc/{pid}/stat") as handle:
            blob = handle.read()
        return int(blob[blob.rindex(")") + 2:].split()[1])
    except (OSError, ValueError, IndexError):
        return None

own = set()
walk = os.getpid()
while walk and walk not in own:
    own.add(walk)
    walk = parent_of(walk)

def sweep():
    reaped = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in own or pid == 1:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                command = handle.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if "python" not in command:
            continue
        print(f"reaping orphaned agent process {pid}: {command[:140]}", flush=True)
        try:
            os.kill(pid, signal.SIGKILL)
            reaped += 1
        except OSError:
            pass
    return reaped

total = 0
for _ in range(6):
    found = sweep()
    total += found
    if not found:
        break
    time.sleep(2)
print(f"reaped {total} orphaned agent process(es)", flush=True)
PY
}
reap_orphaned_agent_processes || true

python3 - <<'PY' || true
import subprocess, time

def used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None

for _ in range(30):
    value = used_mib()
    if value is None or value < 2048:
        break
    time.sleep(2)
value = used_mib()
if value is not None and value >= 2048:
    print(f"WARNING: {value} MiB still resident after reaping; the graded run may OOM.", flush=True)
PY
python3 - <<'PY' || true
import subprocess
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=30).stdout.strip()
    print(f"GPU state entering the graded run: {out}", flush=True)
except Exception as exc:
    print(f"could not read GPU state: {exc}", flush=True)
PY

python3 test_output.py || true

python3 grade.py

python3 apply_rubric_gate.py || true

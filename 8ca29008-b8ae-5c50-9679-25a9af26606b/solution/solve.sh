#!/usr/bin/env bash
# PRIVATE reference solution for S08.
#
# Default path: the real reference. It installs solution/reference_recovery.py as the
# submission, runs the harness runner at the graded scaled profile on the accelerator
# the environment provides, derives submission/report.json independently from the
# telemetry, and hands the run to tests/test.sh for scoring.
#
# BIA_SMOKE=1: the identical code path at the smoke profile, CPU only, with CUDA
# switched off at the environment level. It generates a smoke-sized poisoned
# checkpoint with the same generator, loads it through the same runner, calls the
# same reference recover, and grades with the same grade.py. It is not a stub and it
# is not a separate implementation. It exists so this reference can be proven to
# execute end to end without an accelerator.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "$HERE")"
SMOKE="${BIA_SMOKE:-0}"

if [ "$SMOKE" = "1" ]; then
  PROFILE="smoke"
  WORK="$(mktemp -d -t bia-s08-smoke-XXXXXX)"
  STATE="$WORK/state"
  FIX="$STATE/fixtures"
  TEL="$WORK/telemetry"
  SUB="$WORK/submission"
  LOGS="$WORK/logs/verifier"
  export CUDA_VISIBLE_DEVICES=""
  DEVICE="cpu"
  GEN_THREADS="${S08_GEN_THREADS:-4}"
else
  PROFILE="${S08_PROFILE:-scaled}"
  STATE="${S08_STATE_DIR:-/opt/bia/s08}"
  FIX="${S08_FIXTURE_DIR:-$STATE/fixtures}"
  TEL="${S08_TELEMETRY_DIR:-/telemetry}"
  SUB="${S08_SUBMISSION:-/workspace/submission}"
  LOGS="${LOG_DIR:-/logs/verifier}"
  DEVICE="${S08_DEVICE:-}"
  GEN_THREADS="${S08_GEN_THREADS:-8}"
fi

RUNNER="$STATE/runner"
[ -f "$RUNNER/run_s08.py" ] || RUNNER="$BUNDLE/environment/runner"
# The smoke profile always builds its own fixture into its own scratch tree. It never
# falls back to the shipped fixtures directory, because writing there would mutate the
# frozen bytes the content hash covers.
if [ "$SMOKE" != "1" ] && [ ! -f "$FIX/manifest_$PROFILE.json" ]; then
  FIX="$BUNDLE/environment/fixtures"
fi

mkdir -p "$STATE" "$TEL" "$SUB" "$LOGS" "$FIX" 2>/dev/null || true
DATA="$STATE/data/$PROFILE"

echo "== bia S08 reference =="
echo "profile      $PROFILE"
echo "smoke        $SMOKE"
echo "runner       $RUNNER"
echo "fixtures     $FIX"
echo "telemetry    $TEL"
echo "submission   $SUB"

# 1. The smoke profile regenerates its poisoned checkpoint with the same generator
#    that produced the shipped scaled fixture. The scaled fixture ships and is not
#    regenerated here, because regenerating it is a reproduction check that
#    solution/recompute.py owns.
if [ "$SMOKE" = "1" ]; then
  echo "-- generating the smoke-scale poisoned checkpoint with the shipped generator"
  python3 "$BUNDLE/solution/make_fixture.py" \
    --profile "$PROFILE" --out "$FIX" --data-dir "$DATA" --threads "$GEN_THREADS" || exit 1
fi

if [ ! -f "$FIX/manifest_$PROFILE.json" ]; then
  echo "FAULT: no fixture manifest for profile $PROFILE under $FIX" >&2
  exit 1
fi

# 2. Install the reference recovery as the submission.
cp "$BUNDLE/solution/reference_recovery.py" "$SUB/recover.py" || exit 1

# 3. Run the harness runner. It is the only writer of the telemetry record.
RUN_ARGS=(--profile "$PROFILE" --submission "$SUB" --out "$TEL"
          --fixture "$FIX/ckpt_s08_$PROFILE.pt" --data-dir "$DATA")
if [ -n "$DEVICE" ]; then RUN_ARGS+=(--device "$DEVICE"); fi
echo "-- running the harness"
python3 "$RUNNER/run_s08.py" "${RUN_ARGS[@]}" || exit 1

# 4. Derive report.json from the telemetry. This derivation is deliberately written
#    here rather than imported from the verifier, so the DIVERGENCE checker compares
#    two independent computations of the same quantity instead of one computation
#    against itself.
echo "-- deriving submission/report.json"
python3 - "$TEL/run_record.jsonl" "$FIX/manifest_$PROFILE.json" "$SUB/report.json" "$SUB/recover.py" <<'PY'
import importlib.util, json, sys

tel, manifest_path, out, recover_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
manifest = json.load(open(manifest_path))

# The declared hyperparameters are read from the recovery that actually ran, so the
# declaration cannot drift from the optimizer the harness fingerprinted.
spec = importlib.util.spec_from_file_location("bia_s08_reference", recover_path)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
target = float(manifest["target_loss"])

by_seed = {}
for line in open(tel):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("phase") != "submission" or r.get("kind") != "eval":
        continue
    by_seed.setdefault(int(r["seed"]), {})[int(r["step"])] = float(r["val_loss"])

steps = None
if by_seed:
    shared = sorted(set.intersection(*[set(v) for v in by_seed.values()]))
    ok = []
    for s in shared:
        vals = [by_seed[k][s] for k in by_seed]
        ok.append(max(vals) <= target + 1e-9 and sum(vals) / len(vals) <= target + 1e-9)
    for i, s in enumerate(shared):
        if ok[i] and all(ok[i:]):
            steps = s
            break

report = {
    "steps_to_target": steps if steps is not None else -1,
    "target_loss": target,
    "seeds": sorted(by_seed),
    "optimizer": {
        "lr": ref.BASE_LR,
        "betas": list(ref.BETAS),
        "eps": ref.EPS,
        "weight_decay": ref.WEIGHT_DECAY,
    },
    "recovery_summary": (
        "Rebuilt AdamW with hyperparameters owned by the recovery rather than inherited "
        "from the checkpoint param_groups. Measured the live per-tensor gradient second "
        "moment with a bounded number of probe passes, rescaled the checkpoint second "
        "moment per tensor so its mean matches what the gradients actually are, clipped "
        "the first moment against the repaired preconditioner, wrote a truthful step "
        "counter taken from the manifest checkpoint step, and supplied a warmup with a "
        "late cosine tail keyed to the recovery step index."
    ),
}
json.dump(report, open(out, "w"), indent=1, sort_keys=True)
print(json.dumps({"steps_to_target": report["steps_to_target"], "target_loss": target}))
PY

# 5. Grade with the shipped verifier.
echo "-- grading"
# The verifier now executes its own graded attempt rather than accepting the record the
# agent left behind, so it needs the same runner, fixture, corpus and device bindings the
# agent run used. Everything below is a location, never a value: no number reaches the
# verifier through this block.
export S08_BUNDLE="$BUNDLE"
export S08_TELEMETRY_DIR="$TEL"
export S08_SUBMISSION="$SUB"
export S08_FIXTURE_DIR="$FIX"
export S08_PROFILE="$PROFILE"
export S08_RUNNER_DIR="$RUNNER"
export S08_DATA_DIR="$DATA"
export S08_STATE_DIR="$STATE"
export S08_DEVICE="$DEVICE"
export LOG_DIR="$LOGS"
export SCORE_PATH="$LOGS/score.json"
export S08_OUTCOMES="$LOGS/outcomes.json"
bash "$BUNDLE/tests/test.sh"
RC=$?

echo "== reference finished, verifier exit $RC =="
exit "$RC"

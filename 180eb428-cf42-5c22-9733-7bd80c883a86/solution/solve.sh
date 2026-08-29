#!/usr/bin/env bash
# Reference solution for bia slot S07, mixed-precision-stability.
#
# Default: runs the real reference path at the frozen operating point on one H100.
#   ./solution/solve.sh
#
# Smoke: runs the IDENTICAL code path at the smoke operating point, CPU only, no CUDA.
#   BIA_SMOKE=1 ./solution/solve.sh
#
# Adequacy suite: runs every route in solution/negative_controls.py through the same path.
#   BIA_SMOKE=1 ./solution/solve.sh --controls
#
# The smoke path is not a stub. It calls the same Quantizer, the same reduce_in_format, the
# same overflow ledger, the same train_one and the same grader, and it holds the reduction fan
# in at 16 so the accumulation order argument is exercised rather than skipped. It asserts that
# the shipped standard recipe raises the overflow flag, so a smoke run that passed while the
# gate was dead would fail instead of reporting success.

set -uo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${BIA_WORK:-${TMPDIR:-/tmp}/bia_s07_reference}"
ARTIFACTS="$WORK/artifacts"
LOGS="$WORK/logs/verifier"

export BIA_BUNDLE="$BUNDLE"
export BIA_ENVDIR="$BUNDLE/environment"
export BIA_ARTIFACTS="$ARTIFACTS"
export SCORE_PATH="$LOGS/score.json"
export BIA_OUTCOMES="$LOGS/outcomes.json"
export BIA_REWARD_PATH="$LOGS/reward.txt"

mkdir -p "$ARTIFACTS" "$LOGS"

MODE="full"
if [ "${BIA_SMOKE:-0}" = "1" ]; then
  MODE="smoke"
  export CUDA_VISIBLE_DEVICES=""
fi

echo "=== bia S07 reference solve, mode=$MODE"
echo "    bundle    $BUNDLE"
echo "    workdir   $WORK"
python3 - <<'PY'
import json, os, pathlib
env = pathlib.Path(os.environ["BIA_ENVDIR"])
fmt = json.loads((env / "format.json").read_text())
print("    format    %s  max_normal=%s  min_subnormal=%s  overflow_onset=%s  range=%s decades"
      % (fmt["format_id"], fmt["max_normal"], fmt["min_subnormal"], fmt["overflow_onset"],
         fmt["dynamic_range_decades"]))
PY

if [ "${1:-}" = "--liveness" ]; then
  echo
  echo "=== liveness: every compiled checker must ACCEPT the oracle and REJECT its control"
  python3 - <<'PY'
import json, os, pathlib, shutil, subprocess, sys, tempfile

bundle = pathlib.Path(os.environ["BIA_BUNDLE"])
sys.path.insert(0, str(bundle / "solution"))
import negative_controls as NC

env_src = bundle / "environment"
tests = bundle / "tests"
scratch = pathlib.Path(tempfile.mkdtemp(prefix="bia_s07_liveness_"))
DELIVERABLE = "submission/precision_policy.py"


def stage(policy):
    """A scratch environment whose canonical deliverable path holds the given policy.

    The verifier measures the deliverable itself, so every route has to be INSTALLED at the
    path the verifier resolves rather than passed as a flag. That is the same motion a real
    Harbor run makes, and it is why the oracle and every control share this one helper.
    """
    d = pathlib.Path(tempfile.mkdtemp(dir=scratch)) / "environment"
    shutil.copytree(env_src, d)
    shutil.copy(policy, d / DELIVERABLE)
    return d


def run_attempt(env, art, extra=None):
    e = dict(os.environ)
    e["BIA_ARTIFACTS"] = str(art)
    if extra:
        e.update(extra)
    subprocess.run(
        [sys.executable, str(env / "run_attempt.py"), "--submission", str(env / DELIVERABLE),
         "--out", str(art)],
        capture_output=True, text=True, env=e, cwd=str(env), check=True)


def grade(art, env, extra=None):
    logs = pathlib.Path(tempfile.mkdtemp(dir=scratch)) / "verifier"
    logs.mkdir(parents=True)
    e = dict(os.environ)
    e.update({"BIA_ARTIFACTS": str(art), "BIA_ENVDIR": str(env),
              "SCORE_PATH": str(logs / "score.json"),
              "BIA_OUTCOMES": str(logs / "outcomes.json"),
              "BIA_REWARD_PATH": str(logs / "reward.txt")})
    e.pop("BIA_AUTHORITATIVE_TELEMETRY", None)
    if extra:
        e.update(extra)
    subprocess.run([sys.executable, str(tests / "grade.py")], capture_output=True, text=True,
                   env=e, check=False)
    return json.loads((logs / "outcomes.json").read_text()), logs


print("  building the clean oracle artifact set")
clean_env = stage(bundle / "solution" / "reference_precision_policy.py")
clean_art = pathlib.Path(os.environ["BIA_ARTIFACTS"])
run_attempt(clean_env, clean_art)
oracle, clean_logs = grade(clean_art, clean_env)

print()
print("  ORACLE ACCEPT")
for k in sorted(oracle):
    print("    %-52s %s" % (k, "PASS" if oracle[k] else "FAIL"))
print("    -> %d/%d checkers accept the oracle" % (sum(1 for v in oracle.values() if v), len(oracle)))

fail = sorted(k for k, v in oracle.items() if not v)
rows = []

print()
print("  CONTROL REJECT (policy controls: a real run through the identical code path)")
for name in sorted(NC.POLICY_CONTROL_TARGET):
    target = NC.POLICY_CONTROL_TARGET[name]
    env = stage(bundle / "solution" / "negative_controls.py")
    art = pathlib.Path(tempfile.mkdtemp(dir=scratch)) / "artifacts"
    art.mkdir(parents=True)
    run_attempt(env, art, {"BIA_CONTROL": name})
    out, _ = grade(art, env, {"BIA_CONTROL": name})
    rejected = sorted(k for k, v in out.items() if not v)
    ok = target in rejected
    rows.append((name, target, ok, rejected))
    if not ok:
        fail.append(name)

print("  CONTROL REJECT (artifact controls: a deterministic mutation of a clean copy)")
for name in sorted(NC.TAMPERS):
    target, mutate = NC.TAMPERS[name]
    env = stage(bundle / "solution" / "reference_precision_policy.py")
    art = pathlib.Path(tempfile.mkdtemp(dir=scratch)) / "artifacts"
    shutil.copytree(clean_art, art)
    mutate(art, env)
    out, _ = grade(art, env)
    rejected = sorted(k for k, v in out.items() if not v)
    ok = target in rejected
    rows.append((name, target, ok, rejected))
    if not ok:
        fail.append(name)

print("  CONTROL REJECT (origin controls: a valid record this verifier did not author)")
for name in sorted(NC.EXTERNAL_RECORD_CONTROLS):
    target = NC.EXTERNAL_RECORD_CONTROLS[name]
    borrowed = next(clean_logs.glob("verifier_run_*/telemetry.json"))
    env = stage(bundle / "solution" / "reference_precision_policy.py")
    out, _ = grade(clean_art, env, {"BIA_AUTHORITATIVE_TELEMETRY": str(borrowed)})
    rejected = sorted(k for k, v in out.items() if not v)
    ok = target in rejected
    rows.append((name, target, ok, rejected))
    if not ok:
        fail.append(name)

print()
print("  %-44s %-52s %s" % ("control", "checker it must reject", "verdict"))
for name, target, ok, rejected in rows:
    extra = [r for r in rejected if r != target]
    print("    %-42s %-52s %s%s" % (
        name, target, "REJECTED" if ok else "*** ACCEPTED, INERT ***",
        ("  (also: %s)" % ",".join(extra)) if extra else ""))

covered = {t for _, t, ok, _ in rows if ok}
uncovered = sorted(set(oracle) - covered)
print()
if uncovered:
    print("  checkers with no control that rejects them: %s" % ", ".join(uncovered))
    fail.append("uncovered")
shutil.rmtree(scratch, ignore_errors=True)
print("liveness: %s" % ("BOTH HALVES LIVE" if not fail else "FAILED: " + ",".join(map(str, fail))))
raise SystemExit(1 if fail else 0)
PY
  exit $?
fi

if [ "${1:-}" = "--controls" ]; then
  echo
  echo "=== adequacy suite: every route through the identical code path"
  python3 - <<'PY'
import os, subprocess, sys, json, pathlib
bundle = pathlib.Path(os.environ["BIA_BUNDLE"])
sys.path.insert(0, str(bundle / "solution"))
import negative_controls as NC
env_dir = bundle / "environment"
art = os.environ["BIA_ARTIFACTS"]

def run(sub, extra=None):
    e = dict(os.environ)
    if extra:
        e.update(extra)
    p = subprocess.run(
        [sys.executable, str(env_dir / "run_attempt.py"), "--submission", str(sub), "--out", art],
        capture_output=True, text=True, env=e, cwd=str(env_dir))
    try:
        return json.loads((pathlib.Path(art) / "score.json").read_text())
    except Exception:
        print(p.stdout, p.stderr)
        raise


ref = run(bundle / "solution" / "reference_precision_policy.py")
print("  anchors: baseline=%.4f target=%.4f" % (ref["baseline_metric"], ref["target_metric"]))
print("  %-34s %-9s score=%.4f ovf=%d loss=%.4f" % ("REFERENCE", "accept", ref["score"], ref["agent_overflow_events"], ref["agent_metric"]))
fail = []
for name in sorted(NC.VARIANT_ROUTES):
    r = run(bundle / "solution" / "negative_controls.py", {"BIA_CONTROL": name})
    ok = r["score"] > 0.0 and r["agent_overflow_events"] == 0
    print("  %-34s %-9s score=%.4f ovf=%d loss=%.4f  %s" % (name, "accept", r["score"], r["agent_overflow_events"], r["agent_metric"], "OK" if ok else "UNEXPECTED"))
    if not ok:
        fail.append(name)
for name in sorted(NC.REJECTED_ROUTES):
    r = run(bundle / "solution" / "negative_controls.py", {"BIA_CONTROL": name})
    ok = r["score"] == 0.0
    print("  %-34s %-9s score=%.4f ovf=%d loss=%.4f reason=%s  %s" % (name, "reject", r["score"], r["agent_overflow_events"], r["agent_metric"], r["reason"], "OK" if ok else "UNEXPECTED"))
    if not ok:
        fail.append(name)
if ref["score"] <= 0.0 or ref["agent_overflow_events"] != 0:
    fail.append("REFERENCE")
print()
print("adequacy: %s" % ("BOTH HALVES LIVE" if not fail else "FAILED: " + ",".join(fail)))
raise SystemExit(1 if fail else 0)
PY
  exit $?
fi

# The verifier measures the deliverable path for itself, so the reference has to be INSTALLED
# there rather than passed as a flag. A scratch copy of environment/ keeps the shipped starter
# in the bundle untouched while giving the oracle the same layout a real Harbor run has.
STAGE="$WORK/environment"
rm -rf "$STAGE"
mkdir -p "$(dirname "$STAGE")"
cp -r "$BUNDLE/environment" "$STAGE"
cp "$BUNDLE/solution/reference_precision_policy.py" "$STAGE/submission/precision_policy.py"
export BIA_ENVDIR="$STAGE"

echo
echo "=== phase 1..3: fp32 control, standard control, reference policy"
python3 "$STAGE/run_attempt.py" \
  --submission "$STAGE/submission/precision_policy.py" \
  --out "$ARTIFACTS"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "reference attempt exited $RC" >&2
  exit "$RC"
fi

echo
echo "=== verifier"
bash "$BUNDLE/tests/test.sh"
VRC=$?

echo
echo "=== smoke gate assertions"
python3 - <<'PY'
import json, os, pathlib, sys
art = pathlib.Path(os.environ["BIA_ARTIFACTS"])
logs = pathlib.Path(os.environ["SCORE_PATH"]).parent
score = json.loads((art / "score.json").read_text())
outcomes = json.loads((logs / "outcomes.json").read_text())
graded = json.loads(pathlib.Path(os.environ["SCORE_PATH"]).read_text())

problems = []
if score["standard_control_overflow_events"] <= 0:
    problems.append("the shipped standard recipe did NOT overflow, so the gate proves nothing")
if score["agent_overflow_events"] != 0:
    problems.append("the reference overflowed, which contradicts its bound on the partial sums")
if not all(outcomes.values()):
    problems.append("checkers failed: %s" % sorted(k for k, v in outcomes.items() if not v))
if not (0.0 <= graded["score"] <= 1.0):
    problems.append("graded score outside [0,1]")

print("  standard_control overflow events : %d  (must be > 0)" % score["standard_control_overflow_events"])
print("  reference overflow events        : %d  (must be 0)" % score["agent_overflow_events"])
print("  checkers passed                  : %d/%d" % (sum(1 for v in outcomes.values() if v), len(outcomes)))
print("  baseline / target / agent        : %.4f / %.4f / %.4f" % (score["baseline_metric"], score["target_metric"], score["agent_metric"]))
print("  harness score                    : %.6f" % score["score"])
print("  graded score                     : %.6f" % graded["score"])
print("  mode                             : %s" % score["mode"])
print("  wallclock seconds                : %s" % score["wallclock_seconds"])
if problems:
    print()
    for p in problems:
        print("  FAIL: %s" % p)
    sys.exit(1)
print()
print("  OK: gate is live, reference is clean, every checker passed.")
if score["mode"] == "smoke":
    print("  NOTE: smoke scale. This proves the code path and the gate. It is NOT a reference")
    print("        score and it says nothing about the full operating point.")
PY
ARC=$?

exit $(( VRC != 0 ? VRC : ARC ))

#!/usr/bin/env bash
# Verifier entry point. Harbor runs this and reads the reward from
# $SCORE_PATH. The exit code is not the verdict: a scored zero and a full
# score both exit zero, and only a harness fault exits nonzero.
set -uo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BIA_BUNDLE="${BIA_BUNDLE:-$(dirname "$TESTS_DIR")}"
export BIA_SUBSTRATE_DIR="${BIA_SUBSTRATE_DIR:-/opt/bia/substrate}"
export BIA_SUBMISSION_DIR="${BIA_SUBMISSION_DIR:-/workspace/submission}"
export BIA_WORKDIR="${BIA_WORKDIR:-/workspace/artifacts}"
export BIA_LOGDIR="${BIA_LOGDIR:-/logs/verifier}"
export BIA_ANCHOR_DIR="${BIA_ANCHOR_DIR:-/anchors}"
export BIA_CORPUS="${BIA_CORPUS:-/opt/bia/data/enwik8}"
export SCORE_PATH="${SCORE_PATH:-$BIA_LOGDIR/score.json}"
export BIA_OUTCOMES="${BIA_OUTCOMES:-$BIA_LOGDIR/outcomes.json}"
export BIA_RUN_RECORD="${BIA_RUN_RECORD:-$BIA_LOGDIR/run_record.jsonl}"
export BIA_EVIDENCE="${BIA_EVIDENCE:-$BIA_LOGDIR/evidence.json}"

export REWARD_PATH="${REWARD_PATH:-/logs/verifier/reward.txt}"

mkdir -p "$BIA_LOGDIR" "$BIA_WORKDIR" "$BIA_ANCHOR_DIR"

# Trapped on EXIT so the preflight faults below, which exit 2 before grade.py ever
# runs, still leave one float behind. An absent reward reads as an infrastructure
# fault; a zero reads as the score it is.
write_reward () {
  python3 - "$SCORE_PATH" "${REWARD_PATH:-/logs/verifier/reward.txt}" <<'PY'
import json, pathlib, sys
try:
    score = float(json.loads(pathlib.Path(sys.argv[1]).read_text()).get("score", 0.0))
except Exception:
    score = 0.0
if score != score:
    score = 0.0
out = pathlib.Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("%r\n" % min(max(score, 0.0), 1.0))
PY
}
trap write_reward EXIT

preflight_fault () {
  local reason="$1" msg="$2"
  echo "=====================================================================" >&2
  echo "PREFLIGHT FAULT: $msg" >&2
  echo "  This is a CONFIG fault, not a solver failure. Do not report it as a" >&2
  echo "  score. The bundle emits no number for it." >&2
  echo "=====================================================================" >&2
  printf '{"score":0.0,"reason":"%s","detail":{}}\n' "$reason" > "$SCORE_PATH"
  exit 2
}

[ -d "$BIA_SUBSTRATE_DIR" ] || preflight_fault "preflight_fault_substrate_dir_absent" \
  "BIA_SUBSTRATE_DIR=$BIA_SUBSTRATE_DIR does not exist"
[ -f "$BIA_SUBSTRATE_DIR/SUBSTRATE_MANIFEST.json" ] || preflight_fault \
  "preflight_fault_substrate_manifest_absent" \
  "$BIA_SUBSTRATE_DIR/SUBSTRATE_MANIFEST.json is missing, so the INVARIANT checker cannot pin anything"
[ -f "$BIA_CORPUS" ] || preflight_fault "preflight_fault_corpus_absent" \
  "BIA_CORPUS=$BIA_CORPUS does not exist"

python3 "$TESTS_DIR/grade.py" 2>&1 | tee "$BIA_LOGDIR/grade-stdout.md"
GRADE_RC=${PIPESTATUS[0]}

test -s "$SCORE_PATH" || echo '{"score":0.0,"reason":"verifier_produced_no_score","detail":{}}' > "$SCORE_PATH"

python3 - "$SCORE_PATH" "$BIA_LOGDIR/reward.json" <<'PY'
import json, pathlib, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
d = json.loads(src.read_text())
score = float(d.get("score", 0.0))
score = min(max(score, 0.0), 1.0)
dst.write_text(json.dumps({"score": score}))
print(json.dumps({"reward_path": str(dst), "score": score, "reason": d.get("reason")}, sort_keys=True))
PY

echo "--- score.json ---"; cat "$SCORE_PATH"; echo
echo "--- outcomes.json ---"; cat "$BIA_OUTCOMES" 2>/dev/null; echo

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest "$TESTS_DIR/test_output.py" -v --no-header -p no:cacheprovider \
    2>&1 | tee "$BIA_LOGDIR/test-stdout.md" || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so tests/test_output.py did not run."
    echo "This is advisory only. The reward comes from grade.py and from nothing else."
    echo "Nothing was asserted here. Do not read this as the compiled rubric passing."
  } | tee "$BIA_LOGDIR/test-stdout.md"
fi

exit "$GRADE_RC"

#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
cd /workspace
export TRACK3_BUNDLE=/workspace
export TRACK3_TELEMETRY_DIR="${TRACK3_TELEMETRY_DIR:-/telemetry}"
export TRACK3_TELEMETRY="$TRACK3_TELEMETRY_DIR/run_record.jsonl"
export TRACK3_SUBMISSION=/workspace/submission
export TRACK3_CORPUS_DIR=/tests/corpus
export SCORE_PATH=/logs/verifier/score.json
export TRACK3_OUTCOMES=/logs/verifier/outcomes.json

if [ -z "${TRACK3_CHAIN_KEY:-}" ]; then
  echo "=====================================================================" >&2
  echo "PREFLIGHT FAULT: TRACK3_CHAIN_KEY is unset in the verifier environment" >&2
  echo "  grade.py -> chain_key_absent_cannot_verify -> score 0.0" >&2
  echo "  This is a CONFIG fault, not a solver failure. Do not report it as a" >&2
  echo "  score. Fix: set [verifier].env.TRACK3_CHAIN_KEY in task.toml." >&2
  echo "=====================================================================" >&2
  printf '%s\n' '{"score":0.0,"reason":"preflight_fault_chain_key_unset"}' > /logs/verifier/grade-stdout.md
  exit 2
fi

python3 /tests/grade.py 2>&1 | tee /logs/verifier/grade-stdout.md
GRADE_RC=${PIPESTATUS[0]}
test -s "$SCORE_PATH" || echo '{"score":0.0,"reason":"verifier_produced_no_score"}' > "$SCORE_PATH"
python3 - "$SCORE_PATH" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text())
num={k:v for k,v in d.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
num.setdefault("score", float(d.get("score",0.0)))
p.write_text(json.dumps(num))
PY
cat "$SCORE_PATH"; echo; echo "--- full record ---"; cat /logs/verifier/grade-stdout.md

if python3 -m pytest --version >/dev/null 2>&1; then
  TRACK3_OPTIMIZER=/workspace/submission/optimizer.py \
    python3 -m pytest /tests/test_output.py -v --no-header -p no:cacheprovider \
    2>&1 | tee /logs/verifier/test-stdout.md || true
else
  {
    echo "SKIPPED: pytest is unavailable in this image, so /tests/test_output.py did not run."
    echo "This is advisory only and does not affect the score, which comes from grade.py."
    echo "Nothing was asserted here; do not read this as 17 assertions passing."
  } | tee /logs/verifier/test-stdout.md
fi

python3 /tests/emit_verifier_artifacts.py /logs 2>/dev/null || true

exit "$GRADE_RC"

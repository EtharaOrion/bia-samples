# FORGE-CANARY-BEGIN
# 0: 78a19c95ac0cec56ee530463bc7a95de36579e8d5353d18391db292aa0d5754e
# 1: dcfc4f37e8d5876a20aa6ec8cd80d9833c5ee871216a2912d84affc16d3aeef2
# 2: 1eba2466c8ad267f79175c09f6baa977518f3591944874e8a0b81af1828b90ba
# 3: 047733cac9897780e7e8cde4132ca2e1f0eba8188bdfe24c5a08b40b48871956
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml.
#
# The reference entry point. It emits the reference recipe to the bound
# submission path and screens it against the pinned corpus before exiting, so a
# reference that had become a replay would refuse here rather than at grade time.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${BIA_SUBMISSION:-/workspace/submission/recipe.py}"
mkdir -p "$(dirname "$OUT")"
cp reference.py "$OUT"
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests'))
import fingerprint
bundle = Path(__file__).resolve().parent.parent
corpus = json.loads((bundle / 'tests' / 'corpus.json').read_text())
recipe = fingerprint.recipe_from_source(Path(sys.argv[1]).read_text())
record = fingerprint.screen(recipe, corpus['entries'], corpus['proximity_floor'])
if record['replays']:
    print('reference is a replay of ' + ', '.join(record['replays']), file=sys.stderr)
    raise SystemExit(1)
print(json.dumps({'structural_digest': record['structural_digest'], 'replays': []}, sort_keys=True))
PY

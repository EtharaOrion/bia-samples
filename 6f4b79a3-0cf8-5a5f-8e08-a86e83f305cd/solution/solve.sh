# FORGE-CANARY-BEGIN
# 0: 305da7873cbac09a5e7303f6965e720534fa569afb559a1c35b73828a58a8b2b
# 1: 800f3f996602fe01a073cb3edf20a1c1e8201cd899d08b877003d3eabc355583
# 2: e4c1e72cff9777df4df4e5d5d39dfb7a37f252ebfd98f3db195d34dd53cb822a
# 3: 7564410556951f57dba4b6815d76d342b74587df76f24b88e0f643c460f11815
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

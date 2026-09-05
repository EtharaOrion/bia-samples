# FORGE-CANARY-BEGIN
# 0: eb3f279d1fbcac28975c1f3d6337ffef143213e610752bcd684d48d3073a3018
# 1: bf9347e9847fb805362a3294394df2ac4a4dc15b5ea194d00eadf2a508026e80
# 2: df0860665ab9ab6cafb510811c15db1a86f500c02db4d09b4d2f840ad9e87905
# 3: 16eb03177c022c244c72e8257e05ccd731877130e4086d0ac92137502302a451
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

# FORGE-CANARY-BEGIN
# 0: b74faa8068b79d33f792249974cdb6dfd6a81387ca593b07a497bc29f1d4926f
# 1: 11aa228e733530efae986bf439a0bad1eb210fefc717b54c31423baae33b5bc5
# 2: e3157ef4cf2a718ceffabb77066a6053953a648b2f3db62502d094dea70c5688
# 3: bebe388f8870481e96f9286f0bc2cf5f44b3c13f95ee377391c4d1da52891fde
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

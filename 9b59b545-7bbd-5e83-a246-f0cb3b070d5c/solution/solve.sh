# FORGE-CANARY-BEGIN
# 0: 75ca4eac42d278aa6824aaa7f2652eb99258291ffb981c8f5fcc1992dba0f23b
# 1: 1d7c25e05e49eaf9e6684188a38b8f8e1896d427a53228c27a476764541fd183
# 2: 034816e1740c45a32740557c8e170e76d417eaf8e90ad09f3bb3e6e26105e33c
# 3: b6f7f339d06e8774f60cb8b5103023510ec7e6c663e5bbeeee3bc1c6415b92cd
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

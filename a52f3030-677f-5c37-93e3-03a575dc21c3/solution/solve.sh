# FORGE-CANARY-BEGIN
# 0: f1019d183aea6590a24336c6fdff344fe02b40d60e593083e96c938351cf169c
# 1: 4f3d0a7a8c816d0d825c9e8fa54f18f954294b58ad3f47a8f479a1d7de694646
# 2: 066e60ef508d790ec519bad4d52652d1ca8a289ea24003b8b23b686688cded37
# 3: 8b9e6854373940c8aac266705d8386e58f1a179df22d1a5f0bb8225c7f810e04
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml
#
# Reference solve path for OER-11.
#
# The shape of this script IS the lesson: the snapshot version is asked, work is
# done, and then the snapshot version is asked AGAIN before anything derived from
# the first answer is committed to. solution/reference.py re-resolves the corpus
# whenever the handle it was given has gone stale.
#
# THIS ORACLE INSTALLS NO SUBMISSION, AND THAT IS THE SLOT'S CONTRACT RATHER THAN AN
# OMISSION. Everything tests/grade.py grades is a harness-owned record under
# /logs/harness, listed in tests/checkers.py HANDLE_FILES and described in
# environment/README.md as written by the harness process and not writable from here.
# There is therefore no destination for this script to name and nothing for it to
# copy, install or redirect: it reads the corpus, builds the token stream, and reports
# on stdout. A reader looking for the graded artifact should look at /logs/harness.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${HERE}/../environment:${PYTHONPATH:-}"

# NOTHING THIS SCRIPT RUNS MAY LEAVE A FILE IN THE SHARED WORKSPACE. Without this,
# the two python3 -c steps below import corpus_api and pipeline from /task and the
# interpreter writes /task/__pycache__/*.pyc, which were the only paths the oracle
# phase created. A harness that infers an install destination from what the phase
# wrote then has exactly one parent directory to choose and picks that bytecode
# directory as the submission. An oracle that installs nothing must also WRITE
# nothing, otherwise its incidental output is read as its submission.
export PYTHONDONTWRITEBYTECODE=1
cd "${HERE}"

python3 -c 'import corpus_api; print(corpus_api.snapshot_version())'
python3 reference.py --probe
python3 -c 'import corpus_api; print(corpus_api.snapshot_version())'
python3 reference.py --build

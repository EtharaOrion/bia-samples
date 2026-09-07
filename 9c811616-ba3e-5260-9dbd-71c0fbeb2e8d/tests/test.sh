#!/usr/bin/env bash
# The verifier entry point Harbor runs. Harbor ships no result parser of its own, so
# this file is the whole contract: it terminates by writing the bound reward path.
#
# Bound reward contract path: /logs/verifier/reward.txt, the bare float.
# Companion score document:   /logs/verifier/score.json, reason and metric block.
#
# MEASURED CORRECTION, wave-2: /logs/verifier/ is a single shared host path and
# concurrent lanes collide on it. OER22_REWARD_ROOT redirects the root for a LOCAL
# exercise only. Unset, which is how Harbor runs this, the root is the bound one.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${TESTS_DIR}/.." && pwd)"
REWARD_ROOT="${OER22_REWARD_ROOT:-/logs/verifier}"
WORKSPACE="${OER22_WORKSPACE:-${BUNDLE_DIR}}"

emit_reward() {
  status=$?
  REWARD_FILE="${REWARD_ROOT}/reward.txt"
  if [ -z "${OER22_REWARD_ROOT:-}" ]; then
    REWARD_FILE="/logs/verifier/reward.txt"
  fi
  mkdir -p "${REWARD_ROOT}"
  python3 "${TESTS_DIR}/grade.py" --emit --reward-root "${REWARD_ROOT}" --exit-status "${status}" || true
  if [ ! -s "${REWARD_FILE}" ]; then
    printf '0.000000\n' > "${REWARD_FILE}"
    printf '%s\n' '{"reward": 0.0, "reason": "verifier-aborted-before-grading", "metric": {"graded_mean_degradation": null}}' > "${REWARD_ROOT}/score.json"
  fi
  exit "${status}"
}
trap emit_reward EXIT

mkdir -p "${REWARD_ROOT}"

# 0. The verifier's OWN pristine checkpoint, at the path tests/bound.json pins and from
#    the verifier's own copy of the pinned train shards. Built by the identical
#    environment/bootstrap.py invocation the agent surface runs, under the identical
#    pinned seed and step count, so the two surfaces agree by construction and neither
#    ever loads the other's file. Nothing a submission wrote is read here: the path, the
#    declaration and the shards all come from this image and from tests/bound.json.
#
#    It is built here rather than into a layer of tests/Dockerfile because an image
#    build has no accelerator, and the step count is the one environment/provision.sh
#    runs, where the measurement that fixes it is recorded. It runs once per container.
CHECKPOINT="$(python3 -c "import json,os,sys;b=json.load(open(sys.argv[1]));print(os.path.join(b['checkpoint_root'],b['checkpoint_name']))" "${TESTS_DIR}/bound.json")"
TRAIN_SHARDS="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['calibration_root'])" "${TESTS_DIR}/bound.json")"
if [ ! -s "${CHECKPOINT}" ]; then
  python3 "${BUNDLE_DIR}/environment/bootstrap.py" checkpoint \
    --declaration "${BUNDLE_DIR}/environment/nanogpt_substrate.json" \
    --shards "${TRAIN_SHARDS}" \
    --steps "${OER22_CHECKPOINT_STEPS:-1500}" \
    --seed "${OER22_CHECKPOINT_SEED:-1337}" \
    --out "${CHECKPOINT}" 1>&2
fi

# 0b. The agent-surface paths, resolved inside THIS container.
#
#     solution/solve.sh is the agent's entry point and it resolves the checkpoint and the
#     calibration corpus from agent-surface declarations: environment/substrate.json
#     checkpoint.path and environment/calibration_stats.json corpus_root, both of which
#     name /workspace/... . runner.py hands the submission a small environment allowlist
#     and no OER22_ variable crosses it, which is deliberate: the entry point must behave
#     here exactly as it behaves in the agent image rather than being reconfigured by the
#     grader. So the paths are made to resolve instead of the entry point being changed.
#
#     The link targets are this image's own staged copies. Nothing a submission wrote is
#     reachable through them, and the verifier still resolves every graded input from
#     tests/bound.json and never from these paths.
AGENT_CHECKPOINT_DIR="$(python3 -c "import json,os,sys;print(os.path.dirname(json.load(open(sys.argv[1]))['checkpoint']['path']))" "${BUNDLE_DIR}/environment/substrate.json")"
AGENT_CORPUS_ROOT="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['corpus_root'])" "${BUNDLE_DIR}/environment/calibration_stats.json")"
mkdir -p "$(dirname "${AGENT_CHECKPOINT_DIR}")" "$(dirname "${AGENT_CORPUS_ROOT}")"
[ -e "${AGENT_CHECKPOINT_DIR}" ] || ln -s "$(dirname "${CHECKPOINT}")" "${AGENT_CHECKPOINT_DIR}"
[ -e "${AGENT_CORPUS_ROOT}" ] || ln -s "${TRAIN_SHARDS}" "${AGENT_CORPUS_ROOT}"

# 1. Run the submission in isolation. runner.py never imports it: it copies the entry
#    point alone into a fresh temporary directory, launches it as a new session leader
#    under a small environment allowlist, and kills the whole process group afterwards.
#    OER22_ENTRY names the entry point explicitly and is how the ORACLE exercise of this
#    bundle is run: the reference is solution/solve.sh and it does not sit at the root of
#    an agent workspace. It is opt-in and there is deliberately no fallback to it. A run
#    that does not set it grades whatever the workspace carries, so an agent that produced
#    nothing is graded on nothing and scores zero, which is what
#    empty-submission-validation requires.
python3 "${TESTS_DIR}/runner.py" \
  --workspace "${WORKSPACE}" \
  ${OER22_ENTRY:+--entry "${OER22_ENTRY}"} \
  --report "${REWARD_ROOT}/runner.json" || true

# 2. Grade. Every graded number is recomputed here, inside the verifier, from its own
#    pristine substrate and the held-out evaluation payload. No number the submission
#    reported enters the graded path.
python3 "${TESTS_DIR}/grade.py" \
  --reward-root "${REWARD_ROOT}" \
  --workspace "${WORKSPACE}"

# 3. The compiled tests over the checker fixtures, both halves of every checker.
python3 "${TESTS_DIR}/test_output.py"

# 4. The reward write is the last thing that happens, and the EXIT trap above
#    guarantees it happens on every exit path including failure.

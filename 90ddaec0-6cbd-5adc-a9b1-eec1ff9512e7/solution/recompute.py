#!/usr/bin/env python3
"""Deterministic regeneration of every generated artifact in this bundle.

Source of truth is solution/grounding.yaml. This script reads it and derives:

    solution/TRUTH.md                    the private ground truth projection
    solution/rubrics.json                the reference based rubric
    tests/test_output.py                 the compiled deterministic rubric tests
    tests/checkers/frozen_manifest.json  the digest manifest the invariant checker reads
    solution/solve.sh                    the executable oracle entry point

It invokes no model, no network, no clock, no locale, and no random source, so two runs
under different host identities produce identical bytes. Every prose literal it emits is
frozen in grounding.yaml rather than composed here.

    python3 solution/recompute.py generate --root .
    python3 solution/recompute.py verify   --root .
    python3 solution/recompute.py hash     --root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import uuid

FORGE_TASK_NAMESPACE = uuid.UUID("c53e8f3b-526f-52c0-a04e-89e2269b237d")
HASH_EXCLUDED_PREFIXES = ("trajectories/",)
HASH_EXCLUDED_PATHS = ("solution/provenance.yaml",)
HASH_EXCLUDED_DIR_NAMES = ("__pycache__", ".pytest_cache")

GENERATED_BANNER = "GENERATED SECTION. DO NOT HAND-EDIT."


def load_grounding(root):
    import yaml

    return yaml.safe_load((root / "solution" / "grounding.yaml").read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_content_hash(root):
    """The hash recipe recorded in grounding.yaml and in solution/provenance.yaml.

    Walk every regular file under the bundle root, drop each excluded path, drop any
    path with a __pycache__ or .pytest_cache component, sort the surviving relative
    paths as byte strings, and for each emit the relative path bytes, a single zero
    byte, the lowercase hex sha256 of the file contents, and a newline. The canonical
    content hash is the lowercase hex sha256 of that concatenation.
    """
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in HASH_EXCLUDED_PATHS:
            continue
        if any(rel.startswith(prefix) for prefix in HASH_EXCLUDED_PREFIXES):
            continue
        if any(part in HASH_EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        entries.append((rel, sha256_file(path)))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    blob = b"".join(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n" for rel, digest in entries)
    return hashlib.sha256(blob).hexdigest(), entries


def bundle_uuid(content_hash):
    return str(uuid.uuid5(FORGE_TASK_NAMESPACE, content_hash))


def render_truth(grounding):
    op = grounding["operator"]
    reward = grounding["reward"]
    point = grounding["operating_point"]
    lines = []
    lines.append("# TRUTH: %s, kernel level throughput" % grounding["task_id"])
    lines.append("")
    lines.append(GENERATED_BANNER)
    lines.append("")
    lines.append("Source of truth is `solution/grounding.yaml`. Regenerate with `python3 solution/recompute.py generate --root .`. This file is private and never crosses the agent visible boundary.")
    lines.append("")
    lines.append("## What is frozen and what is free")
    lines.append("")
    lines.append("Frozen is the mathematical output. The operator is a fixed sequence of correctly rounded IEEE-754 binary32 primitives over a reduction whose order is part of the specification, and the graded comparison is on raw 32 bit patterns. Free is the implementation: the chunking, the memory layout, the kernel structure, the launch count, and the library used to express it. The reward is measured wall clock under that correctness gate.")
    lines.append("")
    lines.append("The per row operator, in order:")
    lines.append("")
    for step in op["per_row_sequence"]:
        lines.append("    %s" % step)
    lines.append("")
    lines.append("Constants are eps = %s, alpha = %s, beta = %s, each exactly representable in binary32 so no constant carries an implicit rounding decision. The permitted primitives are %s. The excluded primitives are %s, and each is excluded because its result is either not correctly rounded or not order determined." % (op["constants"]["eps"], op["constants"]["alpha"], op["constants"]["beta"], ", ".join(op["primitives"]), ", ".join(op["excluded_primitives"])))
    lines.append("")
    lines.append(" ".join(op["determinism_argument"].split()))
    lines.append("")
    lines.append("## The ordered path through instruction.md")
    lines.append("")
    for step in grounding["truth_steps"]:
        lines.append("### %s" % step["id"])
        lines.append("")
        lines.append("Action: %s." % " ".join(str(step["action"]).split()))
        lines.append("")
        lines.append("Established state: %s." % " ".join(str(step["established_state"]).split()))
        lines.append("")
        lines.append("Survives: %s." % " ".join(str(step["survives"]).split()))
        lines.append("")
        lines.append("Satisfied checker: `%s`." % step["checker"])
        lines.append("")
    lines.append("## Rejected routes")
    lines.append("")
    lines.append("Each rejected route names the checker that rejects it and the known wrong control whose rejection the control runner measured. A route with no measured control is not listed.")
    lines.append("")
    lines.append("| Route | Rejected by | Control |")
    lines.append("|---|---|---|")
    for route in grounding["rejected_routes"]:
        lines.append("| %s | `%s` | `%s` |" % (" ".join(str(route["route"]).split()), route["rejected_by"], route["control"]))
    lines.append("")
    for route in grounding["rejected_routes"]:
        if route.get("note"):
            lines.append("On %s: %s." % (" ".join(str(route["route"]).split()), " ".join(str(route["note"]).split())))
            lines.append("")
    lines.append("## Reward")
    lines.append("")
    lines.append("The metric is %s. Direction is %s. The formula is `%s`." % (" ".join(str(reward["metric"]).split()), reward["direction"], reward["formula"]))
    lines.append("")
    lines.append(" ".join(str(reward["anchor_standing"]).split()))
    lines.append("")
    cal = reward["reference_calibration"]
    lines.append("## Measured reference calibration")
    lines.append("")
    lines.append("The reference solution was measured on %s at the graded profile, operating point %s, over %s repeats. The median speedup is %s, the minimum is %s, the maximum is %s, the standard deviation is %s, and the relative standard deviation is %s percent." % (cal["device"], " ".join(str(cal["operating_point"]).split()), cal["repeats"], cal["speedup_median"], cal["speedup_min"], cal["speedup_max"], cal["speedup_stdev"], cal["speedup_relative_stdev_percent"]))
    lines.append("")
    lines.append("The target is derived rather than chosen: `%s`. At that target the reference scores %s, which is interior to the clamped interval on both sides, so a submission faster than the reference earns a strictly higher score and a submission slower than the reference earns a strictly lower one." % (cal["derivation"], cal["reference_score_at_this_target"]))
    lines.append("")
    lines.append(" ".join(str(cal["contention_control"]).split()))
    lines.append("")
    lines.append(" ".join(str(cal["contended_observation"]).split()))
    lines.append("")
    lines.append("## Scaled operating point")
    lines.append("")
    lines.append("A session is %s hours across %s attempts, so one graded attempt gets %s minutes of single H100 time, which is budget_hours %s. The source is %s." % (point["session_wallclock_hours"], point["attempts_per_session"], point["per_attempt_budget_minutes"], point["per_attempt_budget_hours"], point["source"]))
    lines.append("")
    lines.append(" ".join(str(point["scaled_point"]).split()))
    lines.append("")
    lines.append(" ".join(str(point["realized_duration"]).split()))
    lines.append("")
    lines.append(" ".join(str(point["rounds_calibration"]).split()))
    lines.append("")
    lines.append("## Checker reduction")
    lines.append("")
    lines.append("| Checker | Kind | Live state read |")
    lines.append("|---|---|---|")
    for checker in grounding["checkers"]:
        lines.append("| `%s` | %s | %s |" % (checker["id"], checker["kind"], " ".join(str(checker["live_state_read"]).split())))
    lines.append("")
    lines.append("Every checker above reduces to exactly one kind. None reduces to two and none reduces to zero.")
    lines.append("")
    lines.append("## Controls and both halves")
    lines.append("")
    lines.append("| Control | Class | Targets |")
    lines.append("|---|---|---|")
    for control in grounding["controls"]:
        lines.append("| `%s` | %s | %s |" % (control["id"], control["class"], control.get("targets", "accepting half")))
    lines.append("")
    lines.append("Run them with `python3 solution/control_runner.py --bundle .`, which stages a throwaway copy of the bundle, runs the smoke profile on CPU, and reports each control as accepted or rejected. It allocates no accelerator memory.")
    lines.append("")
    lines.append("## Defeat mechanism")
    lines.append("")
    lines.append(" ".join(str(grounding["defeat_mechanism"]["statement"]).split()))
    lines.append("")
    lines.append("Status: %s. No pass probability is authored here, no self solve is claimed, and no difficulty tier is assigned. The bundle ships at %s, which is the correct terminal state for a bundle with no external signed pilot." % (grounding["defeat_mechanism"]["status"], grounding["disposition_ceiling"]))
    lines.append("")
    return "\n".join(lines)


def render_rubrics(grounding):
    items = []
    floor = None
    for raw in grounding["rubric_items"]:
        if "floor" in raw:
            floor = float(raw["floor"])
            continue
        item = {
            "id": raw["id"],
            "dimension": raw["dimension"],
            "weight": int(raw["weight"]),
            "evaluation_target": raw["evaluation_target"],
            "criterion": " ".join(str(raw["criterion"]).split()),
            "judgment": " ".join(str(raw["judgment"]).split()),
            "evidence": list(raw["evidence"]),
            "mode": raw["mode"],
        }
        if "outcome_key" in raw:
            item["outcome_key"] = raw["outcome_key"]
        if raw["mode"] == "judged":
            item["residue"] = " ".join(str(raw["residue"]).split())
        items.append(item)
    total = sum(i["weight"] for i in items)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    payload = {
        "source": "solution/grounding.yaml",
        "generator": "solution/recompute.py",
        "banner": GENERATED_BANNER,
        "task_id": grounding["task_id"],
        "evaluation_target_vocabulary": ["verifier_outcomes", "verifier_report", "submission_module", "agent_trajectory"],
        "compilation_floor": floor,
        "compiled_weight_share": round(compiled / total, 6),
        "items": items,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_test_output(grounding):
    compiled = [i for i in grounding["rubric_items"] if i.get("mode") == "compiled"]
    lines = []
    lines.append('"""Compiled deterministic rubric tests.')
    lines.append("")
    lines.append(GENERATED_BANNER)
    lines.append("")
    lines.append("Generated by solution/recompute.py from solution/grounding.yaml. One test per")
    lines.append("compiled rubric item, each asserting over the verifier outcome map or the verifier")
    lines.append("report. No criterion prose and no reference text appears here.")
    lines.append('"""')
    lines.append("")
    lines.append("import json")
    lines.append("import os")
    lines.append("import pathlib")
    lines.append("")
    lines.append("")
    lines.append("def _outcomes():")
    lines.append('    path = os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json")')
    lines.append('    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))')
    lines.append("")
    lines.append("")
    lines.append("def _report():")
    lines.append('    path = os.environ.get("BIA_REPORT", "/logs/verifier/report.json")')
    lines.append('    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))')
    lines.append("")
    for item in compiled:
        lines.append("")
        if item["evaluation_target"] == "verifier_outcomes":
            key = item["outcome_key"]
            lines.append("def test_%s():" % item["id"])
            lines.append('    assert _outcomes()["%s"] is True' % key)
        elif item["id"] == "reward_within_unit_interval":
            lines.append("def test_%s():" % item["id"])
            lines.append('    score = _report()["score"]')
            lines.append("    assert isinstance(score, float)")
            lines.append("    assert 0.0 <= score <= 1.0")
        elif item["id"] == "zero_score_carries_reason":
            lines.append("def test_%s():" % item["id"])
            lines.append('    reason = _report()["reason"]')
            lines.append("    assert isinstance(reason, str) and reason")
        else:
            raise ValueError("no compilation rule for item %s" % item["id"])
        lines.append("")
    return "\n".join(lines)


def render_frozen_manifest(root, grounding):
    files = {}
    for rel in grounding["frozen_surface"]:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError("frozen path absent: %s" % rel)
        files[rel] = sha256_file(path)
    payload = {
        "banner": GENERATED_BANNER,
        "source": "solution/grounding.yaml",
        "algorithm": "sha256",
        "files": files,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


SOLVE_TEMPLATE = """#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Generated by solution/recompute.py from solution/grounding.yaml.
#
# Executable oracle for the {task_id} kernel throughput task. It installs the private
# reference solution as the submission and runs the real verifier path.
#
# Default mode runs the graded profile on one H100.
# BIA_SMOKE=1 runs the identical code path at tiny scale on CPU with no CUDA context,
# which exists so the reference can be proven to execute end to end without an
# accelerator. In smoke mode the correctness gate and the divergence path are fully
# exercised and the timing number produced is not a claim about the operating point.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BUNDLE="$(cd "$SOLUTION_DIR/.." && pwd)"

if [ "${{1:-}}" = "--controls" ]; then
  exec python3 "$SOLUTION_DIR/control_runner.py" --bundle "$BUNDLE"
fi

SMOKE="${{BIA_SMOKE:-0}}"
if [ "$SMOKE" = "1" ]; then
  WORK="$(mktemp -d -t bia-s05-smoke-XXXXXX)"
  LOGS="$WORK/logs"
  export CUDA_VISIBLE_DEVICES=""
else
  WORK="${{BIA_WORK:-/workspace}}"
  LOGS="${{BIA_LOGS:-/logs/verifier}}"
fi

mkdir -p "$WORK/submission" "$WORK/artifacts" "$LOGS"
cp "$SOLUTION_DIR/fast_impl.py" "$WORK/submission/impl.py"

export BIA_SMOKE="$SMOKE"
export BIA_BUNDLE="$BUNDLE"
export BIA_TESTS="${{BIA_TESTS:-$BUNDLE/tests}}"
export BIA_ENV_DIR="${{BIA_ENV_DIR:-$BUNDLE/environment}}"
export BIA_SUBMISSION="$WORK/submission/impl.py"
export BIA_ARTIFACTS="$WORK/artifacts"
export BIA_LOGS="$LOGS"
export SCORE_PATH="${{SCORE_PATH:-$LOGS/score.json}}"
export BIA_REPORT="${{BIA_REPORT:-$LOGS/report.json}}"
export BIA_OUTCOMES="${{BIA_OUTCOMES:-$LOGS/outcomes.json}}"
export BIA_EVENTS="${{BIA_EVENTS:-$LOGS/harness_events.jsonl}}"
export BIA_TIMING_SUMMARY="${{BIA_TIMING_SUMMARY:-$LOGS/timing_summary.json}}"

bash "$BIA_TESTS/test.sh"
RC=$?
echo "score file: $SCORE_PATH"
cat "$SCORE_PATH"
echo
exit $RC
"""


def render_solve(grounding):
    return SOLVE_TEMPLATE.format(task_id=grounding["task_id"])


TARGETS = (
    ("solution/TRUTH.md", render_truth, "grounding"),
    ("solution/rubrics.json", render_rubrics, "grounding"),
    ("tests/test_output.py", render_test_output, "grounding"),
    ("tests/checkers/frozen_manifest.json", render_frozen_manifest, "root_grounding"),
    ("solution/solve.sh", render_solve, "grounding"),
)


def build(root):
    grounding = load_grounding(root)
    pairs = []
    for rel, renderer, mode in TARGETS:
        text = renderer(root, grounding) if mode == "root_grounding" else renderer(grounding)
        pairs.append((rel, text))
    return pairs


def check_byte_identical_pairs(root, grounding):
    problems = []
    for left, right in grounding.get("byte_identical_pairs", []):
        lb = (root / left).read_bytes()
        rb = (root / right).read_bytes()
        if lb != rb:
            problems.append("%s and %s differ" % (left, right))
    return problems


def cmd_generate(root):
    grounding = load_grounding(root)
    problems = check_byte_identical_pairs(root, grounding)
    if problems:
        print("FAIL byte identical pair check: %s" % "; ".join(problems))
        return 1
    for rel, text in build(root):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if rel.endswith(".sh"):
            path.chmod(0o755)
        print("wrote %s" % rel)
    return 0


def cmd_verify(root):
    grounding = load_grounding(root)
    problems = check_byte_identical_pairs(root, grounding)
    drift = []
    for rel, text in build(root):
        path = root / rel
        if not path.is_file():
            drift.append("%s absent" % rel)
            continue
        if path.read_text(encoding="utf-8") != text:
            drift.append("%s drifted" % rel)
    for line in problems + drift:
        print("FAIL %s" % line)
    if problems or drift:
        return 1
    print("all generated artifacts are byte identical to their regeneration")
    return 0


def cmd_hash(root):
    content_hash, entries = canonical_content_hash(root)
    print("files_hashed          : %d" % len(entries))
    print("canonical_content_hash: %s" % content_hash)
    print("namespace             : %s" % FORGE_TASK_NAMESPACE)
    print("uuid5                 : %s" % bundle_uuid(content_hash))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "verify", "hash"])
    parser.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parent.parent))
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    return {"generate": cmd_generate, "verify": cmd_verify, "hash": cmd_hash}[args.command](root)


if __name__ == "__main__":
    sys.exit(main())

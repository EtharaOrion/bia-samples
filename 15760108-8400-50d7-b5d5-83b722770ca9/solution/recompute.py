#!/usr/bin/env python3
"""Recomputation instrument for bia S09 multi-objective-frontier.

Three modes, each independently runnable.

  --reward RECORD [--score SCORE]
      Recompute the reward from the RAW achieved objective values in a sealed
      frontier record: re-derive the normalization from the declared anchors,
      re-filter the Pareto front, re-integrate the hypervolume through the
      verifier's independent implementation, clamp, and compare against the
      score the verifier wrote. A grader can therefore verify the number
      instead of trusting it. Exits non-zero on any disagreement.

  --regen [--check]
      Derive every generated canonical bundle artifact from solution/grounding.yaml:
      solution/TRUTH.md, solution/rubrics.json, tests/rubrics.jsonl and
      tests/test_output.py. With --check it writes nothing and exits non-zero if
      the bytes on disk differ from the bytes it would write.

  --hash [BUNDLE]
      Recompute the canonical content hash and the uuid5 the directory name
      must equal, under the recipe recorded in solution/provenance.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import uuid

HERE = pathlib.Path(__file__).resolve().parent
BUNDLE = HERE.parent
FORGE_TASK_NAMESPACE = uuid.UUID("c53e8f3b-526f-52c0-a04e-89e2269b237d")
HASH_EXCLUDED_TOP_LEVEL = ("trajectories",)
HASH_EXCLUDED_NAMES = ("__pycache__",)
HASH_EXCLUDED_SUFFIXES = (".pyc",)

sys.path.insert(0, str(BUNDLE / "tests"))


# ---------------------------------------------------------------------------
# Mode 1: recompute the reward from raw achieved points.
# ---------------------------------------------------------------------------


def mode_reward(record_path: pathlib.Path, score_path: pathlib.Path | None) -> int:
    from checkers.hypervolume import hypervolume_grid, normalize_point

    record = json.loads(record_path.read_text())
    anc = record["anchors"]

    expected_loss_ref = math.log(record["spec_vocab_size"]) if "spec_vocab_size" in record else None
    raw_points = [
        (r["objectives"]["val_loss"], r["objectives"]["density"]) for r in record["runs"]
    ]
    if not raw_points:
        print("no achieved points in the record; reward is 0.0 by definition")
        return 1

    print("reference point (raw)        ", [anc["loss_ref"], anc["density_ref"]])
    print("ideal point (raw)            ", [anc["loss_ideal"], anc["density_ideal"]])
    print("reference point (normalized) ", [0.0, 0.0])
    print("ideal point (normalized)     ", [1.0, 1.0])
    if expected_loss_ref is not None and abs(expected_loss_ref - anc["loss_ref"]) > 1e-9:
        print("FAIL: recorded loss reference is not ln(vocab_size)")
        return 1

    normalized = [normalize_point(l, d, anc) for l, d in raw_points]
    print()
    print("raw achieved points and their recomputed normalized coordinates")
    for i, ((l, d), (u1, u2)) in enumerate(zip(raw_points, normalized)):
        print(f"  point {i}  val_loss {l:.6f}  density {d:.6f}  ->  u ({u1:.9f}, {u2:.9f})")

    hv = hypervolume_grid(normalized)
    reward = min(max(hv, 0.0), 1.0)
    single = max((u1 * u2 for u1, u2 in normalized), default=0.0)
    print()
    print(f"recomputed hypervolume       {hv:.12f}")
    print(f"recomputed reward, clamped   {reward:.12f}")
    print(f"best single point box        {single:.12f}")
    print(f"multi-point gain over best single point  {reward - single:+.12f}")

    rc = 0
    if abs(hv - float(record["hypervolume_raw"])) > 1e-9:
        print(f"FAIL: record hypervolume {record['hypervolume_raw']!r} disagrees")
        rc = 1
    if score_path is not None and score_path.is_file():
        recorded = float(json.loads(score_path.read_text())["score"])
        if abs(recorded - reward) > 1e-9:
            print(f"FAIL: verifier score {recorded!r} disagrees with the recomputed reward")
            rc = 1
        else:
            print(f"verifier score {recorded:.12f} agrees with the recomputed reward")
    if rc == 0:
        print("REWARD RECOMPUTATION AGREES")
    return rc


# ---------------------------------------------------------------------------
# Mode 2: regenerate every generated canonical artifact from grounding.yaml.
# ---------------------------------------------------------------------------


def _load_grounding() -> dict:
    import yaml

    return yaml.safe_load((HERE / "grounding.yaml").read_text())


def _render_rubrics_json(g: dict) -> str:
    items = g["rubric"]["items"]
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    total = sum(i["weight"] for i in items)
    payload = {
        "schema": "bia.rubric/v1",
        "task": g["task"]["name"],
        "present": g["rubric"]["present"],
        "compilation_floor": g["rubric"]["compilation_floor"],
        "evaluation_targets": g["rubric"]["evaluation_targets"],
        "total_weight": round(total, 6),
        "compiled_weight_share": round(compiled / total, 6),
        "meets_compilation_floor": (compiled / total) >= g["rubric"]["compilation_floor"],
        "items": [
            {
                "id": i["id"],
                "dimension": i["dimension"],
                "weight": i["weight"],
                "evaluation_target": i["evaluation_target"],
                "criterion": i["criterion"].strip(),
                "judgment": i["judgment"],
                "evidence": i["evidence"],
                "mode": i["mode"],
                **({"outcome_key": i["outcome_key"]} if "outcome_key" in i else {}),
                **({"residue": " ".join(i["residue"].split())} if "residue" in i else {}),
            }
            for i in items
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _render_rubrics_jsonl(g: dict) -> str:
    """Agent-facing roster: one object per line, exactly the keys id and rubric."""
    lines = []
    for i in g["rubric"]["items"]:
        if i["mode"] != "judged":
            continue
        lines.append(json.dumps({"id": i["id"], "rubric": " ".join(i["criterion"].split())}))
    return "\n".join(lines) + "\n"


def _render_test_output(g: dict) -> str:
    items = [i for i in g["rubric"]["items"] if i["mode"] == "compiled"]
    body = [
        '"""Deterministic rubric tests compiled from solution/grounding.yaml.',
        "",
        "GENERATED SECTION. DO NOT HAND-EDIT.",
        "Regenerate with: python3 solution/recompute.py --regen",
        "",
        "Each test asserts one compiled rubric item against the outcomes map grade.py",
        "wrote. These tests never re-derive an outcome, so a pytest verdict can never",
        'disagree with the score the grader recorded.',
        '"""',
        "",
        "import json",
        "import os",
        "import pathlib",
        "import sys",
        "",
        "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))",
        "",
        "from checkers import outcomes",
        "",
        "",
        "def _o():",
        "    return outcomes.load()",
        "",
    ]
    for i in items:
        key = i["outcome_key"]
        if key == "_score":
            body += [
                "",
                f"def test_{i['id']}():",
                f"    # {' '.join(i['criterion'].split())}",
                '    s = _o()["_score"]',
                "    assert isinstance(s, float)",
                "    assert 0.0 <= s <= 1.0",
                "",
            ]
        elif key == "best_single_point_hypervolume":
            body += [
                "",
                f"def test_{i['id']}():",
                f"    # {' '.join(i['criterion'].split())}",
                '    score_path = pathlib.Path(os.environ.get("BIA_S09_SCORE", "/logs/verifier/score.json"))',
                "    if not score_path.is_file():",
                "        return",
                "    d = json.loads(score_path.read_text())",
                '    if "best_single_point_hypervolume" not in d:',
                "        return",
                '    assert d["best_single_point_hypervolume"] <= d["score"] + 1e-9',
                "",
            ]
        else:
            body += [
                "",
                f"def test_{i['id']}():",
                f"    # {' '.join(i['criterion'].split())}",
                f'    assert _o()["{key}"] is True',
                "",
            ]
    return "\n".join(body).rstrip("\n") + "\n"


def _para(text: str) -> str:
    """One paragraph is one continuous line. No newline inside a paragraph."""
    return " ".join(str(text).split())


def _render_truth(g: dict) -> str:
    t, o, n, h, b, d = (
        g["task"],
        g["objectives"],
        g["normalization"],
        g["hypervolume"],
        g["budget"],
        g["defeat_mechanism"],
    )
    tr = g["truth"]
    items = g["rubric"]["items"]
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    total = sum(i["weight"] for i in items)
    lines = [
        "# TRUTH: bia S09 multi-objective-frontier",
        "",
        "GENERATED SECTION. DO NOT HAND-EDIT. Regenerate with `python3 solution/recompute.py --regen`, whose sole source is `solution/grounding.yaml`.",
        "",
        "This file is private. It never reaches the agent.",
        "",
        "## What is frozen and what is free",
        "",
        _para(t["frozen"]) + ".",
        "",
        _para(t["free"]) + ".",
        "",
        "## The two objectives",
        "",
    ]
    for obj in o:
        lines += [
            f"**{obj['id']}, {obj['name']}, {obj['direction']}.** " + _para(obj["definition"]) + f". Measured by `{obj['measured_by']}`.",
            "",
        ]
        if "scale_invariance_note" in obj:
            lines += [_para(obj["scale_invariance_note"]) + ".", ""]
    lines += [
        "## The declared tradeoff surface",
        "",
        _para(g["tradeoff_surface"]["statement"]) + ".",
        "",
        _para(g["tradeoff_surface"]["endpoints_are_forced"]) + ".",
        "",
        _para(g["tradeoff_surface"]["curvature_is_the_open_question"]) + ".",
        "",
        "## The pinned normalization",
        "",
        _para(n["form"]) + ".",
        "",
        f"Loss axis. Reference coordinate is {_para(n['loss_axis']['reference_coordinate'])}. Ideal coordinate is {_para(n['loss_axis']['ideal_coordinate'])}. The entropy rate is {_para(n['loss_axis']['entropy_rate_derivation'])}.",
        "",
        f"Density axis. Reference coordinate is {_para(n['density_axis']['reference_coordinate'])}. Ideal coordinate is {_para(n['density_axis']['ideal_coordinate'])}.",
        "",
        _para(n["measurement_free_note"]) + ".",
        "",
        "## The pinned hypervolume",
        "",
        f"Reference point, normalized: `{h['reference_point_normalized']}`. Ideal point, normalized: `{h['ideal_point_normalized']}`. Reference point, raw: `{h['reference_point_raw']}`. Ideal point, raw: `{h['ideal_point_raw']}`.",
        "",
        _para(h["definition"]) + ".",
        "",
        f"Reward is `{h['reward']}`. The engine computes it at `{h['engine_implementation']}` and the verifier recomputes it independently at `{h['verifier_implementation']}`, and the DIVERGENCE checker requires the two to agree.",
        "",
        "## The degenerate case, stated plainly",
        "",
        _para(h["degenerate_case"]),
        "",
        "## The budget",
        "",
        f"Per-attempt budget is `{b['per_attempt_budget_hours']}` hours, which is {b['per_attempt_budget_minutes']} minutes. " + _para(b["derivation"]) + ".",
        "",
        f"A graded attempt evaluates **{b['frontier_points_per_graded_attempt']} frontier points**. " + _para(b["arithmetic"]) + ".",
        "",
        _para(b["enforcement"]) + ".",
        "",
        _para(b["shared_overhead_measurement_note"]) + ".",
        "",
        "UNMEASURED. " + _para(b["graded_wall_clock_unmeasured_reason"]) + ".",
        "",
        "## The reference solution",
        "",
        f"`{tr['reference_solution']}`.",
        "",
        _para(tr["reference_design"]),
        "",
        _para(tr["why_the_sweep_is_the_answer"]),
        "",
        "## The reference score",
        "",
        f"**{_para(tr['reference_score_graded'])}.** " + _para(tr["reference_score_graded_reason"]),
        "",
        _para(tr["smoke_note"]),
        "",
        "## Refusing control, at smoke scale",
        "",
        _para(tr["negative_control_smoke_note"]),
        "",
        f"| Submission at the smoke fixture | Hypervolume |",
        "|---|---|",
        f"| reference, six points spread along the surface | {tr['reference_smoke_score']} |",
        f"| negative control, six loss-only points | {tr['negative_control_smoke_score']} |",
        "",
        "## The defeat mechanism, as design intent",
        "",
        _para(d["design_intent"]),
        "",
        f"Designed against the pinned cohort {', '.join(d['cohort'])}, valid to {d['cohort_validity_end']}.",
        "",
        _para(d["measured_note"]),
        "",
        "## Graded assertions",
        "",
        "| Checker | Kind |",
        "|---|---|",
    ]
    for c in g["checkers"]:
        lines.append(f"| `{c['id']}` | {c['kind']} |")
    lines += [
        "",
        f"Rubric compiled weight share is {compiled / total:.4f} over total weight {total:.2f}, against a floor of {g['rubric']['compilation_floor']}.",
        "",
        "## Disposition",
        "",
        f"`{g['disposition']['value']}`. Reasons: " + ", ".join(f"`{r}`" for r in g["disposition"]["reasons"]) + ".",
        "",
    ]
    return "\n".join(lines)


def _render_frozen_bytes(g):
    """Pin the executable harness bytes the graded measurement runs through.

    engine.py, run_frontier.py and frontier_worker.py are agent-visible files that decide the
    objectives. Nothing hashed them before, so a submission could rewrite the density
    measurement rather than earn it. The pin is generated here so it can never drift from the
    bytes it pins.
    """
    harness = BUNDLE / "environment" / "harness"
    files = {
        name: hashlib.sha256((harness / name).read_bytes()).hexdigest()
        for name in ("engine.py", "run_frontier.py", "frontier_worker.py")
    }
    return json.dumps(
        {
            "_comment": "GENERATED SECTION. DO NOT HAND-EDIT. Regenerate with solution/recompute.py --regen.",
            "purpose": "pins the executable harness bytes for the frozen_harness_bytes_unmodified checker",
            "files": files,
        },
        indent=1,
        sort_keys=True,
    ) + "\n"


GENERATED = {
    "tests/checkers/frozen_bytes.json": _render_frozen_bytes,
    "solution/TRUTH.md": _render_truth,
    "solution/rubrics.json": _render_rubrics_json,
    "tests/rubrics.jsonl": _render_rubrics_jsonl,
    "tests/test_output.py": _render_test_output,
}


def mode_regen(check_only: bool) -> int:
    g = _load_grounding()
    rc = 0
    for rel, render in GENERATED.items():
        want = render(g)
        target = BUNDLE / rel
        have = target.read_text() if target.is_file() else None
        if have == want:
            print(f"  match      {rel}")
            continue
        if check_only:
            print(f"  DIFFERS    {rel}")
            rc = 1
        else:
            target.write_text(want)
            print(f"  written    {rel}")
    return rc


# ---------------------------------------------------------------------------
# Mode 3: canonical content hash and uuid5.
# ---------------------------------------------------------------------------


def canonical_files(bundle: pathlib.Path):
    out = []
    for p in sorted(bundle.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(bundle)
        parts = rel.parts
        if parts[0] in HASH_EXCLUDED_TOP_LEVEL:
            continue
        if any(part in HASH_EXCLUDED_NAMES for part in parts):
            continue
        if p.suffix in HASH_EXCLUDED_SUFFIXES:
            continue
        out.append(rel.as_posix())
    return sorted(out)


def canonical_content_hash(bundle: pathlib.Path) -> str:
    h = hashlib.sha256()
    for rel in canonical_files(bundle):
        data = (bundle / rel).read_bytes()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(len(data)).encode("ascii"))
        h.update(b"\x00")
        h.update(data)
    return h.hexdigest()


def mode_hash(bundle: pathlib.Path) -> int:
    files = canonical_files(bundle)
    digest = canonical_content_hash(bundle)
    task_uuid = uuid.uuid5(FORGE_TASK_NAMESPACE, digest)
    for rel in files:
        print(f"  {rel}")
    print()
    print(f"files in hash domain      {len(files)}")
    print(f"canonical_content_hash    {digest}")
    print(f"uuid5                     {task_uuid}")
    print(f"directory name            {bundle.name}")
    if bundle.name == str(task_uuid):
        print("DIRECTORY NAME MATCHES uuid5 OVER THE FROZEN CONTENT HASH")
        return 0
    print("directory name does not match; rename the bundle to the uuid5 above")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward", metavar="RECORD")
    ap.add_argument("--score", metavar="SCORE")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--hash", nargs="?", const=str(BUNDLE), metavar="BUNDLE")
    a = ap.parse_args()
    if not (a.reward or a.regen or a.hash):
        ap.print_help()
        return 2
    rc = 0
    if a.reward:
        rc |= mode_reward(pathlib.Path(a.reward), pathlib.Path(a.score) if a.score else None)
    if a.regen:
        rc |= mode_regen(a.check)
    if a.hash:
        rc |= mode_hash(pathlib.Path(a.hash).resolve())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

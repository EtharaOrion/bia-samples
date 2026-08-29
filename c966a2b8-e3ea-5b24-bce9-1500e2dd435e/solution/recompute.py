"""Regenerates every generated artifact in this bundle from solution/grounding.yaml.

Generated, never hand edited:
  solution/TRUTH.md          the private ground truth
  solution/rubrics.json      the private reference-based rubric
  tests/test_output.py       the deterministic tests compiled from that rubric

Run with --check to fail when the committed bytes differ from the regenerated
bytes, which is what proves the generated artifacts did not drift.

  python3 solution/recompute.py
  python3 solution/recompute.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
GROUNDING = os.path.join(HERE, "grounding.yaml")

BANNER_MD = "<!-- GENERATED SECTION. DO NOT HAND-EDIT. Regenerate with: python3 solution/recompute.py -->"
BANNER_PY = "# GENERATED FILE. DO NOT HAND-EDIT. Regenerate with: python3 solution/recompute.py"


def load():
    with open(GROUNDING) as f:
        return yaml.safe_load(f)


def render_truth(g) -> str:
    s = g["slot"]
    pr = g["published_result"]
    ab = g["ablated_component"]
    en = g["enforcement"]
    re_ = g["reachability"]
    op = g["operating_point"]
    an = g["anchors"]
    ob = g["optimization_bindings"]
    co = g["cohort"]
    ct = g["contamination"]
    L = []
    A = L.append
    A(f"# TRUTH: slot {s['id']}, {s['name']}")
    A("")
    A(BANNER_MD)
    A("")
    A("This file is private ground truth. It is generated from solution/grounding.yaml by solution/recompute.py and it never crosses the agent-visible boundary.")
    A("")
    A("## What is frozen and what is free")
    A("")
    A(f"Frozen: {s['frozen']}. Free: {s['free']}. The reward is on {s['reward_on']}.")
    A("")
    A("## The published result this slot freezes")
    A("")
    A(pr["identity"])
    A("")
    A("Lineage:")
    A("")
    for item in pr["lineage"]:
        A(f"- {item}")
    A("")
    A(pr["how_it_is_reproduced_here"])
    A("")
    A("## The ablated component")
    A("")
    A(ab["identity"] + " " + ab["why_this_component"])
    A("")
    A(f"Precise scope: {ab['precise_scope']}")
    A("")
    A(f"Explicitly out of scope: {ab['explicitly_out_of_scope']}")
    A("")
    A("## How the harness enforces the ablation")
    A("")
    A(f"The enforcement is {en['mechanism']}. The instrument is {en['instrument']}. Two independent interception points are installed so a single bypass does not blind it.")
    A("")
    for item in en["interception_points"]:
        A(f"- {item}")
    A("")
    A(en["phase_tagging"])
    A("")
    A(f"Live-state read: {en['live_state_read']}")
    A("")
    A(f"Liveness witness: {en['liveness_witness']}")
    A("")
    A(en["why_the_agent_cannot_reintroduce_it"])
    A("")
    A("## Is the target reachable without the ablated component")
    A("")
    A(f"Status: {re_['claim_status']}.")
    A("")
    A(re_["argument"])
    A("")
    A(re_["why_this_is_not_a_proof"])
    A("")
    A(f"How a grader confirms it on an idle GPU: {re_['how_a_grader_confirms_it_on_an_idle_gpu']}")
    A("")
    A(re_["partial_credit_note"])
    A("")
    A("## The scaled operating point")
    A("")
    A(f"Binding source: {op['binding_source']}. {op['derivation']}")
    A("")
    A("| Quantity | Value |")
    A("|---|---|")
    for k, v in op["scaled_point"].items():
        A(f"| {k} | {v} |")
    A("")
    A(f"Wallclock status: {op['wallclock_status']}")
    A("")
    A(f"First orchestrator action: {op['first_orchestrator_action']}")
    A("")
    A("## The anchors and the reward")
    A("")
    A(f"Policy: {an['policy']}. Target metric: {an['target_metric']}. Baseline metric: {an['baseline_metric']}.")
    A("")
    A(f"Reward: `{an['reward']}`")
    A("")
    A(an["baseline_divergence_fallback"])
    A("")
    A(an["load_bearing_guard"])
    A("")
    A("## Optimization bindings")
    A("")
    A("| Binding | Value |")
    A("|---|---|")
    for k in ("max_timeout", "budget_hours", "max_attempts", "final_selection",
              "solver_egress", "compute_envelope", "reward"):
        A(f"| {k} | {ob[k]} |")
    A("")
    A(ob["role_note"])
    A("")
    A("## Cohort")
    A("")
    A(f"Pinned by name: {', '.join(co['pinned'])}, valid to {co['valid_to']}. Registry digest status: {co['registry_digest_status']}")
    A("")
    A("## Contamination")
    A("")
    A(f"Primary leakage surface: {ct['primary_leakage_surface']}")
    A("")
    A(ct["why_replay_does_not_work_here"])
    A("")
    A(ct["what_remains_available_and_should"])
    A("")
    A(f"Fingerprint check status: {ct['fingerprint_check_status']}")
    A("")
    A("## Rejected routes, and what rejects them")
    A("")
    A("| Route a submission might take | What rejects it |")
    A("|---|---|")
    A("| Reintroduce activation normalization into the forward pass by any means | The ABSENCE checker ablation_absent_from_graded_run, reading the live NormProbe trace for the agent arm |")
    A("| Edit the substrate to remove the probe or change the model | The INVARIANT checker substrate_bytes_unmodified, which rehashes every substrate file at run start and run end against the manifest baked into the image |")
    A("| Submit a rule that leaves the weights alone so a stale evaluation is graded | The EFFECT checker submitted_rule_moved_the_weights, reading the measured parameter delta at every sampled step |")
    A("| Report a validation loss the checkpoint does not reproduce | The DIVERGENCE checker validation_loss_recompute_agrees, comparing the streaming value against a fresh recompute from the saved checkpoint |")
    A("| Degrade the target anchor instead of improving the agent arm | The ORDERING checker anchors_measured_before_agent_arm, which requires both anchors to be ready before the agent arm starts |")
    A("| Port a published record recipe unchanged | Nothing rejects it, and nothing needs to. That recipe is the measured baseline anchor, so it lands at raw zero by construction |")
    A("| Emit a score outside the unit interval or one that is not the clamp of raw | The VALUE checker reward_is_clamped_unit_float |")
    A("")
    A("## Defeat mechanism, stated as design intent")
    A("")
    A(s["defeat_mechanism_design_intent"])
    A("")
    A(f"Status: {s['defeat_mechanism_status']}. No pass probability is authored here, no self-solve is claimed, and no difficulty tier is assigned. Difficulty is measured by an external signed pilot over frozen bytes and by nothing else.")
    A("")
    A("## Declared gaps")
    A("")
    A("| Gap | Statement |")
    A("|---|---|")
    for gap in g["declared_gaps"]:
        A(f"| {gap['id']} | {gap['text']} |")
    A("")
    A("## Disposition")
    A("")
    A(f"{g['disposition']}. {g['disposition_reason']}")
    A("")
    return "\n".join(L)


def render_rubrics(g) -> str:
    items = []
    for it in g["rubric_items"]["items"]:
        row = {
            "id": it["id"],
            "dimension": it["dimension"],
            "weight": float(it["weight"]),
            "evaluation_target": it["evaluation_target"],
            "criterion": it["criterion"],
            "judgment": it["judgment"],
            "evidence": list(it["evidence"]),
            "mode": it["mode"],
        }
        if it["mode"] == "judged":
            row["residue"] = it["residue"]
        items.append(row)
    total = sum(i["weight"] for i in items)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    doc = {
        "schema_version": "1.0",
        "slot": g["slot"]["id"],
        "compilation_floor": float(g["rubric_items"]["compilation_floor"]),
        "compiled_weight_share": round(compiled / total, 6),
        "items": items,
    }
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


COMPILED_TESTS = {
    "checkers_roster_matches_the_registry": '''
def test_checkers_roster_matches_the_registry():
    declared = [c["id"] for c in _checkers_yaml()["checkers"]]
    registry = [n for n, _, _ in _checkers_module().REGISTRY]
    assert declared == registry, (declared, registry)
    assert len(declared) == len(registry) and declared
''',
    "each_checker_reduces_to_exactly_one_kind": '''
def test_each_checker_reduces_to_exactly_one_kind():
    import re
    closed = {"VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"}
    for c in _checkers_yaml()["checkers"]:
        assert isinstance(c["kind"], str), c["id"]
        assert c["kind"] in closed, (c["id"], c["kind"])
    mod = _checkers_module()
    reg = {n: k for n, k, _ in mod.REGISTRY}
    kebab = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    seen = set()
    for c in _checkers_yaml()["checkers"]:
        assert reg[c["id"]] == c["kind"], c["id"]
        assert c["reduction"] == c["kind"], c["id"]
        assert c["zero_reason"] == mod.ZERO_REASON[c["id"]], c["id"]
        assert kebab.match(c["zero_reason"]), (c["id"], c["zero_reason"])
        assert c["zero_reason"] not in seen, c["id"]
        seen.add(c["zero_reason"])
        carrier = os.path.join(BUNDLE, c["reached_by"])
        assert os.path.isfile(carrier), (c["id"], c["reached_by"])
        assert "def %s(" % c["selector"] in open(carrier).read(), (c["id"], c["selector"])
''',
    "every_checker_names_a_live_state_read": '''
def test_every_checker_names_a_live_state_read():
    for c in _checkers_yaml()["checkers"]:
        assert c.get("live_state_read"), c["id"]
        assert len(c["live_state_read"].strip()) > 40, c["id"]
''',
    "kind_census_matches_declared_kinds": '''
def test_kind_census_matches_declared_kinds():
    y = _checkers_yaml()
    counted = {}
    for c in y["checkers"]:
        counted[c["kind"]] = counted.get(c["kind"], 0) + 1
    declared = dict(y["kind_census"])
    total = declared.pop("total")
    assert counted == declared, (counted, declared)
    assert total == sum(counted.values()) == len(y["checkers"])
    assert set(counted) == {"VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"}
''',
    "trajectory_rubrics_shape": '''
def test_trajectory_rubrics_shape():
    import json
    raw = open(os.path.join(BUNDLE, "tests", "rubrics.jsonl")).read()
    assert not raw.lstrip().startswith("["), "rubrics.jsonl must not be a wrapping array"
    ids = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        assert set(obj.keys()) == {"id", "rubric"}, obj.keys()
        assert isinstance(obj["id"], str) and isinstance(obj["rubric"], str)
        ids.append(obj["id"])
    assert len(ids) == len(set(ids))
    assert len(ids) >= 1
''',
    "seven_optimization_bindings_present": '''
def test_seven_optimization_bindings_present():
    t = _task_toml()
    o = t["optimization"]
    assert o["max_timeout"] == 6.0
    assert o["budget_hours"] == 0.12
    assert o["max_attempts"] == 50
    assert o["final_selection"] == "best"
    assert o["solver_egress"] == "setup-only"
    assert o["compute_envelope"] == "one-H100"
    assert o["reward_range"] == [0.0, 1.0]
    assert o["reward_clamp"] == "min(max(raw,0),1)"
    assert t["environment"]["gpus"] == 1
    assert t["environment"]["gpu_types"] == ["H100"]
''',
    "budget_hours_bound_by_role": '''
def test_budget_hours_bound_by_role():
    o = _task_toml()["optimization"]
    assert o["budget_hours"] != 6
    assert o["budget_hours"] < o["max_timeout"]
    assert abs(o["budget_hours"] * o["max_attempts"] - o["max_timeout"]) < 1e-9
''',
    "reward_clamp_is_the_bound_formula": '''
def test_reward_clamp_is_the_bound_formula():
    C = _checkers_module()
    for raw in (-2.0, -0.001, 0.0, 0.37, 1.0, 4.2):
        good = {"raw": raw, "score": min(max(raw, 0.0), 1.0), "anchors": {}, "agent_metric": None}
        ok, reason, _ = C.check_reward_is_clamped_unit_float(good)
        assert ok, (raw, reason)
    bad = {"raw": 0.25, "score": 0.9, "anchors": {}, "agent_metric": None}
    ok, reason, _ = C.check_reward_is_clamped_unit_float(bad)
    assert not ok and reason == "score_not_clamped_raw"
    out = {"raw": 0.25, "score": 1.7, "anchors": {}, "agent_metric": None}
    ok, reason, _ = C.check_reward_is_clamped_unit_float(out)
    assert not ok and reason == "score_outside_unit_interval"
''',
    "substrate_manifest_covers_substrate": '''
def test_substrate_manifest_covers_substrate():
    import hashlib, json
    d = os.path.join(BUNDLE, "environment", "substrate")
    pinned = json.load(open(os.path.join(d, "SUBSTRATE_MANIFEST.json")))["files"]
    on_disk = sorted(n for n in os.listdir(d) if n.endswith(".py"))
    assert sorted(pinned) == on_disk, (sorted(pinned), on_disk)
    for name, digest in pinned.items():
        got = hashlib.sha256(open(os.path.join(d, name), "rb").read()).hexdigest()
        assert got == digest, name
''',
    "smoke_fixture_digest_matches": '''
def test_smoke_fixture_digest_matches():
    import hashlib, json
    f = os.path.join(BUNDLE, "environment", "fixtures")
    meta = json.load(open(os.path.join(f, "FIXTURES.json")))
    blob = open(os.path.join(f, meta["fixture"]), "rb").read()
    assert len(blob) == meta["bytes"]
    assert hashlib.sha256(blob).hexdigest() == meta["sha256"]
''',
    "ablation_scope_is_closed_and_shared": '''
def test_ablation_scope_is_closed_and_shared():
    C = _substrate_config()
    assert len(C.ABLATED_MODULE_CLASSES) >= 6
    assert len(C.ABLATED_FUNCTIONALS) >= 5
    assert "RMSNorm" in C.ABLATED_MODULE_CLASSES
    assert "layer_norm" in C.ABLATED_FUNCTIONALS
    assert set(C.FORBIDDEN_PHASES) == {"forward", "backward", "eval"}
    src = open(os.path.join(BUNDLE, "environment", "substrate", "model.py")).read()
    assert 'norm_mode == "rmsnorm"' in src
    assert 'norm_mode == "none"' in src
''',
    "every_checker_has_a_rejecting_control": '''
def test_every_checker_has_a_rejecting_control():
    controls = _controls_module()
    targets = {t for t, _, _, _ in controls.DEFECTS}
    registry = {n for n, _, _ in _checkers_module().REGISTRY}
    assert registry <= targets, registry - targets
''',
    "emitted_markdown_prose_rule": '''
def test_emitted_markdown_prose_rule():
    offenders = []
    for root, dirs, files in os.walk(BUNDLE):
        dirs[:] = [d for d in dirs if d not in ("trajectories", "__pycache__")]
        for name in files:
            if not name.endswith(".md"):
                continue
            p = os.path.join(root, name)
            text = open(p, encoding="utf-8").read()
            if "\\u2014" in text:
                offenders.append((p, "em-dash"))
            in_fence = False
            prev_kind = None
            for line in text.split("\\n"):
                if line.startswith("```"):
                    in_fence = not in_fence
                    prev_kind = None
                    continue
                if in_fence:
                    continue
                kind = "blank" if not line.strip() else "prose"
                if kind == "prose" and prev_kind == "prose":
                    if not (line.lstrip().startswith(("|", "-", "*", "#", ">", "<!--"))
                            or line.lstrip()[:2].rstrip(".").isdigit()):
                        offenders.append((p, "intra-paragraph newline: " + line[:60]))
                prev_kind = kind
    assert not offenders, offenders
''',
}


def render_test_output(g) -> str:
    items = g["rubric_items"]["items"]
    compiled = [i for i in items if i["mode"] == "compiled"]
    L = []
    A = L.append
    A('"""Deterministic tests compiled from solution/rubrics.json.')
    A("")
    A("Advisory to the reward: the reward comes from tests/grade.py and from")
    A("nothing here. These tests are the compiled half of the reference rubric and")
    A("they assert structure over frozen bundle bytes, never criterion prose and")
    A("never a reference answer.")
    A('"""')
    A("")
    A(BANNER_PY)
    A("")
    A("import functools")
    A("import os")
    A("import sys")
    A("")
    A("HERE = os.path.dirname(os.path.abspath(__file__))")
    A("BUNDLE = os.path.dirname(HERE)")
    A("")
    A("")
    A("@functools.lru_cache(maxsize=1)")
    A("def _checkers_yaml():")
    A("    import yaml")
    A('    with open(os.path.join(BUNDLE, "tests", "checkers.yaml")) as f:')
    A("        return yaml.safe_load(f)")
    A("")
    A("")
    A("@functools.lru_cache(maxsize=1)")
    A("def _task_toml():")
    A("    import tomllib")
    A('    with open(os.path.join(BUNDLE, "task.toml"), "rb") as f:')
    A("        return tomllib.load(f)")
    A("")
    A("")
    A("@functools.lru_cache(maxsize=1)")
    A("def _checkers_module():")
    A('    sys.path.insert(0, os.path.join(BUNDLE, "tests"))')
    A("    import checkers")
    A("    return checkers")
    A("")
    A("")
    A("@functools.lru_cache(maxsize=1)")
    A("def _controls_module():")
    A('    sys.path.insert(0, os.path.join(BUNDLE, "tests"))')
    A("    import controls")
    A("    return controls")
    A("")
    A("")
    A("@functools.lru_cache(maxsize=1)")
    A("def _substrate_config():")
    A('    sys.path.insert(0, os.path.join(BUNDLE, "environment"))')
    A("    from substrate import config")
    A("    return config")
    A("")
    for it in compiled:
        body = COMPILED_TESTS[it["id"]]
        A("")
        A(body.strip("\n"))
        A("")
    return "\n".join(L).rstrip("\n") + "\n"


TARGETS = [
    ("solution/TRUTH.md", render_truth),
    ("solution/rubrics.json", render_rubrics),
    ("tests/test_output.py", render_test_output),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    g = load()
    drift = []
    for rel, fn in TARGETS:
        path = os.path.join(BUNDLE, rel)
        new = fn(g)
        if args.check:
            old = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
            status = "SAME" if old == new else "DRIFT"
            print(f"{status}  {rel}  sha256={hashlib.sha256(new.encode()).hexdigest()[:16]}")
            if old != new:
                drift.append(rel)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            print(f"WROTE {rel}  sha256={hashlib.sha256(new.encode()).hexdigest()[:16]}")
    if drift:
        print("DRIFT DETECTED: " + ", ".join(drift))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

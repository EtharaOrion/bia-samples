"""PRIVATE. Derive every generated canonical artifact of this bundle from
solution/grounding.yaml, and reproduce the shipped fixture from the pinned recipe.

Modes:

  --check              regenerate the generated artifacts into memory and compare them
                       against the bytes on disk, exiting non zero on any difference.
  --write              write the generated artifacts to disk.
  --verify-fixture     regenerate the shipped checkpoint from the pinned seed and thread
                       count and compare it against the shipped blob, reporting both a
                       byte comparison and an exact structural comparison of the poison
                       transform. This is CPU only and it allocates no CUDA memory.
  --content-hash       print the canonical content hash and the uuid5 derived from it,
                       using exactly the recipe solution/provenance.yaml records.

The generated artifacts are solution/TRUTH.md, solution/rubrics.json,
tests/rubrics.jsonl, tests/checkers.yaml, tests/checkers/registry.py,
tests/frozen_digests.json and tests/test_output.py. The compiled rubric lane, the
checker declaration, the executable checker registry and the compiled tests all
descend from the single `checkers` list in solution/grounding.yaml, so the identifier
set they carry is one set by construction rather than by reconciliation.

Every string this script emits is either a literal frozen in solution/grounding.yaml
or a value computed from bundle bytes. It invokes no model, no network, no clock, no
locale and no random source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)

FORGE_TASK_NAMESPACE = uuid.UUID("c53e8f3b-526f-52c0-a04e-89e2269b237d")
HASH_EXCLUDED_DIRS = ("trajectories",)


def load_grounding():
    import yaml

    with open(os.path.join(HERE, "grounding.yaml")) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Canonical content hash. This recipe is recorded verbatim in provenance.yaml.
# ---------------------------------------------------------------------------


def iter_hashed_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in HASH_EXCLUDED_DIRS and d != "__pycache__")
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if rel.split(os.sep)[0] in HASH_EXCLUDED_DIRS:
                continue
            yield rel.replace(os.sep, "/"), full


def canonical_content_hash(root):
    h = hashlib.sha256()
    for rel, full in sorted(iter_hashed_files(root)):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        with open(full, "rb") as f:
            body = f.read()
        h.update(str(len(body)).encode("ascii"))
        h.update(b"\x00")
        h.update(body)
        h.update(b"\x00")
    return h.hexdigest()


def derived_uuid(root):
    digest = canonical_content_hash(root)
    return digest, str(uuid.uuid5(FORGE_TASK_NAMESPACE, digest))


# ---------------------------------------------------------------------------
# Generated artifacts
# ---------------------------------------------------------------------------

KINDS = ("VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE")

BANNER = "GENERATED FROM solution/grounding.yaml BY solution/recompute.py. DO NOT HAND-EDIT."

HARNESS_CODE = {
    "run_s08.py": "environment/runner/run_s08.py",
    "s08_core.py": "environment/runner/s08_core.py",
    "recover_worker.py": "environment/runner/recover_worker.py",
}

DIGEST_PINNED_PROFILES = ("scaled",)

AGGREGATION_MODES = ("weighted_mean", "all_pass", "any_pass", "threshold", "required_pass")

REASON_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def flow(text):
    return " ".join(str(text).split())


def carrier(c):
    """Split `implemented_by` into the bundle-relative carrier path and its selector.

    Both halves are contract fields seed/forge/verifier.py resolves independently:
    the path must sit in the reachability closure of tests/test.sh, and the selector
    must appear literally in that file. Splitting one grounding value rather than
    authoring two keeps the manifest and the code that grades from drifting apart.
    """
    path, _, selector = flow(c["implemented_by"]).partition(" ")
    return path, selector


def gen_rubrics_jsonl(g):
    lines = [
        json.dumps({"id": item["id"], "rubric": flow(item["criterion"])}, sort_keys=False)
        for item in g["rubric_items"]
    ]
    return "\n".join(lines) + "\n"


def rubric_items(g):
    compiled = [
        {
            "id": c["id"],
            "dimension": "outcome",
            "weight": float(g["rubric_policy"]["item_weight"]),
            "evaluation_target": "verifier_outcomes",
            "criterion": flow(c["criterion"]),
            "judgment": "decided by tests/grade.py, recorded in /logs/verifier/outcomes.json",
            "evidence": ["verifier_outcomes"],
            "mode": "compiled",
            "kind": c["kind"],
        }
        for c in g["checkers"]
    ]
    judged = [
        {
            "id": item["id"],
            "dimension": item["dimension"],
            "weight": float(g["rubric_policy"]["item_weight"]),
            "evaluation_target": "agent_trajectory",
            "criterion": flow(item["criterion"]),
            "judgment": flow(item["judgment"]),
            "evidence": ["agent_trajectory"],
            "mode": "judged",
            "residue": flow(item["residue"]),
        }
        for item in g["rubric_items"]
    ]
    return compiled + judged


def compiled_share(g):
    items = rubric_items(g)
    total = sum(i["weight"] for i in items)
    compiled = sum(i["weight"] for i in items if i["mode"] == "compiled")
    return compiled, total, compiled / total


def gen_rubrics_json(g):
    items = rubric_items(g)
    compiled, total, share = compiled_share(g)
    floor = float(g["rubric_policy"]["compilation_floor"])
    doc = {
        "schema": "bia.rubric/v1",
        "slot": g["slot"],
        "evaluation_targets": ["verifier_outcomes", "agent_trajectory"],
        "compilation_floor": floor,
        "compiled_weight_share": round(share, 6),
        "compilation_floor_met": share >= floor,
        "compiled_item_count": sum(1 for i in items if i["mode"] == "compiled"),
        "judged_item_count": sum(1 for i in items if i["mode"] == "judged"),
        "compilation_floor_note": flow(g["rubric_policy"]["weight_rule"]) + " " + flow(g["rubric_policy"]["lane_rule"]),
        "items": items,
    }
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def gen_registry(g):
    lines = [
        '"""' + BANNER + '"""',
        "",
        "from __future__ import annotations",
        "",
        f"KINDS = {KINDS!r}",
        "",
        "CHECKERS = {",
    ]
    for c in g["checkers"]:
        path, selector = carrier(c)
        lines.append(f"    {c['id']!r}: {{")
        lines.append(f"        \"kind\": {c['kind']!r},")
        lines.append(f"        \"zero_reason\": {c['zero_reason']!r},")
        lines.append(f"        \"reached_by\": {path!r},")
        lines.append(f"        \"selector\": {selector!r},")
        lines.append(f"        \"live_state_read\": {flow(c['live_state_read'])!r},")
        lines.append(f"        \"asserts\": {flow(c['asserts'])!r},")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def gen_checkers_yaml(g):
    published = g["reward_rule"]["published"]
    lines = [
        "# " + BANNER,
        "#",
        "# Every graded assertion reduces to exactly one of VALUE, EFFECT, ABSENCE,",
        "# INVARIANT, ORDERING, DIVERGENCE, and names the live-state read it traces to.",
        "# tests/checkers/registry.py is the executable twin of this file and both are",
        "# generated from the same source, so the declaration and the code that grades",
        "# cannot drift.",
        "#",
        "# Every checker is a hard gate. A single failure emits a score of exactly 0.0.",
        "",
        f"schema: {g['checker_manifest_schema']}",
        f"task_slot: {g['slot']}",
        "kinds: [" + ", ".join(KINDS) + "]",
        f"reward_path: {g['checker_reward_path']}",
        "",
        "aggregation:",
        f"  mode: {g['checker_aggregation_mode']}",
        "",
        "checkers:",
    ]
    for c in g["checkers"]:
        path, selector = carrier(c)
        lines.append(f"  - id: {c['id']}")
        lines.append(f"    statement: {json.dumps(flow(c['criterion']))}")
        lines.append(f"    reduction: {c['kind']}")
        lines.append(f"    kind: {c['kind']}")
        lines.append(f"    reached_by: {path}")
        lines.append(f"    selector: {selector}")
        lines.append(f"    zero_reason: {c['zero_reason']}")
        lines.append(f"    live_state_read: {json.dumps(flow(c['live_state_read']))}")
        lines.append(f"    asserts: {json.dumps(flow(c['asserts']))}")
        lines.append(f"    evidence: {c['evidence']}")
        lines.append("    on_failure: score 0.0")
        lines.append(f"    implemented_by: {json.dumps(flow(c['implemented_by']))}")
        if c.get("bounded_claim"):
            lines.append(f"    bounded_claim: {json.dumps(flow(c['bounded_claim']))}")
        lines.append("")
    lines.append("reward:")
    lines.append(f"  form: {json.dumps(published['form'])}")
    lines.append(f"  raw: {json.dumps(published['raw'])}")
    lines.append(f"  baseline_steps: {json.dumps(published['baseline_steps'])}")
    lines.append(f"  full_reward_at_steps: {json.dumps(published['full_reward_at_steps'])}")
    lines.append(f"  range: {published['range']}")
    return "\n".join(lines) + "\n"


def gen_test_output(g):
    lines = [
        '"""' + BANNER,
        "",
        "One test per mode=compiled rubric item, carrying that item's identifier. The",
        "set of test names is the set of compiled identifiers, which is how completeness",
        "is proven. Each test asserts the one relation the item implies: the verifier",
        "recorded that outcome as true for the graded run.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "import os",
        "",
        "import pytest",
        "",
        'LOG_DIR = os.environ.get("LOG_DIR", "/logs/verifier")',
        'OUTCOMES_PATH = os.environ.get("S08_OUTCOMES", os.path.join(LOG_DIR, "outcomes.json"))',
        "",
        "",
        "def _outcome(name):",
        "    if not os.path.exists(OUTCOMES_PATH):",
        '        pytest.skip("no graded run present")',
        "    with open(OUTCOMES_PATH) as f:",
        "        outcomes = json.load(f)",
        "    assert name in outcomes, name",
        "    return outcomes[name]",
        "",
    ]
    for c in g["checkers"]:
        lines.append("")
        lines.append(f"def test_{c['id']}():")
        lines.append(f"    assert _outcome({c['id']!r}) is True")
        lines.append("")
    return "\n".join(lines)


def gen_frozen_digests(g):
    doc = {"_source": BANNER, "harness_code": {}, "profiles": {}}
    for name, rel in sorted(HARNESS_CODE.items()):
        doc["harness_code"][name] = file_digest(os.path.join(BUNDLE, rel))
    for profile in DIGEST_PINNED_PROFILES:
        fixtures = os.path.join(BUNDLE, "environment", "fixtures")
        doc["profiles"][profile] = {
            "checkpoint": file_digest(os.path.join(fixtures, f"ckpt_s08_{profile}.pt")),
            "manifest": file_digest(os.path.join(fixtures, f"manifest_{profile}.json")),
        }
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_truth(g):
    p = g["scaled_operating_point"]
    lines = []
    a = lines.append
    a("# S08 ground truth, private")
    a("")
    a("GENERATED FROM solution/grounding.yaml BY solution/recompute.py. DO NOT HAND-EDIT.")
    a("")
    a("## What this slot freezes and what it leaves free")
    a("")
    a(f"Frozen: {g['frozen_surface']}. Free: {g['free_surface']}. The metric is {g['metric']}.")
    a("")
    a("## The poison, in full")
    a("")
    a("Four components ship inside the optimizer state of the frozen checkpoint. None of them touches the weights, which is the whole point: the validation loss at the moment of the checkpoint is exactly the loss the healthy pretraining curve ends on, so the curve the agent is handed is a true record that carries no information about the damage.")
    a("")
    a("| Id | Applies to | Transform | Effect |")
    a("|---|---|---|---|")
    for item in g["fixture"]["poison"]:
        a(f"| {item['id']} | {item['applies_to']} | {item['transform']} | {item['effect']} |")
    a("")
    a("## Reference recovery")
    a("")
    a("solution/reference_recovery.py is the oracle. It never adopts the checkpoint's stored hyperparameters, it measures the live per-tensor gradient second moment with a bounded number of probe passes that are charged against the same budget the reward counts, it rescales the stored second moment per tensor so its mean matches the measured gradients, it clips the first moment against the repaired preconditioner, it writes a truthful step counter taken from the manifest, and it supplies its own warmup with a late cosine tail keyed to the recovery step index rather than to the counter carried in the checkpoint.")
    a("")
    a("## Scaled operating point")
    a("")
    a("| Quantity | Value |")
    a("|---|---|")
    for key in (
        "parameters", "vocab_size", "sequence_length", "layers", "model_width",
        "batch_sequences", "tokens_per_step", "checkpoint_step", "step_ceiling_per_seed",
        "graded_seeds", "harness_control_runs", "probe_batches_max",
        "total_graded_steps_per_attempt",
    ):
        a(f"| {key} | {p[key]} |")
    a(f"| wallclock at graded scale | {p['wallclock_at_graded_scale']} |")
    a("")
    a("## Golden trajectory, and the checker each step binds")
    a("")
    a("The reference run reaches every step below in this order. The right column is the checker identifier the step satisfies, and the set of identifiers named here is exactly the compiled checker set the bundle commits.")
    a("")
    a("| # | Step the reference takes | Established state | Checker satisfied |")
    a("|---|---|---|---|")
    for i, step in enumerate(g["golden_trajectory"], start=1):
        a(f"| {i} | {flow(step['action'])} | {flow(step['establishes'])} | `{step['checker']}` |")
    a("")
    a("## Rejected routes")
    a("")
    a("Each route is a known-wrong control in solution/negative_controls.py, and the checker named is the one whose rejection is measured rather than asserted.")
    a("")
    a("| Control | Route | Rejected by |")
    a("|---|---|---|")
    for control in g["negative_controls"]:
        a(f"| `{control['id']}` | {flow(control['statement'])} | {', '.join('`' + c + '`' for c in control['rejected_by'])} |")
    a("")
    a("## Rubric compilation")
    a("")
    compiled, total, share = compiled_share(g)
    floor = float(g["rubric_policy"]["compilation_floor"])
    a(f"| Quantity | Value |")
    a("|---|---|")
    a(f"| Compiled items | {sum(1 for i in rubric_items(g) if i['mode'] == 'compiled')} |")
    a(f"| Judged items | {sum(1 for i in rubric_items(g) if i['mode'] == 'judged')} |")
    a(f"| Item weight | {g['rubric_policy']['item_weight']}, uniform |")
    a(f"| Compiled weight share | {round(share, 6)} |")
    a(f"| Compilation floor | {floor} |")
    a(f"| Floor met | {share >= floor} |")
    a("")
    a(flow(g["rubric_policy"]["weight_rule"]))
    a("")
    a(flow(g["rubric_policy"]["lane_rule"]))
    a("")
    a("## Defeat mechanism")
    a("")
    a(g["defeat_mechanism"]["statement"].strip().replace("\n", " "))
    a("")
    a(f"Status: {g['defeat_mechanism']['status']}. Measured by: {g['defeat_mechanism']['measured_by']}. The only admissible evidence is {g['defeat_mechanism']['admissible_evidence']}.")
    a("")
    a("## Known gaps")
    a("")
    a("| Id | Statement | Caps at |")
    a("|---|---|---|")
    for gap in g["known_gaps"]:
        a(f"| {gap['id']} | {gap['statement'].strip().replace(chr(10), ' ')} | {gap['caps_at']} |")
    a("")
    a("## Disposition")
    a("")
    a(f"This bundle ships at {g['disposition']}, which is the correct terminal state for a locally verified bundle carrying no external signed pilot. No difficulty tier is claimed, no pass probability is authored, and no self-solve result is offered as evidence.")
    a("")
    return "\n".join(lines)


GENERATED = {
    "tests/rubrics.jsonl": gen_rubrics_jsonl,
    "tests/checkers.yaml": gen_checkers_yaml,
    "tests/checkers/registry.py": gen_registry,
    "tests/test_output.py": gen_test_output,
    "tests/frozen_digests.json": gen_frozen_digests,
    "solution/rubrics.json": gen_rubrics_json,
    "solution/TRUTH.md": gen_truth,
}


def _reward_binding_problems(g):
    """The reward path must be bound here and actually written by the entry point."""
    problems = []
    reward_path = str(g.get("checker_reward_path", "")).strip()
    if not reward_path:
        problems.append("no checker_reward_path, so the manifest would bind no reward path")
    else:
        entry = os.path.join(BUNDLE, "tests", "test.sh")
        if reward_path not in open(entry).read():
            problems.append(f"tests/test.sh never writes the bound reward path {reward_path}")
    mode = str(g.get("checker_aggregation_mode", "")).strip()
    if mode not in AGGREGATION_MODES:
        problems.append(f"checker_aggregation_mode {mode!r} is not one of {list(AGGREGATION_MODES)}")
    if mode == "threshold" and g.get("checker_aggregation_threshold") is None:
        problems.append("threshold aggregation with no contract-bound threshold")
    return problems


def _carrier_problems(g):
    """Every checker names a real carrier, a real selector and an emitted zero reason.

    seed/forge/verifier.py resolves all three against frozen bytes, so a value that
    reads plausibly but does not appear in the code it names is caught here rather
    than in a manifest that already shipped.
    """
    problems = []
    seen = {}
    for c in g["checkers"]:
        path, selector = carrier(c)
        full = os.path.join(BUNDLE, path)
        if not os.path.isfile(full):
            problems.append(f"{c['id']} names carrier {path}, which is not a file in this bundle")
        elif f"def {selector}(" not in open(full).read():
            problems.append(f"{c['id']} names selector {selector}, which {path} does not define")
        reason = str(c.get("zero_reason", "")).strip()
        if not REASON_PATTERN.match(reason):
            problems.append(f"{c['id']} zero_reason {reason!r} is not a lowercase kebab reason code")
        elif reason in seen:
            problems.append(f"{c['id']} reuses the zero_reason of {seen[reason]}, so a zero would be ambiguous")
        else:
            seen[reason] = c["id"]
    return problems


def validate(g):
    """Refuse to generate from a grounding file whose identifier sets do not close.

    Every obligation below is one FORGE emission rule, and each is cheaper to catch
    here than in a downstream artifact that already shipped.
    """
    checkers = [c["id"] for c in g["checkers"]]
    judged = [item["id"] for item in g["rubric_items"]]
    problems = []
    if not str(g.get("checker_manifest_schema", "")).strip():
        problems.append("no checker_manifest_schema, so the checker manifest would declare no schema")
    problems.extend(_reward_binding_problems(g))
    problems.extend(_carrier_problems(g))
    if len(set(checkers)) != len(checkers):
        problems.append("duplicate checker id")
    if len(set(judged)) != len(judged):
        problems.append("duplicate judged rubric id")
    overlap = sorted(set(checkers) & set(judged))
    if overlap:
        problems.append(f"ids in both lanes: {overlap}")
    for c in g["checkers"]:
        if c["kind"] not in KINDS:
            problems.append(f"{c['id']} reduces to {c['kind']!r}, which is not one of the six kinds")
    uncovered = sorted(set(KINDS) - {c["kind"] for c in g["checkers"]})
    if uncovered:
        problems.append(f"kinds with no checker: {uncovered}")
    for item in g["rubric_items"]:
        if not str(item.get("residue", "")).strip():
            problems.append(f"judged item {item['id']} names no residue")
    walked = [step["checker"] for step in g["golden_trajectory"]]
    if sorted(walked) != sorted(checkers):
        problems.append(
            "TRUTH.md golden trajectory does not reconcile with the checker set; "
            f"only in trajectory: {sorted(set(walked) - set(checkers))}; "
            f"only in checkers: {sorted(set(checkers) - set(walked))}"
        )
    targets = {t for control in g["negative_controls"] for t in control["rejected_by"]}
    stray = sorted(targets - set(checkers))
    if stray:
        problems.append(f"controls naming no checker: {stray}")
    unmeasured = sorted(set(checkers) - targets)
    if unmeasured:
        problems.append(f"checkers with no known-wrong control to reject: {unmeasured}")
    _, _, share = compiled_share(g)
    floor = float(g["rubric_policy"]["compilation_floor"])
    if share < floor:
        problems.append(f"compiled weight share {share} is below the floor {floor}")
    if problems:
        raise SystemExit("grounding.yaml does not close:\n  " + "\n  ".join(problems))


def run_generated(write: bool) -> int:
    g = load_grounding()
    validate(g)
    bad = 0
    for rel, fn in sorted(GENERATED.items()):
        want = fn(g)
        path = os.path.join(BUNDLE, rel)
        if write:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(want)
            print(f"wrote {rel}")
            continue
        have = open(path).read() if os.path.exists(path) else None
        if have != want:
            print(f"MISMATCH {rel}")
            bad += 1
        else:
            print(f"ok       {rel}")
    return bad


def verify_fixture(profile: str) -> int:
    import shutil
    import tempfile

    import torch

    sys.path.insert(0, os.path.join(BUNDLE, "environment", "runner"))
    import s08_core as core

    import make_fixture

    fixtures = os.path.join(BUNDLE, "environment", "fixtures")
    with open(os.path.join(fixtures, f"manifest_{profile}.json")) as f:
        shipped_manifest = json.load(f)
    tmp = tempfile.mkdtemp(prefix="bia-s08-verify-")
    try:
        data_dir = os.path.join(tmp, "data", profile)
        out = os.path.join(tmp, "fixtures")
        regenerated = make_fixture.build(
            profile, out, data_dir, int(shipped_manifest["generator_threads"])
        )
        byte_identical = regenerated["fixture_sha256"] == shipped_manifest["fixture_sha256"]
        print(json.dumps({
            "byte_identical": byte_identical,
            "shipped_sha256": shipped_manifest["fixture_sha256"],
            "regenerated_sha256": regenerated["fixture_sha256"],
            "shipped_bytes": shipped_manifest["fixture_bytes"],
            "regenerated_bytes": regenerated["fixture_bytes"],
            "shipped_torch": shipped_manifest["torch_version"],
            "regenerated_torch": regenerated["torch_version"],
        }, indent=1, sort_keys=True))

        # Structural check of the poison transform. This is exact regardless of any
        # last-bit float difference between builds, because it reads the ratio the
        # transform imposes rather than the absolute tensor values.
        ck = torch.load(os.path.join(fixtures, f"ckpt_s08_{profile}.pt"), map_location="cpu", weights_only=False)
        names = ck["param_names"]
        report = {"p1_ratios": [], "p2_ratios": [], "p3_steps": set(), "p4_weight_decay": []}
        rk = torch.load(os.path.join(out, f"ckpt_s08_{profile}.pt"), map_location="cpu", weights_only=False)
        for idx, st in ck["optimizer"]["state"].items():
            name = names[int(idx)]
            rst = rk["optimizer"]["state"][idx]
            if torch.is_tensor(st.get("exp_avg_sq")) and st["exp_avg_sq"].dim() >= 2:
                a = float(st["exp_avg_sq"].double().mean())
                b = float(rst["exp_avg_sq"].double().mean())
                key = "p2_ratios" if name.endswith(make_fixture.AMPLIFIED_SUFFIX) else "p1_ratios"
                report[key].append(round(a / b, 9) if b else None)
            s = st.get("step")
            report["p3_steps"].add(float(s) if not torch.is_tensor(s) else float(s.item()))
        for grp in ck["optimizer"]["param_groups"]:
            report["p4_weight_decay"].append(float(grp["weight_decay"]))
        report["p3_steps"] = sorted(report["p3_steps"])
        report["p3_expected"] = make_fixture.POISON["p3_recorded_step"]
        report["p4_expected"] = make_fixture.POISON["p4_param_group_weight_decay"]
        report["structural_match"] = (
            all(r == 1.0 for r in report["p1_ratios"] if r is not None)
            and all(r == 1.0 for r in report["p2_ratios"] if r is not None)
            and report["p3_steps"] == [report["p3_expected"]]
            and all(w == report["p4_expected"] for w in report["p4_weight_decay"])
        )
        print(json.dumps(report, indent=1, sort_keys=True, default=str))
        return 0 if (byte_identical or report["structural_match"]) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify-fixture", action="store_true")
    ap.add_argument("--content-hash", action="store_true")
    ap.add_argument("--profile", default="scaled")
    ap.add_argument("--root", default=BUNDLE)
    args = ap.parse_args(argv)

    if not any((args.check, args.write, args.verify_fixture, args.content_hash)):
        args.check = True

    rc = 0
    if args.write:
        run_generated(write=True)
    if args.check:
        rc |= 1 if run_generated(write=False) else 0
    if args.content_hash:
        digest, uid = derived_uuid(args.root)
        print(json.dumps({"canonical_content_hash": digest, "uuid": uid}, indent=1))
    if args.verify_fixture:
        rc |= verify_fixture(args.profile)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

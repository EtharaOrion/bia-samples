#!/usr/bin/env python3
"""Derive every generated canonical artifact of bia slot S07 from solution/grounding.yaml.

Generated: tests/checkers/format_digest.json, tests/checkers/frozen_bytes.json,
tests/checkers.yaml, tests/rubrics.jsonl, tests/test_output.py, solution/TRUTH.md and
solution/rubrics.json. Run it twice over frozen inputs and the bytes are identical, which is
what makes the generated artifacts recomputable rather than authored.

The rubric comes from the `rubric` block of solution/grounding.yaml and from nowhere else, as
FORGE 10f requires. Every criterion string is read as a frozen literal of that file. Nothing
here invokes a model, a network, a clock, a locale or a random source.

    python3 solution/recompute.py [--check]

--check regenerates into memory and exits non-zero if any on-disk artifact differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import textwrap
from typing import Any, Dict, List

BUNDLE = pathlib.Path(__file__).resolve().parent.parent
GROUNDING = BUNDLE / "solution" / "grounding.yaml"

BANNER = "GENERATED SECTION. DO NOT HAND-EDIT. Regenerate with solution/recompute.py."


FORGE_TASK_NAMESPACE = "c53e8f3b-526f-52c0-a04e-89e2269b237d"

HASH_EXCLUDED_PREFIXES = ("trajectories/",)
HASH_EXCLUDED_PATHS = ("solution/provenance.yaml",)
HASH_EXCLUDED_NAMES = ("__pycache__", ".DS_Store", ".pyc")


def hash_members() -> List[pathlib.Path]:
    """Every file inside the canonical hash domain, in byte-wise relative path order."""
    out = []
    for p in BUNDLE.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(BUNDLE).as_posix()
        if any(rel.startswith(x) for x in HASH_EXCLUDED_PREFIXES):
            continue
        if rel in HASH_EXCLUDED_PATHS:
            continue
        if any(x in rel for x in HASH_EXCLUDED_NAMES):
            continue
        out.append(rel)
    return sorted(out, key=lambda r: r.encode("utf-8"))


def canonical_content_hash() -> str:
    """SHA-256 over length-prefixed (path, bytes) pairs in sorted path order.

    Framing each member as path bytes, a NUL, an 8 byte big-endian length, then the content
    makes the digest injective over the file set: no rename or content shift can produce the
    same stream, which a bare concatenation would allow.
    """
    h = hashlib.sha256()
    for rel in hash_members():
        data = (BUNDLE / rel).read_bytes()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def bundle_uuid() -> str:
    import uuid as _uuid

    return str(_uuid.uuid5(_uuid.UUID(FORGE_TASK_NAMESPACE), canonical_content_hash()))


def load_grounding() -> Dict[str, Any]:
    """Minimal reader for the flat scalar fields TRUTH.md needs. No yaml dependency."""
    text = GROUNDING.read_text(encoding="utf-8")
    out: Dict[str, Any] = {}
    for key in (
        "slot",
        "slot_name",
        "task_name",
        "disposition",
        "fixture_id",
        "per_attempt_budget_hours",
        "per_attempt_budget_minutes",
        "session_bound_hours",
        "attempts_per_session",
        "cohort_validity_end",
        "measured_reference_score",
        "wallclock_seconds_one_h100_contended",
        "measured_baseline_metric",
        "measured_target_metric",
        "measured_agent_metric",
        "chunk_absmax_spread",
        "reference_zero_fraction",
        "chain_zero_fraction",
    ):
        m = re.search(r"^\s*%s:\s*(.+)$" % re.escape(key), text, re.M)
        if m:
            out[key] = m.group(1).strip()
    return out


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def adequacy_table() -> List[Dict[str, Any]]:
    """Read the measured adequacy table out of grounding.yaml. Never hardcode it here."""
    text = GROUNDING.read_text(encoding="utf-8")
    m = re.search(r"adequacy_table_json:\s*>-\s*\n((?:\s{6,}.*\n)+)", text)
    if not m:
        raise RuntimeError("grounding.yaml carries no adequacy_table_json block")
    return json.loads(" ".join(line.strip() for line in m.group(1).splitlines()))


def adequacy_anchors() -> List[str]:
    text = GROUNDING.read_text(encoding="utf-8")
    m = re.search(r"anchors:\s*\{baseline:\s*([0-9.]+),\s*target:\s*([0-9.]+)\}", text)
    if not m:
        raise RuntimeError("grounding.yaml carries no adequacy anchors")
    return [m.group(1), m.group(2)]


def _jsonl_block(key: str) -> List[Dict[str, Any]]:
    """Read a `<key>: |` literal block out of grounding.yaml as one JSON object per line.

    The block ends at the first line that is neither blank nor indented past the key. This is
    the same frozen-literal pattern grounding.yaml already uses for adequacy_table_json, and it
    keeps the generator free of a YAML dependency so it re-runs on any clean host.
    """
    lines = GROUNDING.read_text(encoding="utf-8").splitlines()
    start = None
    indent = 0
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)%s:\s*\|\s*$" % re.escape(key), line)
        if m:
            start, indent = i + 1, len(m.group(1))
            break
    if start is None:
        raise RuntimeError("grounding.yaml carries no %s block" % key)

    out: List[Dict[str, Any]] = []
    for line in lines[start:]:
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        out.append(json.loads(line.strip()))
    if not out:
        raise RuntimeError("grounding.yaml %s block is empty" % key)
    return out


def rubric_scalar(key: str) -> str:
    text = GROUNDING.read_text(encoding="utf-8")
    m = re.search(r"^\s{2}%s:\s*(\S+)\s*$" % re.escape(key), text, re.M)
    if not m:
        raise RuntimeError("grounding.yaml rubric block carries no %s" % key)
    return m.group(1)


def grounding_scalar(key: str) -> str:
    text = GROUNDING.read_text(encoding="utf-8")
    m = re.search(r"^%s:\s*(\S+)\s*$" % re.escape(key), text, re.M)
    if not m:
        raise RuntimeError("grounding.yaml carries no %s" % key)
    return m.group(1)


def checker_manifest_schema() -> str:
    return grounding_scalar("checker_manifest_schema")


def checker_carrier() -> str:
    return grounding_scalar("checker_carrier")


def checker_reward_path() -> str:
    return grounding_scalar("checker_reward_path")


def checker_aggregation_mode() -> str:
    return grounding_scalar("checker_aggregation_mode")


def selector_for(ident: str) -> str:
    return "chk_" + ident


def compiled_items() -> List[Dict[str, Any]]:
    return _jsonl_block("compiled_items_jsonl")


def judged_items() -> List[Dict[str, Any]]:
    return _jsonl_block("judged_items_jsonl")


def checker_kinds() -> Dict[str, str]:
    return {it["id"]: it["kind"] for it in compiled_items()}


SIX_KINDS = ("VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE")

AGGREGATION_MODES = ("weighted_mean", "all_pass", "any_pass", "threshold", "required_pass")

FROZEN_BYTES_FILES = (
    "harness.py",
    "run_attempt.py",
    "policy_worker.py",
    "baseline_recipe.py",
    "operating_point.json",
    "corpus/moby_dick.txt",
)


def validate_rubric() -> None:
    """Fail closed on every rubric defect FORGE 7e and 10f make BLOCK:INVALID_TASK.

    Checked here rather than downstream because a generator that emits a defective rubric and
    reports success is worse than one that refuses: the drift check would then confirm the
    defect as canonical.
    """
    comp, judged = compiled_items(), judged_items()
    ids = [it["id"] for it in comp] + [it["id"] for it in judged]
    if len(set(ids)) != len(ids):
        raise RuntimeError("rubric item identifiers are not unique")

    vocab = set(re.findall(r"^\s{4}-\s*(\S.*?)\s*$", _targets_block(), re.M))
    for it in comp + judged:
        for e in it.get("evidence", ["agent trajectory"]):
            if e not in vocab:
                raise RuntimeError("item %s names evidence %r outside the closed vocabulary" % (it["id"], e))
    for it in comp:
        if it["kind"] not in SIX_KINDS:
            raise RuntimeError("compiled item %s reduces to %r, not one of the six" % (it["id"], it["kind"]))
    for it in judged:
        if not str(it.get("residue", "")).strip():
            raise RuntimeError("judged item %s names no residue, which makes it decorative" % it["id"])

    floor = float(rubric_scalar("compilation_floor"))
    share = len(comp) / float(len(comp) + len(judged))
    if share < floor:
        raise RuntimeError(
            "compiled weight share %.4f is below the compilation floor %.4f" % (share, floor)
        )

    declared = set(re.findall(r'^\s*\("([a-z0-9_]+)",\s*chk_', (BUNDLE / "tests" / "grade.py").read_text(encoding="utf-8"), re.M))
    if declared != {it["id"] for it in comp}:
        raise RuntimeError(
            "tests/grade.py CHECKERS %s do not match the compiled rubric ids %s"
            % (sorted(declared), sorted(it["id"] for it in comp))
        )

    validate_checker_manifest_bindings(comp)


REASON_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

ENTRY_POINT = "tests/test.sh"


def reachable_from_entry_point() -> Dict[str, str]:
    """The closure seed/forge/verifier.py computes: tests/ files the entry point names, transitively.

    Reproduced here so the generator refuses a manifest whose carrier the verifier would find
    unreachable, instead of emitting one and letting the drift surface downstream as a BLOCK.
    """
    entry = BUNDLE / ENTRY_POINT
    pool = [p for p in sorted((BUNDLE / "tests").rglob("*")) if p.is_file()]
    seen, queue = {entry}, [entry]
    while queue:
        text = queue.pop().read_text(encoding="utf-8", errors="replace")
        for cand in pool:
            if cand in seen:
                continue
            named = _names(text, cand.name) or (cand.suffix == ".py" and _names(text, cand.stem))
            if named:
                seen.add(cand)
                queue.append(cand)
    return {
        p.relative_to(BUNDLE).as_posix(): p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(seen)
    }


def _names(text: str, token: str) -> bool:
    return bool(re.search(r"(?<![\w.-])" + re.escape(token) + r"(?![\w-])", text))


def validate_checker_manifest_bindings(comp: List[Dict[str, Any]]) -> None:
    """Refuse to emit a manifest that claims a carrier, a selector or a reason the bytes lack.

    A zero_reason the carrier never prints is worse than an absent one: it reads as attributed
    while the run still emits a bare zero. So each of the three bindings is checked against the
    reachable bytes here, and a failure stops generation rather than producing a lying manifest.
    """
    carrier, reward = checker_carrier(), checker_reward_path()
    reach = reachable_from_entry_point()

    if carrier not in reach:
        raise RuntimeError(
            "checker_carrier %s is not reachable from %s; reachable: %s"
            % (carrier, ENTRY_POINT, sorted(reach))
        )
    if checker_aggregation_mode() not in AGGREGATION_MODES:
        raise RuntimeError("checker_aggregation_mode %r is outside the closed set" % checker_aggregation_mode())

    carrier_text = reach[carrier]
    corpus = "\n".join(reach.values())

    for it in comp:
        code = it.get("zero_reason", "")
        if not REASON_PATTERN.match(str(code)):
            raise RuntimeError("item %s carries zero_reason %r, which is not a kebab reason code" % (it["id"], code))
        if not _names(corpus, code):
            raise RuntimeError(
                "item %s declares zero_reason %s, which no file reachable from %s emits"
                % (it["id"], code, ENTRY_POINT)
            )
        if not _names(carrier_text, selector_for(it["id"])):
            raise RuntimeError("carrier %s defines no %s" % (carrier, selector_for(it["id"])))

    codes = [it["zero_reason"] for it in comp]
    if len(set(codes)) != len(codes):
        raise RuntimeError("zero_reason codes are not unique, so a zero would not name its checker")

    if not any(reward in text for text in reach.values()):
        raise RuntimeError("no file reachable from %s writes the bound reward path %s" % (ENTRY_POINT, reward))


def _targets_block() -> str:
    text = GROUNDING.read_text(encoding="utf-8")
    m = re.search(r"^  evaluation_targets:\s*\n((?:^    -.*\n)+)", text, re.M)
    if not m:
        raise RuntimeError("grounding.yaml rubric block carries no evaluation_targets")
    return m.group(1)


def gen_format_digest() -> str:
    files = {
        name: sha256_file(BUNDLE / "environment" / name)
        for name in ("format.json", "bia_numerics.py")
    }
    fmt = json.loads((BUNDLE / "environment" / "format.json").read_text())
    payload = {
        "_comment": BANNER,
        "purpose": "pins the frozen numeric surface for the frozen_format_digest_matches checker",
        "format_id": fmt["format_id"],
        "max_normal": fmt["max_normal"],
        "min_subnormal": fmt["min_subnormal"],
        "overflow_onset": fmt["overflow_onset"],
        "files": files,
    }
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def gen_frozen_bytes() -> str:
    payload = {
        "_comment": BANNER,
        "purpose": "pins the frozen files instruction.md red-lines that format_digest.json does not cover, for the frozen_harness_bytes_unmodified checker",
        "excluded": {
            "environment/submission/precision_policy.py": "the one file instruction.md tells the agent to edit, so its bytes are not frozen and no digest of it is gradable"
        },
        "files": {name: sha256_file(BUNDLE / "environment" / name) for name in FROZEN_BYTES_FILES},
    }
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def gen_checkers_yaml() -> str:
    comp = compiled_items()
    carrier, reward, mode = checker_carrier(), checker_reward_path(), checker_aggregation_mode()
    P = ["--- # %s" % BANNER]
    A = P.append
    A("# checkers.yaml for bia slot S07, mixed-precision-stability.")
    A("#")
    A("# Every graded assertion below reduces to EXACTLY ONE of VALUE, EFFECT, ABSENCE,")
    A("# INVARIANT, ORDERING, DIVERGENCE, and names the live-state read it traces to. A checker")
    A("# reducing to none or to two kinds is invalid. Several checkers share a kind; that is")
    A("# lawful, because the rule binds each checker to one kind and does not require the six")
    A("# kinds to partition the set.")
    A("#")
    A("# Each row also carries the three facts seed/forge/verifier.py reads from frozen bytes")
    A("# rather than from an author's word: reached_by, the carrier file the entry point can")
    A("# actually reach; selector, the function that carrier actually defines; and zero_reason,")
    A("# the machine-readable code that carrier actually emits alongside the zero it explains.")
    A("# recompute.py refuses to emit this file unless all three are true of the carrier bytes.")
    A("#")
    A("# Source of truth is the `rubric` block of solution/grounding.yaml. This file is")
    A("# generated from it, tests/grade.py implements it, and recompute.py refuses to emit")
    A("# anything unless the two identifier sets are equal.")
    A("")
    A("schema: %s" % checker_manifest_schema())
    A("schema_version: 1")
    A("slot: S07")
    A("task: bia/mixed-precision-stability")
    A("implementation: %s" % carrier)
    A("outcome_carrier: /logs/verifier/outcomes.json")
    A("reward_path: %s" % reward)
    A("gate_rule: every checker must pass; any failure emits score 0.0 with reason checker_failed:<names>")
    A("")
    A("aggregation:")
    A("  mode: %s" % mode)
    A("  note: >-")
    A(
        _wrap(
            "Every checker below is a gate rather than a weighted contributor: tests/grade.py "
            "returns 0.0 the moment any one of them is false, and only a run in which all of "
            "them hold reaches the metric ratio at all. No threshold is declared, because this "
            "bundle binds no contract threshold and authoring one here would invent the bar "
            "instead of carrying it.",
            "    ",
        )
    )
    A("")
    A("checkers:")
    for it in comp:
        A("")
        A("  - id: %s" % it["id"])
        A("    reduction: %s" % it["kind"])
        A("    reached_by: %s" % carrier)
        A("    selector: %s" % selector_for(it["id"]))
        A("    zero_reason: %s" % it["zero_reason"])
        A("    statement: >-")
        A(_wrap(it["asserts"], "      "))
        for field in ("live_state_read", "accepting_half", "rejecting_half"):
            A("    %s: >-" % field)
            A(_wrap(it[field], "      "))
        A("    negative_control: %s" % it["control"])
    return "\n".join(P) + "\n"


def _wrap(text: str, indent: str, width: int = 96) -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def gen_rubrics_jsonl() -> str:
    return "".join(
        json.dumps({"id": it["id"], "rubric": it["criterion"]}, sort_keys=True) + "\n"
        for it in judged_items()
    )


def gen_test_output() -> str:
    kinds = checker_kinds()
    lines = [
        '"""%s' % BANNER,
        "",
        "One assertion per checker declared in tests/checkers.yaml. These read the outcome",
        "carrier that tests/grade.py wrote; they never recompute a verdict, so a disagreement",
        "between this file and the carrier is impossible by construction rather than by luck.",
        '"""',
        "",
        "import json",
        "import os",
        "import pathlib",
        "",
        "import pytest",
        "",
        'CARRIER = pathlib.Path(os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json"))',
        "",
        "",
        "def _outcomes():",
        "    if not CARRIER.is_file():",
        '        pytest.skip("no outcome carrier at %s" % CARRIER)',
        "    return json.loads(CARRIER.read_text())",
        "",
        "",
        "def test_every_declared_checker_reported():",
        "    got = set(_outcomes())",
        "    want = set(DECLARED)",
        '    assert got == want, "carrier reported %s, checkers.yaml declares %s" % (sorted(got), sorted(want))',
        "",
        "",
        "DECLARED = %r" % (sorted(kinds),),
        "",
        "REDUCTION_KIND = %s" % json.dumps(dict(sorted(kinds.items())), indent=4, sort_keys=True),
        "",
        "",
        "def test_each_checker_reduces_to_exactly_one_of_the_six_kinds():",
        '    six = {"VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"}',
        "    for name, kind in REDUCTION_KIND.items():",
        '        assert kind in six, "%s reduces to %r which is not one of the six" % (name, kind)',
        "",
    ]
    for name in sorted(kinds):
        lines += [
            "",
            "def test_%s():" % name,
            '    assert _outcomes().get("%s") is True' % name,
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def gen_rubrics_json() -> str:
    comp, judged = compiled_items(), judged_items()
    floor = float(rubric_scalar("compilation_floor"))
    total = len(comp) + len(judged)
    share = len(comp) / float(total)
    targets = [t.strip() for t in _targets_block().splitlines() if t.strip()]
    payload = {
        "_comment": BANNER,
        "slot": "S07",
        "task": "bia/mixed-precision-stability",
        "evaluation_targets": [t.lstrip("- ").strip() for t in targets],
        "compilation_floor": floor,
        "compiled_items": len(comp),
        "judged_items": len(judged),
        "total_items": total,
        "compiled_weight_share": round(share, 4),
        "compiled_weight_share_clears_floor": share >= floor,
        "compiled_weight_share_margin": round(share - floor, 4),
        "compiled_weight_share_note": (
            "Every item carries unit weight, so the share is the plain count ratio %d/%d and no "
            "reweighting can move it. The %d compiled items are decided by code in tests/grade.py "
            "and are emitted one-for-one as tests in tests/test_output.py; the %d judged items are "
            "decided by a model reading the agent trajectory. Each judged item names the specific "
            "semantic residue that blocks its compilation, and the decidable half of every "
            "obligation it was merged from was moved into the compiled lane rather than dropped: "
            "see merged_from and compiled_into on each judged item."
        ) % (len(comp), total, len(comp), len(judged)),
        "items": [
            {
                "id": it["id"],
                "dimension": "outcome",
                "weight": 1,
                "evaluation_target": "logs/verifier/outcomes.json",
                "criterion": it["criterion"],
                "judgment": "deterministic",
                "evidence": it["evidence"],
                "mode": "compiled",
                "reduction_kind": it["kind"],
            }
            for it in comp
        ]
        + [
            {
                "id": it["id"],
                "dimension": "process",
                "weight": 1,
                "evaluation_target": "agent trajectory",
                "criterion": it["criterion"],
                "judgment": "judged",
                "evidence": ["agent trajectory"],
                "mode": "judged",
                "residue": it["residue"],
                "merged_from": it["merged_from"],
                "compiled_into": it["compiled_into"],
            }
            for it in judged
        ],
    }
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def gen_truth() -> str:
    g = load_grounding()
    fmt = json.loads((BUNDLE / "environment" / "format.json").read_text())
    op = json.loads((BUNDLE / "environment" / "operating_point.json").read_text())
    anchors = adequacy_anchors()
    P = []
    A = P.append
    A("# Ground truth: %s" % g.get("task_name", "bia/mixed-precision-stability"))
    A("")
    A(BANNER)
    A("")
    A("## Objective")
    A("")
    A("Reach the target validation loss without a single overflow event under the frozen numeric format %s. The target is the final validation loss of a full precision control run measured inside the same attempt, and the baseline is the final validation loss of the standard mixed precision recipe measured inside the same attempt. Neither anchor is an authored constant, so no number in this bundle is an estimated loss." % fmt["format_id"])
    A("")
    A("## Graded outcome")
    A("")
    A("One float on the closed interval zero to one, written to the verifier score path, computed as `min(max((baseline - agent) / (baseline - target), 0), 1)` and gated to exactly zero by any overflow event in the graded phase.")
    A("")
    A("## The frozen surface")
    A("")
    A("The format is pinned as a standalone fixture at `environment/format.json` rather than left implicit in code. It carries %d exponent bits and %d mantissa bits, a maximum normal of %s, a minimum subnormal of %s, an overflow onset of %s, a machine epsilon of %s, and %s decades of dynamic range. It does not saturate and it encodes no infinity, so an out of range partial sum is a counted event rather than a silently clipped value." % (fmt["exponent_bits"], fmt["mantissa_bits"], fmt["max_normal"], fmt["min_subnormal"], fmt["overflow_onset"], fmt["machine_epsilon"], fmt["dynamic_range_decades"]))
    A("")
    A("## The scaled operating point")
    A("")
    A("The session bound is %s hours and the attempt count is %s, so the per attempt budget is %s hours, which is %s minutes of single H100 time. The upstream nanogpt operating point costs 84 minutes per graded run, 11.7 times over that budget, so this slot is authored at a scaled point instead: a %d layer, %d wide byte level transformer of about %d parameters over %d steps at %d tokens per step, with a reduction fan in of %d." % (g.get("session_bound_hours"), g.get("attempts_per_session"), g.get("per_attempt_budget_hours"), g.get("per_attempt_budget_minutes"), op["model"]["n_layer"], op["model"]["d_model"], op["model"]["approx_parameters"], op["schedule"]["steps"], op["schedule"]["tokens_per_step"], op["schedule"]["micro_batches_per_step"]))
    A("")
    A("The micro batch of %d sequences is a measured choice rather than a default. The reduction error a chain accumulation makes under round to nearest is close to unbiased, so it only becomes visible in the metric once it exceeds the gradient's own sampling noise. At 2 sequences per micro batch it did not: the float32 control finished at 2.4147 while the standard narrow format recipe finished at 1.8868, the anchor span was inverted and no submission could score. Lowering the learning rate to 0.001 narrowed that inversion to 0.057 at 400 steps and to 0.005 at 1200 steps without clearing it. Raising the micro batch to %d sequences drops the sampling noise below the reduction error, and the float32 control then leads the standard recipe by 0.436. The step count falls to %d so the token budget and the wall clock stay inside the per attempt bound." % (op["schedule"]["micro_batch_sequences"], op["schedule"]["micro_batch_sequences"], op["schedule"]["steps"]))
    A("")
    A("## Golden path")
    A("")
    A("**Step 1. Read the fixture before reading the code.** Establishes that the dynamic range is %s decades and that a single global loss scale is one degree of freedom against a gradient tensor whose element magnitudes span more than that. Survives the `frozen_format_digest_matches` checker, which fails any run that widened the range instead of solving the problem." % fmt["dynamic_range_decades"])
    A("")
    A("**Step 2. Recognise that the recalled recipe fails on its own mechanism.** Dynamic loss scaling discovers its scale by overflowing and backing off. The harness counts every overflow event on a ledger no policy can reach, so the backoff that makes the recipe work elsewhere is what disqualifies it here. Survives the `no_overflow_flag_raised_in_agent_phase` checker.")
    A("")
    A("**Step 3. Reject the two easy escapes.** Scaling down far enough to avoid overflow lands at the baseline, because values under the minimum subnormal flush to zero and a machine epsilon of one sixteenth swamps small contributions into a large running sum. Moving master weights to float32 is correct and changes nothing, because the damage happens in the reduction before the optimizer sees a gradient. Survives the `agent_improves_on_standard_control` checker.")
    A("")
    A("**Step 4. Derive the bound on the partial sums.** Pre scale each leaf from its own reported magnitude so its stored value is at most 32, two binades under the ceiling. Combine with alignment to the smaller exponent and a node exponent of minus one, so both operands are shifted by a strictly negative amount, neither rescale can grow a word, each lands at or below 16, and their sum at or below 32. Induction carries the bound to every node. Survives the `format_in_force_at_every_reduction` checker.")
    A("")
    A("**Step 5. Notice that the shape and the node exponent are only correct together.** A pairwise shape without the per level renormalization still lets the partial sum grow with the fan in and still overflows. The same renormalization on a chain shrinks the running sum by two to the minus fifteen and flushes the gradient to zero. This is the accumulation order argument, and it is the step a recalled configuration does not supply.")
    A("")
    A("**Step 6. Recover the precision the halving spent.** Carry a compensated two word accumulator entirely inside the frozen format, so each level's discarded bit is folded back rather than lost. Survives the `reduction_agrees_with_independent_replay` checker, which replays the recorded plan through a scalar reducer sharing no code with the harness.")
    A("")
    A("**Step 7. Report the outcome truthfully.** Survives `phases_ran_in_declared_order`, `master_weights_updated_in_declared_dtype` and `reported_outcome_agrees_with_telemetry`, which re-derives the reported score and reason from the telemetry under the harness reward rule, and the trajectory rubrics in `tests/rubrics.jsonl`.")
    A("")
    A("**Step 8. Leave the instrument alone.** The routes that defeat the grader rather than the problem are each bound to a checker: the frozen numeric surface to `frozen_format_digest_matches`, the training loop, the attempt driver, the operating point and the corpus to `frozen_harness_bytes_unmodified`, the overflow ledger to `overflow_ledger_accounting_is_consistent`, the declared schedule to `agent_phase_covered_every_declared_step`, and the two scoring anchors to `fp32_control_ran_outside_the_frozen_format`, `baseline_anchor_ran_the_shipped_narrow_recipe`, `anchor_span_is_positive_and_finite`, `all_phases_started_from_the_same_initialization` and `control_cache_agrees_with_graded_telemetry`. Staying inside the budget the operating point declares survives `attempt_wallclock_within_declared_budget`.")
    A("")
    A("## Checker reductions")
    A("")
    A("| Checker | Reduction kind | Negative control it must reject |")
    A("|---|---|---|")
    A("\n".join("| `%s` | %s | `%s` |" % (it["id"], it["kind"], it["control"]) for it in compiled_items()))
    A("")
    A("## Rubric composition")
    A("")
    A(
        "The rubric carries %d items of unit weight: %d graded by compiled deterministic tests and %d graded by a model reading the trajectory. The compiled weight share is therefore %.4f against a compilation floor of %.2f, clearing it by %.4f. The share is a plain count ratio and no reweighting can move it."
        % (
            len(compiled_items()) + len(judged_items()),
            len(compiled_items()),
            len(judged_items()),
            len(compiled_items()) / float(len(compiled_items()) + len(judged_items())),
            float(rubric_scalar("compilation_floor")),
            len(compiled_items()) / float(len(compiled_items()) + len(judged_items())) - float(rubric_scalar("compilation_floor")),
        )
    )
    A("")
    A("Each judged item below names the one semantic residue that blocks its compilation, and records which earlier obligations it consolidates and which compiled checkers now carry their decidable half. Consolidation here is a merge, never a deletion: every obligation the nine original items stated still binds.")
    A("")
    A("| Judged item | Consolidates | Decidable half now compiled into |")
    A("|---|---|---|")
    for it in judged_items():
        A(
            "| `%s` | %s | %s |"
            % (
                it["id"],
                ", ".join("`%s`" % m for m in it["merged_from"]),
                ", ".join("`%s`" % c for c in it["compiled_into"]),
            )
        )
    A("")
    A("## Routes bound to a measured control")
    A("")
    A("Every row was measured by running the route end to end through the identical code path in smoke mode on CPU. These numbers establish that the instrument is live in both directions. They are smoke scale and they are not evidence about difficulty.")
    A("")
    A("| Route | Verdict | Score | Overflow events | Final validation loss | Reason |")
    A("|---|---|---|---|---|---|")
    for r in adequacy_table():
        A(
            "| %s | %s | %.4f | %d | %.4f | `%s` |"
            % (r["route"], r["verdict"], r["score"], r["overflow_events"], r["final_val_loss"], r["reason"])
        )
    A("")
    A("Both anchors for that table were measured in the same runs: baseline %s and target %s." % (anchors[0], anchors[1]))
    A("")
    A("The last two rows are recorded as accepted rather than reclassified as defeats. Both carry the load bearing insight and differ only in a second order detail, and a suite that rejected them would be a suite that rejects everything not byte identical to the reference.")
    A("")
    A("## What this document does not claim")
    A("")
    A("- **It does not estimate the reference score.** The reference score on the one H100 envelope is MEASURED at %s, from one full graded attempt run end to end with no control cache and no smoke switch. The anchors of that same attempt were a standard control baseline of %s and a float32 control target of %s against a reference of %s. That is a single seed measurement taken while an unrelated production campaign held the accelerator at full utilization, so it is a real number and not a difficulty claim." % (g.get("measured_reference_score"), g.get("measured_baseline_metric"), g.get("measured_target_metric"), g.get("measured_agent_metric")))
    A("")
    A("- **It does not estimate a wall clock.** One full graded attempt is MEASURED at %s seconds on a contended H100 against a per attempt bound of 432 seconds. The accelerator was carrying an unrelated production campaign at full utilization for the whole measurement, so the number is an upper bound and an idle accelerator will be faster." % g.get("wallclock_seconds_one_h100_contended"))
    A("")
    A("- **It does not claim a difficulty tier or a pass probability.** The defeat mechanism above is design intent and is marked UNMEASURED. The bundle ships at %s, which is the correct terminal state." % g.get("disposition", "HOLD:PILOT_REQUIRED"))
    A("")
    A("- **It does not claim the smoke numbers describe the full operating point.** Smoke runs %d steps of a %d layer, %d wide model on CPU. It proves the code path executes, the gate fires, and the instrument decides in both directions. It proves nothing about how a frontier model fares at %d steps." % (op["smoke"]["steps"], op["smoke"]["n_layer"], op["smoke"]["d_model"], op["schedule"]["steps"]))
    A("")
    A("- **It does not claim the alignment knob is load bearing.** Measured at full model dimensions, the sixteen micro batch chunks have magnitudes within a factor of %s of each other, so their leaf exponents are nearly equal and alignment to the smaller or the larger exponent produced the same result. The knob is free and honest, and on this data it is close to inert. It would bite on a run whose chunk magnitudes diverge." % g.get("chunk_absmax_spread"))
    A("")
    A("## Residual limitations, recorded rather than repaired")
    A("")
    A("**The graded measurement is authored by the verifier, not read off the agent.** The verifier starts its own attempt of the frozen driver, in a directory it names from system entropy, with the anchor cache disabled, and stamps it with a nonce nothing outside the verifier process has seen. The driver in turn hosts every policy in a separate `policy_worker.py` interpreter, so a submission is two process boundaries away from the telemetry, the overflow ledger, the clock, the replay-step draw and the reward arithmetic. The agent's own attempt record is still graded, because a truthful report is part of the deliverable, but the reward is only ever read off the verifier-authored record and `agent_report_reproduces_under_verifier_execution` requires the two to state the same experiment. What used to be a residual here, an anchor cache in the agent's writable space that could move its own baseline, is no longer on the reward path at all: the graded run measures both anchors inside itself, and the baseline recipe lives in its own frozen red-lined `baseline_recipe.py` rather than in the file the agent is told to edit.")
    A("")
    A("**What the isolation does not close.** The verifier pays for a second full attempt on every graded run, which is a real wall-clock cost that the 0.12 hour per attempt bound does not cover and the verifier window absorbs. The replayed reduction steps are drawn from system entropy inside the measuring process, which a co-resident reader could not predict, but the worker runs under the same operating system user and this bundle does not claim kernel-level isolation between them; what it claims is that nothing the worker can reach is on the reward path.")
    A("")
    A("**Just over twelve percent of gradient elements flush to zero even under the reference.** Measured at full model dimensions, the zero fraction under the reference plan is %s against %s under a chain carrying no per chunk exponent. That is the intrinsic floor of a format with %s decades of range applied to a gradient tensor whose within chunk magnitude spread is wider, and no plan removes it. It is recorded because a reader could otherwise mistake the reference for a lossless path." % (g.get("reference_zero_fraction"), g.get("chain_zero_fraction"), fmt["dynamic_range_decades"]))
    A("")
    A("**The two scoring anchors are stochastic.** Both are single seed measurements at a fixed seed. Kernel nondeterminism on an accelerator makes the anchors and therefore the score reproducible only to within run to run noise, and no seed ladder is bound for this slot, because a fifty attempt session at 7.2 minutes per attempt does not buy one. The measured variance is UNMEASURED and is owed before any difficulty claim.")
    return "\n".join(P) + "\n"


TARGETS = {
    "tests/checkers/format_digest.json": gen_format_digest,
    "tests/checkers/frozen_bytes.json": gen_frozen_bytes,
    "tests/checkers.yaml": gen_checkers_yaml,
    "tests/rubrics.jsonl": gen_rubrics_jsonl,
    "tests/test_output.py": gen_test_output,
    "solution/rubrics.json": gen_rubrics_json,
    "solution/TRUTH.md": gen_truth,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--hash", action="store_true")
    args = ap.parse_args()

    if args.hash:
        members = hash_members()
        digest = canonical_content_hash()
        print("members            %d" % len(members))
        print("namespace          %s" % FORGE_TASK_NAMESPACE)
        print("canonical_content_hash %s" % digest)
        print("uuid               %s" % bundle_uuid())
        return 0

    validate_rubric()
    comp, judged = compiled_items(), judged_items()
    share = len(comp) / float(len(comp) + len(judged))
    print(
        "rubric  compiled=%d judged=%d share=%.4f floor=%s"
        % (len(comp), len(judged), share, rubric_scalar("compilation_floor"))
    )

    bad = []
    for rel, fn in TARGETS.items():
        p = BUNDLE / rel
        want = fn()
        if args.check:
            have = p.read_text(encoding="utf-8") if p.is_file() else None
            status = "OK   " if have == want else "DRIFT"
            if have != want:
                bad.append(rel)
            print("%s %s" % (status, rel))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(want, encoding="utf-8")
            print("wrote %s (%d bytes)" % (rel, len(want.encode())))

    if args.check and bad:
        print("DRIFT in: " + ", ".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

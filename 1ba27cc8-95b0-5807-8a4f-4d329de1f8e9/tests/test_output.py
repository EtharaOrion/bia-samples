from __future__ import annotations

import ast
import json
import pathlib
import re
import tempfile

import checkers
import score

HERE = pathlib.Path(__file__).resolve().parent

REASONS = {
    "carried_summary_is_current": checkers.STALE_SUMMARY,
    "bound_envelope_respected": checkers.BUDGET_EXCEEDED,
    "frozen_axes_untouched": checkers.FROZEN_AXIS_MOVED,
    "graded_crossing_recomputed_unsmoothed": checkers.READOUT_SMOOTHED,
    "evaluated_weights_owned_by_the_harness": checkers.WEIGHTS_NOT_OWNED,
    "crossing_sustained_across_scheduled_evaluations": checkers.NOT_SUSTAINED,
    "early_halt_yields_no_crossing": checkers.EARLY_STOP,
    "reported_crossing_reconciles_with_verifier_measurement": checkers.REPORT_DIVERGES,
}

ALLOWED_IMPORTS = {"__future__", "dataclasses", "hashlib", "json", "math", "pathlib", "typing"}
FORBIDDEN_IMPORTS = {"datetime", "numpy", "os", "random", "requests", "socket",
                     "subprocess", "time", "torch", "urllib"}


def selector(name: str):
    return getattr(checkers, "check_" + name)


def source_tree(filename: str) -> ast.Module:
    return ast.parse((HERE / filename).read_text())


def test_chain_declares_the_schema_the_instrument_reads():
    assert checkers.SCHEMA == "bia.session/v1"
    assert len(checkers.CHAIN) == len(REASONS)


def test_every_checker_in_the_chain_declares_a_zero_reason():
    declared = {name for name, _ in checkers.CHAIN}
    assert declared == set(REASONS), \
        f"chain and reason table disagree: {declared ^ set(REASONS)}"


def test_every_checker_reduces_to_exactly_one_reason():
    reasons = list(REASONS.values())
    assert len(set(reasons)) == len(reasons), "two checkers share a zero reason"
    published = {value for name, value in vars(checkers).items()
                 if name.isupper() and isinstance(value, str)}
    for reason in reasons:
        assert reason in published, f"{reason!r} is not a declared constant in checkers.py"


def test_every_declared_selector_exists_in_the_checkers_module():
    defined = {node.name for node in source_tree("checkers.py").body
               if isinstance(node, ast.FunctionDef) and node.name.startswith("check_")}
    for name, function in checkers.CHAIN:
        assert "check_" + name in defined, f"check_{name} is not defined in checkers.py"
        assert function is selector(name), f"CHAIN entry {name} is not checkers.check_{name}"


def test_the_grader_imports_the_chain_and_nothing_else():
    imported = set()
    for node in ast.walk(source_tree("grade.py")):
        if isinstance(node, ast.ImportFrom) and node.module == "checkers":
            imported.update(alias.name for alias in node.names)
    expected = {"check_" + name for name, _ in checkers.CHAIN}
    assert imported == expected, f"grade.py imports {imported ^ expected} off the chain"


def test_checkers_module_imports_only_the_allowlist():
    seen = set()
    for node in ast.walk(source_tree("checkers.py")):
        if isinstance(node, ast.Import):
            seen.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            seen.add(node.module.split(".")[0])
    assert not (seen & FORBIDDEN_IMPORTS), f"checkers.py imports {seen & FORBIDDEN_IMPORTS}"
    assert seen <= ALLOWED_IMPORTS, f"checkers.py imports {seen - ALLOWED_IMPORTS} off the allowlist"


def test_grade_module_never_imports_the_submission():
    text = (HERE / "grade.py").read_text()
    assert "import refine" not in text
    assert "importlib" not in text


def test_recipe_schema_closes_the_frozen_axes():
    schema = json.loads((HERE / "recipe_schema.json").read_text())
    frozen = set(schema["frozen_axis_keys"])
    assert frozen & {"batch_tokens_per_step", "num_layers", "seq_len"}
    assert not (set(schema["fields"]) & frozen), "a frozen axis is writable through fields"


def _floor_literals() -> dict:
    pattern = re.compile(r"printf\s+'(\{.*\})\\n'.*SCORE_DIR\}/([\w.]+)")
    literals = {}
    for line in (HERE / "test.sh").read_text().splitlines():
        found = pattern.search(line)
        if found:
            literals[found.group(2)] = found.group(1)
    return literals


def _duplicate_keys(literal: str) -> list:
    repeated = set()

    def hook(pairs):
        names = [name for name, _ in pairs]
        repeated.update(name for name in names if names.count(name) > 1)
        return dict(pairs)

    json.loads(literal, object_pairs_hook=hook)
    return sorted(repeated)


def test_the_abort_floor_writes_every_score_document():
    written = set(_floor_literals())
    expected = {score.SCORE_NUMERIC_JSON, score.SCORE_FULL, score.SCORE_DOC}
    assert written == expected, \
        f"test.sh abort floor writes {sorted(written)}, expected {sorted(expected)}"


def test_the_abort_floor_repeats_no_json_key():
    for name, literal in sorted(_floor_literals().items()):
        repeated = _duplicate_keys(literal)
        assert not repeated, f"test.sh floor for {name} repeats JSON key(s) {repeated}"


def test_the_abort_floor_matches_what_the_grader_would_write():
    floor = {name: json.loads(text) for name, text in _floor_literals().items()}
    assert floor[score.SCORE_DOC] == floor[score.SCORE_FULL], \
        "the score.json and score_full.json floors disagree; score.write emits one document to both"

    original = score.SCORE_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            score.SCORE_DIR = pathlib.Path(tmp)
            score.write(0.0, floor[score.SCORE_FULL]["reason"],
                        floor[score.SCORE_FULL]["metric"], [])
            graded = {p.name: json.loads(p.read_text())
                      for p in score.SCORE_DIR.glob("*.json")}
            bare = (score.SCORE_DIR / score.SCORE_FLOAT).read_text()
    finally:
        score.SCORE_DIR = original

    for name in sorted(floor):
        assert set(floor[name]) == set(graded[name]), \
            f"floor {name} carries {sorted(floor[name])}, grader writes {sorted(graded[name])}"
    assert floor[score.SCORE_FULL] == graded[score.SCORE_FULL], \
        "the floor document is not what score.write emits for the same inputs"
    assert bare == "0.000000\n", f"grader writes bare score {bare!r}"


def main() -> int:
    failed = 0
    passed = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
        except Exception as exc:
            print(f"FAIL {name}: {exc}", flush=True)
            failed += 1
        else:
            print(f"ok   {name}", flush=True)
            passed += 1
    print(f"{passed} passed, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

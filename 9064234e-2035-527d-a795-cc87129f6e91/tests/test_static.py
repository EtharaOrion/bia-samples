"""Static surface of the grading tree. Reports; never scores.

Nothing here contributes to the reward. It exists so that a malformed grading
tree is loud at the top of the run instead of surfacing as a confusing zero
three minutes later.
"""
from __future__ import annotations

import ast
import json
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent

CHECKER_IMPORT_ALLOWLIST = {"json", "math", "hashlib", "pathlib", "dataclasses", "typing",
                            "__future__"}
FORBIDDEN_IN_CHECKERS = {"random", "time", "datetime", "os", "socket", "subprocess",
                         "urllib", "requests", "torch", "numpy"}


def _manifest() -> dict:
    return yaml.safe_load((HERE / "checkers.yaml").read_text(encoding="utf-8"))


def test_manifest_declares_the_schema_the_instrument_reads():
    assert _manifest()["schema"] == "forge.checkers/v1"


def test_every_checker_reduces_to_exactly_one_closed_kind():
    closed = {"VALUE", "EFFECT", "ABSENCE", "INVARIANT", "ORDERING", "DIVERGENCE"}
    for row in _manifest()["checkers"]:
        assert row["reduction"] in closed, row["id"]


def test_checkers_module_imports_only_the_allowlist():
    tree = ast.parse((HERE / "checkers.py").read_text(encoding="utf-8"))
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            seen.add(node.module.split(".")[0])
    assert seen <= CHECKER_IMPORT_ALLOWLIST, sorted(seen - CHECKER_IMPORT_ALLOWLIST)
    assert not (seen & FORBIDDEN_IN_CHECKERS)


def test_grade_module_never_imports_the_submission():
    text = (HERE / "grade.py").read_text(encoding="utf-8")
    assert "import refine" not in text
    assert "importlib" not in text


def test_every_declared_selector_exists_in_the_checkers_module():
    tree = ast.parse((HERE / "checkers.py").read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for row in _manifest()["checkers"]:
        assert row["selector"] in defined, row["selector"]


def test_recipe_schema_closes_the_frozen_axes():
    schema = json.loads((HERE.parent / "environment" / "recipe_schema.json").read_text(encoding="utf-8"))
    assert set(schema["frozen_axis_keys"]) & {"batch_tokens_per_step", "num_layers", "seq_len"}
    assert not set(schema["fields"]) & set(schema["frozen_axis_keys"])

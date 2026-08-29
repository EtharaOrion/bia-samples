"""Static and dynamic checks the verifier runs directly against the submitted rule.

Three obligations live here, each reading live state rather than the author's
word: the AST of the submitted bytes, the parameter tensors after a real
optimizer step, and the AST-normalized distance to the pinned record corpus.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import pathlib

IMPORT_ALLOWLIST = {
    "torch", "math", "typing", "__future__", "collections", "functools",
    "itertools", "warnings", "dataclasses", "numbers", "abc",
}

BANNED_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "getenv", "environ",
    "input", "globals", "locals", "vars",
}

BANNED_ATTRS = {"environ", "getenv", "system", "popen"}

COPY_SIMILARITY_MAX = 0.92


def _root_module(name: str) -> str:
    return (name or "").split(".")[0]


HORIZON_NAMES = ("total_steps", "max_steps", "num_steps", "horizon")

DYNAMIC_STRING_METHODS = {"join", "format", "format_map", "encode", "decode",
                          "replace", "ljust", "rjust", "zfill"}

DYNAMIC_NAME_BUILTINS = {"chr", "str", "bytes", "bytearray", "format", "repr"}


def _foldable(node):
    """Value of a constant-foldable numeric expression, or None.

    A denylist of names is defeated by writing the number, and a denylist of
    numbers is defeated by writing an arithmetic expression, so the audit folds
    literal arithmetic before it compares. This is the reason the check is
    structural rather than a list of forbidden spellings.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, (int, float)) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _foldable(node.operand)
        if inner is None:
            return None
        return inner if isinstance(node.op, ast.UAdd) else -inner
    if isinstance(node, ast.BinOp):
        left, right = _foldable(node.left), _foldable(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Pow) and abs(right) < 16:
                return left ** right
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
    return None


def horizon_value_set(total_steps: int):
    """Every value that stands in for the denied horizon closely enough to matter."""
    out = set()
    for k in (1, 2, 3, 4):
        out.add(float(total_steps) * k)
        out.add(float(total_steps) / k)
    for delta in (-1, 0, 1):
        out.add(float(total_steps + delta))
    return out


def audit_horizon_leak(path: str, total_steps: int = None):
    """ABSENCE: no static route by which the update rule can obtain the horizon.

    Three structural rules replace the string denylist an earlier revision
    carried, because a denylist of four spellings is defeated by writing a
    fifth. They are, in order: an import allowlist; a ban on constructing names
    at runtime at all, which removes the whole class of assembling a key from
    fragments rather than the one assembly that was tried; and rejection of any
    constant-foldable numeric literal that equals the denied horizon or a
    simple multiple of it, which removes writing the number instead of naming
    it.

    Bounded claim: this rejects the named static routes. A horizon reconstructed
    through data-dependent computation the folder cannot evaluate is outside
    this check, and is what the behavioural schedule-ownership probe in
    tests/harness/audit_child.py is for. Neither check alone is the control;
    the pair is, and the pair is what the checker reports.
    """
    src = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return False, "update_rule_unparseable_%s" % exc.lineno
    forbidden_numbers = horizon_value_set(total_steps) if total_steps else set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root_module(alias.name) not in IMPORT_ALLOWLIST:
                    return False, "disallowed_import_%s" % alias.name
        elif isinstance(node, ast.ImportFrom):
            if _root_module(node.module or "") not in IMPORT_ALLOWLIST:
                return False, "disallowed_import_from_%s" % (node.module or "relative")
            if node.level and node.level > 0:
                return False, "relative_import_not_allowed"
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                return False, "banned_name_%s" % node.id
            if node.id in HORIZON_NAMES:
                return False, "horizon_name_%s" % node.id
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                return False, "banned_attribute_%s" % node.attr
            if node.attr in HORIZON_NAMES:
                return False, "horizon_attribute_%s" % node.attr
        elif isinstance(node, ast.JoinedStr):
            return False, "runtime_string_construction_fstring"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod, ast.Mult)):
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, (str, bytes)):
                    return False, "runtime_string_construction_binop"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in DYNAMIC_STRING_METHODS:
                return False, "runtime_string_construction_%s" % func.attr
            if isinstance(func, ast.Name) and func.id in DYNAMIC_NAME_BUILTINS:
                return False, "runtime_string_construction_%s" % func.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in HORIZON_NAMES:
                return False, "horizon_key_%s" % node.value
    if forbidden_numbers:
        for node in ast.walk(tree):
            value = _foldable(node)
            if value is None:
                continue
            for candidate in forbidden_numbers:
                if abs(float(value) - candidate) <= 1e-9 * max(1.0, abs(candidate)):
                    return False, "horizon_literal_%s" % value
    return True, None


def _ast_norm(path):
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.dump(tree, annotate_fields=False)


def audit_record_replay(path: str, corpus_dir: str, extra_refs=()):
    """ABSENCE: no pinned published record is present in the submission bytes."""
    submitted = _ast_norm(path)
    if submitted is None:
        return False, "update_rule_unreadable_for_replay_check"
    refs = []
    corpus = pathlib.Path(corpus_dir)
    if corpus.is_dir():
        refs.extend(sorted(corpus.glob("*.py")))
    for extra in extra_refs:
        p = pathlib.Path(extra)
        if p.is_file():
            refs.append(p)
    if not refs:
        return False, "no_corpus_available_for_replay_check"
    worst = None
    for ref in refs:
        other = _ast_norm(str(ref))
        if other is None:
            continue
        ratio = difflib.SequenceMatcher(None, submitted, other).ratio()
        if ratio >= COPY_SIMILARITY_MAX and (worst is None or ratio > worst[0]):
            worst = (ratio, ref.stem)
    if worst is None:
        return True, None
    return False, "record_replay_of_%s_similarity_%.3f" % (worst[1], worst[0])


def audit_lr_homogeneity(path: str, tolerance: float = 1e-8):
    """EFFECT: the parameter delta a real step produces is linear in the applied lr."""
    try:
        import torch
    except Exception:
        return False, "torch_unavailable_for_homogeneity_check"

    spec = importlib.util.spec_from_file_location("bia_submission_audit", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return False, "update_rule_import_failed_%s" % type(exc).__name__
    build = getattr(module, "build_update_rule", None)
    if not callable(build):
        return False, "build_update_rule_absent"

    def delta_at(lr):
        torch.manual_seed(11)
        # float64 so the measured delta is not swamped by the cancellation error of
        # subtracting two nearby float32 parameter tensors.
        weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float64))
        vector = torch.nn.Parameter(torch.ones(8, dtype=torch.float64))
        grads = [torch.randn(8, 4, dtype=torch.float64), torch.randn(8, dtype=torch.float64)]
        weight.grad = grads[0].clone()
        vector.grad = grads[1].clone()
        groups = [
            {"name": "hidden_matrix", "params": [weight], "lr": 0.0},
            {"name": "vector", "params": [vector], "lr": 0.0},
        ]
        opt = build(groups)
        if not isinstance(opt, torch.optim.Optimizer):
            raise TypeError(type(opt).__name__)
        before = [weight.detach().clone(), vector.detach().clone()]
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        moved = [(weight.detach() - before[0]), (vector.detach() - before[1])]
        return moved

    try:
        base_lr = 1e-3
        d1 = delta_at(base_lr)
        d2 = delta_at(2.0 * base_lr)
    except TypeError as exc:
        return False, "build_update_rule_returned_%s" % exc
    except Exception as exc:
        return False, "update_rule_step_failed_%s" % type(exc).__name__

    for a, b in zip(d1, d2):
        scale = a * 2.0
        num = float((b - scale).abs().max())
        den = float(scale.abs().max()) + 1e-12
        if float(a.abs().max()) == 0.0:
            return False, "update_rule_produced_no_parameter_change"
        if num / den > tolerance:
            return False, "update_not_linear_in_lr_residual_%.3e" % (num / den)
    return True, None


def file_digest(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

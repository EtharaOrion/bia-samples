"""Structural fingerprint of a submitted recipe, and numeric constant extraction.

WHY THIS IS NOT THE PREVIOUS FINGERPRINT. The previous one stripped comments,
string literals and whitespace and then hashed the remaining bytes. It was
defeated by appending eleven characters of dead code, because a no-op assignment
survives that normalization and moves the digest. A digest that any semantic
no-op can move enforces digest novelty, not algorithmic novelty.

This one normalizes the parsed syntax tree instead. It drops docstrings and bare
constant statements, prunes assignments to names nothing ever reads until the
set stops shrinking, erases every literal value to its type, and rewrites every
locally bound identifier to its order of first appearance. What survives is the
shape of the computation and the library calls it makes. Renaming variables,
retuning hyperparameters, restyling, reformatting and padding with dead code all
leave the fingerprint where it was.

WHAT IT DOES NOT REACH, stated rather than implied. A submission that
reimplements a published record in a different structure, inlines its helpers,
or reorders independent statements produces a different tree and a different
fingerprint. Structural fingerprinting catches replay, not recall. That residue
is carried as a rubric obligation, and the load-bearing defense against recall
in this bundle is that the published corpus is written against a different
operating point and a different loop contract and cannot be executed here at
all.

Nothing in this module executes the submission. It parses it.
"""
from __future__ import annotations

import ast
import hashlib


class _Canonicaliser(ast.NodeTransformer):
    def __init__(self, keep: set[str]):
        self.keep = keep
        self.mapping: dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name in self.keep:
            return name
        if name not in self.mapping:
            self.mapping[name] = f"n{len(self.mapping)}"
        return self.mapping[name]

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(
            ast.Name(id=self._rename(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._rename(node.arg)
        node.annotation = None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self._rename(node.name)
        node.returns = None
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = self._rename(node.name)
        return self.generic_visit(node)


def _imported_names(tree: ast.AST) -> set[str]:
    """Module and symbol names an import bound. These carry the library identity."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module.split(".")[0] if node.module else "")
            for alias in node.names:
                out.add(alias.asname or alias.name)
    return {name for name in out if name}


def _strip_docstrings_and_bare_constants(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [s for s in body
                         if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            if not node.body and not isinstance(node, ast.Module):
                node.body = [ast.Pass()]


def _loaded_names(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _is_pure(node: ast.AST) -> bool:
    """True when evaluating this expression cannot have an effect worth keeping."""
    return all(not isinstance(n, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom))
               for n in ast.walk(node))


def _prune_dead_assignments(tree: ast.AST) -> None:
    """Remove assignments to names nothing reads, until the set stops shrinking.

    This is the clause that closes the eleven-character defeat. `_unused = 1`
    binds a name no Load ever mentions and its value cannot escape, so it
    carries no computation and is not part of the algorithm.
    """
    while True:
        loaded = _loaded_names(tree)
        removed = 0
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            kept = []
            for stmt in body:
                dead = False
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and _is_pure(stmt.value or ast.Pass()):
                    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    names = [t for t in targets if isinstance(t, ast.Name)]
                    dead = (len(names) == len(targets)
                            and all(t.id not in loaded for t in names))
                if dead:
                    removed += 1
                else:
                    kept.append(stmt)
            if not kept:
                kept = [ast.Pass()]
            node.body = kept
        if not removed:
            return


def fingerprint(source: str) -> str:
    """Structural digest. Raises SyntaxError on source that does not parse."""
    tree = ast.parse(source)
    _strip_docstrings_and_bare_constants(tree)
    _prune_dead_assignments(tree)
    keep = _imported_names(tree)
    tree = _Canonicaliser(keep).visit(tree)
    ast.fix_missing_locations(tree)
    dumped = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(dumped.encode()).hexdigest()


def numeric_constants(source: str) -> set[float]:
    """Every number the source names, including constant arithmetic it folds.

    A regular expression over the surface form cannot see that `2000 + 690` is
    2690 or that `3.280` is 3.28. Folding the constant subtrees can. Only
    expressions built entirely from literals are folded, so nothing is executed.
    """
    tree = ast.parse(source)
    found: set[float] = set()

    def fold(node: ast.AST):
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, (int, float)) and not isinstance(node.value, bool) else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = fold(node.operand)
            if inner is None:
                return None
            return inner if isinstance(node.op, ast.UAdd) else -inner
        if isinstance(node, ast.BinOp):
            left, right = fold(node.left), fold(node.right)
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
                if isinstance(node.op, ast.Pow):
                    return left ** right
            except (ZeroDivisionError, OverflowError, ValueError):
                return None
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.BinOp, ast.UnaryOp)):
            value = fold(node)
            if value is not None:
                found.add(float(value))
    return found


def names_a_forbidden_constant(source: str, forbidden: dict[str, float],
                               rel_tol: float = 1e-9) -> list[str]:
    """Which graded constants the source names, folded rather than matched."""
    present = numeric_constants(source)
    out = []
    for label, target in sorted(forbidden.items()):
        if any(abs(v - target) <= max(rel_tol * abs(target), 1e-12) for v in present):
            out.append(label)
    return out

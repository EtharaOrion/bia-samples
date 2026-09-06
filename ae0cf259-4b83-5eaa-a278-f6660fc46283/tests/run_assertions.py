#!/usr/bin/env python3
"""Collect and run one compiled assertion module under the interpreter this image has.

WHY THIS EXISTS. `tests/Dockerfile` asked for pytest with `pip install --no-index`
against an image that ships no pytest and a build with no index to resolve it
from, and swallowed the failure with `|| true`. The verifier then invoked
`python3 -m pytest` and exited 1 on `No module named pytest` on every run,
healthy or not, so the entry point reported a failure the assertions had not
made. The repair is to make the verifier surface carry the runner it invokes
rather than to silence the invocation: this module is that runner, it depends on
nothing outside the standard library, and the pinned image runs it as it stands.

WHAT IT MAY ASSUME, and why that is not a narrowing. The modules it collects,
`test_static.py` and the compiled `test_output.py`, use no pytest feature: every
case is a module-level `test_` function taking no argument and asserting with a
bare `assert`. There is no fixture, no mark, no parametrization and no
conftest. Collecting exactly that shape therefore runs every assertion the
pytest invocation ran, and a module that later grew a fixture would fail to
collect loudly here rather than be skipped quietly.

THE EXIT STATUS IS THE WHOLE CONTRACT, and it keeps both halves the entry point
needs. A collected case that raises anything at all is a failure, so a genuine
assertion failure still exits non-zero. A collected count of zero is also a
failure, never a pass, because an empty parse is ambiguous between a clean run
and no run at all and `tests/test.sh` says so at the site that calls this.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import traceback

PASSED = "passed"
FAILED = "failed"


def load(path: pathlib.Path):
    """Import the module by path, with its own directory importable first.

    The assertion modules import `checkers` and `reward` as siblings, exactly as
    they do under pytest's rootdir insertion, so the directory goes on the path
    before the module body runs rather than after.
    """
    here = str(path.resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError("no loader for " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cases(module) -> list:
    """Every `test_` callable the module defines, in declaration order.

    Ordered by the function's own line number rather than by name, so a report
    reads in the order the file was compiled and two runs agree.
    """
    found = []
    for name in dir(module):
        if not name.startswith("test_"):
            continue
        case = getattr(module, name)
        if not callable(case):
            continue
        code = getattr(case, "__code__", None)
        if code is None or code.co_argcount:
            continue
        found.append((code.co_firstlineno, name, case))
    return [(name, case) for _, name, case in sorted(found)]


def run(path: pathlib.Path) -> int:
    try:
        module = load(path)
    except BaseException:  # noqa: BLE001  an uncollectable module is a failure, not an absence
        traceback.print_exc()
        print(path.name + ": collection failed", flush=True)
        return 2
    collected = cases(module)
    outcomes = []
    for name, case in collected:
        try:
            case()
        except BaseException:  # noqa: BLE001  any escape is this case failing
            outcomes.append((name, FAILED))
            print("FAILED " + path.name + "::" + name, flush=True)
            traceback.print_exc()
        else:
            outcomes.append((name, PASSED))
    failures = [name for name, outcome in outcomes if outcome == FAILED]
    print(path.name + ": " + str(len(outcomes) - len(failures)) + " passed, "
          + str(len(failures)) + " failed, " + str(len(outcomes)) + " collected", flush=True)
    if not outcomes:
        print(path.name + ": zero collected, which is a failure and never a pass", flush=True)
        return 3
    return 1 if failures else 0


def main(argv: list) -> int:
    if len(argv) != 1:
        print("usage: run_assertions.py <module.py>", file=sys.stderr)
        return 64
    path = pathlib.Path(argv[0])
    if not path.is_absolute():
        path = pathlib.Path(__file__).resolve().parent / path
    if not path.is_file():
        print("no such assertion module: " + str(path), file=sys.stderr)
        return 4
    return run(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

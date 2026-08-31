#!/usr/bin/env python3
"""Execute the submission in isolation. The grading process never imports it.

The submission is copied alone into a fresh temporary directory, launched as a new session
leader under a small environment allowlist, its output captured, and the whole process group
killed in a `finally` block whether it exited, hung or raised. Nothing it writes reaches the
grader's import path, and nothing it prints is read as a graded number.

Two things this module must do that the delivered version did not, and both were measured
failures rather than hypotheses.

FIRST, the submission has to be able to reach the frozen environment it is graded on. The
entry point is copied alone into the sandbox, so a submission that resolves its bundle from
its own file location resolves it to the sandbox's parent and finds nothing there. The
delivered OER-09 oracle did exactly that and died with
`can't open file '/tmp/environment/curate.py'`, exit status 2, on every run. The bundle root
is therefore handed over as `OER09_BUNDLE` on the environment allowlist, which is the same
idiom the sibling slot's runner uses for `OER08_CORPUS`. Isolation is unchanged: the sandbox
still holds the entry point and nothing else, and the bundle is read-only to the child.

SECOND, the graded state has to survive the sandbox. The submission's curated pool, its
curation report and its training receipt are written inside the sandbox, which this module
deletes. They are copied out into a verifier-owned directory BEFORE the teardown, and
tests/harness.py recomputes the graded records from them. That capture is what lets the
verifier grade what the submission actually produced rather than what it said it produced.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent

# The whole environment the submission is given. Anything not on this list is not inherited.
ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "PYTHONHASHSEED",
    "OER_SLOT",
    "OER_FAMILY",
    "OER09_BUNDLE",
)

DEFAULT_TIMEOUT_SECONDS = 420  # the bound per-attempt budget, 0.12 h = 7.2 min

# What the verifier lifts out of the sandbox before tearing it down. Each is an artifact
# instruction.md names as part of the deliverable.
CAPTURED = ("work/curated_pool.jsonl", "work/curation_report.json", "work/train_receipt.json")


def child_environment(bundle: Path) -> dict:
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OER09_BUNDLE"] = str(bundle)
    return env


def capture(workspace: Path, artifacts: Path) -> list:
    """Copy the submission's deliverables out of the sandbox. Returns what was found."""
    artifacts.mkdir(parents=True, exist_ok=True)
    found = []
    for relative in CAPTURED:
        source = workspace / relative
        if source.is_file():
            shutil.copy2(source, artifacts / Path(relative).name)
            found.append(relative)
    return found


def run_submission(entry: Path, bundle: Path, artifacts: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    workspace = Path(tempfile.mkdtemp(prefix="oer09-submission-"))
    process = None
    try:
        staged = workspace / entry.name
        shutil.copy2(entry, staged)
        staged.chmod(0o755)
        process = subprocess.Popen(
            ["/bin/bash", str(staged)],
            cwd=str(workspace),
            env=child_environment(bundle),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            out, err = process.communicate(timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            out, err, code = "", "submission exceeded the bound per-attempt timeout", 124
        captured = capture(workspace, artifacts)
        return {"exit_code": code, "stdout": out, "stderr": err, "graded": False, "captured": captured}
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        shutil.rmtree(workspace, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="run the submission in isolation")
    parser.add_argument("--entry", required=True)
    parser.add_argument("--record", default="/logs/verifier/submission_run.json")
    parser.add_argument("--artifacts", default="/logs/verifier/submission_artifacts")
    parser.add_argument("--bundle", default=str(BUNDLE))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    entry = Path(args.entry)
    artifacts = Path(args.artifacts)
    if not entry.is_file():
        artifacts.mkdir(parents=True, exist_ok=True)
        record = {
            "exit_code": 127,
            "stdout": "",
            "stderr": "submission entry absent",
            "graded": False,
            "captured": [],
        }
    else:
        record = run_submission(entry, Path(args.bundle), artifacts, args.timeout)

    target = Path(args.record)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The exit code of the submission is recorded and deliberately not propagated: a run that
    # failed is graded, with a reason, rather than reported as an infrastructure fault.
    return 0


if __name__ == "__main__":
    sys.exit(main())

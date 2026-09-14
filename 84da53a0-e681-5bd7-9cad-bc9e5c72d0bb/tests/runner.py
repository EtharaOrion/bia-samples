"""Run the submission as a separate, isolated, short-lived process. Never import it.

The grading process must not import the submission, because an imported submission runs
inside the process that is about to grade it and can reach every object that process
holds. So the submission is copied alone into a fresh temporary directory, launched as a
new session leader under a small environment allowlist, given a bound wall-clock ceiling,
and its whole process group is killed in a `finally` block whether it exited, hung, or
crashed. Nothing it wrote outside its own directory is read.

The kill is what makes the ORDERING checker's window closed: by the time tests/grade.py
digests the corpus, no process belonging to the submission is alive to rewrite it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ENTRY = "generator.py"
CORPUS_NAME = "corpus.jsonl"
MANIFEST_NAME = "coverage_manifest.json"
RUN_RECORD_NAME = "run.json"

# The bound wall-clock ceiling for one generator invocation, in seconds. It is a bound
# value rather than a tuned one: a generator that cannot emit the frozen token budget
# inside it has not produced a corpus.
GENERATOR_TIMEOUT_SECONDS = 180

# The whole environment the submission sees. Nothing else is inherited, so no secret,
# no host locale and no salted hash seed reaches it.
# TIKTOKEN_CACHE_DIR is forwarded because both Dockerfiles bake the GPT-2 BPE into the
# image at build time precisely so the graded, egress-free run never resolves a tokenizer
# over the network. Withholding it from the generator subprocess made every submission die
# on a DNS lookup for openaipublic.blob.core.windows.net and emit no corpus at all, which
# the grader then read as an empty corpus rather than as a broken harness. This widens no
# grading surface: it exposes a read-only cache path, not a submission-controlled input.
ENV_ALLOWLIST = ("PATH", "TIKTOKEN_CACHE_DIR")


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    workdir: Path
    timed_out: bool = False
    launched: bool = True
    failure: str = ""
    artifacts: dict = field(default_factory=dict)

    def artifact(self, name: str):
        path = self.artifacts.get(name)
        return path if path is not None and path.is_file() else None


def _environment(home: Path) -> dict:
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["HOME"] = str(home)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _stage(submission: Path, workdir: Path) -> Path:
    sandbox = workdir / "sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    for entry in sorted(submission.iterdir()):
        if entry.name in ("__pycache__", ".git"):
            continue
        if entry.is_dir():
            shutil.copytree(entry, sandbox / entry.name, symlinks=False)
        else:
            shutil.copy2(entry, sandbox / entry.name)
    return sandbox


def run(submission: Path, workdir: Path) -> RunResult:
    """Launch the submission's generator once and collect the artifacts it left behind."""
    workdir.mkdir(parents=True, exist_ok=True)
    sandbox = _stage(Path(submission), workdir)
    artifacts = {
        "corpus": sandbox / CORPUS_NAME,
        "manifest": sandbox / MANIFEST_NAME,
        "run_record": sandbox / RUN_RECORD_NAME,
    }
    entry = sandbox / ENTRY
    if not entry.is_file():
        return RunResult(
            returncode=127,
            stdout="",
            stderr="",
            workdir=sandbox,
            launched=False,
            failure="submission-entry-absent",
            artifacts=artifacts,
        )

    argv = [
        "python3",
        ENTRY,
        "--out",
        CORPUS_NAME,
        "--manifest",
        MANIFEST_NAME,
        "--run-record",
        RUN_RECORD_NAME,
    ]
    process = None
    timed_out = False
    stdout, stderr, code = "", "", 1
    try:
        process = subprocess.Popen(  # noqa: S603 - argv is fixed, never shell-interpolated
            argv,
            cwd=str(sandbox),
            env=_environment(sandbox),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=GENERATOR_TIMEOUT_SECONDS)
            code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr, code = "", "generator-timeout", 124
    finally:
        if process is not None:
            _terminate_group(process)

    return RunResult(
        returncode=code,
        stdout=stdout or "",
        stderr=stderr or "",
        workdir=sandbox,
        timed_out=timed_out,
        launched=True,
        failure="generator-timeout" if timed_out else "",
        artifacts=artifacts,
    )


def _terminate_group(process) -> None:
    """Kill the whole process group, so a forked child cannot outlive the grade."""
    try:
        group = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        group = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if group is None:
            break
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, PermissionError, OSError):
            break
        try:
            process.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    except OSError:
        pass

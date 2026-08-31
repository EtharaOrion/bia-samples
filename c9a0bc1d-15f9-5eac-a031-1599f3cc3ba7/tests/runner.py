#!/usr/bin/env python3
"""Isolation. The grading process never imports the submission.

The submission is data. The verifier does not run it, does not read the store the agent's own
image carried, and does not accept any digest or order the submission asserted. It launches
the BUNDLE's own minter, over the BUNDLE's frozen ledger source and frozen instance family, as
a NEW SESSION LEADER under a five-key environment allowlist, materialises the store into a
fresh temporary directory, reads it back, and kills the whole process group in a `finally`
block whether the run succeeded, failed or timed out.

That is what makes the realised attestation order and every atom digest quantities the
verifier established rather than quantities the submission reported. A submission that hands
back a store of its own has handed back nothing the grader looks at.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")

DEFAULT_TIMEOUT_SECONDS = 300


class RunFailed(Exception):
    """The harness mint did not produce a store. Never a score, always a reason."""


def _environment():
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONHASHSEED"] = "0"
    return env


def _read_store(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    atoms = {}
    for entry in sorted((root / "atoms").glob("*.json")):
        record = json.loads(entry.read_text(encoding="utf-8"))
        atoms[str(record["atom_id"])] = record
    attestations = []
    for entry in sorted((root / "seals").glob("*.json")):
        attestations.append(json.loads(entry.read_text(encoding="utf-8")))
    return {
        "schema": manifest.get("schema"),
        "atom_count": int(manifest.get("atom_count", 0)),
        "atoms": atoms,
        "attestations": attestations,
    }


def materialise(bundle: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Re-mint the store from the bundle's frozen inputs and read it back."""
    bundle = Path(bundle).resolve()
    minter = bundle / "environment" / "mint_store.py"
    source = bundle / "environment" / "ledger_source.json"
    instances = bundle / "environment" / "instances.json"
    for required in (minter, source, instances):
        if not required.is_file():
            raise RunFailed("harness input absent: " + required.as_posix())

    scratch = tempfile.mkdtemp(prefix="oer30-mint-")
    process = None
    try:
        out = Path(scratch) / "store"
        argv = [
            sys.executable,
            "-I",
            "-S",
            str(minter),
            "--source",
            str(source),
            "--instances",
            str(instances),
            "--out",
            str(out),
        ]
        process = subprocess.Popen(
            argv,
            cwd=scratch,
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunFailed("the harness mint exceeded " + str(timeout) + " seconds")
        if process.returncode != 0:
            raise RunFailed(
                "the harness mint exited "
                + str(process.returncode)
                + ": "
                + stderr.decode("utf-8", "replace").strip()[:400]
            )
        if not (out / "manifest.json").is_file():
            raise RunFailed("the harness mint produced no store manifest")
        return _read_store(out)
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for entry in sorted(Path(scratch).rglob("*"), reverse=True):
            try:
                entry.unlink() if entry.is_file() else entry.rmdir()
            except OSError:
                pass
        try:
            Path(scratch).rmdir()
        except OSError:
            pass

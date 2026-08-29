#!/usr/bin/env python3
"""Run the submission in isolation, then run the FROZEN harness in the verifier's process.

The isolation boundary is the whole point of this file and it has two halves.

First half: the submission is copied ALONE into a fresh temporary directory and
launched as a NEW SESSION LEADER under a small environment allowlist. Its stdout is
captured. The whole process group is killed in a `finally` block, so a submission
that forks, backgrounds a child, or blocks on something leaves nothing behind. The
subprocess is asked for exactly one thing: the list of byte strings its
`build_vocab` returned, plus two optional declarations it is allowed to make about
itself.

Second half: the harness is NOT run inside that subprocess. It is imported from the
delivery unit and run here, in the verifier's own process, over the vocabulary the
subprocess handed back. That is what makes every number in the telemetry a number
the verifier produced. A submission cannot reach the compute counter, the model
state, the evaluation corpus, the denominator or the schedule, because none of them
ever exist inside its process.

tests/grade.py never imports the submission. Neither does this file: the submission
is executed as a separate program and is only ever read from through a pipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

EXTRACT = '''
import json, sys, importlib.util
spec = importlib.util.spec_from_file_location("submission_tokenizer", "tokenizer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
train = open(sys.argv[1], "rb").read()
budget = int(sys.argv[2])
vocab = module.build_vocab(train, budget)
rows = []
for item in vocab or []:
    rows.append((item.encode("utf-8") if isinstance(item, str) else bytes(item)).hex())
halt = getattr(module, "HALT_AT_UPDATES", None)
report = getattr(module, "REPORT", None)
print("OER15-VOCAB " + json.dumps({
    "vocab": rows,
    "halt_at": None if halt is None else int(halt),
    "report": report if isinstance(report, dict) else {},
}, separators=(",", ":")))
'''

ENV_ALLOWLIST = ("PATH", "LANG", "TMPDIR")
TIMEOUT_SECONDS = 600


def _environment() -> dict:
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LC_ALL"] = "C"
    return env


def extract(submission: Path, train_path: Path, budget: int) -> dict:
    """Everything the submission is allowed to contribute, and nothing else."""
    workdir = tempfile.mkdtemp(prefix="oer15-submission-")
    process = None
    try:
        source = submission / "tokenizer.py"
        if not source.is_file():
            return {"error": "submission-absent", "detail": str(source) + " is not a file"}
        shutil.copy2(source, Path(workdir) / "tokenizer.py")
        (Path(workdir) / "extract.py").write_text(EXTRACT, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "extract.py", str(train_path), str(budget)],
            cwd=workdir,
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            out, err = process.communicate(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return {"error": "submission-timeout", "detail": "no vocabulary within the bound wall time"}
        if process.returncode != 0:
            return {"error": "submission-crashed", "detail": (err or "").strip()[-400:]}
        for line in (out or "").splitlines():
            if line.startswith("OER15-VOCAB "):
                return {"payload": json.loads(line[len("OER15-VOCAB "):])}
        return {"error": "submission-returned-no-vocabulary", "detail": (out or "").strip()[-400:]}
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        shutil.rmtree(workdir, ignore_errors=True)


def run(delivery_root: Path, submission: Path, out: Path) -> int:
    environment = delivery_root / "environment"
    sys.path.insert(0, str(environment))
    import harness  # frozen bundle code, run in the verifier's own process

    manifest = harness.manifest()
    train_path = environment / "corpus" / "train.txt"
    eval_bytes = (environment / "corpus" / "eval.txt").read_bytes()

    got = extract(submission, train_path, int(manifest["vocab_budget"]))
    if "payload" not in got:
        record = {
            "schema": "oer15.run_record/v1",
            "error": got.get("error"),
            "detail": got.get("detail"),
            "manifest": manifest,
            "telemetry": {},
            "submitted_vocab_digest": "",
            "eval_corpus_digest": hashlib.sha256(eval_bytes).hexdigest(),
            "eval_corpus_bytes": len(eval_bytes),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return 2

    payload = got["payload"]
    entries = [bytes.fromhex(item) for item in payload.get("vocab") or []]
    telemetry = harness.run(
        entries, halt_at=payload.get("halt_at"), reported=payload.get("report")
    )
    record = {
        "schema": "oer15.run_record/v1",
        "error": None,
        "detail": "",
        "manifest": manifest,
        "telemetry": telemetry,
        "submitted_vocab_digest": harness.submission_digest(entries),
        "eval_corpus_digest": hashlib.sha256(eval_bytes).hexdigest(),
        "eval_corpus_bytes": len(eval_bytes),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolate the submission, run the frozen harness.")
    parser.add_argument("--delivery-root", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return run(Path(args.delivery_root), Path(args.submission), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The verifier-owned harness for OER-12.

Two jobs, kept apart on purpose.

1. It is the ONLY implementation of the harness's own three-way parse classification.
   That classification is computed by the verifier's process from the frozen corpus
   bytes, independently of whatever the submission produced, so the DIVERGENCE checker
   is comparing two independent readings of the same bytes rather than a reading
   against itself.

2. It launches the submission in isolation. The submission is COPIED ALONE into a
   fresh temporary directory, launched as a new session leader under a small
   environment allowlist, its output is captured, and the whole process group is killed
   in a `finally` block. THE GRADING PROCESS NEVER IMPORTS THE SUBMISSION.

Everything this module writes into the harness directory carries writer
"verifier-harness". The checkers refuse any document whose writer is not that, which
is how "never a number the submission reported" is made machine-checkable rather than
asserted.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field

HARNESS_WRITER = "verifier-harness"
SUBMISSION_WRITER = "submission"

KNOWN_KEYS = ("id", "lang", "title", "body", "url")
REQUIRED_KEYS = ("id", "lang", "title", "body")

VERDICT_FAILED = "parse-failed"
VERDICT_VALID = "parse-valid"
VERDICT_PARTIAL = "parse-partial"
VERDICTS = (VERDICT_FAILED, VERDICT_VALID, VERDICT_PARTIAL)

FAILED_REASONS = (
    "directive-malformed",
    "field-key-unknown",
    "record-id-absent",
    "record-id-duplicated",
)
VALID_REASONS = ("record-complete",)
PARTIAL_REASONS = (
    "required-field-absent",
    "record-unterminated",
    "continuation-dangling",
)

POLICY_OUTCOME = {
    "quarantine-partial": "partial-quarantined",
    "complete-partial": "partial-completed",
    "truncate-partial": "partial-truncated",
}
POLICY_ADMITS = {
    "quarantine-partial": False,
    "complete-partial": True,
    "truncate-partial": True,
}

# The environment allowlist handed to a launched submission. Nothing else crosses.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED")

SUBMISSION_TIMEOUT_SEC = 432


@dataclass
class Record:
    ordinal: str
    directives: list = field(default_factory=list)
    malformed: bool = False
    terminated: bool = False


def split_records(text: str) -> list:
    """Split the raw corpus into records. Comments and blank lines never enter one."""
    records: list = []
    current: Record | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if current is None:
            current = Record(ordinal="r%02d" % (len(records) + 1))
        if line == "@end":
            current.terminated = True
            records.append(current)
            current = None
            continue
        if line.startswith("@") or line.startswith("+"):
            marker, rest = line[0], line[1:]
            key, _, payload = rest.partition(" ")
            current.directives.append((marker, key.strip(), payload.strip()))
            continue
        current.malformed = True
    if current is not None:
        records.append(current)
    return records


def classify_record(record: Record, schema_version: int) -> tuple:
    """The declared classification order, first rule that fires decides the verdict."""
    if record.malformed:
        return VERDICT_FAILED, "directive-malformed"
    keys = [key for _marker, key, _payload in record.directives]
    if any(key not in KNOWN_KEYS for key in keys):
        return VERDICT_FAILED, "field-key-unknown"
    ids = [p for marker, key, p in record.directives if marker == "@" and key == "id"]
    if not ids:
        return VERDICT_FAILED, "record-id-absent"
    if len(ids) > 1:
        return VERDICT_FAILED, "record-id-duplicated"
    opened = {key for marker, key, _p in record.directives if marker == "@"}
    if any(key not in opened for key in REQUIRED_KEYS):
        return VERDICT_PARTIAL, "required-field-absent"
    if not record.terminated:
        return VERDICT_PARTIAL, "record-unterminated"
    if int(schema_version) >= 2:
        body_lines = [
            (marker, payload)
            for marker, key, payload in record.directives
            if key == "body"
        ]
        if body_lines and body_lines[-1][0] == "+" and body_lines[-1][1] == "":
            return VERDICT_PARTIAL, "continuation-dangling"
    return VERDICT_VALID, "record-complete"


def record_id(record: Record) -> str:
    for marker, key, payload in record.directives:
        if marker == "@" and key == "id":
            return payload
    return ""


def record_text(record: Record) -> str:
    """The trainable text of a record: title then body, continuations joined."""
    parts = []
    for _marker, key, payload in record.directives:
        if key in ("title", "body") and payload:
            parts.append(payload)
    return " ".join(parts)


def classify_corpus(text: str, schema_version: int) -> list:
    """The harness's own three-way classification of the frozen corpus bytes."""
    rows = []
    for record in split_records(text):
        verdict, reason = classify_record(record, schema_version)
        rows.append(
            {
                "ordinal": record.ordinal,
                "id": record_id(record),
                "verdict": verdict,
                "reason": reason,
            }
        )
    return rows


def tokenize(text: str) -> list:
    """The harness token accounting. Whitespace tokens, lowercased, order preserved."""
    return [token for token in text.lower().split() if token]


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_ledger(text: str, schema_version: int) -> dict:
    return {
        "writer": HARNESS_WRITER,
        "schema_version": int(schema_version),
        "corpus_digest": digest_text(text),
        "records": classify_corpus(text, schema_version),
    }


def admitted_ordinals(rows: list, policy: str, eval_ordinals: list) -> list:
    """Which records reach the training feed under a declared policy."""
    admits_partial = POLICY_ADMITS.get(policy, False)
    out = []
    for row in rows:
        if row["ordinal"] in eval_ordinals:
            continue
        if row["verdict"] == VERDICT_VALID:
            out.append(row["ordinal"])
        elif row["verdict"] == VERDICT_PARTIAL and admits_partial:
            out.append(row["ordinal"])
    return out


def feed_ledger(text: str, schema_version: int, policy: str, eval_ordinals: list,
                budget_tokens: int) -> dict:
    """The feed the harness data loader would carry, counted AS FED."""
    records = {r.ordinal: r for r in split_records(text)}
    rows = classify_corpus(text, schema_version)
    ordinals = admitted_ordinals(rows, policy, eval_ordinals)
    tokens: list = []
    for ordinal in ordinals:
        tokens.extend(tokenize(record_text(records[ordinal])))
    return {
        "writer": HARNESS_WRITER,
        "accounting": "as-fed",
        "budget_tokens": int(budget_tokens),
        "fed_tokens": len(tokens),
        "fed_ordinals": ordinals,
        "feed_digest": digest_text(" ".join(tokens)),
    }


def _child_environment() -> dict:
    return {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}


def run_submission(submission: pathlib.Path, timeout_sec: int = SUBMISSION_TIMEOUT_SEC) -> dict:
    """Copy the submission ALONE into a fresh directory and run it in its own session.

    The grading process never imports the submission. It is launched as a separate
    process in a directory it does not share with the grader, under an environment
    allowlist, and its whole process group is killed in a finally block whether it
    exits, times out, or raises.
    """
    workdir = tempfile.mkdtemp(prefix="oer12-submission-")
    process = None
    try:
        target = pathlib.Path(workdir) / "submission.py"
        shutil.copyfile(str(submission), str(target))
        process = subprocess.Popen(
            ["python3", "submission.py"],
            cwd=workdir,
            env=_child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            out, err = process.communicate(timeout=timeout_sec)
            code = process.returncode
        except subprocess.TimeoutExpired:
            out, err, code = "", "submission-timeout", 124
        return {"exit_code": code, "stdout": out, "stderr": err, "workdir": workdir}
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def write_harness_documents(target: pathlib.Path, documents: dict) -> list:
    """Write every harness telemetry document, each stamped with the harness writer."""
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in sorted(documents.items()):
        body = dict(payload)
        body["writer"] = HARNESS_WRITER
        path = target / (name + ".json")
        path.write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(path.name)
    return written


def main() -> int:
    print(json.dumps({"harness": "oer12", "writer": HARNESS_WRITER}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

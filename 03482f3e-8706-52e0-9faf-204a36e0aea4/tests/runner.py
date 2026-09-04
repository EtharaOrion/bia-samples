#!/usr/bin/env python3
"""Run the submission in isolation, then run the FROZEN harness in the verifier's process.

The isolation boundary is the whole point of this file and it has three parts.

First: the submission is copied ALONE into a fresh temporary directory and
launched as a NEW SESSION LEADER under a small environment allowlist. Its stdout
is captured. The whole process group is killed in a `finally` block, so a
submission that forks, backgrounds a child, or blocks on something leaves nothing
behind. The subprocess is asked for exactly one thing: the list of byte strings
its `build_vocab` returned, plus two optional declarations it is allowed to make
about itself.

Second: the training run is NOT performed inside that subprocess. The harness is
imported from the delivery unit and run here, in the verifier's own process, over
the vocabulary the subprocess handed back. That is what makes every number in the
telemetry a number the verifier produced. A submission cannot reach the step
counter, the parameters, the evaluation slice, the denominator or the schedule,
because none of them ever exist inside its process.

Third: the EVALUATION SLICE is resolved here and only here, from the pin in
tests/anchors.json, over a FineWeb validation shard that tests/Dockerfile stages
into the verifier image alone. `environment/harness.py` resolves no evaluation
split of its own; it is handed the bytes. A delivery unit whose `environment/`
was tampered with therefore cannot move the surface the grade is computed on.

This file also runs the PAIRED BASELINE: a second training run of the same
canonical decoder, at the same seed, over the same shards, at the same step
budget, against the same held-out slice, on the vocabulary the handed default
construction produces at its grid maximum. Its bits per byte is the floor of the
reward scale. It is measured rather than recalled, so no reading of the graded
metric appears as a literal anywhere in this bundle.

tests/grade.py never imports the submission. Neither does this file: the
submission is executed as a separate program and is only ever read from through a
pipe.
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

ANCHORS_PATH = Path(__file__).resolve().parent / "anchors.json"

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
halt = getattr(module, "HALT_AT_STEPS", None)
report = getattr(module, "REPORT", None)
print("OER15-VOCAB " + json.dumps({
    "vocab": rows,
    "halt_at": None if halt is None else int(halt),
    "report": report if isinstance(report, dict) else {},
}, separators=(",", ":")))
'''

ENV_ALLOWLIST = ("PATH", "LANG", "TMPDIR")
TIMEOUT_SECONDS = 3600


def anchors() -> dict:
    payload = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    block = payload.get("verifier_operating_points")
    if not isinstance(block, dict):
        raise SystemExit(
            "verifier anchors malformed: " + str(ANCHORS_PATH)
            + " carries no verifier_operating_points block"
        )
    return block


def held_out_root() -> Path:
    return Path(os.environ.get("OER15_HELDOUT_ROOT", "/verifier/data/fineweb10B"))


def held_out_slice(harness, block: dict) -> dict:
    """The evaluation bytes, resolved from the verifier's pin and from nowhere else."""
    path = held_out_root() / str(block["held_out_shard"])
    if not path.is_file():
        raise SystemExit(
            "the held-out FineWeb validation shard is absent at " + str(path)
            + "; the verifier image stages it and grading cannot proceed without it"
        )
    shard_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    pinned = block.get("held_out_shard_sha256")
    if isinstance(pinned, str) and pinned and pinned != shard_sha:
        raise SystemExit(
            "the held-out shard digests " + shard_sha[:16] + " against the pinned "
            + str(pinned)[:16]
        )
    ids = harness.read_shard(path)
    start = int(block["held_out_token_offset"])
    count = int(block["held_out_token_count"])
    data = harness.gpt2_decoder().decode_bytes(
        [int(value) for value in ids[start:start + count]]
    )
    return {
        "bytes": data,
        "source": "verifier:" + path.name + "[" + str(start) + ":" + str(start + count) + "]",
        "shard_sha256": shard_sha,
        "shard_sha256_pinned": bool(isinstance(pinned, str) and pinned),
        "slice_sha256": hashlib.sha256(data).hexdigest(),
        "slice_bytes": len(data),
    }


def _environment() -> dict:
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LC_ALL"] = "C"
    return env


def extract(submission: Path, fit_bytes: bytes, budget: int) -> dict:
    """Everything the submission is allowed to contribute, and nothing else."""
    workdir = tempfile.mkdtemp(prefix="oer15-submission-")
    process = None
    try:
        source = submission / "tokenizer.py"
        if not source.is_file():
            return {"error": "submission-absent", "detail": str(source) + " is not a file"}
        shutil.copy2(source, Path(workdir) / "tokenizer.py")
        fit_path = Path(workdir) / "fit.bin"
        fit_path.write_bytes(fit_bytes)
        (Path(workdir) / "extract.py").write_text(EXTRACT, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "extract.py", str(fit_path), str(budget)],
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


def paired_baseline(harness, environment: Path, fit_bytes: bytes, budget: int,
                    block: dict, evaluation: dict) -> dict:
    """The floor of the reward scale, trained here rather than recalled from a file.

    The option row comes from the VERIFIER's copy in tests/anchors.json, not from
    environment/default_tokenizer.py, so a tampered delivery unit cannot lower the
    bar by handing the baseline a worse construction to run.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "oer15_default", str(environment / "default_tokenizer.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = dict(block["paired_baseline_row"])
    entries = module.build_vocab(fit_bytes, budget, **row)
    telemetry = harness.run(
        entries,
        eval_bytes=evaluation["bytes"],
        eval_source=evaluation["source"],
    )
    graded = telemetry.get("graded") or {}
    return {
        "construction": str(block.get("paired_baseline_construction") or ""),
        "option_row": row,
        "vocabulary_size": (telemetry.get("vocabulary") or {}).get("size"),
        "bits_per_byte": graded.get("bits_per_byte"),
        "bits": graded.get("bits"),
        "denominator_bytes": graded.get("denominator_bytes"),
        "steps": graded.get("steps"),
        "snapshot_digest": graded.get("snapshot_digest"),
    }


def run(delivery_root: Path, submission: Path, out: Path) -> int:
    environment = delivery_root / "environment"
    sys.path.insert(0, str(environment))
    import harness  # frozen bundle code, run in the verifier's own process

    manifest = harness.manifest()
    block = anchors()
    budget = int(manifest["tokenizer"]["vocab_budget"])
    fit_bytes = harness.fit_bytes()
    evaluation = held_out_slice(harness, block)

    skeleton = {
        "schema": "oer15.run_record/v2",
        "manifest": manifest,
        "substrate": harness.substrate(),
        "substrate_digest": harness.substrate_digest(),
        "held_out": {
            "source": evaluation["source"],
            "slice_sha256": evaluation["slice_sha256"],
            "slice_bytes": evaluation["slice_bytes"],
            "shard_sha256": evaluation["shard_sha256"],
            "shard_sha256_pinned": evaluation["shard_sha256_pinned"],
            "resolved_from": "tests/anchors.json",
            "reachable_from_environment": (
                environment / "corpus" / str(block["held_out_shard"])
            ).exists(),
        },
    }

    got = extract(submission, fit_bytes, budget)
    if "payload" not in got:
        record = dict(skeleton)
        record.update(
            {
                "error": got.get("error"),
                "detail": got.get("detail"),
                "telemetry": {},
                "paired_baseline": {},
                "submitted_vocab_digest": "",
            }
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return 2

    payload = got["payload"]
    entries = [bytes.fromhex(item) for item in payload.get("vocab") or []]
    telemetry = harness.run(
        entries,
        eval_bytes=evaluation["bytes"],
        eval_source=evaluation["source"],
        halt_at=payload.get("halt_at"),
        reported=payload.get("report"),
        snapshot_path=str(out.parent / "snapshot.pt"),
    )
    baseline = paired_baseline(harness, environment, fit_bytes, budget, block, evaluation)

    record = dict(skeleton)
    record.update(
        {
            "error": None,
            "detail": "",
            "telemetry": telemetry,
            "paired_baseline": baseline,
            "submitted_vocab_digest": harness.submission_digest(entries),
        }
    )
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

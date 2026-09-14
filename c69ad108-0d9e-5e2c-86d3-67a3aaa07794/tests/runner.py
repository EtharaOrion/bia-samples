"""Isolated execution of the submission, and the telemetry the verifier itself produces.

Two jobs live here and they are kept apart on purpose.

`run_submission` executes the agent's generator. The grading process never imports
it. The submission file alone is copied into a fresh temporary directory, launched
as a new session leader under a small environment allowlist, its output captured,
and the whole process group killed in a `finally` block whether it exited, hung or
crashed. The one thing placed alongside it is `corpus_diversity.py` on PYTHONPATH,
read-only, because that is exactly the frozen tool the agent's own sandbox provides
and a submission graded without the tool the task tells it to use would be graded
on a surface the agent never had.

`produce_telemetry` is the verifier measuring. Every number a checker later reads is
computed here, in the verifier's own process. The corpus bytes the run left on disk
are packed into a nanoGPT training shard by the frozen `shards` module, the frozen
decoder is trained on that shard by the frozen `trainer` module, and the frozen
evaluator reads the trained parameters against the held-out token stream the
verifier owns. Nothing the submission printed reaches a graded number. The
submission's run report is carried into the telemetry under `self_report` and is
used for exactly one thing: the divergence checker compares it against the
recomputation. It is never a source of a score.

The held-out surface is pinned to the verifier's own copy and is absent from every
agent-visible container. `tests/benchmark_held_out.jsonl` carries FineWeb validation
text, the substrate's declared validation split, and it is read from this directory
rather than resolved from anything the submission can reach. The agent's
`environment/frozen/benchmark_dev.jsonl` is authored dev material, disjoint from it,
and is never read on this path.

The exit status is recorded and never trusted. That is the whole slot: a generator
that collapses to a low-diversity mode partway through keeps emitting, keeps the
document count and the format right, and exits zero. The telemetry therefore carries
the exit status beside the recomputed diversity profile so a checker can grade the
corpus rather than the status.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import diversity
import shards
import trainer

TELEMETRY_SCHEMA = "forge.oer19.telemetry/v2"
PRODUCED_BY = "tests/runner.py"

# The environment a submission is launched under. Nothing else crosses, so a
# submission cannot read a verifier secret or a host path out of the environment.
ENV_ALLOWLIST = ("PATH", "PYTHONPATH", "PYTHONHASHSEED", "LC_ALL", "LANG", "HOME", "TMPDIR")

DEFAULT_TIMEOUT_SECONDS = 900


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_submission(script: Path, tools: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Execute the submission in isolation and return what it left behind.

    The temporary directory is fresh per call and removed afterwards, so no run
    can read a previous run's corpus and present it as its own.
    """
    script = Path(script)
    workspace = Path(tempfile.mkdtemp(prefix="oer19-run-"))
    process = None
    try:
        local = workspace / script.name
        shutil.copy2(script, local)
        env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(tools) + (os.pathsep + existing if existing else "")
        if "TIKTOKEN_CACHE_DIR" in os.environ:
            env["TIKTOKEN_CACHE_DIR"] = os.environ["TIKTOKEN_CACHE_DIR"]
        process = subprocess.Popen(
            [sys.executable, str(local), str(workspace)],
            cwd=str(workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            status = process.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, status = b"", b"submission-timeout", 124
        return {
            "exit_status": int(status),
            "stdout": stdout.decode("utf-8", "replace")[-4000:],
            "stderr": stderr.decode("utf-8", "replace")[-4000:],
            "sha256": _sha256(script),
            "corpus": read_corpus(workspace / "corpus.jsonl"),
            "self_report": read_report(workspace / "run_report.json"),
        }
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        shutil.rmtree(workspace, ignore_errors=True)


def read_corpus(path: Path) -> list:
    """Parse the emitted corpus. A malformed line is dropped and counted, never guessed."""
    rows = []
    if not Path(path).is_file():
        return rows
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "text" in row:
                rows.append({"seq": row.get("seq"), "text": str(row.get("text"))})
    return rows


def read_report(path: Path):
    if not Path(path).is_file():
        return None
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def load_benchmark(path: Path) -> list:
    """The verifier's held-out FineWeb validation items, provenance line excluded.

    The first line of the file is a provenance record naming the upstream dataset
    and carries no item. It is dropped here by shape rather than by position, so a
    file whose provenance line moves is still read correctly.
    """
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if row.get("kind") == "provenance" or "text" not in row:
                continue
            rows.append({"id": str(row.get("id")), "text": str(row["text"])})
    return rows


def produce_telemetry(
    corpus_rows,
    held_out,
    generator_sha256: str,
    generator_exit_status: int,
    self_report,
    bounds: dict,
) -> dict:
    """The verifier's own measurement of one run. Every graded number originates here."""
    texts = [str(row["text"]) for row in corpus_rows]

    # The diversity statistic reads a bound excerpt of each document rather than the
    # whole of it. The statistic is quadratic in the document count and the bound
    # document is a full 1024-token sequence, so an unbounded read would make the
    # measurement cost dominate the run without making it more discriminating. The
    # excerpt length is bound in tests/checkers.yaml and published to the agent, so
    # the agent computes the identical number over its own corpus.
    excerpt = int(bounds["diversity_excerpt_chars"])
    sampled = [text[:excerpt] for text in texts]

    profile = diversity.profile(
        sampled,
        int(bounds["segment_count"]),
        int(bounds["prefix_points"]),
        float(bounds["mode_share_threshold"]),
    )
    hits = diversity.near_duplicates(
        sampled,
        [(row["id"], row["text"][:excerpt]) for row in held_out],
        float(bounds["near_duplicate_threshold"]),
    )

    workspace = Path(tempfile.mkdtemp(prefix="oer19-shard-"))
    try:
        built = shards.build_train_shard(texts, workspace)
        shard_tokens = shards.read_shard(built["shard"])
        val_tokens = shards.build_val_tokens([row["text"] for row in held_out])
        trained = trainer.train(
            shard_tokens,
            val_tokens,
            steps=int(bounds["bound_steps"]),
            points=tuple(int(value) for value in bounds["scheduled_points"]),
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    points, digests = [], {}
    for reading in trained.readings:
        points.append(
            {
                "step": reading.step,
                "nll_sum": reading.nll_sum,
                "tokens": reading.tokens,
                "loss": reading.loss,
            }
        )
        digests[str(reading.step)] = reading.digest

    bound_point = int(bounds["bound_evaluation_point"])
    graded = trained.at(bound_point)

    return {
        "schema": TELEMETRY_SCHEMA,
        "produced_by": PRODUCED_BY,
        "substrate": {
            "declaration_version": trainer.DECLARATION_VERSION,
            "vocab_size": trainer.VOCAB_SIZE,
            "num_layers": trainer.NUM_LAYERS,
            "model_dim": trainer.MODEL_DIM,
            "head_dim": trainer.HEAD_DIM,
            "num_heads": trainer.NUM_HEADS,
            "seq_len": trainer.SEQ_LEN,
            "batch_tokens_per_step": trainer.BATCH_TOKENS_PER_STEP,
        },
        "generator": {
            "exit_status": int(generator_exit_status),
            "sha256": str(generator_sha256),
            "exit_status_is_not_evidence": True,
        },
        "corpus": {
            "documents": profile.samples,
            "emission_sequence": [row.get("seq") for row in corpus_rows],
            "normalization": profile.normalization,
            "diversity_excerpt_chars": excerpt,
            "segment_count": profile.segment_count,
            "segment_distinct_ngram_ratio": list(profile.segment_distinct_ngram_ratio),
            "corpus_distinct_ngram_ratio": profile.corpus_distinct_ngram_ratio,
            "prefix_checkpoints": list(profile.prefix_checkpoints),
            "prefix_mode_share": list(profile.prefix_mode_share),
            "max_mode_share": profile.max_mode_share,
            "benchmark_near_duplicate_hits": hits,
            "near_duplicate_threshold": float(bounds["near_duplicate_threshold"]),
            "mode_share_threshold": float(bounds["mode_share_threshold"]),
            "shard_tokens": int(built["shard_tokens"]),
            "short_documents": int(built["short_documents"]),
        },
        "training": {
            "forward_passes": trained.counters.forward_passes,
            "backward_passes": trained.counters.backward_passes,
            "steps": trained.counters.optimizer_steps,
            "tokens_fed": trained.counters.tokens_fed,
            "epochs_completed": trained.counters.epochs_completed,
            "terminated_early": bool(trained.counters.terminated_early),
            "counter_source": trainer.COUNTER_SOURCE,
            "bound_steps": int(bounds["bound_steps"]),
            "bound_tokens_fed": int(bounds["bound_tokens_fed"]),
            "bound_epochs": int(bounds["bound_epochs"]),
            "bound_corpus_documents": int(bounds["corpus_documents"]),
        },
        "evaluation": {
            "held_out_items": len(held_out),
            "held_out_source": "tests/benchmark_held_out.jsonl",
            "bound_point": bound_point,
            "scheduled_points": [int(value) for value in bounds["scheduled_points"]],
            "points": points,
            "graded_point": bound_point,
            "graded_nll_sum": (graded.nll_sum if graded else None),
            "graded_tokens": (graded.tokens if graded else None),
            "graded_loss": (graded.loss if graded else None),
            "readout": dict(trainer.READOUT),
            "weights_source": trainer.WEIGHTS_SOURCE,
            "parameter_shape_signature": trained.shape_signature,
            "parameter_count": trained.parameter_count,
            "model_state_digest": digests,
            "graded_model_state_digest": digests.get(str(bound_point)),
            "sustain_tolerance": float(bounds["sustain_tolerance"]),
        },
        "self_report": (
            {"present": False}
            if not isinstance(self_report, dict)
            else {
                "present": True,
                "declared_corpus_distinct_ngram_ratio": self_report.get(
                    "declared_corpus_distinct_ngram_ratio"
                ),
                "declared_max_mode_share": self_report.get("declared_max_mode_share"),
                "declared_documents": self_report.get("declared_documents"),
            }
        ),
    }

"""The verifier's own held-out evaluation. VERIFIER IMAGE ONLY.

`tests/grade.py:_held_out_loss` has always done

    import evaluator  # type: ignore  # verifier-image module, GPU side
    return float(evaluator.held_out_loss(state_path, params_digest))

and the pinned image
pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
carries no module of that name. Measured directly in the image:

    python3 -c "import evaluator"  ->  ModuleNotFoundError: No module named 'evaluator'

`_held_out_loss` swallows that ImportError and returns None, which drops every
point, which empties the evaluation series, which is exactly the
`run-produced-no-verifier-state` zero the delivered oracle drive recorded. This
file is that module.

WHAT IT COMPUTES. A real cross-entropy. It rebuilds the frozen architecture at
the executed operating point, loads the parameter tensors the PINNED HARNESS
wrote at that step, and evaluates them on a held-out token stream that exists in
this image only. There is no curve fit, no lookup of an expected series, and no
reference to the submission's own reported numbers. Two submissions whose
weights differ get different losses, and a submission whose weights are worse
gets a worse loss.

WHAT IT REFUSES. The ledger names a `params_digest` for every step. This module
recomputes that digest from the tensor bytes it actually loaded, using the
PINNED HARNESS's own `bia_harness.params_digest`, and returns NaN when the two
disagree. `tests/checkers.py:evaluations` drops a NaN row, so a ledger entry that
does not correspond to the bytes on disk contributes no evaluation instead of
contributing a fabricated one. An unreadable or structurally wrong state file is
refused the same way. This module can therefore lower a score and never raise
one.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any, Dict, Optional

HERE = pathlib.Path(__file__).resolve().parent
AGENT_ENV = pathlib.Path("/app/env")
for _candidate in (HERE, AGENT_ENV):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

HOLDOUT_DOCUMENT = HERE / "holdout.json"

_STATE: Dict[str, Any] = {}


def holdout_document() -> Dict[str, Any]:
    return json.loads(HOLDOUT_DOCUMENT.read_text(encoding="utf-8"))


def _resolved():
    """Model config, held-out batches and device. Built once, reused per state."""
    if _STATE:
        return _STATE
    import torch

    import bia_corpus
    import gpt as gpt_module

    document = holdout_document()
    point = bia_corpus.operating_point()
    if not point.get("corpus"):
        raise RuntimeError(
            "operating point " + repr(point.get("name")) + " binds no corpus, so no held-out split "
            "can be drawn from the delivered closure"
        )
    directory = pathlib.Path(str(document["holdout_shard_dir"]))
    bia_corpus.materialize_shard(
        directory,
        point,
        int(document["holdout_stream_seed"]),
        int(document["holdout_tokens"]),
        name="holdout_000.bin",
    )
    shard = sorted(directory.glob("*.bin"))[0]
    tokens = torch.from_numpy(_read_shard(shard))

    sequences = int(document["eval_batch_sequences"])
    seq_len = int(point["seq_len"])
    span = sequences * seq_len
    batches = []
    for index in range(int(document["eval_batches"])):
        start = index * span
        stop = start + span
        if stop + 1 > tokens.numel():
            break
        batches.append(
            (
                tokens[start:stop].view(sequences, seq_len),
                tokens[start + 1 : stop + 1].view(sequences, seq_len),
            )
        )
    if not batches:
        raise RuntimeError("the held-out stream is shorter than one evaluation batch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = [(inputs.to(device), targets.to(device)) for inputs, targets in batches]
    config = gpt_module.GPTConfig(
        block_size=seq_len,
        vocab_size=int(point["vocab_size"]),
        n_layer=int(point["n_layer"]),
        n_head=int(point["n_head"]),
        n_embd=int(point["n_embd"]),
    )
    _STATE.update(
        {
            "config": config,
            "batches": batches,
            "device": device,
            "gpt": gpt_module,
            "point": point,
            "tokens_evaluated": sum(int(inputs.numel()) for inputs, _ in batches),
            "entropy_floor_nats": bia_corpus.resolved_entropy_floor(point),
        }
    )
    return _STATE


def _read_shard(path: pathlib.Path):
    import numpy

    with path.open("rb") as handle:
        header = numpy.frombuffer(handle.read(1024), dtype=numpy.int32)
        count = int(header[2])
        return numpy.frombuffer(handle.read(count * 2), dtype=numpy.uint16).astype(numpy.int64)


def entropy_floor_nats() -> float:
    """The Bayes-optimal loss of the held-out language. Analytic, not estimated."""
    return float(_resolved()["entropy_floor_nats"])


def held_out_loss(state_path, params_digest: str = "") -> float:
    """The unsmoothed held-out cross-entropy of one harness-owned parameter set.

    Returns NaN, which tests/checkers.py drops, when the state file cannot be
    read, does not fit the frozen architecture, or does not reproduce the
    params_digest the harness recorded for that step.
    """
    import torch

    resolved = _resolved()
    try:
        tensors = torch.load(str(state_path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  an unreadable state drops its point, never grades
        return float("nan")
    if not isinstance(tensors, dict):
        return float("nan")

    if params_digest:
        try:
            import bia_harness

            if bia_harness.params_digest(tensors) != str(params_digest):
                return float("nan")
        except Exception:  # noqa: BLE001
            return float("nan")

    model = resolved["gpt"].GPT(resolved["config"])
    try:
        model.load_state_dict(tensors, strict=True)
    except Exception:  # noqa: BLE001  weights that are not this architecture's drop their point
        return float("nan")
    model.to(resolved["device"]).eval()

    total = 0.0
    counted = 0
    with torch.no_grad():
        for inputs, targets in resolved["batches"]:
            loss = model(inputs, targets)
            count = int(targets.numel())
            total += float(loss) * count
            counted += count
    if counted <= 0:
        return float("nan")
    value = total / counted
    if math.isnan(value) or math.isinf(value):
        return float("nan")
    return float(value)


def describe() -> Dict[str, Any]:
    """What this evaluator is, for a reader who wants it in one place."""
    resolved = _resolved()
    return {
        "kind": "real-held-out-cross-entropy",
        "is_accelerator_measurement": bool(str(resolved["device"]) .startswith("cuda")),
        "operating_point": resolved["point"]["name"],
        "device": str(resolved["device"]),
        "tokens_evaluated_per_state": int(resolved["tokens_evaluated"]),
        "entropy_floor_nats": float(resolved["entropy_floor_nats"]),
        "readout": "raw, unsmoothed, one mean over the whole held-out stream",
        "binds_state_to_ledger": "recomputes bia_harness.params_digest over the loaded tensors and returns NaN on mismatch",
    }


if __name__ == "__main__":
    sys.stdout.write(json.dumps(describe(), indent=2, sort_keys=True) + "\n")

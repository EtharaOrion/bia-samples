"""The verifier's own held-out evaluation. VERIFIER IMAGE ONLY.

`tests/grade.py:_held_out_loss` has always done

    import evaluator  # type: ignore  # verifier-image module, GPU side
    return float(evaluator.held_out_loss(state_path, params_digest))

and the pinned image
pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
carries no module of that name. Measured directly in the image:

    python3 -c "import evaluator"  ->  ModuleNotFoundError: No module named 'evaluator'

`_held_out_loss` catches that ImportError and returns None, `evaluate_ledger`
drops every point that returns None, and the evaluation series therefore comes
out empty on every run whatever the run did. An empty series is the
`run-produced-no-verifier-state` zero, which means no submission could ever be
graded on its merits and the slot could not grade at all. This file is the
missing module.

WHAT IT COMPUTES. A real cross-entropy, unsmoothed, over real held-out tokens.
It rebuilds the frozen architecture, loads the parameter tensors the PINNED
HARNESS wrote at that step, and evaluates them on the held-out split carried at
tests/corpus/held_out_split.json. There is no curve fit, no lookup of the
`golden` series in solution/grounding.yaml, no reference to anything the
submission printed, and no constant. Two states whose weights differ get
different losses, and a worse state gets a worse loss.

WHAT THE SPLIT IS. tests/corpus/held_out_split.json carries 1048576 GPT-2 BPE
token ids copied without transformation out of the upstream FineWeb10B shard
`fineweb_val_000000.bin` at token offset 800000, recorded in that file's own
`held_out_split.provenance` block with its `slice_sha256`. It is upstream data,
not a generated stand-in, and this module verifies that digest over the decoded
bytes before it evaluates anything, so a substituted or truncated payload is
refused rather than silently evaluated.

WHAT IT REFUSES. The harness ledger names a `params_digest` for every step. This
module recomputes that digest from the tensor bytes it actually loaded, using the
PINNED HARNESS's own `bia_harness.params_digest`, and returns NaN when the two
disagree. `tests/checkers.py:evaluations` drops a NaN row, so a ledger entry that
does not correspond to the bytes on disk contributes no evaluation instead of
contributing a fabricated one. An unreadable state file, or one that is not this
architecture's, is refused the same way. This module can therefore only ever
lower a score, never raise one.

WHAT IT BINDS. Every quantity it reads is bound in solution/grounding.yaml under
`bound_state` and mirrored in environment/shape.json, which is the copy this
image carries at /app/env/shape.json: the frozen batch axis is not this module's
to choose, and the readout filter it declares is `raw`, matching
`bound_state.readout_filter_admitted`. The evaluation grid, the sustain window
and the target bar are the grader's and the checkers', never this module's; this
module answers exactly one question, which is what one parameter set scores.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent

# The pinned runtime modules. The verifier image carries its own byte-identical
# copy of the agent-surface modules at /app/env, because Harbor mounts
# /app/submission.py into this image and does not mount /app/env.
AGENT_ENV = pathlib.Path("/app/env")
for _candidate in (HERE, AGENT_ENV):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# The held-out carrier. Verifier side only: tests/Dockerfile COPYs it into this
# image and nothing else does.
SPLIT_DOCUMENT = HERE / "corpus" / "held_out_split.json"

# The readout this module produces, named so a reader can check it against
# solution/grounding.yaml bound_state.readout_filter_admitted rather than trust
# the docstring. No filter, no blend, no window.
READOUT = "raw"

# How many held-out sequences enter one forward pass. This is a MEMORY schedule
# and not a measurement choice: the mean is taken over every token of the split
# whatever this value is, because each partial mean is reweighted by its own
# token count below. It is fixed here rather than read from the environment so
# two verifier runs over the same state produce the same number.
EVAL_SEQUENCES_PER_FORWARD = 32

_STATE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# The held-out split
# ---------------------------------------------------------------------------


def split_document() -> Dict[str, Any]:
    """The carrier. An absent or malformed carrier raises rather than defaults.

    A default here would be a fabricated measurement: the module would go on to
    report a loss over tokens nobody shipped.
    """
    with SPLIT_DOCUMENT.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    block = document["held_out_split"]
    if not isinstance(block, dict):
        raise RuntimeError(
            str(SPLIT_DOCUMENT) + " carries no held_out_split mapping, so this image holds no "
            "split to evaluate on"
        )
    return block


def held_out_tokens() -> Tuple[Any, Dict[str, Any]]:
    """Decode the split and verify it against the digest its own provenance names."""
    import numpy

    block = split_document()
    provenance = dict(block["provenance"])
    raw = base64.b64decode(block["payload_base64"], validate=True)

    declared_bytes = int(provenance["byte_length"])
    if len(raw) != declared_bytes:
        raise RuntimeError(
            "the held-out payload decodes to " + str(len(raw)) + " bytes and its provenance "
            "declares " + str(declared_bytes) + ", so the carrier is not the split it names"
        )
    observed = hashlib.sha256(raw).hexdigest()
    declared_digest = str(provenance["slice_sha256"])
    if observed != declared_digest:
        raise RuntimeError(
            "the held-out payload hashes " + observed + " and its provenance declares "
            + declared_digest + ", so the bytes in this image are not the bytes that were frozen"
        )

    tokens = numpy.frombuffer(raw, dtype=numpy.uint16).astype(numpy.int64)
    declared_count = int(provenance["token_count"])
    if int(tokens.size) != declared_count:
        raise RuntimeError(
            "the held-out payload carries " + str(int(tokens.size)) + " tokens and its provenance "
            "declares " + str(declared_count)
        )
    return tokens, provenance


# ---------------------------------------------------------------------------
# The frozen architecture and the evaluation batches, built once
# ---------------------------------------------------------------------------


def _resolved() -> Dict[str, Any]:
    """Model config, held-out batches and device. Built once, reused per state."""
    if _STATE:
        return _STATE
    import torch

    import bia_loader
    import gpt as gpt_module

    tokens_array, provenance = held_out_tokens()
    tokens = torch.from_numpy(tokens_array.copy())

    seq_len = int(bia_loader.SEQUENCE_LENGTH)
    span = int(EVAL_SEQUENCES_PER_FORWARD) * seq_len
    batches: List[Tuple[Any, Any]] = []
    start = 0
    # One target is the next token, so the last usable window stops one token
    # short of the end. The remainder is dropped rather than padded: a padded
    # row would contribute a token nobody wrote to the mean.
    while start + span + 1 <= int(tokens.numel()):
        stop = start + span
        batches.append(
            (
                tokens[start:stop].view(EVAL_SEQUENCES_PER_FORWARD, seq_len),
                tokens[start + 1 : stop + 1].view(EVAL_SEQUENCES_PER_FORWARD, seq_len),
            )
        )
        start = stop
    if not batches:
        raise RuntimeError("the held-out split is shorter than one evaluation forward pass")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = gpt_module.GPTConfig(
        block_size=seq_len,
        vocab_size=int(bia_loader.VOCAB_SIZE),
        n_layer=int(bia_loader.N_LAYER),
        n_head=int(bia_loader.N_HEAD),
        n_embd=int(bia_loader.N_EMBD),
    )
    _STATE.update(
        {
            "config": config,
            "batches": batches,
            "device": device,
            "gpt": gpt_module,
            "provenance": provenance,
            "sequence_length": seq_len,
            "tokens_evaluated": sum(int(inputs.numel()) for inputs, _ in batches),
        }
    )
    return _STATE


# ---------------------------------------------------------------------------
# The one function tests/grade.py calls
# ---------------------------------------------------------------------------


def held_out_loss(state_path, params_digest: str = "") -> float:
    """The unsmoothed held-out cross-entropy of one harness-owned parameter set.

    Returns NaN, which tests/checkers.py:evaluations drops, when the state file
    cannot be read, does not fit the frozen architecture, or does not reproduce
    the params_digest the harness recorded for that step. Dropping a point is
    always safe for the grade: it can only remove an evaluation, never invent
    one, and a series with too few points is graded as a non-convergence with a
    reason rather than as an absent result.
    """
    import torch

    resolved = _resolved()
    try:
        tensors = torch.load(str(state_path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  an unreadable state drops its point, never grades
        return float("nan")
    if not isinstance(tensors, dict):
        return float("nan")

    # The binding to the ledger. Recomputed with the PINNED HARNESS's own
    # function rather than restated here, because two expressions of one
    # derivation is how two things that must agree come to disagree.
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

    device = resolved["device"]
    total = 0.0
    counted = 0
    try:
        with torch.no_grad():
            for inputs, targets in resolved["batches"]:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                loss = model(inputs, targets)
                count = int(targets.numel())
                total += float(loss) * count
                counted += count
    except Exception:  # noqa: BLE001  a state this architecture cannot run drops its point
        return float("nan")
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if counted <= 0:
        return float("nan")
    value = total / counted
    if math.isnan(value) or math.isinf(value):
        return float("nan")
    return float(value)


def describe() -> Dict[str, Any]:
    """What this evaluator is, in one place, for a reader and for the build check."""
    resolved = _resolved()
    provenance = dict(resolved["provenance"])
    return {
        "kind": "real-held-out-cross-entropy",
        "readout": READOUT,
        "device": str(resolved["device"]),
        "is_accelerator_measurement": str(resolved["device"]).startswith("cuda"),
        "sequence_length": int(resolved["sequence_length"]),
        "sequences_per_forward": int(EVAL_SEQUENCES_PER_FORWARD),
        "forward_passes_per_state": len(resolved["batches"]),
        "tokens_evaluated_per_state": int(resolved["tokens_evaluated"]),
        "split_shard": str(provenance.get("shard", "")),
        "split_token_offset": int(provenance.get("token_offset", 0)),
        "split_slice_sha256": str(provenance.get("slice_sha256", "")),
        "split_is_generated": bool(provenance.get("generated", False)),
        "binds_state_to_ledger": (
            "recomputes bia_harness.params_digest over the loaded tensors and returns NaN on "
            "mismatch, which tests/checkers.py drops"
        ),
    }


if __name__ == "__main__":
    sys.stdout.write(json.dumps(describe(), indent=2, sort_keys=True) + "\n")

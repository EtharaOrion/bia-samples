"""Harness-owned measurement for slot OER-14. Every number on the graded path is born here.

This module is the verifier's own process. It imports the FROZEN substrate from
`environment/substrate.py`, trains the canonical nanoGPT decoder over FineWeb10B text
retokenized with the submitted vocabulary, and hands the resulting telemetry to
`checkers.py`.

THE HELD-OUT SPLIT LIVES HERE AND NOWHERE ELSE. `environment/` carries the FineWeb training
shards and the frozen vocabulary construction window; it carries no validation shard and no
evaluation text. The slice the graded reading is computed over is materialised only in the
verifier image, from the validation shard and the token range pinned in `tests/controls.json`,
which is verifier-only. A solver cannot read it, and no dynamic lookup resolves it: the shard
name, the token offset and the token count are constants of the verifier's own control table.

The second half of that separation is `probe_reading_for`, below. An oracle is an AGENT-PHASE
artifact: Harbor runs `solution/solve.sh` inside the image `environment/Dockerfile` builds,
where the validation shard is absent and must stay absent. So the reference session reads its
own attempts back over a TRAIN shard slice pinned at `session_probe`, and only the verifier
ever resolves `corpus_construction`. Two pins, two surfaces, and the graded one is reachable
from exactly one of them.

It never imports the submission. A submission is a JSON DOCUMENT, and this module reads that
document only to learn which allocation to measure and which attempt record to re-measure.
No number the document carries is copied onto the graded path: every bits-per-byte figure in
the context this builds is one this process computed, from parameters this process trained,
over held-out text this process decoded.

It reads no clock, no network and no environment secret. Model initialization is seeded from
the substrate's frozen seed, so two runs over the same document produce the same parameters
and the same context.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
TESTS = BUNDLE / "tests"
sys.path.insert(0, str(BUNDLE / "environment"))

import substrate  # noqa: E402

CONTROLS = TESTS / "controls.json"

#: The FIXED key list the grading path consumes out of the context. It is a constant of this
#: module, so no submission can add itself to it: there is no code path that appends. Every
#: entry names a harness-owned surface. A key naming a submission field would be a defect and
#: `no_submission_reported_number_on_graded_path` is the checker that reads this list.
CONSUMED_KEYS = (
    "harness.frozen",
    "harness.points",
    "harness.meter",
    "harness.anchors",
    "harness.attempts",
    "harness.graded.bpb",
    "harness.corpus",
    "harness.corpus_probe",
)

#: Fields of the submission document that a grading path must never consume. Mirrors
#: `substrate.VocabSpec.reported_fields()`, which is the frozen statement of the same set.
FORBIDDEN_CONSUMED = (
    "denominator_override",
    "graded_readout_filter",
    "notes",
    "reported_bpb",
)

_MEASURED: dict = {}
_TEXT: dict = {}
_PROBED: dict = {}
_PROBE_TEXT: dict = {}


def controls() -> dict:
    """The harness control table. Generated from solution/grounding.yaml by recompute.py."""
    with CONTROLS.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# The corpora. Two are agent-reachable; the third is the verifier's alone.
# ---------------------------------------------------------------------------


def _heldout_slice_tokens():
    """The raw uint16 token ids of the verifier-owned held-out FineWeb slice."""
    pin = controls()["corpus_construction"]
    return substrate.read_shard_tokens(pin["shard"], int(pin["offset"]), int(pin["stride"]))


def corpora() -> tuple:
    """The construction window, the training text, and the held-out text, in that order.

    The first two come from the train shards the agent image also stages. The third comes
    from the validation shard, which is staged in the verifier image only.
    """
    if not _TEXT:
        table = controls()
        _TEXT["window"] = substrate.vocabulary_window_text()
        _TEXT["train"] = substrate.training_text(int(table["train_text_gpt2_tokens"]))
        _TEXT["heldout"] = substrate.detokenize(_heldout_slice_tokens())
    return _TEXT["window"], _TEXT["train"], _TEXT["heldout"]


def heldout_byte_count() -> int:
    """The denominator. It is the UTF-8 byte length of the held-out FineWeb text, always."""
    return len(corpora()[2])


def heldout_digest() -> str:
    return hashlib.sha256(corpora()[2]).hexdigest()


def _key(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def corpus_probe() -> dict:
    """Measure the construction phase and period off the LIVE staged FineWeb shards.

    Two quantities are read back, each isolated so that its outcome is a function of exactly
    one frozen construction parameter:

      phase  -- the token offset at which the pinned held-out slice actually sits in the
                staged validation shard. The probe reads `stride` tokens at the frozen
                `offset` and hashes the raw uint16 bytes; if that digest is the pinned
                `slice_sha256` the phase is established at that offset, and otherwise it is
                absent. The period is held fixed while this is read, so the outcome moves
                with the offset alone.
      period -- the token count each staged shard carries, read from the shard headers. It
                is the period at which the shard set repeats and it is a property of the
                staged bytes only, so the offset cancels identically.

    Pure and deterministic: it reads frozen bytes off disk and does integer arithmetic. It
    reads no clock, no random source, no network and no environment secret.
    """
    pin = controls()["corpus_construction"]
    row = {
        "shard": pin["shard"],
        "phase": None,
        "period": None,
        "slice_sha256": None,
        "slice_tokens": None,
        "header": None,
        "heldout_sha256": None,
    }
    try:
        header = substrate.shard_header(pin["shard"])
        tokens = _heldout_slice_tokens()
    except (OSError, ValueError, FileNotFoundError):
        return row
    row["header"] = header
    row["slice_tokens"] = int(len(tokens))
    row["slice_sha256"] = hashlib.sha256(tokens.tobytes()).hexdigest()
    row["heldout_sha256"] = heldout_digest()
    if header["magic"] != substrate.SHARD_MAGIC or header["version"] != substrate.SHARD_VERSION:
        return row
    if row["slice_sha256"] == str(pin["slice_sha256"]):
        row["phase"] = int(pin["offset"])
    row["period"] = int(len(tokens))
    return row


# ---------------------------------------------------------------------------
# The SESSION probe. Agent-reachable by construction, and never a graded number.
# ---------------------------------------------------------------------------


def _session_probe_tokens():
    """The raw uint16 token ids of the agent-side probe slice, cut from a TRAIN shard."""
    pin = controls()["session_probe"]
    return substrate.read_shard_tokens(
        pin["shard"], int(pin["offset"]), int(pin["token_count"])
    )


def probe_corpora() -> tuple:
    """The construction window, the probe training text, and the probe text.

    All three are cut from the TRAIN shards the agent image stages, so this path reads
    nothing the solver surface does not already carry and it never touches the held-out
    split. That is what lets `solution/reference.py` run in the AGENT phase, which is where
    Harbor runs an oracle, rather than in the verifier image where the validation shard is.
    """
    if not _PROBE_TEXT:
        pin = controls()["session_probe"]
        _PROBE_TEXT["window"] = substrate.vocabulary_window_text()
        _PROBE_TEXT["train"] = substrate.training_text(int(pin["train_text_gpt2_tokens"]))
        _PROBE_TEXT["probe"] = substrate.detokenize(_session_probe_tokens())
    return _PROBE_TEXT["window"], _PROBE_TEXT["train"], _PROBE_TEXT["probe"]


def probe_reading_for(allocation: dict) -> float:
    """One allocation's bits per byte AS THE SESSION READS IT, at the probe budget.

    This is the state the reference session carries across its attempts. It is not the
    graded reading and no grading path consumes it: `context_for` computes every graded
    number from `corpora()` at the frozen budget, and `_attempt_rows` re-measures every
    allocation the record names rather than reading the number the record carries.
    """
    key = _key({"allocations": dict(allocation)})
    cached = _PROBED.get(key)
    if cached is not None:
        return cached
    units = int(controls()["session_probe"]["budget_units"])
    window, train, probe = probe_corpora()
    spec = substrate.VocabSpec.from_payload(
        {"allocations": dict(allocation), "train_units_requested": units}
    )
    telemetry = substrate.measure(spec, window, train, probe, substrate.Meter(units))
    reading = substrate.graded_bpb(telemetry["points"], len(probe))
    _PROBED[key] = reading
    return reading


# ---------------------------------------------------------------------------
# Measurement. One real training run per distinct allocation, memoised.
# ---------------------------------------------------------------------------


def telemetry_for(payload: dict) -> dict:
    """Train the frozen decoder over one submission payload and return its telemetry.

    Memoised on the canonical payload, because the control table and the attempt record ask
    for the same allocation more than once and a training run at the frozen seed and the
    frozen budget is a pure function of it. The memo is what keeps the verifier's cost at one
    training run per DISTINCT allocation rather than one per mention.
    """
    cached = _MEASURED.get(_key(payload))
    if cached is not None:
        return cached
    window, train, heldout = corpora()
    spec = substrate.VocabSpec.from_payload(payload)
    meter = substrate.Meter(substrate.COMPUTE_BUDGET_UNITS)
    result = substrate.measure(spec, window, train, heldout, meter)
    _MEASURED[_key(payload)] = result
    return result


def reading_for(allocation: dict) -> float:
    """The sustained bits per byte of one allocation, as the harness reads it."""
    telemetry = telemetry_for({"allocations": dict(allocation)})
    return substrate.graded_bpb(telemetry["points"], heldout_byte_count())


def anchors() -> dict:
    """Resolve the reward formula's two anchors BY MEASUREMENT, never from a constant.

    `baseline_metric` is the best reading any single-direction sweep of the whole slot budget
    reaches, so a greedy sweep of one direction scores exactly 0.0 by construction.
    `target_metric` is the harness's own reallocated reference at the same frozen budget.
    Neither number is written anywhere in this bundle; both are produced by this process, by
    training the frozen decoder once per distinct allocation.
    """
    table = controls()
    sweep = {
        name: reading_for(allocation)
        for name, allocation in sorted(table["single_direction_sweep"].items())
    }
    return {
        "sweep": sweep,
        "baseline_metric": min(sweep.values()),
        "baseline_source": "measured: best single-direction sweep of the whole slot budget",
        "target_metric": reading_for(table["reference_allocation"]),
        "target_source": "measured: harness reference reallocation at the frozen compute budget",
        "separation_margin": float(table["separation_margin"]),
        "anchors_state": table["anchors_state"],
        "anchors_gap": table["anchors_gap"],
    }


def _attempt_rows(document: dict, flattened: str) -> list:
    """Re-measure every attempt the record carries. A fabricated frontier does not survive."""
    rows, previous = [], None
    for raw in document.get("attempts") or []:
        row = raw if isinstance(raw, dict) else {}
        allocation = {name: 0 for name in substrate.DIRECTIONS}
        for name, value in (row.get("allocation") or {}).items():
            if name in allocation:
                allocation[name] = int(value)
        reading = reading_for(allocation)
        carried = row.get("carried_frontier")
        rows.append(
            {
                "index": int(row.get("index", len(rows) + 1)),
                "direction": str(row.get("direction", "")),
                "allocation": allocation,
                "reallocated_from": row.get("reallocated_from"),
                "bpb": reading,
                "marginal_gain_bits": 0.0 if previous is None else previous - reading,
                "carried_frontier": dict(carried) if isinstance(carried, dict) else {},
                "first": previous is None,
            }
        )
        previous = reading
    del flattened
    return rows


def context_for(document) -> dict:
    """Assemble the whole harness-owned context one grading pass reads."""
    payload = document if isinstance(document, dict) else {}
    table = controls()
    telemetry = telemetry_for(payload)
    denominator = heldout_byte_count()
    flattened = str(table["drift_state_final"].get("paying_direction", ""))
    return {
        "frozen": {
            "eval_bytes": denominator,
            "eval_sha256": heldout_digest(),
            "budget_units": substrate.COMPUTE_BUDGET_UNITS,
            "band_tolerance_bpb": substrate.BAND_TOLERANCE_BPB,
            "scheduled_points": len(substrate.EVALUATION_POINT_FRACTIONS),
            "slot_budget": substrate.SLOT_BUDGET,
            "directions": list(substrate.DIRECTIONS),
            "meter_source": "harness-counter",
            "denominator_key": "heldout_fineweb_byte_count",
            "architecture": {
                "vocab_size": substrate.VOCAB_SIZE,
                "num_layers": substrate.NUM_LAYERS,
                "model_dim": substrate.MODEL_DIM,
                "head_dim": substrate.HEAD_DIM,
                "num_heads": substrate.NUM_HEADS,
            },
            "batch_tokens_per_step": substrate.BATCH_TOKENS_PER_STEP,
            "forward_passes_per_step": substrate.FORWARD_PASSES_PER_STEP,
            "backward_passes_per_step": substrate.BACKWARD_PASSES_PER_STEP,
        },
        "points": [dict(row) for row in telemetry["points"]],
        "meter": dict(telemetry["meter"]),
        "spec_requests": dict(telemetry["spec_requests"]),
        "graded": {
            "bpb": substrate.graded_bpb(telemetry["points"], denominator),
            "allocations": {
                name: int((payload.get("allocations") or {}).get(name, 0) or 0)
                for name in substrate.DIRECTIONS
            },
            "slots_requested": telemetry["slots_requested"],
            "vocab_size": telemetry["vocab_size"],
            "vocab_ceiling": telemetry["vocab_ceiling"],
            "vocab_dropped_at_ceiling": telemetry["vocab_dropped_at_ceiling"],
            "vocab_digest": telemetry["vocab_digest"],
            "architecture": dict(telemetry["architecture"]),
        },
        "anchors": anchors(),
        "attempts": _attempt_rows(payload, flattened),
        "flatten": {
            "epsilon_bits": float(table["flatten_epsilon_bits"]),
            "run_required": int(table["flatten_run_required"]),
            "direction": "merge_depth",
            "note": (
                "the merge family admits a byte pair only at or above the frozen frequency "
                "floor, so it saturates at a finite depth and nothing announces it"
            ),
        },
        "corpus": {
            "stride": int(table["corpus_construction"]["stride"]),
            "offset": int(table["corpus_construction"]["offset"]),
            "shard": str(table["corpus_construction"]["shard"]),
            "slice_sha256": str(table["corpus_construction"]["slice_sha256"]),
        },
        "corpus_probe": corpus_probe(),
        "consumed_keys": list(CONSUMED_KEYS),
        "forbidden_consumed": list(FORBIDDEN_CONSUMED),
    }


def reward_for(context: dict) -> float:
    """The bound formula, over anchors this process measured, clipped to the closed interval."""
    baseline = context["anchors"]["baseline_metric"]
    target = context["anchors"]["target_metric"]
    agent = context["graded"]["bpb"]
    if agent is None or baseline is None or target is None:
        return 0.0
    span = baseline - target
    if span <= 0.0:
        return 0.0
    raw = (baseline - agent) / span
    return min(max(raw, 0.0), 1.0)

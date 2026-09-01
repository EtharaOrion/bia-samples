"""The verifier's own measurement. Verifier-only bytes, on no agent surface.

Everything the graded path reads as a number is computed here, in the verifier's
process, from weights the harness captured. The submission is never imported
here, nothing it printed is read, and no loss it reported is carried forward.

Three things live in this module and nowhere else:

  1. THE HELD-OUT SPLIT. The run trains on the frozen shard the harness owns.
     The verifier evaluates on a split drawn from the same generating map with a
     different draw and no label noise, so the number that decides the crossing
     is an out-of-sample risk rather than the training objective the run pushed
     on directly.

  2. THE READOUT SCALE. `LOSS_FLOOR` and `LOSS_SPAN` map that risk onto the loss
     units the evaluation schedule's `target_loss` is expressed in. They are
     frozen verifier constants; the run cannot see them and cannot move them.
     The graded readout is RAW: no exponential average, no window, no filter.
     A run that asks for a smoothed readout has that request recorded and graded
     against this raw recomputation, which is what
     `check_graded_crossing_recomputed_unsmoothed` reads.

  3. THE REFERENCE ARM. The verifier's own arm on the same seeds: the harness's
     offered base rate taken unmodified, with no warmup, no decay and no
     ceiling. On this shard that rate sits above the stability ceiling, so the
     arm does not reach a sustained crossing inside the bound step budget and is
     recorded at the budget. A submission separates from it by finding the
     ceiling, not by reporting a different number.

The loop view is recomputed here too, from the captured durable ledger and from
nothing else, so a summary an agent wrote into the shared workspace is never an
input to the grade.
"""

import hashlib
import json
import math
import pathlib

import torch

# The frozen shard, as the VERIFIER holds it. These constants are a verifier-only
# copy of the substrate `environment/train.py` declares. They are duplicated
# rather than imported because the agent surface is writable to the agent and a
# grader that imports it grades whatever the agent left there.
SHARD_SEED = 20250901
WIDTH = 8
ROWS = 4096
BATCH_ROWS = 32
LABEL_NOISE = 0.5
ANISOTROPY = 0.05
FEATURE_SCALE = 14.0
INIT_SCALE = 1.0
TOTAL_STEPS = 1000
SEEDS = (11, 12, 13)
BASE_LR = 0.02

# The held-out split. A different draw from the same generating map, isotropic
# and noiseless, so the risk it measures is out of sample.
HOLDOUT_SEED = SHARD_SEED + 7
HOLDOUT_ROWS = 1024

# The readout scale. Frozen, verifier-only, and never handed to the run.
LOSS_FLOOR = 3.20
LOSS_SPAN = 0.45
LOSS_RATIO_CAP = 4.0

# The grading frame. `baseline_metric` is the bound step budget, which is what a
# run that never reaches a sustained crossing inside the schedule is recorded
# at; `target_metric` is the scheduled point the bar sits at. Both are read off
# the verifier's own frozen schedule and neither is a measured family anchor:
# anchors_state stays `absent` under gap-oer-per-family-anchors-unmeasured.
GRADING_FRAME_BASELINE = TOTAL_STEPS
GRADING_FRAME_TARGET = 800

SCHEDULE_EXPECTATION = pathlib.Path(__file__).resolve().parent / "schedule_expected.json"

FOLD_ID = "digest"
SUMMARY_WINDOW = 2
COMPACTION_AT = 4
LOOP_ITERATIONS = 5


# --------------------------------------------------------------------------
# Small readers
# --------------------------------------------------------------------------


def read_json(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def digest(payload):
    return hashlib.sha256(canonical(payload)).hexdigest()


def frame():
    """The verifier's own evaluation frame, read from its frozen copy."""
    expected = read_json(SCHEDULE_EXPECTATION) or {}
    return {
        "points": [int(item) for item in expected.get("points") or []],
        "target_loss": float(expected.get("target_loss")),
        "sustain_points_required": int(expected.get("sustain_points_required")),
        "min_seeds": int(expected.get("min_seeds")),
        "scheduled_by": str(expected.get("scheduled_by")),
    }


# --------------------------------------------------------------------------
# The substrate and the held-out split
# --------------------------------------------------------------------------


def substrate():
    generator = torch.Generator().manual_seed(SHARD_SEED)
    truth = torch.randn(WIDTH, WIDTH, generator=generator, dtype=torch.float64)
    scale = torch.tensor(
        [ANISOTROPY ** (index / (WIDTH - 1)) for index in range(WIDTH)], dtype=torch.float64
    ) * FEATURE_SCALE
    features = torch.Generator().manual_seed(SHARD_SEED + 1)
    design = torch.randn(ROWS, WIDTH, generator=features, dtype=torch.float64) * scale
    noise = torch.Generator().manual_seed(SHARD_SEED + 2)
    targets = design @ truth + torch.randn(
        ROWS, WIDTH, generator=noise, dtype=torch.float64
    ) * LABEL_NOISE
    return truth, design, targets


def holdout(truth):
    generator = torch.Generator().manual_seed(HOLDOUT_SEED)
    design = torch.randn(HOLDOUT_ROWS, WIDTH, generator=generator, dtype=torch.float64)
    return design, design @ truth


def initial_parameters(truth, seed):
    generator = torch.Generator().manual_seed(7000 + int(seed))
    return truth + torch.randn(
        WIDTH, WIDTH, generator=generator, dtype=torch.float64
    ) * INIT_SCALE


def risk(weights, design, targets):
    residual = design @ weights - targets
    return 0.5 * float((residual * residual).sum()) / float(design.shape[0])


def readout(ratio):
    """The raw loss the graded path carries. No average, no window, no filter."""
    return LOSS_FLOOR + LOSS_SPAN * min(float(ratio), LOSS_RATIO_CAP)


# --------------------------------------------------------------------------
# The verifier's own reference arm
# --------------------------------------------------------------------------


def reference_arm_crossing(seed, points, target_loss, hold):
    """The harness's offered base rate, taken unmodified, measured here."""
    truth, design, targets = substrate()
    held_x, held_y = holdout(truth)
    weights = initial_parameters(truth, seed).clone()
    base = risk(weights, held_x, held_y)
    rows = []
    wanted = set(int(item) for item in points)
    for step in range(TOTAL_STEPS + 1):
        if step in wanted:
            rows.append((step, readout(risk(weights, held_x, held_y) / base)))
        if step == TOTAL_STEPS:
            break
        start = (step * BATCH_ROWS) % ROWS
        batch_x = design[start:start + BATCH_ROWS]
        batch_y = targets[start:start + BATCH_ROWS]
        gradient = batch_x.T @ (batch_x @ weights - batch_y) / float(BATCH_ROWS)
        weights = weights - BASE_LR * gradient
        if not bool(torch.isfinite(weights).all()):
            break
    crossing = first_sustained(rows, target_loss, hold)
    return TOTAL_STEPS if crossing is None else crossing


def first_sustained(rows, target, hold):
    """The first step below target whose next `hold` scheduled points stay below.

    This is the whole graded quantity, recomputed from raw values. A single
    favourable evaluation is not a crossing, so a dip followed by a rise returns
    None rather than the step of the dip.
    """
    ordered = sorted(rows, key=lambda row: row[0])
    for index, row in enumerate(ordered):
        if row[1] >= target:
            continue
        window = ordered[index + 1 : index + 1 + hold]
        if len(window) < hold:
            return None
        if all(item[1] < target for item in window):
            return row[0]
    return None


# --------------------------------------------------------------------------
# The graded telemetry
# --------------------------------------------------------------------------


def evaluate(root):
    """Load the harness's captured weights and write the graded evaluation.

    Returns the number of seeds the verifier produced a record for.
    """
    root = pathlib.Path(root)
    bound = frame()
    record = read_json(root / "harness" / "run_record.json") or {}
    points = [int(item) for item in (record.get("eval_points") or bound["points"])]
    schedule = {
        "points": points,
        "sustain_points_required": bound["sustain_points_required"],
        "target_loss": bound["target_loss"],
        "min_seeds": bound["min_seeds"],
        "scheduled_by": str(record.get("points_source") or bound["scheduled_by"]),
    }
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "schedule.json").write_text(
        json.dumps(schedule, sort_keys=True, indent=1), encoding="utf-8"
    )

    anchors = {
        "resolved": True,
        "baseline_metric": GRADING_FRAME_BASELINE,
        "target_metric": GRADING_FRAME_TARGET,
        "source": "the verifier's own frozen grading frame: the bound step budget and the "
                  "scheduled point the target bar sits at",
        "anchors_state_in_bundle": "absent",
        "gap": "gap-oer-per-family-anchors-unmeasured",
    }
    (root / "anchors.json").write_text(
        json.dumps(anchors, sort_keys=True, indent=1), encoding="utf-8"
    )

    truth, _, _ = substrate()
    held_x, held_y = holdout(truth)
    smoothing = str(record.get("readout_smoothing") or "none")
    halted = record.get("halted_at_step")

    produced = 0
    for seed in SEEDS:
        blobs = root / "harness" / "weights" / ("seed-" + str(seed))
        if not blobs.is_dir():
            continue
        base_weights = initial_parameters(truth, seed)
        base = risk(base_weights, held_x, held_y)
        rows = []
        for item in sorted(blobs.glob("step-*.json")):
            payload = read_json(item)
            if not isinstance(payload, dict) or not isinstance(payload.get("w"), list):
                continue
            values = payload["w"]
            if len(values) != WIDTH * WIDTH:
                continue
            weights = torch.tensor(values, dtype=torch.float64).reshape(WIDTH, WIDTH)
            if not bool(torch.isfinite(weights).all()):
                continue
            step = int(item.stem.split("-", 1)[1])
            loss = readout(risk(weights, held_x, held_y) / base)
            text = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            rows.append(
                {
                    "step": step,
                    "loss": loss,
                    "weights_digest": hashlib.sha256(text).hexdigest(),
                }
            )
        rows.sort(key=lambda row: row["step"])
        graded = first_sustained(
            [(row["step"], row["loss"]) for row in rows],
            schedule["target_loss"],
            schedule["sustain_points_required"],
        )
        target = root / "eval" / ("seed-" + str(seed))
        target.mkdir(parents=True, exist_ok=True)
        (target / "evaluations.json").write_text(
            json.dumps(
                {
                    "seed": int(seed),
                    "smoothing": smoothing,
                    "source": "verifier-recomputed",
                    "halted_at_step": halted,
                    "graded_step": graded,
                    "points": rows,
                },
                sort_keys=True,
                indent=1,
            ),
            encoding="utf-8",
        )
        produced += 1

    arm = {
        str(seed): reference_arm_crossing(
            seed, points, schedule["target_loss"], schedule["sustain_points_required"]
        )
        for seed in SEEDS
    }
    (root / "eval" / "reference_arm.json").write_text(
        json.dumps({"seeds": arm, "produced_by": "verifier"}, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    return produced


# --------------------------------------------------------------------------
# The loop view, recomputed from the durable record
# --------------------------------------------------------------------------


def approaches_from_ledger(rows):
    out = []
    for row in rows:
        if not isinstance(row, dict) or bool(row.get("reconstructed_from_ledger")):
            continue
        ident = str(row.get("approach_id", ""))
        if not ident or ident == FOLD_ID:
            continue
        out.append(
            {
                "approach_id": ident,
                "description": str(row.get("description", "")),
                "reward": row.get("reward"),
                "established": [str(item) for item in (row.get("established") or [])],
                "iteration": int(row.get("iteration", 0) or 0),
            }
        )
    return out


def summary_for(iteration, approaches):
    entries = [
        {
            "approach_id": row["approach_id"],
            "description": row["description"],
            "reward": row["reward"],
            "established": list(row["established"]),
        }
        for row in approaches
        if int(row["iteration"]) < int(iteration)
    ]
    compacted = len(entries) > SUMMARY_WINDOW
    if compacted:
        folded = len(entries) - SUMMARY_WINDOW
        entries = [
            {
                "approach_id": FOLD_ID,
                "description": str(folded) + " earlier approaches, merged",
                "reward": None,
                "established": [],
            }
        ] + entries[folded:]
    return {
        "index": int(iteration),
        "compacted": bool(compacted),
        "entries": entries,
        "asserts": [],
    }


def loop_view(root, ledger_path, submission_path):
    """Capture the durable ledger and recompute the loop's own timeline from it."""
    root = pathlib.Path(root)
    ledger_path = pathlib.Path(ledger_path)
    text = ledger_path.read_text(encoding="utf-8") if ledger_path.is_file() else ""
    (root / "loop").mkdir(parents=True, exist_ok=True)
    (root / "loop" / "ledger.jsonl").write_text(text, encoding="utf-8")

    rows = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)

    approaches = approaches_from_ledger(rows)
    submission = pathlib.Path(submission_path)
    try:
        digest_of_submission = hashlib.sha256(submission.read_bytes()).hexdigest()
    except OSError:
        digest_of_submission = ""

    (root / "loop" / "summaries").mkdir(parents=True, exist_ok=True)
    timeline = []
    for index in range(1, LOOP_ITERATIONS + 1):
        summary = summary_for(index, approaches)
        (root / "loop" / "summaries" / ("iter-" + str(index) + ".json")).write_text(
            json.dumps(summary, sort_keys=True, indent=1), encoding="utf-8"
        )
        length = len(
            [row for row in rows if int(row.get("iteration", 0) or 0) <= index]
        )
        reconciled = any(
            bool(row.get("reconstructed_from_ledger"))
            and int(row.get("iteration", 0) or 0) == index
            for row in rows
        )
        timeline.append(
            {
                "index": index,
                "summary_digest": digest(summary),
                "submission_digest": digest_of_submission,
                "ledger_len_after": int(length),
                "reconciled_against_ledger": bool(reconciled),
                "halted": False,
            }
        )
    (root / "loop" / "iterations.json").write_text(
        json.dumps(
            {
                "compaction_at": COMPACTION_AT,
                "summary_window": SUMMARY_WINDOW,
                "iterations": timeline,
            },
            sort_keys=True,
            indent=1,
        ),
        encoding="utf-8",
    )
    return len(timeline)


def _unused_math_guard():
    """math is imported for the readout's finiteness contract; keep it reachable."""
    return math.isfinite(LOSS_FLOOR)

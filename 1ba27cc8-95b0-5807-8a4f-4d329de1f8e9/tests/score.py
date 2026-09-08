from __future__ import annotations

import json
import os
import pathlib

SCORE_DIR = pathlib.Path(os.environ.get("BIA_SCORE_DIR", "/logs/verifier"))
SCORE_NUMERIC_JSON = "score_numeric.json"
SCORE_FLOAT = "score.md"
SCORE_DOC = "score.json"
SCORE_FULL = "score_full.json"

BOUND_SCORE_PATH = "/logs/verifier/score.md"
BOUND_SCORE_NUMERIC_DOCUMENT = "/logs/verifier/score_numeric.json"
BOUND_SCORE_DOCUMENT = "/logs/verifier/score.json"
BOUND_SCORE_FULL_DOCUMENT = "/logs/verifier/score_full.json"

REASON_CODES = (
    ("consolidated-crossing-graded", 0),
    ("no-improvement-over-baseline", 1),
    ("loss-band-graded-below-crossing", 20),
    ("submission-missing", 2),
    ("verifier-aborted-before-grading", 3),
    ("grader-internal-error", 4),
    ("session-record-unreadable", 5),
    ("readout-smoothed-on-the-graded-path", 10),
    ("crossing-not-sustained", 11),
    ("early-stop-claimed-as-crossing", 12),
    ("weights-not-harness-owned", 13),
    # 14 and 15 are retired: the two multi-attempt checkers that emitted them were
    # removed from checkers.CHAIN. The codes are kept rather than reused so that the
    # numbering of everything after them stays stable.
    ("single-attempt-metric-unconsolidated", 14),
    ("graded-outcome-not-carried", 15),
    ("reported-crossing-diverges-from-measurement", 16),
    ("attempt-budget-exceeded", 17),
    ("frozen-axis-moved", 18),
    ("stale-summary-carried", 19),
)
REASON_CODE_MISSING = 98
REASON_CODE_UNRECOGNISED = 99

def reason_code(reason: str) -> int:
    if not reason:
        return REASON_CODE_MISSING
    for prefix, code in REASON_CODES:
        if reason.startswith(prefix):
            return code
    return REASON_CODE_UNRECOGNISED

LOWER_BAND = 0.4

NO_PROGRESS_LOSS = 6.8

def anchored(agent_metric, baseline: int, target: int,
             final_loss=None, target_loss=None,
             no_progress_loss: float = NO_PROGRESS_LOSS) -> float:
    span = float(baseline) - float(target)
    if agent_metric is not None and span > 0.0:
        raw = (float(baseline) - float(agent_metric)) / span
        return LOWER_BAND + (1.0 - LOWER_BAND) * min(max(raw, 0.0), 1.0)

    if final_loss is None or target_loss is None:
        return 0.0
    loss_span = float(no_progress_loss) - float(target_loss)
    if loss_span <= 0.0:
        return 0.0
    progress = (float(no_progress_loss) - float(final_loss)) / loss_span
    return LOWER_BAND * min(max(progress, 0.0), 1.0)

def _numeric_document(value: float, reason: str, metric: dict) -> dict:
    document = {"score": value, "reason_code": reason_code(reason)}
    for key, item in (metric or {}).items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        document[key] = item
    return document

def write(score: float, reason: str, metric: dict, checkers: list | None = None) -> pathlib.Path:
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    value = min(max(float(score), 0.0), 1.0)
    rich = {"score": value, "reason": reason,
            "reason_code": reason_code(reason), "metric": metric,
            "metrics": {"n_seeds": int((metric or {}).get("attempts") or 0),
                        "seeds_required": 1},
            "checkers": checkers or []}

    (SCORE_DIR / SCORE_NUMERIC_JSON).write_text(
        json.dumps(_numeric_document(value, reason, metric), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (SCORE_DIR / SCORE_FULL).write_text(json.dumps(rich, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    (SCORE_DIR / SCORE_DOC).write_text(json.dumps(rich, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    (SCORE_DIR / SCORE_FLOAT).write_text("{:.6f}\n".format(value), encoding="utf-8")
    return SCORE_DIR / SCORE_FLOAT

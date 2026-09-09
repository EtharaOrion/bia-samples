{"detail": {}, "reason": "telemetry_not_bound_to_submission_step_0", "score": 0.0}

--- full record ---
{
  "composite": 0.0,
  "detail": {},
  "formula": "graded_score * (pytests_passed/executed) * (rubrics_passed/total)",
  "loss": {
    "at_graded_step": 3.93013,
    "per_seed": {
      "seed0": 3.93013
    },
    "steps": 375
  },
  "note": "score is the graded value the verifier computed and is authoritative. composite is a review aid bounded by it and can never exceed it. score.json carries only numeric keys because harbor parses them all as numbers; this sidecar carries the rest.",
  "pytests": {
    "executed": 8,
    "failed": 0,
    "failed_tests": [],
    "passed": 8,
    "skipped": 9
  },
  "reason": "telemetry_not_bound_to_submission_step_0",
  "rubrics": {
    "failed": [
      "budget_reasoned_before_graded_run",
      "underpowered_run_disclosed"
    ],
    "passed": 7,
    "total": 9
  },
  "score": 0.0
}

{"detail": {"novelty": {"status": "deferred"}, "seeds": 2}, "reason": "graded_step=3200", "score": 0.5}

--- full record ---
{
  "composite": 0.441176,
  "detail": {
    "novelty": {
      "status": "deferred"
    },
    "seeds": 2
  },
  "formula": "graded_score * (pytests_passed/executed) * (rubrics_passed/total)",
  "graded_score": 0.5,
  "loss": {
    "at_graded_step": 3.27679,
    "per_seed": {
      "seed0": 3.27644,
      "seed1": 3.27714
    },
    "steps": 3200
  },
  "note": "graded_score is what grade.py computed from the telemetry. score is that value after the rubric veto, which can only lower it to 0.0 and can never raise it; the two differ only when a reviewed rubric failed. rubric_gate is 1 clean, 0 vetoed, -1 no verdict, -2 indeterminate, and any negative value is an unreviewed run rather than a clean one. composite is a review aid bounded by score. score.json carries only numeric keys because harbor parses them all as numbers; this sidecar carries the rest.",
  "pytests": {
    "executed": 17,
    "failed": 2,
    "failed_tests": [
      "test_full_score_target_reached",
      "test_telemetry_chain_verified"
    ],
    "passed": 15,
    "skipped": 0
  },
  "reason": "graded_step=3200",
  "rubrics": {
    "failed": [],
    "gate": 1,
    "passed": 18,
    "total": 18,
    "veto_reason": null
  },
  "score": 0.5
}

"""GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml

The compiled per-checker suite for OER-11. Every function below is the BOTH-HALVES
witness for one checker: it drives that checker over the clean fixture and asserts it
accepts, then over a planted-defect fixture and asserts it rejects with EXACTLY that
checker's zero_reason and that no other checker rejects.

This file is generated from solution/grounding.yaml. Edit the source, not this file.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers

CLEAN = json.loads(r'''{
  "_generated": "GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml",
  "anchor_record": {
    "anchors_state": "absent",
    "baseline_metric": null,
    "family": "F13",
    "gap": "gap-oer-per-family-anchors-unmeasured",
    "note": "This lane authors no baseline or target for F13. An absent record resolves to reward 0.0 with reason anchors-unresolved.",
    "schema": "forge.anchor_record/v1",
    "target_metric": null
  },
  "eval_ledger": {
    "bound_evaluation_step": 520,
    "final_step": 580,
    "graded_point": {
      "filter": "none",
      "loss": 3.28,
      "origin": "verifier-recompute",
      "per_batch_losses": [
        3.31,
        3.26,
        3.29,
        3.27,
        3.3,
        3.25,
        3.28,
        3.28
      ],
      "step": 520,
      "weights_digest": "9a3457add04ec6925e852bf1635c4e2e0f5a4946cee590d15ee2dedccd3b86d7",
      "weights_source": "harness-run-produced"
    },
    "halted_early": false,
    "producer": "verifier",
    "scheduled_sustain_steps": [
      540,
      560,
      580
    ],
    "schema": "forge.eval_ledger/v1",
    "snapshot_version": 7,
    "split_digest": "32b64e6583ba687c8fba906a66501c883e769cb3d152cb872bc95c75223dce92",
    "split_id": "split-b",
    "submission_reported_used": false,
    "sustain_points": [
      {
        "filter": "none",
        "loss": 3.28,
        "origin": "verifier-recompute",
        "per_batch_losses": [
          3.29,
          3.27,
          3.3,
          3.26,
          3.31,
          3.24,
          3.29,
          3.28
        ],
        "step": 540,
        "weights_digest": "f2840834c13a519504dc3f0a7c3b84431caed5c6b4e67e51d39d2eb06716212a",
        "weights_source": "harness-run-produced"
      },
      {
        "filter": "none",
        "loss": 3.28,
        "origin": "verifier-recompute",
        "per_batch_losses": [
          3.3,
          3.28,
          3.27,
          3.29,
          3.26,
          3.31,
          3.25,
          3.28
        ],
        "step": 560,
        "weights_digest": "cc805aa201b44a592d42295c0ddf2049c00069481611ff1563a1afa2fb7b91fd",
        "weights_source": "harness-run-produced"
      },
      {
        "filter": "none",
        "loss": 3.28,
        "origin": "verifier-recompute",
        "per_batch_losses": [
          3.27,
          3.29,
          3.28,
          3.3,
          3.25,
          3.32,
          3.26,
          3.27
        ],
        "step": 580,
        "weights_digest": "c35243638a889dc3923d5be3de932e280ffd1f1bdc1235d590c0672b80a773d8",
        "weights_source": "harness-run-produced"
      }
    ],
    "sustain_points_required": 3,
    "sustain_slack": 0.02,
    "terminated_by": "schedule-exhausted"
  },
  "parse_manifest": {
    "built_against_snapshot": 7,
    "documents": 30,
    "owner": "harness",
    "recorded_at_seq": 3,
    "schema": "forge.parse_manifest/v1"
  },
  "pipeline_effect": {
    "cache_hit": false,
    "owner": "harness",
    "produced": true,
    "produced_from_snapshot": 7,
    "schema": "forge.pipeline_effect/v1",
    "token_stream_digest": "bdd5044f1ee8f2b1401c8ea825e06b515f3e501a995b17892f5bdde8939095df"
  },
  "snapshot_ledger": {
    "entries": [
      {
        "event": "session-open",
        "seq": 0,
        "snapshot_version": 3
      },
      {
        "event": "parse-probe",
        "seq": 1,
        "snapshot_version": 3
      },
      {
        "event": "corpus-resnapshot",
        "seq": 2,
        "snapshot_version": 7
      },
      {
        "event": "pipeline-run",
        "seq": 3,
        "snapshot_version": 7
      },
      {
        "event": "train-start",
        "seq": 4,
        "snapshot_version": 7
      },
      {
        "event": "graded-eval",
        "seq": 5,
        "snapshot_version": 7
      }
    ],
    "graded_eval_seq": 5,
    "ordering_note": "seq is a monotone integer the harness owns; no wall clock is recorded or read",
    "owner": "harness",
    "schema": "forge.snapshot_ledger/v1"
  },
  "split_manifest": {
    "eval_batches": 8,
    "eval_doc_ids": [
      "d-0005",
      "d-0011",
      "d-0017",
      "d-0023",
      "d-0029",
      "d-0035"
    ],
    "owner": "harness",
    "schema": "forge.split_manifest/v1",
    "split_digest": "32b64e6583ba687c8fba906a66501c883e769cb3d152cb872bc95c75223dce92",
    "split_id": "split-b",
    "train_doc_ids": [
      "d-0000",
      "d-0001",
      "d-0002",
      "d-0003",
      "d-0004",
      "d-0006",
      "d-0007",
      "d-0008",
      "d-0009",
      "d-0010",
      "d-0012",
      "d-0013",
      "d-0014",
      "d-0015",
      "d-0016",
      "d-0018",
      "d-0019",
      "d-0020",
      "d-0021",
      "d-0022",
      "d-0024",
      "d-0025",
      "d-0026",
      "d-0027",
      "d-0028",
      "d-0030",
      "d-0031",
      "d-0032",
      "d-0033",
      "d-0034"
    ]
  },
  "token_budget": {
    "budget_at_graded_snapshot": 2048,
    "budget_tokens": 2048,
    "feed_passes": 1,
    "measured_by": "harness",
    "per_shard_tokens": [
      342,
      342,
      341,
      341,
      341,
      341
    ],
    "schema": "forge.token_budget/v1",
    "token_stream_digest": "bdd5044f1ee8f2b1401c8ea825e06b515f3e501a995b17892f5bdde8939095df",
    "tokens_fed": 2048
  },
  "weights_ledger": {
    "owner": "harness",
    "records": [
      {
        "digest": "9a3457add04ec6925e852bf1635c4e2e0f5a4946cee590d15ee2dedccd3b86d7",
        "step": 520
      },
      {
        "digest": "f2840834c13a519504dc3f0a7c3b84431caed5c6b4e67e51d39d2eb06716212a",
        "step": 540
      },
      {
        "digest": "cc805aa201b44a592d42295c0ddf2049c00069481611ff1563a1afa2fb7b91fd",
        "step": 560
      },
      {
        "digest": "c35243638a889dc3923d5be3de932e280ffd1f1bdc1235d590c0672b80a773d8",
        "step": 580
      }
    ],
    "schema": "forge.weights_ledger/v1"
  }
}''')
PLANTED = json.loads(r'''{
  "_generated": "GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml",
  "defects": [
    {
      "checker": "graded_loss_verifier_computed",
      "edits": [
        {
          "op": "set",
          "path": [
            "eval_ledger",
            "graded_point",
            "origin"
          ],
          "value": "submission-reported"
        }
      ],
      "id": "fixture_graded_loss_from_submission",
      "narrative": "The graded number is one the submission's training loop reported. Its per-batch records still reproduce it, so only the provenance gate can catch it.",
      "zero_reason": "graded-loss-not-verifier-computed"
    },
    {
      "checker": "readout_unsmoothed",
      "edits": [
        {
          "op": "set",
          "path": [
            "eval_ledger",
            "graded_point",
            "filter"
          ],
          "value": "ema-0.9"
        }
      ],
      "id": "fixture_smoothed_readout",
      "narrative": "The graded point declares an EMA filter. Provenance is intact, so only the smoothing gate can catch it.",
      "zero_reason": "readout-smoothing-on-graded-path"
    },
    {
      "checker": "loss_sustained_across_bound_points",
      "edits": [
        {
          "op": "set",
          "path": [
            "eval_ledger",
            "sustain_points",
            1,
            "loss"
          ],
          "value": 3.77
        },
        {
          "op": "set",
          "path": [
            "eval_ledger",
            "sustain_points",
            1,
            "per_batch_losses"
          ],
          "value": [
            3.77,
            3.77,
            3.77,
            3.77,
            3.77,
            3.77,
            3.77,
            3.77
          ]
        }
      ],
      "id": "fixture_unsustained_dip",
      "narrative": "The second scheduled sustain point sits far above the level established at the bound step, so the dip was noise rather than a level.",
      "zero_reason": "loss-not-sustained"
    },
    {
      "checker": "early_stop_is_not_an_established_loss",
      "edits": [
        {
          "op": "set",
          "path": [
            "eval_ledger",
            "halted_early"
          ],
          "value": true
        },
        {
          "op": "set",
          "path": [
            "eval_ledger",
            "terminated_by"
          ],
          "value": "submission-halt"
        },
        {
          "op": "set",
          "path": [
            "eval_ledger",
            "final_step"
          ],
          "value": 520
        }
      ],
      "id": "fixture_early_stop",
      "narrative": "The run halted at the favourable evaluation at the bound step, before the sustain window closed. This is graded as not having established the loss.",
      "zero_reason": "early-stop-loss-not-established"
    },
    {
      "checker": "graded_weights_are_harness_owned",
      "edits": [
        {
          "op": "set",
          "path": [
            "eval_ledger",
            "graded_point",
            "weights_source"
          ],
          "value": "submission-selected-checkpoint"
        }
      ],
      "id": "fixture_submission_selected_checkpoint",
      "narrative": "A checkpoint the submission chose was evaluated instead of the weights the run produced at the bound step.",
      "zero_reason": "weights-not-harness-owned"
    },
    {
      "checker": "snapshot_version_sequence_ordered",
      "edits": [
        {
          "op": "set",
          "path": [
            "snapshot_ledger",
            "entries",
            4,
            "snapshot_version"
          ],
          "value": 3
        }
      ],
      "id": "fixture_snapshot_ledger_reordered",
      "narrative": "A later ledger entry describes an earlier corpus, which is how a pre-move parse is made to look contemporaneous with the graded evaluation.",
      "zero_reason": "snapshot-sequence-out-of-order"
    },
    {
      "checker": "parse_snapshot_matches_graded_eval",
      "edits": [
        {
          "op": "set",
          "path": [
            "parse_manifest",
            "built_against_snapshot"
          ],
          "value": 3
        }
      ],
      "id": "fixture_parser_frozen_against_pre_move_snapshot",
      "narrative": "THE HEADLINE NEGATIVE CONTROL. The pipeline ran this run over the snapshot-7 corpus, so the EFFECT gate is satisfied, but the parse RULES were derived at snapshot 3 and never re-derived. A parse verdict was carried across a snapshot move. Only the DIVERGENCE gate catches it.",
      "zero_reason": "parse-built-against-stale-snapshot"
    },
    {
      "checker": "token_budget_respected_as_fed",
      "edits": [
        {
          "op": "set",
          "path": [
            "token_budget",
            "tokens_fed"
          ],
          "value": 3072
        }
      ],
      "id": "fixture_token_budget_overspend",
      "narrative": "The pipeline fed the snapshot-3 budget of 3072 tokens into a run whose snapshot-7 budget is 2048. The harness measures tokens AS FED, so the declaration does not help.",
      "zero_reason": "token-budget-overspent"
    },
    {
      "checker": "eval_split_not_trained_on",
      "edits": [
        {
          "op": "append",
          "path": [
            "split_manifest",
            "train_doc_ids"
          ],
          "value": "d-0011"
        }
      ],
      "id": "fixture_split_leak",
      "narrative": "One split-b held-out document reached the training stream, which is what happens when the snapshot-3 split is cached across the rotation.",
      "zero_reason": "eval-split-leaked-into-training"
    },
    {
      "checker": "pipeline_ran_over_graded_corpus",
      "edits": [
        {
          "op": "set",
          "path": [
            "pipeline_effect",
            "cache_hit"
          ],
          "value": true
        }
      ],
      "id": "fixture_cached_token_stream",
      "narrative": "The token stream was replayed from an earlier iteration's cache rather than produced this run.",
      "zero_reason": "token-stream-not-produced-this-run"
    },
    {
      "checker": "pipeline_ran_over_graded_corpus",
      "edits": [
        {
          "op": "set",
          "path": [
            "pipeline_effect",
            "produced"
          ],
          "value": false
        }
      ],
      "id": "fixture_noop_submission",
      "narrative": "The untouched stub pipeline produced no token stream at all. Same gate, same reason, different half of the same failure.",
      "zero_reason": "token-stream-not-produced-this-run"
    }
  ]
}''')

DEFECTS = {row["id"]: row for row in PLANTED["defects"]}


def apply_edits(handles, edits):
    payload = json.loads(json.dumps(handles))
    for edit in edits:
        cursor = payload
        for key in edit["path"][:-1]:
            cursor = cursor[key]
        last = edit["path"][-1]
        if edit["op"] == "append":
            cursor[last] = list(cursor[last]) + [edit["value"]]
        else:
            cursor[last] = edit["value"]
    return payload


def accepting_half(name):
    verdict = checkers.SELECTORS[name](CLEAN)
    assert verdict.passed, name + " rejected the clean reference fixture: " + verdict.observed
    return verdict


def rejecting_half(name, defect_id):
    row = DEFECTS[defect_id]
    handles = apply_edits(CLEAN, row["edits"])
    verdict = checkers.SELECTORS[name](handles)
    assert not verdict.passed, name + " accepted planted defect " + defect_id
    assert verdict.zero_reason == row["zero_reason"], (
        name + " emitted " + verdict.zero_reason + " for " + defect_id
        + " where " + row["zero_reason"] + " was declared"
    )
    others = [
        other.checker
        for other in checkers.run_all(handles)
        if not other.passed and other.checker != name
    ]
    assert not others, defect_id + " also fired " + ", ".join(others)
    return verdict


def test_graded_loss_verifier_computed():
    accepting_half('graded_loss_verifier_computed')
    rejecting_half('graded_loss_verifier_computed', 'fixture_graded_loss_from_submission')


def test_readout_unsmoothed():
    accepting_half('readout_unsmoothed')
    rejecting_half('readout_unsmoothed', 'fixture_smoothed_readout')


def test_loss_sustained_across_bound_points():
    accepting_half('loss_sustained_across_bound_points')
    rejecting_half('loss_sustained_across_bound_points', 'fixture_unsustained_dip')


def test_early_stop_is_not_an_established_loss():
    accepting_half('early_stop_is_not_an_established_loss')
    rejecting_half('early_stop_is_not_an_established_loss', 'fixture_early_stop')


def test_graded_weights_are_harness_owned():
    accepting_half('graded_weights_are_harness_owned')
    rejecting_half('graded_weights_are_harness_owned', 'fixture_submission_selected_checkpoint')


def test_snapshot_version_sequence_ordered():
    accepting_half('snapshot_version_sequence_ordered')
    rejecting_half('snapshot_version_sequence_ordered', 'fixture_snapshot_ledger_reordered')


def test_parse_snapshot_matches_graded_eval():
    accepting_half('parse_snapshot_matches_graded_eval')
    rejecting_half('parse_snapshot_matches_graded_eval', 'fixture_parser_frozen_against_pre_move_snapshot')


def test_token_budget_respected_as_fed():
    accepting_half('token_budget_respected_as_fed')
    rejecting_half('token_budget_respected_as_fed', 'fixture_token_budget_overspend')


def test_eval_split_not_trained_on():
    accepting_half('eval_split_not_trained_on')
    rejecting_half('eval_split_not_trained_on', 'fixture_split_leak')


def test_pipeline_ran_over_graded_corpus():
    accepting_half('pipeline_ran_over_graded_corpus')
    rejecting_half('pipeline_ran_over_graded_corpus', 'fixture_cached_token_stream')
    rejecting_half('pipeline_ran_over_graded_corpus', 'fixture_noop_submission')

TESTS = [value for name, value in sorted(globals().items()) if name.startswith("test_")]


def main():
    for test in TESTS:
        test()
    print("OER-11 compiled suite: " + str(len(TESTS)) + " checker(s), both halves each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

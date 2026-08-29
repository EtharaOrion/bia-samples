#!/usr/bin/env python3
"""BIA S01 verifier.

The verifier re-executes the submission itself. It never grades an artifact the
agent produced, it never imports a submitted byte into this interpreter, and it
never parses a graded number out of anything the submission wrote or printed.

The shape of one grading pass is:

  1. this process pins the frozen substrate by digest and refuses to grade a
     bundle whose substrate bytes moved,
  2. it screens the submitted file against the exclusion corpus before any
     training happens, so a replay costs no compute,
  3. it creates a private root the agent has never seen, and spawns one
     separate interpreter per seed that loads the submission and trains,
     streaming its records back over a pipe this process created,
  4. it writes the graded record file itself from that stream,
  5. it spawns a third interpreter, into which no submitted byte is loaded,
     that recomputes the validation loss series, the initialization digest,
     the architecture signature and the parameter movement from the
     checkpointed parameters,
  6. it computes the reward from step 5 and from nothing else.

Artifacts the agent produced are read for exactly one purpose: the
reconciliation checker, which compares the agent's reported curve against the
curve this verifier measured. That checker carries no reward of its own.
"""

from __future__ import annotations

import ast
import difflib
import glob
import hashlib
import json
import math
import os
import pathlib
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.environ.get("BIA_BUNDLE", os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BUNDLE, "environment"))

from checkers.parser import parse_file  # noqa: E402
from harness import supervisor as sup  # noqa: E402

SCORE_PATH = os.environ.get("BIA_SCORE", "/logs/verifier/score.json")
OUTCOMES_PATH = os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json")
REWARD_PATH = os.environ.get("BIA_REWARD", "/logs/verifier/reward.txt")
SUBMISSION = os.environ.get("BIA_SUBMISSION", os.path.join(
    os.environ.get("BIA_WORKSPACE", "/workspace"), "submission"))
CORPUS_DIR = os.environ.get("BIA_EXCLUSION_CORPUS", os.path.join(HERE, "corpus"))
MANIFEST = os.environ.get("BIA_FROZEN_MANIFEST",
                          os.path.join(BUNDLE, "environment", "frozen_manifest.json"))
SCALE = os.environ.get("BIA_SCALE", "full")
DEVICE = os.environ.get("BIA_DEVICE", "auto")
KEEP_ROOT = os.environ.get("BIA_KEEP_VERIFIER_ROOT") == "1"

COPY_SIMILARITY_MAX = 0.90

OUTCOME_KEYS = [
    "frozen_batch_contract",
    "frozen_init_digest",
    "frozen_architecture_signature",
    "frozen_corpus_digest",
    "telemetry_bound_to_submission",
    "no_submission_writes",
    "optimizer_step_moves_parameters",
    "crossing_sustained",
    "report_matches_telemetry",
    "steps_to_target_within_baseline",
    "no_verbatim_record_copy",
]

ZERO_REASON = {
    "frozen_batch_contract": "frozen-batch-contract-violated",
    "frozen_init_digest": "frozen-init-digest-unproven",
    "frozen_architecture_signature": "frozen-architecture-signature-changed",
    "frozen_corpus_digest": "frozen-corpus-digest-changed",
    "telemetry_bound_to_submission": "telemetry-not-bound-to-submission",
    "no_submission_writes": "no-submission-writes-unproven",
    "optimizer_step_moves_parameters": "optimizer-step-movement-unproven",
    "crossing_sustained": "crossing-not-sustained",
    "report_matches_telemetry": "report-telemetry-reconciliation-unproven",
    "steps_to_target_within_baseline": "steps-to-target-not-within-baseline",
    "no_verbatim_record_copy": "no-verbatim-record-copy-unproven",
}


def emit(score, reason, detail=None, outcomes=None, reason_code="unattributed-zero"):
    vector = outcomes if outcomes is not None else {k: False for k in OUTCOME_KEYS}
    payload = {
        "score": float(score),
        "reason": reason,
        "reason_code": reason_code,
        "zero_reasons": [ZERO_REASON[k] for k in OUTCOME_KEYS if not vector.get(k)],
        "detail": detail or {},
    }
    for target in (SCORE_PATH, OUTCOMES_PATH, REWARD_PATH):
        d = os.path.dirname(target)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(SCORE_PATH, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    with open(OUTCOMES_PATH, "w") as fh:
        json.dump(vector, fh, indent=1, sort_keys=True)
    with open(REWARD_PATH, "w") as fh:
        fh.write(repr(float(score)) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return payload


def ast_normalize(path):
    try:
        tree = ast.parse(pathlib.Path(path).read_text())
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if ast.get_docstring(node):
                node.body = node.body[1:]
    return ast.dump(tree, annotate_fields=False)


def check_no_verbatim_copy(optimizer_path):
    mine = ast_normalize(optimizer_path)
    if mine is None:
        return False, "optimizer_unparseable_for_exclusion_check"
    refs = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.py")))
    baseline = os.path.join(BUNDLE, "environment", "baseline_optimizer.py")
    if os.path.isfile(baseline):
        refs.append(baseline)
    if not refs:
        return False, "exclusion_corpus_absent"
    worst = None
    for ref in refs:
        other = ast_normalize(ref)
        if other is None:
            continue
        ratio = difflib.SequenceMatcher(None, mine, other).ratio()
        if ratio >= COPY_SIMILARITY_MAX and (worst is None or ratio > worst[0]):
            worst = (ratio, os.path.basename(ref))
    if worst is None:
        return True, None
    return False, f"excluded_recipe_match_{worst[1]}_similarity_{worst[0]:.3f}"


def check_batch_contract(recs, cfg):
    expect = {
        "tokens_per_step": cfg["tokens_per_step"],
        "sequences_per_forward": cfg["seqs_per_forward"],
        "sequence_length": cfg["seq_len"],
        "forward_backward_per_step": cfg["forward_backward_per_step"],
        "forward_calls_this_step": cfg["forward_backward_per_step"],
        "backward_calls_this_step": cfg["forward_backward_per_step"],
        "vocab_size": cfg["vocab"],
    }
    for r in recs:
        for key, want in expect.items():
            if r.get(key) != want:
                return False, f"batch_contract_{key}_step_{r.get('step')}_saw_{r.get(key)}"
        if r.get("early_stop") is not False:
            return False, f"early_stop_flag_step_{r.get('step')}"
        if r.get("eval_weights_source") != "live_parameters":
            return False, f"eval_weights_source_{r.get('eval_weights_source')}"
    return True, None


def check_binding(recs, digest_before, digest_after):
    """The bytes the verifier executed are the bytes on disk, at both ends."""
    if digest_before != digest_after:
        return False, "submission_file_mutated_during_grading"
    for r in recs:
        for field in ("submission_digest", "submission_digest_at_load",
                      "submission_digest_after_load"):
            if r.get(field) != digest_before:
                return False, f"{field}_mismatch_step_{r.get('step')}"
    return True, None


def check_no_writes(recs, parent_observed):
    """Two independent surfaces, and either one firing is a violation.

    `parent_observed` is the verifier's own before-and-after diff of the roots
    the child could reach. The per-record fields are the in-child guard. The
    parent surface is the load-bearing one, because it observes an effect
    rather than a self-report.
    """
    if parent_observed:
        return False, "verifier_observed_write_" + os.path.basename(parent_observed[0])
    for r in recs:
        writes = r.get("submission_writes")
        if writes is None:
            return False, "write_guard_record_absent"
        if writes:
            return False, f"submission_wrote_{os.path.basename(str(writes[0]))}"
        loaded = r.get("guard_observed_writes")
        if loaded is None:
            return False, "import_guard_record_absent"
        if loaded:
            return False, f"submission_wrote_at_import_{os.path.basename(str(loaded[0]))}"
    return True, None


def check_param_movement(measured):
    seen = 0
    for seed, block in sorted(measured["seeds"].items()):
        deltas = block.get("deltas") or {}
        if not deltas:
            return False, f"no_param_delta_observations_seed_{seed}"
        for step, value in sorted(deltas.items(), key=lambda kv: int(kv[0])):
            seen += 1
            if not (float(value) > 0.0):
                return False, f"optimizer_step_moved_nothing_by_step_{step}_seed_{seed}"
    if seen == 0:
        return False, "no_param_delta_observations"
    return True, None


def check_arch_and_corpus(measured, recs, manifest, scale):
    want_arch = manifest["scales"][scale]["arch_signature"]
    want_corpus = manifest["scales"][scale]["corpus_digest"]
    arch_ok, arch_reason = True, None
    for seed, block in sorted(measured["seeds"].items()):
        signatures = block.get("arch_signatures") or []
        if signatures != [want_arch]:
            arch_ok = False
            arch_reason = f"architecture_changed_seed_{seed}_saw_{signatures}"
            break
    corpus_ok, corpus_reason = True, None
    if measured.get("corpus_digest") not in (None, want_corpus):
        corpus_ok, corpus_reason = False, "corpus_changed_under_verifier_measurement"
    else:
        for r in recs:
            if r.get("corpus_digest") != want_corpus:
                corpus_ok = False
                corpus_reason = f"corpus_changed_step_{r.get('step')}"
                break
    return (arch_ok, arch_reason), (corpus_ok, corpus_reason)


def check_init_digest(measured, recs):
    for seed, block in sorted(measured["seeds"].items()):
        got, want = block.get("init_digest"), block.get("expected_init_digest")
        if got is None:
            return False, f"init_checkpoint_absent_seed_{seed}"
        if got != want:
            return False, f"init_digest_mismatch_seed_{seed}"
    by_seed = {}
    for r in recs:
        by_seed.setdefault(str(r["seed"]), set()).add(r.get("init_digest"))
    for seed, digests in sorted(by_seed.items()):
        if len(digests) != 1:
            return False, f"init_digest_not_constant_for_seed_{seed}"
        recorded = digests.pop()
        expected = (measured["seeds"].get(seed) or {}).get("expected_init_digest")
        if recorded != expected:
            return False, f"recorded_init_digest_mismatch_seed_{seed}"
    return True, None


def measured_series(measured):
    out = {}
    for seed, block in measured["seeds"].items():
        series = block.get("series") or {}
        out[int(seed)] = {int(k): float(v) for k, v in series.items()}
    return out


def check_reconciliation(series, seed_logs, tol):
    """The agent's reported curve against the curve this verifier measured.

    This is the one checker that reads an agent artifact. It carries no reward:
    it establishes agreement between two accounts, and it can no longer be
    satisfied by regenerating one account from the other, because the account
    it is compared against is produced after the agent's container is gone.
    """
    for seed, reported in sorted(seed_logs.items()):
        if seed not in series:
            return False, f"reported_seed_{seed}_absent_from_verifier_measurement"
        for step, loss in sorted(reported.items()):
            if step not in series[seed]:
                return False, f"reported_step_{step}_seed_{seed}_absent_from_verifier_measurement"
            if abs(series[seed][step] - float(loss)) > tol:
                return False, f"reported_loss_diverges_seed_{seed}_step_{step}"
    for seed, seen in sorted(series.items()):
        if seed not in seed_logs:
            return False, f"verifier_seed_{seed}_absent_from_report"
        for step in sorted(seen):
            if step not in seed_logs[seed]:
                return False, f"verifier_step_{step}_seed_{seed}_absent_from_report"
    return True, None


def graded_step(series, target_loss, cfg):
    if len(series) < cfg["min_seeds"]:
        return None, f"need_at_least_{cfg['min_seeds']}_seeds_got_{len(series)}"
    common = sorted(set.intersection(*[set(d) for d in series.values()]))
    if not common:
        return None, "no_step_logged_by_every_seed"

    def at(step):
        vals = [series[s][step] for s in sorted(series)]
        return sum(vals) / len(vals), max(vals), len(vals)

    def holds(step):
        mean, worst, _ = at(step)
        return mean <= target_loss and worst <= target_loss

    rejected = None
    for i, step in enumerate(common):
        mean, worst, n = at(step)
        if mean > target_loss:
            continue
        if (target_loss - mean) * math.sqrt(n) < cfg["sig_margin"]:
            continue
        if worst > target_loss:
            rejected = rejected or f"seed_above_target_at_step_{step}_worst_{worst:.6f}"
            continue
        broke = next((t for t in common[i + 1:] if not holds(t)), None)
        if broke is not None:
            rejected = rejected or f"crossing_at_{step}_not_sustained_at_step_{broke}"
            continue
        return step, None
    return None, rejected or "no_step_clears_noise_floor"


def load_seed_logs(sub_dir):
    paths = sorted(glob.glob(os.path.join(sub_dir, "logs", "full_seed*.log")))
    out, seen = {}, {}
    for p in paths:
        base = os.path.basename(p)
        seed = int(base.split("seed")[-1].split(".")[0])
        if seed in seen:
            return None, f"seed_log_collision_{seed}"
        seen[seed] = base
        try:
            out[seed] = parse_file(p)
        except ValueError as exc:
            return None, f"reported_log_malformed_seed_{seed}_{exc}".replace(" ", "_")
    if not out:
        return None, "no_reported_seed_logs"
    return out, None


def main():
    outcomes = {k: False for k in OUTCOME_KEYS}

    if not os.path.isfile(MANIFEST):
        return emit(0.0, "frozen_manifest_absent", {}, outcomes, "frozen-manifest-absent")
    manifest = json.loads(open(MANIFEST).read())

    core_path = os.path.join(BUNDLE, "environment", "bia_core.py")
    if not os.path.isfile(core_path):
        return emit(0.0, "frozen_substrate_absent", {}, outcomes, "frozen-substrate-absent")
    if sup.sha256_file(core_path) != manifest.get("core_sha256"):
        return emit(0.0, "frozen_substrate_edited", {"path": core_path}, outcomes,
                    "frozen-substrate-edited")

    if SCALE not in manifest["scales"]:
        return emit(0.0, f"unknown_scale_{SCALE}", {}, outcomes, "unknown-scale")

    opt_path = os.path.join(SUBMISSION, "optimizer.py")
    if not os.path.isfile(opt_path):
        return emit(0.0, "submission_optimizer_absent", {}, outcomes,
                    "submission-optimizer-absent")

    copy_ok, copy_reason = check_no_verbatim_copy(opt_path)
    outcomes["no_verbatim_record_copy"] = bool(copy_ok)
    if not copy_ok:
        return emit(0.0, copy_reason, {"scale": SCALE}, outcomes,
                    ZERO_REASON["no_verbatim_record_copy"])

    import bia_core as core
    cfg = core.resolve_scale(SCALE)
    target_loss = float(manifest["scales"][SCALE]["target_loss"])
    seeds = list(range(cfg["min_seeds"]))

    digest_before = sup.sha256_file(opt_path)
    supervisor = sup.Supervisor(BUNDLE, SCALE, DEVICE, SUBMISSION, keep_root=KEEP_ROOT)
    records = []
    measured = {"seeds": {}}
    try:
        try:
            ready = supervisor.start_measure(supervisor.cache)
            measured["corpus_digest"] = ready.get("corpus_digest")
            for seed in seeds:
                block = measured["seeds"].setdefault(
                    str(seed), {"series": {}, "deltas": {}, "arch_signatures": set(),
                                "init_digest": None, "expected_init_digest": None,
                                "load_errors": []})

                def on_measured(doc, block=block):
                    if doc.get("load_error"):
                        block["load_errors"].append(doc["load_error"])
                        return
                    if doc.get("arch_signature"):
                        block["arch_signatures"].add(doc["arch_signature"])
                    if int(doc["step"]) == 0:
                        block["init_digest"] = doc.get("init_digest")
                        block["expected_init_digest"] = doc.get("expected_init_digest")
                    elif doc.get("val_loss") is not None:
                        block["series"][str(int(doc["step"]))] = doc["val_loss"]
                    if doc.get("delta_l1") is not None:
                        block["deltas"][str(int(doc["step"]))] = doc["delta_l1"]

                records.extend(supervisor.run_seed(seed, supervisor.cache, on_measured))
            supervisor.write_records(records)
        except sup.ChildFailure as exc:
            return emit(0.0, exc.code, exc.detail, outcomes, exc.code)
        finally:
            supervisor.stop_measure()

        for block in measured["seeds"].values():
            block["arch_signatures"] = sorted(block["arch_signatures"])
        errors = [e for b in measured["seeds"].values() for e in b["load_errors"]]
        if errors:
            return emit(0.0, "verifier_could_not_load_parameters",
                        {"errors": errors[:4]}, outcomes,
                        "verifier-measurement-failed")

        digest_after = sup.sha256_file(opt_path)

        bind_ok, bind_reason = check_binding(records, digest_before, digest_after)
        outcomes["telemetry_bound_to_submission"] = bool(bind_ok)

        batch_ok, batch_reason = check_batch_contract(records, cfg)
        outcomes["frozen_batch_contract"] = bool(batch_ok)

        (arch_ok, arch_reason), (corpus_ok, corpus_reason) = check_arch_and_corpus(
            measured, records, manifest, SCALE)
        outcomes["frozen_architecture_signature"] = bool(arch_ok)
        outcomes["frozen_corpus_digest"] = bool(corpus_ok)

        writes_ok, writes_reason = check_no_writes(records, supervisor.write_effects)
        outcomes["no_submission_writes"] = bool(writes_ok)

        move_ok, move_reason = check_param_movement(measured)
        outcomes["optimizer_step_moves_parameters"] = bool(move_ok)

        init_ok, init_reason = check_init_digest(measured, records)
        outcomes["frozen_init_digest"] = bool(init_ok)

        series = measured_series(measured)
        seed_logs, log_reason = load_seed_logs(SUBMISSION)
        if seed_logs is None:
            recon_ok, recon_reason = False, log_reason
        else:
            recon_ok, recon_reason = check_reconciliation(
                series, seed_logs, cfg["recon_tol"])
        outcomes["report_matches_telemetry"] = bool(recon_ok)

        step, step_reason = graded_step(series, target_loss, cfg)
        outcomes["crossing_sustained"] = step is not None
        outcomes["steps_to_target_within_baseline"] = (
            step is not None and step <= cfg["baseline_steps"])

        gates = [
            ("telemetry_bound_to_submission", bind_ok, bind_reason),
            ("frozen_batch_contract", batch_ok, batch_reason),
            ("frozen_architecture_signature", arch_ok, arch_reason),
            ("frozen_corpus_digest", corpus_ok, corpus_reason),
            ("no_submission_writes", writes_ok, writes_reason),
            ("optimizer_step_moves_parameters", move_ok, move_reason),
            ("frozen_init_digest", init_ok, init_reason),
            ("report_matches_telemetry", recon_ok, recon_reason),
        ]
        for key, ok, reason in gates:
            if not ok:
                return emit(0.0, reason, {"scale": SCALE}, outcomes, ZERO_REASON[key])
        if step is None:
            return emit(0.0, step_reason, {"scale": SCALE, "target_loss": target_loss},
                        outcomes, ZERO_REASON["crossing_sustained"])

        span = cfg["baseline_steps"] - cfg["target_steps"]
        raw = (cfg["baseline_steps"] - step) / span
        score = min(max(raw, 0.0), 1.0)
        code = "graded-step-scored"
        if not outcomes["steps_to_target_within_baseline"]:
            code = ZERO_REASON["steps_to_target_within_baseline"]
        return emit(score, f"graded_step={step}", {
            "scale": SCALE,
            "graded_step": step,
            "baseline_steps": cfg["baseline_steps"],
            "target_steps": cfg["target_steps"],
            "target_loss": target_loss,
            "seeds": sorted(series),
            "measured_by": "verifier re-execution, tests/harness/measure_child.py",
        }, outcomes, code)
    finally:
        if not KEEP_ROOT:
            shutil.rmtree(supervisor.root, ignore_errors=True)


if __name__ == "__main__":
    main()

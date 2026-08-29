"""S08 verifier. Reads live run state, runs every registered checker, emits one float
on [0,1].

Every number this file produces comes from a read of state the run left behind:
the harness telemetry record, the resume digest file, the recover effect file, the
corpus on disk and the agent's own report. Nothing rests on the author's word.

The reward baseline is not an authored constant. It is the step count the
harness-owned naive warm start needed in this same graded attempt, measured fresh
every run, so a reward of zero means the submission did no better than resuming
from the checkpoint unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkers.reduce import CHECKERS, reduce_all  # noqa: E402

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.environ.get("S08_BUNDLE", "/workspace")
TELEMETRY_DIR = os.environ.get("S08_TELEMETRY_DIR", "/telemetry")
TELEMETRY = os.path.join(TELEMETRY_DIR, "run_record.jsonl")
SUBMISSION = os.environ.get("S08_SUBMISSION", "/workspace/submission")
FIXTURE_DIR = os.environ.get("S08_FIXTURE_DIR", "/opt/bia/s08/fixtures")
PROFILE = os.environ.get("S08_PROFILE", "scaled")
SCORE_PATH = os.environ.get("SCORE_PATH", "/logs/verifier/score.json")
OUTCOMES_PATH = os.environ.get("S08_OUTCOMES", "/logs/verifier/outcomes.json")
RUNNER_DIR = os.environ.get("S08_RUNNER_DIR", os.path.join(BUNDLE, "environment", "runner"))
DATA_DIR = os.environ.get("S08_DATA_DIR", "")
STATE_DIR = os.environ.get("S08_STATE_DIR", "/opt/bia/s08")
DEVICE = os.environ.get("S08_DEVICE", "")
VERIFIER_RUN_TIMEOUT = float(os.environ.get("S08_VERIFIER_RUN_TIMEOUT", "1500"))

# CONTROL-ONLY INPUT. When set, the named telemetry directory is used as the authoritative
# record instead of running the attempt, and the authoritative source is recorded as
# `external`. The origin checker requires the source to be `verifier_execution`, so this
# switch can only ever produce a zero. It exists so the rejecting half of that checker is
# executable under frozen bytes.
EXTERNAL_AUTHORITATIVE = os.environ.get("S08_AUTHORITATIVE_TELEMETRY_DIR", "")

EPS = 1e-9


def emit(score, reason, detail=None, outcomes=None):
    outcomes = outcomes or {k: False for k in CHECKERS}
    for path, payload in (
        (SCORE_PATH, {"score": float(score), "reason": reason, "detail": detail or {}}),
        (OUTCOMES_PATH, outcomes),
    ):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
    print(json.dumps({"score": float(score), "reason": reason, "detail": detail or {}}, sort_keys=True))
    return float(score)


def zero_reason_for(name):
    """The kebab reason code this checker emits when it scores zero.

    The code is read from the generated registry rather than written here, so the
    reason tests/checkers.yaml declares and the reason a run actually emits are one
    string from one source and cannot drift.
    """
    return CHECKERS.get(name, {}).get("zero_reason", "checker-zero-reason-undeclared")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    if not os.path.exists(path):
        return None, "telemetry_absent"
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                return None, "telemetry_malformed"
    if not recs:
        return None, "telemetry_empty"
    return recs, None


def check_chain(recs):
    prev = "genesis"
    for i, r in enumerate(recs):
        if r.get("record_index") != i:
            return False, f"record_index_out_of_order_at_{i}"
        body = {k: v for k, v in r.items() if k != "chain"}
        expect = hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        if r.get("chain") != expect:
            return False, f"chain_break_at_record_{i}"
        prev = expect
    return True, None


# --------------------------------------------------------------------------
# One function per registered checker. Every one reduces to exactly one of the six
# kinds and reads state the run left behind, never an author's constant.
# --------------------------------------------------------------------------


def k_frozen_invariant(recs, manifest):
    """INVARIANT. Live-state read: every telemetry record's frozen block."""
    frozen = manifest["frozen"]
    for r in recs:
        for key, want in frozen.items():
            if r.get(key) != want:
                return False, f"frozen_{key}_moved_at_record_{r.get('record_index')}"
        if r.get("target_loss") != manifest["target_loss"]:
            return False, f"target_loss_moved_at_record_{r.get('record_index')}"
    return True, None


def k_ordering(recs):
    """ORDERING. Live-state read: the record_index sequence and phase labels."""
    seals = [r for r in recs if r.get("kind") == "phase_seal" and r.get("phase") == "control"]
    if len(seals) != 1:
        return False, f"control_phase_seal_count_{len(seals)}"
    seal_at = seals[0]["record_index"]
    subs = [r["record_index"] for r in recs if r.get("phase") == "submission"]
    ctrl = [r["record_index"] for r in recs if r.get("phase") == "control"]
    if not subs:
        return False, "no_submission_phase_records"
    if min(subs) <= seal_at:
        return False, f"submission_record_{min(subs)}_precedes_control_seal_{seal_at}"
    if max(ctrl) != seal_at:
        return False, f"control_record_{max(ctrl)}_written_after_its_own_seal_{seal_at}"
    return True, None


def k_absence_weight_surgery(digests, manifest):
    """ABSENCE. Live-state read: resume_digests.json, written by the runner from the
    live model immediately before and immediately after recover returned."""
    if not digests:
        return False, "resume_digests_absent"
    offenders = []
    for key, d in sorted(digests.items()):
        if d.get("weight_digest_before_recover") != manifest["weight_digest"]:
            offenders.append(f"{key}:load")
        if d.get("weight_digest_after_recover") != manifest["weight_digest"]:
            offenders.append(f"{key}:recover")
    if offenders:
        return False, "weight_surgery_" + ",".join(offenders)
    return True, None


def k_value_val_shard(manifest, corpus_meta):
    """VALUE. Live-state read: the sha256 the verifier recomputes over the
    validation shard the run actually read."""
    if not corpus_meta:
        return False, "corpus_meta_absent"
    path = os.path.join(corpus_meta.get("data_dir", ""), "val.bin")
    if not os.path.exists(path):
        return False, "val_shard_absent"
    got = sha256_file(path)
    want = manifest["corpus"]["val_sha256"]
    if got != want:
        return False, f"val_shard_digest_{got[:12]}_expected_{want[:12]}"
    return True, None


def k_effect_recover(effects):
    """EFFECT. Live-state read: recover_effect.json, the optimizer fingerprint taken
    from the poisoned checkpoint state and from the optimizer recover returned."""
    if not effects:
        return False, "recover_effect_absent"
    changed_any = False
    for key, e in sorted(effects.items()):
        if not key.startswith("submission"):
            continue
        before = e["checkpoint_optimizer_fingerprint"]
        after = e["recovered_optimizer_fingerprint"]
        changed = []
        for idx, b in before["state"].items():
            a = after["state"].get(idx)
            if a is None:
                changed.append(f"{idx}:state_dropped")
                continue
            for field in ("exp_avg_rms", "exp_avg_sq_rms", "step"):
                if field in b and field in a and abs(a[field] - b[field]) > 1e-12 * max(1.0, abs(b[field])):
                    changed.append(f"{idx}:{field}")
            if "step" in b and "step" not in a:
                changed.append(f"{idx}:step_dropped")
        for i, (gb, ga) in enumerate(zip(before["param_groups"], after["param_groups"])):
            for field in ("weight_decay", "lr", "eps"):
                if abs(ga[field] - gb[field]) > 1e-15:
                    changed.append(f"group{i}:{field}")
        if not changed:
            return False, f"{key}_recover_left_optimizer_state_identical_to_checkpoint"
        changed_any = True
    if not changed_any:
        return False, "no_submission_recover_effect_recorded"
    return True, None


def series(recs, phase):
    out = {}
    for r in recs:
        if r.get("phase") != phase or r.get("kind") != "eval":
            continue
        out.setdefault(int(r["seed"]), {})[int(r["step"])] = float(r["val_loss"])
    return out


def crossing(seed_map, target):
    """Earliest step at which the seed mean and every individual seed sit at or below
    target, and stay there at every later step common to all seeds."""
    if not seed_map:
        return None, "no_eval_records"
    common = sorted(set.intersection(*[set(d) for d in seed_map.values()]))
    if not common:
        return None, "no_step_evaluated_by_every_seed"

    def holds(s):
        vals = [seed_map[k][s] for k in seed_map]
        return (sum(vals) / len(vals)) <= target + EPS and max(vals) <= target + EPS

    for i, s in enumerate(common):
        if not holds(s):
            continue
        if all(holds(t) for t in common[i + 1 :]):
            return s, None
        return None, f"crossing_at_{s}_not_sustained"
    return None, "target_never_reached"


def k_value_crossing(sub_steps, manifest, max_steps_ceiling):
    """VALUE. Live-state read: the submission phase validation series in telemetry."""
    if sub_steps is None:
        return False, "submission_never_reached_target"
    if sub_steps > max_steps_ceiling:
        return False, f"graded_step_{sub_steps}_exceeds_ceiling_{max_steps_ceiling}"
    return True, None


def k_value_train_shard(manifest, corpus_meta):
    """VALUE. Live-state read: the sha256 the verifier recomputes over the training
    shard the run actually read."""
    if not corpus_meta:
        return False, "corpus_meta_absent"
    path = os.path.join(corpus_meta.get("data_dir", ""), "train.bin")
    if not os.path.exists(path):
        return False, "train_shard_absent"
    got = sha256_file(path)
    want = manifest["corpus"]["train_sha256"]
    if got != want:
        return False, f"train_shard_digest_{got[:12]}_expected_{want[:12]}"
    return True, None


def k_invariant_chain(recs):
    """INVARIANT. Live-state read: the record_index and chain field of every record,
    rehashed by the verifier from the genesis seed.

    The chain is unkeyed. It defeats truncation, reordering and edits in place, and it
    does not defeat a forger who rewrites the whole record from genesis.
    """
    ok, why = check_chain(recs)
    return (True, None) if ok else (False, why)


def phase_seeds(recs, phase):
    out = {}
    for r in recs:
        if r.get("phase") != phase or r.get("kind") not in ("resume", "eval", "phase_end"):
            continue
        out.setdefault(int(r["seed"]), []).append(r)
    return out


def k_invariant_record_shape(recs, manifest):
    """INVARIANT. Live-state read: the phase, seed, kind and step fields of every
    record against the seeds, eval_every and max_steps the manifest binds."""
    eval_every = int(manifest["eval_every"])
    max_steps = int(manifest["max_steps"])
    expected = {"control": {0}, "submission": {int(s) for s in manifest["seeds"]}}
    for phase, want_seeds in expected.items():
        groups = phase_seeds(recs, phase)
        if set(groups) != want_seeds:
            return False, f"{phase}_seed_set_{sorted(groups)}_expected_{sorted(want_seeds)}"
        for seed, rows in sorted(groups.items()):
            resumes = [r for r in rows if r.get("kind") == "resume"]
            ends = [r for r in rows if r.get("kind") == "phase_end"]
            if len(resumes) != 1:
                return False, f"{phase}_seed{seed}_resume_record_count_{len(resumes)}"
            if len(ends) != 1:
                return False, f"{phase}_seed{seed}_phase_end_count_{len(ends)}"
            probe = int(resumes[0]["step"])
            cadence = {probe + i for i in range(0, max_steps + 1, eval_every)}
            cadence.add(probe + max_steps)
            for r in rows:
                if r.get("kind") != "eval":
                    continue
                if int(r["step"]) not in cadence:
                    return False, f"{phase}_seed{seed}_eval_step_{r['step']}_off_frozen_cadence"
    return True, None


def integrity_record(recs):
    found = [r for r in recs if r.get("kind") == "integrity"]
    if len(found) != 1:
        return None, f"integrity_record_count_{len(found)}"
    digests = found[0].get("digests")
    if not isinstance(digests, dict):
        return None, "integrity_record_carries_no_digests"
    return digests, None


def recheck_digest(recorded, label):
    """Recompute the digest of a file the runner recorded loading, and compare it
    against what the runner recorded at that moment."""
    if not isinstance(recorded, dict) or "path" not in recorded or "sha256" not in recorded:
        return None, f"{label}_not_recorded_at_load"
    path = recorded["path"]
    if not os.path.exists(path):
        return None, f"{label}_absent_at_grading"
    now = sha256_file(path)
    if now != recorded["sha256"]:
        return None, f"{label}_changed_after_load_{recorded['sha256'][:12]}_to_{now[:12]}"
    return now, None


def frozen_digests():
    path = os.path.join(TESTS_DIR, "frozen_digests.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def k_invariant_harness_code(recs):
    """INVARIANT. Live-state read: the run_s08.py and s08_core.py files the run itself
    recorded loading, rehashed at grading against the frozen bundle digests."""
    recorded, why = integrity_record(recs)
    if recorded is None:
        return False, why
    frozen = frozen_digests()
    if not frozen or "harness_code" not in frozen:
        return False, "frozen_digests_absent_from_verifier_tree"
    for name, want in sorted(frozen["harness_code"].items()):
        now, why = recheck_digest(recorded.get(name), name)
        if now is None:
            return False, why
        if now != want:
            return False, f"{name}_digest_{now[:12]}_is_not_the_frozen_{want[:12]}"
    return True, None


def k_value_checkpoint_bytes(recs, manifest, profile):
    """VALUE. Live-state read: the checkpoint file the run recorded loading, rehashed
    at grading against the manifest binding and against the frozen bundle digest."""
    recorded, why = integrity_record(recs)
    if recorded is None:
        return False, why
    now, why = recheck_digest(recorded.get("checkpoint"), "checkpoint")
    if now is None:
        return False, why
    want = manifest["fixture_sha256"]
    if now != want:
        return False, f"checkpoint_digest_{now[:12]}_does_not_match_manifest_{want[:12]}"
    frozen = (frozen_digests() or {}).get("profiles", {}).get(profile)
    if frozen and now != frozen["checkpoint"]:
        return False, f"checkpoint_digest_{now[:12]}_is_not_the_frozen_{frozen['checkpoint'][:12]}"
    return True, None


def k_value_manifest_bytes(recs, manifest_path, profile):
    """VALUE. Live-state read: the manifest path the runner recorded beside the
    manifest path the verifier grades from."""
    recorded, why = integrity_record(recs)
    if recorded is None:
        return False, why
    now, why = recheck_digest(recorded.get("manifest"), "manifest")
    if now is None:
        return False, why
    graded = sha256_file(manifest_path)
    if graded != now:
        return False, f"graded_manifest_{graded[:12]}_is_not_the_loaded_manifest_{now[:12]}"
    frozen = (frozen_digests() or {}).get("profiles", {}).get(profile)
    if frozen and now != frozen["manifest"]:
        return False, f"manifest_digest_{now[:12]}_is_not_the_frozen_{frozen['manifest'][:12]}"
    return True, None


def k_value_probe_budget(recs, digests, manifest):
    """VALUE. Live-state read: the probe_steps of resume_digests.json beside the
    probe_steps carried on every telemetry record of the same phase and seed."""
    if not digests:
        return False, "resume_digests_absent"
    limit = int(manifest["probe_batches_max"])
    for key, d in sorted(digests.items()):
        used = d.get("probe_steps")
        if not isinstance(used, int) or isinstance(used, bool):
            return False, f"{key}_probe_steps_not_an_integer"
        if used < 0 or used > limit:
            return False, f"{key}_charged_{used}_probes_against_an_allowance_of_{limit}"
    for phase in ("control", "submission"):
        for seed, rows in sorted(phase_seeds(recs, phase).items()):
            seen = {r.get("probe_steps") for r in rows}
            if len(seen) != 1:
                return False, f"{phase}_seed{seed}_records_disagree_on_probe_steps_{sorted(seen, key=str)}"
            key = f"{phase}_seed{seed}"
            if key in digests and seen != {digests[key]["probe_steps"]}:
                return False, f"{key}_probe_steps_disagree_between_telemetry_and_resume_digests"
    return True, None


def submission_effects(effects):
    return sorted((k, v) for k, v in effects.items() if k.startswith("submission"))


def k_absence_stale_step(effects, manifest):
    """ABSENCE. Live-state read: the per-parameter step field of the recovered
    optimizer fingerprint, against the checkpoint fingerprint and the manifest's
    checkpoint step."""
    if not effects:
        return False, "recover_effect_absent"
    true_step = float(int(manifest["checkpoint_step"]))
    pairs = submission_effects(effects)
    if not pairs:
        return False, "no_submission_recover_effect_recorded"
    for key, e in pairs:
        before = e["checkpoint_optimizer_fingerprint"]["state"]
        after = e["recovered_optimizer_fingerprint"]["state"]
        for idx, b in sorted(before.items(), key=lambda kv: int(kv[0])):
            stale = b.get("step")
            if stale is None or abs(stale - true_step) <= 1e-9:
                continue
            carried = after.get(idx, {}).get("step")
            if carried is not None and abs(carried - stale) <= 1e-9:
                return False, f"{key}_param{idx}_resumed_the_stale_step_counter_{stale}"
    return True, None


def k_effect_preconditioner(effects):
    """EFFECT. Live-state read: the per-parameter exp_avg_sq_rms of the checkpoint
    fingerprint beside the recovered fingerprint."""
    if not effects:
        return False, "recover_effect_absent"
    pairs = submission_effects(effects)
    if not pairs:
        return False, "no_submission_recover_effect_recorded"
    for key, e in pairs:
        before = e["checkpoint_optimizer_fingerprint"]["state"]
        after = e["recovered_optimizer_fingerprint"]["state"]
        changed = 0
        for idx, b in before.items():
            if "exp_avg_sq_rms" not in b:
                continue
            a = after.get(idx)
            if a is None or "exp_avg_sq_rms" not in a:
                changed += 1
                continue
            was = b["exp_avg_sq_rms"]
            if abs(a["exp_avg_sq_rms"] - was) > 1e-12 * max(1.0, abs(was)):
                changed += 1
        if not changed:
            return False, f"{key}_resumed_every_second_moment_the_checkpoint_carried"
    return True, None


HYPERPARAMETER_FIELDS = ("lr", "weight_decay", "eps", "betas")


def agrees(declared, recorded):
    if isinstance(recorded, list):
        if not isinstance(declared, (list, tuple)) or len(declared) != len(recorded):
            return False
        return all(agrees(d, r) for d, r in zip(declared, recorded))
    if not isinstance(recorded, (int, float)) or isinstance(recorded, bool):
        return False
    if math.isnan(recorded):
        return declared is None
    if not isinstance(declared, (int, float)) or isinstance(declared, bool):
        return False
    return abs(float(declared) - float(recorded)) <= 1e-9 * max(1.0, abs(float(recorded)))


def k_divergence_hyperparameters(effects):
    """DIVERGENCE. Live-state read: the optimizer block of submission/report.json
    beside the recovered param_groups the runner recorded for every submission run."""
    path = os.path.join(SUBMISSION, "report.json")
    if not os.path.exists(path):
        return False, "agent_report_absent"
    try:
        with open(path) as f:
            rep = json.load(f)
    except Exception:
        return False, "agent_report_malformed"
    declared = rep.get("optimizer")
    if not isinstance(declared, dict):
        return False, "agent_report_declares_no_optimizer_hyperparameters"
    if "weight_decay" not in declared:
        return False, "agent_report_optimizer_block_omits_weight_decay"
    named = [f for f in HYPERPARAMETER_FIELDS if f in declared]
    if not effects:
        return False, "recover_effect_absent"
    pairs = submission_effects(effects)
    if not pairs:
        return False, "no_submission_recover_effect_recorded"
    for key, e in pairs:
        groups = e["recovered_optimizer_fingerprint"]["param_groups"]
        for i, group in enumerate(groups):
            for field in named:
                if not agrees(declared[field], group.get(field)):
                    return False, f"{key}_group{i}_{field}_declared_{declared[field]}_in_force_{group.get(field)}"
    return True, None


def k_divergence_report(sub_steps, manifest):
    """DIVERGENCE. Live-state read: the agent's submission/report.json against the
    verifier's own recomputation from telemetry."""
    path = os.path.join(SUBMISSION, "report.json")
    if not os.path.exists(path):
        return False, "agent_report_absent"
    try:
        with open(path) as f:
            rep = json.load(f)
    except Exception:
        return False, "agent_report_malformed"
    if "steps_to_target" not in rep:
        return False, "agent_report_missing_steps_to_target"
    try:
        claimed = int(rep["steps_to_target"])
    except (TypeError, ValueError):
        return False, "agent_report_steps_to_target_not_an_integer"
    if sub_steps is None:
        return False, "agent_report_claims_a_crossing_the_telemetry_does_not_carry"
    if claimed != int(sub_steps):
        return False, f"agent_report_{claimed}_diverges_from_telemetry_{int(sub_steps)}"
    if "target_loss" in rep and abs(float(rep["target_loss"]) - float(manifest["target_loss"])) > 1e-6:
        return False, "agent_report_target_loss_diverges_from_manifest"
    return True, None


class Record:
    """One complete run record set, together with where it came from.

    The verifier grades two of these. `verifier` is the run this process started, in a
    directory it named from system entropy, stamped with a nonce nothing outside this process
    has seen. `agent` is whatever the agent left behind. The reward is read off the verifier
    record alone; the agent record is graded because a truthful account of the attempt is part
    of the deliverable and every artifact-tampering control binds to it.
    """

    def __init__(self, label, telemetry_dir):
        self.label = label
        self.dir = telemetry_dir
        self.recs, self.error = read_jsonl(os.path.join(telemetry_dir, "run_record.jsonl"))
        self.effects = _maybe(telemetry_dir, "recover_effect.json")
        self.digests = _maybe(telemetry_dir, "resume_digests.json")
        self.corpus_meta = _maybe(telemetry_dir, "corpus_meta.json")
        self.origin = _maybe(telemetry_dir, "run_origin.json")

    @property
    def present(self):
        return self.recs is not None


def _maybe(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def k_invariant_recover_isolation(rec):
    """INVARIANT. Live-state read: the recover_isolation block resume_digests.json carries for
    every submission run.

    A recovery that shares an interpreter with the measuring loop can rebind the evaluation the
    crossing is computed from, so a record that does not name a distinct worker process for
    every submission run is not evidence of a measurement. This is the structural property the
    rest of the roster now rests on.
    """
    if not rec.digests:
        return False, "resume_digests_absent"
    seen = 0
    for key, d in sorted(rec.digests.items()):
        if not key.startswith("submission"):
            continue
        iso = d.get("recover_isolation") or {}
        runner, worker = iso.get("runner_pid"), iso.get("worker_pid")
        if iso.get("mode") != "subprocess" or iso.get("worker") != "recover_worker.py":
            return False, f"{key}_recover_isolation_mode_{iso.get('mode')}"
        if not isinstance(worker, int) or worker <= 0 or worker == runner:
            return False, f"{key}_recover_worker_pid_{worker}_runner_pid_{runner}"
        seen += 1
    if seen == 0:
        return False, "no_submission_recover_isolation_recorded"
    return True, None


def k_value_record_origin(rec, expected_nonce, source, submission_dir):
    """VALUE. Live-state read: run_origin.json beside the graded telemetry, against the nonce
    this verifier process minted after it started and the live digest of the deliverable.

    Nothing written before this process existed can carry that nonce, so a planted, replayed or
    hand-authored record set fails here however internally consistent its hash chain is. An
    unkeyed forward chain seals a forgery; an origin stamp the forger has never seen does not.
    """
    if source != "verifier_execution":
        return False, f"authoritative_record_source_{source}"
    origin = rec.origin
    if not isinstance(origin, dict):
        return False, "run_origin_absent"
    if origin.get("run_nonce") != expected_nonce or not expected_nonce:
        return False, "run_nonce_absent_or_mismatched"
    live = os.path.join(submission_dir, "recover.py")
    if not os.path.exists(live):
        return False, "deliverable_absent_at_grading"
    if origin.get("submission_sha256") != sha256_file(live):
        return False, "graded_deliverable_is_not_the_measured_deliverable"
    return True, None


def k_invariant_chain_terminated(rec):
    """INVARIANT. Live-state read: the record count and terminal chain value of the telemetry,
    against the record_count and chain_head the runner wrote outside the chain.

    The forward chain (chain_n = sha256(chain_(n-1) + payload_n)) is complete under insertion,
    reordering and edit in place, and open under truncation: dropping the trailing record leaves
    every surviving link intact. Binding the head and the count outside the chain is what makes
    the terminal phase seal load bearing instead of decorative.
    """
    origin = rec.origin
    if not isinstance(origin, dict):
        return False, "run_origin_absent"
    want_n, want_head = origin.get("record_count"), origin.get("chain_head")
    if not isinstance(want_n, int) or not isinstance(want_head, str) or not want_head:
        return False, "run_origin_carries_no_chain_terminator"
    if len(rec.recs) != want_n:
        return False, f"record_count_{len(rec.recs)}_expected_{want_n}"
    if not rec.recs:
        return False, "telemetry_empty"
    tail = rec.recs[-1]
    if tail.get("chain") != want_head:
        return False, "terminal_chain_does_not_equal_recorded_head"
    if tail.get("kind") != "phase_seal" or tail.get("phase") != "submission":
        return False, f"terminal_record_is_{tail.get('phase')}_{tail.get('kind')}"
    return True, None


def k_divergence_reproduction(agent, verifier, manifest):
    """DIVERGENCE. Live-state read: the crossing series of the agent's own record beside the
    crossing series of the run this verifier executed for itself.

    Independence here is independence of PRODUCER, not merely of arithmetic. One record the
    submission could write, the other it could not, because the verifier names its own output
    directory from system entropy and hosts `recover` behind a process boundary. The tolerance
    is one evaluation window, because the cadence is the finest resolution either record has.
    """
    if not agent.present:
        return False, "agent_record_absent"
    window = int(manifest["eval_every"])
    target = float(manifest["target_loss"])
    rows = {}
    for phase in ("control", "submission"):
        a, _ = crossing(series(agent.recs, phase), target)
        v, _ = crossing(series(verifier.recs, phase), target)
        rows[phase] = (a, v)
        if a is None and v is None:
            continue
        if a is None or v is None:
            return False, f"{phase}_crossing_{a}_vs_verifier_{v}"
        if abs(int(a) - int(v)) > window:
            return False, f"{phase}_crossing_{a}_vs_verifier_{v}_beyond_one_window_{window}"
    return True, None


def run_graded_attempt(nonce, logs_dir, manifest):
    """Start one graded attempt under this process and return where it wrote its record.

    The output directory is named from the same entropy as the nonce, so nothing that existed
    before this process ran can have pre-planted a record there.
    """
    rundir = os.path.join(logs_dir, "verifier_run_%s" % nonce)
    if EXTERNAL_AUTHORITATIVE:
        return EXTERNAL_AUTHORITATIVE, "external", None
    recover = os.path.join(SUBMISSION, "recover.py")
    if not os.path.exists(recover):
        return rundir, "submission_absent", None
    os.makedirs(rundir, exist_ok=True)
    data_dir = DATA_DIR or os.path.join(STATE_DIR, "data", PROFILE)
    if not os.path.exists(os.path.join(data_dir, "val.bin")):
        meta = _maybe(TELEMETRY_DIR, "corpus_meta.json") or {}
        data_dir = meta.get("data_dir", data_dir)
    argv = [
        sys.executable,
        os.path.join(RUNNER_DIR, "run_s08.py"),
        "--profile", PROFILE,
        "--submission", SUBMISSION,
        "--out", rundir,
        "--fixture", os.path.join(FIXTURE_DIR, "ckpt_s08_%s.pt" % PROFILE),
        "--data-dir", data_dir,
    ]
    if DEVICE:
        argv += ["--device", DEVICE]
    env = dict(os.environ)
    env["S08_RUN_NONCE"] = nonce
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("S08_AUTHORITATIVE_TELEMETRY_DIR", None)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, env=env,
                              timeout=VERIFIER_RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return rundir, "timeout", None
    with open(os.path.join(logs_dir, "verifier-attempt-stdout.md"), "w") as f:
        f.write((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""))
    if not os.path.exists(os.path.join(rundir, "run_record.jsonl")):
        return rundir, "attempt_produced_no_record", proc.returncode
    return rundir, "verifier_execution", proc.returncode


def main():
    outcomes = {k: False for k in CHECKERS}

    manifest_path = os.path.join(FIXTURE_DIR, f"manifest_{PROFILE}.json")
    if not os.path.exists(manifest_path):
        return emit(0.0, "fixture_manifest_absent", {"path": manifest_path}, outcomes)
    with open(manifest_path) as f:
        manifest = json.load(f)

    logs_dir = os.path.dirname(SCORE_PATH) or "."
    os.makedirs(logs_dir, exist_ok=True)
    nonce = secrets.token_hex(16)
    auth_dir, source, rc = run_graded_attempt(nonce, logs_dir, manifest)
    verifier = Record("verifier", auth_dir)
    origin_detail = {"authoritative_source": source, "authoritative_dir": auth_dir,
                     "attempt_returncode": rc, "nonce_issued": bool(nonce)}
    if not verifier.present:
        return emit(0.0, f"verifier_execution_failed:{source}",
                    {"authoritative_record": origin_detail, "read_error": verifier.error}, outcomes)

    agent = Record("agent", TELEMETRY_DIR)

    target = float(manifest["target_loss"])
    max_steps = int(manifest["max_steps"])
    ceiling = max_steps + int(manifest["probe_batches_max"])
    ctrl_steps, ctrl_why = crossing(series(verifier.recs, "control"), target)
    sub_steps, sub_why = crossing(series(verifier.recs, "submission"), target)
    agent_sub_steps, _ = (crossing(series(agent.recs, "submission"), target)
                          if agent.present else (None, "agent_record_absent"))

    results = {}

    def run(name, fn, *args):
        """A checker that raises has not decided its question, and an undecided
        question is a failure with a reason rather than a crashed verifier."""
        try:
            results[name] = fn(*args)
        except Exception as exc:  # noqa: BLE001
            results[name] = (False, f"checker_raised_{type(exc).__name__}")

    def both(name, fn, per_record):
        """Decide once over the verifier-authored record and once over the agent-reported one.

        A checker passes only when both halves hold. The verifier half is what a forged agent
        record cannot reach; the agent half is what every artifact-tampering control binds to.
        """
        verdicts = []
        for rec in (verifier, agent):
            if rec.label == "agent" and not rec.present:
                verdicts.append((False, "agent_record_absent"))
                continue
            try:
                verdicts.append(fn(*per_record(rec)))
            except Exception as exc:  # noqa: BLE001
                verdicts.append((False, f"checker_raised_{type(exc).__name__}"))
        passed = all(v[0] for v in verdicts)
        why = None if passed else "; ".join(
            f"{lbl}:{v[1]}" for lbl, v in zip(("verifier", "agent"), verdicts) if not v[0]
        )
        results[name] = (passed, why)

    both("frozen_training_contract_held", k_frozen_invariant, lambda r: (r.recs, manifest))
    both("control_phase_sealed_before_submission_phase", k_ordering, lambda r: (r.recs,))
    both("no_weight_surgery_before_resume", k_absence_weight_surgery, lambda r: (r.digests, manifest))
    both("val_shard_digest_matches_manifest", k_value_val_shard, lambda r: (manifest, r.corpus_meta))
    both("train_shard_digest_matches_manifest", k_value_train_shard, lambda r: (manifest, r.corpus_meta))
    both("recover_mutated_optimizer_state", k_effect_recover, lambda r: (r.effects,))
    both("telemetry_hash_chain_unbroken", k_invariant_chain, lambda r: (r.recs,))
    both("telemetry_record_shape_matches_frozen_protocol", k_invariant_record_shape, lambda r: (r.recs, manifest))
    both("frozen_harness_code_unmodified", k_invariant_harness_code, lambda r: (r.recs,))
    both("shipped_checkpoint_bytes_unmodified", k_value_checkpoint_bytes, lambda r: (r.recs, manifest, PROFILE))
    both("fixture_manifest_bytes_unmodified", k_value_manifest_bytes, lambda r: (r.recs, manifest_path, PROFILE))
    both("probe_budget_within_declared_allowance", k_value_probe_budget, lambda r: (r.recs, r.digests, manifest))
    both("stale_step_counter_not_resumed", k_absence_stale_step, lambda r: (r.effects, manifest))
    both("optimizer_preconditioner_repaired", k_effect_preconditioner, lambda r: (r.effects,))
    both("declared_hyperparameters_match_optimizer_in_force", k_divergence_hyperparameters, lambda r: (r.effects,))
    both("recover_ran_outside_the_measuring_process", k_invariant_recover_isolation, lambda r: (r,))
    both("telemetry_chain_terminated_at_the_recorded_head", k_invariant_chain_terminated, lambda r: (r,))

    # The crossing is graded on the record this verifier authored, because that is the number
    # the reward is read off. The report is graded against the agent's own record, because the
    # report is a claim about the attempt the agent ran and nothing else.
    run("graded_crossing_within_ceiling", k_value_crossing, sub_steps, manifest, ceiling)
    run("agent_report_agrees_with_verifier_recompute", k_divergence_report, agent_sub_steps, manifest)
    run("graded_record_authored_by_the_verifier", k_value_record_origin, verifier, nonce, source, SUBMISSION)
    run("agent_run_reproduces_under_verifier_execution", k_divergence_reproduction, agent, verifier, manifest)

    for name, (passed, _) in results.items():
        outcomes[name] = bool(passed)

    kinds_ok, kinds_why = reduce_all(results)
    detail = {
        "authoritative_record": origin_detail,
        "control_steps_to_target": ctrl_steps,
        "control_reason": ctrl_why,
        "submission_steps_to_target": sub_steps,
        "submission_reason": sub_why,
        "agent_reported_run_steps_to_target": agent_sub_steps,
        "target_loss": target,
        "checker_reasons": {n: r for n, (p, r) in results.items() if not p},
        "zero_reasons": {n: zero_reason_for(n) for n, (p, _) in results.items() if not p},
    }
    if not kinds_ok:
        return emit(0.0, kinds_why, detail, outcomes)

    failed = [n for n, (p, _) in results.items() if not p]
    if failed:
        codes = sorted({zero_reason_for(n) for n in failed})
        return emit(0.0, " ".join(codes), detail, outcomes)

    baseline = ctrl_steps if ctrl_steps is not None else max_steps
    full_reward_at = max(1, int(math.ceil(float(manifest["target_fraction_of_control"]) * baseline)))
    detail["baseline_steps"] = baseline
    detail["full_reward_at_steps"] = full_reward_at
    if baseline <= full_reward_at:
        return emit(0.0, "degenerate_baseline_control_reached_target_immediately", detail, outcomes)

    raw = (baseline - sub_steps) / (baseline - full_reward_at)
    score = min(max(raw, 0.0), 1.0)
    detail["raw"] = raw
    return emit(score, f"graded_step={sub_steps}", detail, outcomes)


if __name__ == "__main__":
    main()

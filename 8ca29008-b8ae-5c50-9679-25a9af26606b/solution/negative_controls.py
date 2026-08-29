"""PRIVATE. Known-wrong routes for the bia S08 adequacy suite, and the driver that
measures their rejection.

Each entry is a route a solver plausibly takes, or a route the integrity lane exists
to reject. The liveness obligation is two-sided: the reference must be accepted by
every compiled checker, and every control here must be rejected by the checker its
table row names. A checker that accepts both is inert.

The table is the executable twin of `negative_controls` in solution/grounding.yaml
and `python3 solution/negative_controls.py --check-table` asserts the two agree by
identifier and by target checker.

    python3 solution/negative_controls.py --list
    python3 solution/negative_controls.py --all
    python3 solution/negative_controls.py --only naive_warm_start,telemetry_forged

The driver runs at the smoke operating point on CPU. Every case gets its own copy of
the harness tree, the fixture and the corpus, so a control that tampers with a frozen
artifact tampers with its own copy and never with the bundle bytes.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
PROFILE = "smoke"


# ---------------------------------------------------------------------------
# The routes. Each `recover` is installed as the submission's entry point.
# ---------------------------------------------------------------------------


def _reference():
    sys.path.insert(0, HERE)
    import reference_recovery

    return reference_recovery


def _fixture_path():
    return os.environ["S08_FIXTURE"]


def _manifest_path():
    return os.environ["S08_MANIFEST"]


def _data_dir():
    return os.environ["S08_DATA_DIR"]


def _telemetry_path():
    return os.path.join(os.environ["S08_TELEMETRY_DIR"], "run_record.jsonl")


def _load_verbatim(model, ckpt_state):
    import torch

    opt = torch.optim.AdamW(list(model.parameters()), lr=3.0e-3)
    opt.load_state_dict(copy.deepcopy(ckpt_state))
    return opt


def naive_warm_start(model, ckpt_state, cfg, probe):
    """The pattern-matched resume. It inherits the poison whole."""
    return _load_verbatim(model, ckpt_state)


def step_counter_only_fix(model, ckpt_state, cfg, probe):
    """Notices the impossible counter, corrects it, and leaves the real damage."""
    import torch

    opt = _load_verbatim(model, ckpt_state)
    true_step = float(int(cfg.get("checkpoint_step", 0)))
    for state in opt.state.values():
        if "step" in state:
            step = state["step"]
            state["step"] = (
                torch.full_like(step, true_step) if torch.is_tensor(step) else true_step
            )
    return opt


def hyperparameters_only_fix(model, ckpt_state, cfg, probe):
    """Replaces the stored hyperparameters and resumes the poisoned state under them."""
    import torch

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=3.0e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    src = ckpt_state.get("state", {})
    for i, p in enumerate(params):
        entry = src.get(i, src.get(str(i), {}))
        state = opt.state[p]
        for key in ("exp_avg", "exp_avg_sq", "step"):
            if key in entry:
                value = entry[key]
                state[key] = value.detach().clone() if torch.is_tensor(value) else value
    return opt


def weight_surgery(model, ckpt_state, cfg, probe):
    """A real repair, plus a nudge to the weights the harness handed over."""
    import torch

    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1.0001)
    return _reference().recover(model, ckpt_state, cfg, probe)


def probe_budget_overrun(model, ckpt_state, cfg, probe):
    """Buys more measurement than the allowance grants by raising the cap on the
    budget object the harness handed in."""
    probe.limit = 10**6
    probe(int(cfg.get("probe_batches_max", 8)) + 4)
    return _reference().recover(model, ckpt_state, cfg, probe)


def _rewrite_telemetry_on_exit(mutate):
    def rewrite():
        path = _telemetry_path()
        if not os.path.exists(path):
            return
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        records = mutate(records)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    atexit.register(rewrite)


def telemetry_forged(model, ckpt_state, cfg, probe):
    """Edits the finished record in place to claim the target was already reached."""

    def mutate(records):
        for record in records:
            if record.get("phase") == "submission" and record.get("kind") == "eval":
                record["val_loss"] = 0.001
        return records

    _rewrite_telemetry_on_exit(mutate)
    return _reference().recover(model, ckpt_state, cfg, probe)


def telemetry_record_inserted(model, ckpt_state, cfg, probe):
    """Appends chain-consistent records off the frozen evaluation cadence, so the
    forgery survives the hash chain and has to be caught by the record shape."""

    def mutate(records):
        chain = records[-1]["chain"] if records else "genesis"
        index = len(records)
        evals = [r for r in records if r.get("phase") == "submission" and r.get("kind") == "eval"]
        for seed in sorted({r["seed"] for r in evals}):
            body = {k: v for k, v in evals[0].items() if k != "chain"}
            body.update(
                {"seed": seed, "step": int(body["step"]) + 7, "val_loss": 0.001, "record_index": index}
            )
            payload = json.dumps(body, sort_keys=True)
            chain = hashlib.sha256((chain + payload).encode()).hexdigest()
            body["chain"] = chain
            records.append(body)
            index += 1
        return records

    _rewrite_telemetry_on_exit(mutate)
    return _reference().recover(model, ckpt_state, cfg, probe)


def _rechain(records):
    """Rebuild the whole hash chain from genesis, so a rewritten record set still
    verifies. This is the forgery the unkeyed chain does not defeat, and it is here so
    the checkers that must catch such a rewrite are measured against one."""
    chain = "genesis"
    out = []
    for index, record in enumerate(records):
        body = {k: v for k, v in record.items() if k != "chain"}
        body["record_index"] = index
        payload = json.dumps(body, sort_keys=True)
        chain = hashlib.sha256((chain + payload).encode()).hexdigest()
        body["chain"] = chain
        out.append(body)
    return out


def frozen_contract_moved(model, ckpt_state, cfg, probe):
    """Rewrites the frozen operating point onto every record and rebuilds the chain,
    so only the frozen-contract checker can catch it."""

    def mutate(records):
        for record in records:
            if "d_model" in record:
                record["d_model"] = int(record["d_model"]) * 2
        return _rechain(records)

    _rewrite_telemetry_on_exit(mutate)
    return _reference().recover(model, ckpt_state, cfg, probe)


def control_phase_seal_removed(model, ckpt_state, cfg, probe):
    """Drops the control-phase seal and rebuilds the chain, so the baseline is no
    longer provably fixed before the submission was observed."""

    def mutate(records):
        kept = [
            r for r in records
            if not (r.get("kind") == "phase_seal" and r.get("phase") == "control")
        ]
        return _rechain(kept)

    _rewrite_telemetry_on_exit(mutate)
    return _reference().recover(model, ckpt_state, cfg, probe)


def crossing_never_reached(model, ckpt_state, cfg, probe):
    """A recovery that abandons the objective: the state is replaced, and the run then
    never moves because every step is taken at a learning rate of zero."""
    import torch

    opt = torch.optim.AdamW(list(model.parameters()), lr=0.0, weight_decay=0.01)
    return opt, lambda step: 0.0


def checkpoint_bytes_edited(model, ckpt_state, cfg, probe):
    """Rewrites the shipped checkpoint from inside the entry point."""
    with open(_fixture_path(), "ab") as f:
        f.write(b"\x00")
    return _reference().recover(model, ckpt_state, cfg, probe)


def fixture_manifest_edited(model, ckpt_state, cfg, probe):
    """Lowers the bar by editing the manifest the verifier grades against."""
    with open(_manifest_path()) as f:
        manifest = json.load(f)
    manifest["target_loss"] = round(float(manifest["target_loss"]) + 2.0, 6)
    with open(_manifest_path(), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return _reference().recover(model, ckpt_state, cfg, probe)


def harness_code_edited(model, ckpt_state, cfg, probe):
    """Edits the frozen core the run imported."""
    sys.path.insert(0, os.path.join(os.environ["S08_RUNNER_DIR"]))
    import s08_core

    with open(s08_core.__file__, "a") as f:
        f.write("\nPATCHED_BY_SUBMISSION = True\n")
    return _reference().recover(model, ckpt_state, cfg, probe)


def val_shard_edited(model, ckpt_state, cfg, probe):
    """Rewrites the validation shard the evaluation protocol reads."""
    with open(os.path.join(_data_dir(), "val.bin"), "ab") as f:
        f.write(b"\x00\x00")
    return _reference().recover(model, ckpt_state, cfg, probe)


def train_shard_edited(model, ckpt_state, cfg, probe):
    """Rewrites the training shard the graded steps consume."""
    with open(os.path.join(_data_dir(), "train.bin"), "ab") as f:
        f.write(b"\x00\x00")
    return _reference().recover(model, ckpt_state, cfg, probe)


def reference(model, ckpt_state, cfg, probe):
    return _reference().recover(model, ckpt_state, cfg, probe)


def _understate_steps(report):
    report["steps_to_target"] = max(0, int(report["steps_to_target"]) - 20)
    return report


def _false_hyperparameters(report):
    report["optimizer"] = dict(report["optimizer"], weight_decay=0.5)
    return report


def _drop_hyperparameters(report):
    report.pop("optimizer", None)
    return report


CONTROLS = {
    "naive_warm_start": {
        "recover": naive_warm_start,
        "rejected_by": [
            "recover_mutated_optimizer_state",
            "stale_step_counter_not_resumed",
            "optimizer_preconditioner_repaired",
        ],
    },
    "step_counter_only_fix": {
        "recover": step_counter_only_fix,
        "rejected_by": ["optimizer_preconditioner_repaired"],
    },
    "hyperparameters_only_fix": {
        "recover": hyperparameters_only_fix,
        "rejected_by": ["stale_step_counter_not_resumed", "optimizer_preconditioner_repaired"],
    },
    "weight_surgery": {
        "recover": weight_surgery,
        "rejected_by": ["no_weight_surgery_before_resume"],
    },
    "probe_budget_overrun": {
        "recover": probe_budget_overrun,
        "rejected_by": ["probe_budget_within_declared_allowance"],
    },
    "telemetry_forged": {
        "recover": telemetry_forged,
        "rejected_by": ["telemetry_hash_chain_unbroken"],
    },
    "telemetry_record_inserted": {
        "recover": telemetry_record_inserted,
        "rejected_by": ["telemetry_record_shape_matches_frozen_protocol"],
    },
    "frozen_contract_moved": {
        "recover": frozen_contract_moved,
        "rejected_by": ["frozen_training_contract_held"],
    },
    "control_phase_seal_removed": {
        "recover": control_phase_seal_removed,
        "rejected_by": ["control_phase_sealed_before_submission_phase"],
    },
    "crossing_never_reached": {
        "recover": crossing_never_reached,
        "rejected_by": ["graded_crossing_within_ceiling"],
    },
    "checkpoint_bytes_edited": {
        "recover": checkpoint_bytes_edited,
        "rejected_by": ["shipped_checkpoint_bytes_unmodified"],
    },
    "fixture_manifest_edited": {
        "recover": fixture_manifest_edited,
        "rejected_by": ["fixture_manifest_bytes_unmodified"],
    },
    "harness_code_edited": {
        "recover": harness_code_edited,
        "rejected_by": ["frozen_harness_code_unmodified"],
    },
    "val_shard_edited": {
        "recover": val_shard_edited,
        "rejected_by": ["val_shard_digest_matches_manifest"],
    },
    "train_shard_edited": {
        "recover": train_shard_edited,
        "rejected_by": ["train_shard_digest_matches_manifest"],
    },
    "report_step_count_false": {
        "recover": reference,
        "report": _understate_steps,
        "rejected_by": ["agent_report_agrees_with_verifier_recompute"],
    },
    "report_hyperparameters_false": {
        "recover": reference,
        "report": _false_hyperparameters,
        "rejected_by": ["declared_hyperparameters_match_optimizer_in_force"],
    },
    "report_hyperparameters_absent": {
        "recover": reference,
        "report": _drop_hyperparameters,
        "rejected_by": ["declared_hyperparameters_match_optimizer_in_force"],
    },
}


def build_recover(name):
    if name == "reference":
        return reference
    if name not in CONTROLS:
        raise SystemExit(f"unknown control {name!r}, choose from {sorted(CONTROLS)}")
    return CONTROLS[name]["recover"]


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------

RECOVER_STUB = """import os, sys
sys.path.insert(0, os.environ["BIA_CONTROLS_DIR"])
from negative_controls import build_recover

recover = build_recover(os.environ["BIA_CONTROL"])
"""


def derive_report(telemetry, manifest, declaration):
    """The submission's own derivation of its step count, written independently of the
    verifier so the DIVERGENCE checker compares two computations rather than one."""
    target = float(manifest["target_loss"])
    by_seed = {}
    with open(telemetry) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("phase") != "submission" or record.get("kind") != "eval":
                continue
            by_seed.setdefault(int(record["seed"]), {})[int(record["step"])] = float(record["val_loss"])
    steps = None
    if by_seed:
        shared = sorted(set.intersection(*[set(v) for v in by_seed.values()]))
        holds = []
        for step in shared:
            values = [by_seed[s][step] for s in by_seed]
            holds.append(max(values) <= target + 1e-9 and sum(values) / len(values) <= target + 1e-9)
        for i, step in enumerate(shared):
            if holds[i] and all(holds[i:]):
                steps = step
                break
    return {
        "steps_to_target": steps if steps is not None else -1,
        "target_loss": target,
        "seeds": sorted(by_seed),
        "optimizer": dict(declaration),
    }


def build_base(work, threads):
    sys.path.insert(0, HERE)
    import make_fixture

    data = os.path.join(work, "base", "data", PROFILE)
    fixtures = os.path.join(work, "base", "fixtures")
    os.makedirs(data, exist_ok=True)
    make_fixture.build(PROFILE, fixtures, data, threads)
    return fixtures, data


def run_case(name, work, base_fixtures, base_data, verbose=False):
    control = CONTROLS.get(name, {"recover": reference, "rejected_by": []})
    case = os.path.join(work, "case-" + name)
    runner = os.path.join(case, "runner")
    fixtures = os.path.join(case, "fixtures")
    data = os.path.join(case, "data", PROFILE)
    submission = os.path.join(case, "submission")
    telemetry = os.path.join(case, "telemetry")
    logs = os.path.join(case, "logs")

    shutil.copytree(os.path.join(BUNDLE, "environment", "runner"), runner)
    shutil.copytree(base_fixtures, fixtures)
    shutil.copytree(base_data, data)
    for path in (submission, telemetry, logs):
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(submission, "recover.py"), "w") as f:
        f.write(RECOVER_STUB)

    fixture = os.path.join(fixtures, f"ckpt_s08_{PROFILE}.pt")
    manifest_path = os.path.join(fixtures, f"manifest_{PROFILE}.json")

    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "BIA_CONTROL": name,
            "BIA_CONTROLS_DIR": HERE,
            "S08_PROFILE": PROFILE,
            "S08_RUNNER_DIR": runner,
            "S08_FIXTURE": fixture,
            "S08_MANIFEST": manifest_path,
            "S08_FIXTURE_DIR": fixtures,
            "S08_DATA_DIR": data,
            "S08_TELEMETRY_DIR": telemetry,
            "S08_SUBMISSION": submission,
            "S08_BUNDLE": BUNDLE,
            "LOG_DIR": logs,
            "SCORE_PATH": os.path.join(logs, "score.json"),
            "S08_OUTCOMES": os.path.join(logs, "outcomes.json"),
        }
    )

    run = subprocess.run(
        [
            sys.executable, os.path.join(runner, "run_s08.py"),
            "--profile", PROFILE, "--submission", submission, "--out", telemetry,
            "--fixture", fixture, "--data-dir", data, "--device", "cpu",
        ],
        env=env, capture_output=True, text=True,
    )
    if verbose or run.returncode != 0:
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)

    runner_failed = run.returncode != 0
    telemetry_file = os.path.join(telemetry, "run_record.jsonl")
    if not runner_failed and os.path.exists(telemetry_file):
        with open(manifest_path) as f:
            manifest = json.load(f)
        reference_module = _reference()
        declaration = {
            "lr": reference_module.BASE_LR,
            "betas": list(reference_module.BETAS),
            "eps": reference_module.EPS,
            "weight_decay": reference_module.WEIGHT_DECAY,
        }
        report = derive_report(telemetry_file, manifest, declaration)
        if control.get("report"):
            report = control["report"](report)
        with open(os.path.join(submission, "report.json"), "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)

    grade = subprocess.run(
        [sys.executable, os.path.join(BUNDLE, "tests", "grade.py")],
        env=env, capture_output=True, text=True,
    )
    if verbose:
        sys.stdout.write(grade.stdout)

    outcomes, score = {}, None
    outcomes_path = os.path.join(logs, "outcomes.json")
    score_path = os.path.join(logs, "score.json")
    if os.path.exists(outcomes_path):
        with open(outcomes_path) as f:
            outcomes = json.load(f)
    if os.path.exists(score_path):
        with open(score_path) as f:
            score = json.load(f).get("score")
    reasons = {}
    try:
        reasons = json.loads(grade.stdout.strip().splitlines()[-1])["detail"]["checker_reasons"]
    except Exception:
        pass
    return {
        "control": name,
        "runner_exit": run.returncode,
        "score": score,
        "outcomes": outcomes,
        "reasons": reasons,
        "rejected_by": control["rejected_by"],
    }


def check_table():
    import yaml

    with open(os.path.join(HERE, "grounding.yaml")) as f:
        grounding = yaml.safe_load(f)
    declared = {c["id"]: sorted(c["rejected_by"]) for c in grounding["negative_controls"]}
    implemented = {name: sorted(c["rejected_by"]) for name, c in CONTROLS.items()}
    if declared != implemented:
        only_declared = sorted(set(declared) - set(implemented))
        only_implemented = sorted(set(implemented) - set(declared))
        differing = sorted(
            k for k in set(declared) & set(implemented) if declared[k] != implemented[k]
        )
        print(json.dumps({
            "declared_only": only_declared,
            "implemented_only": only_implemented,
            "target_checkers_disagree": differing,
        }, indent=1))
        return 1
    checkers = {c["id"] for c in grounding["checkers"]}
    unknown = sorted({t for targets in declared.values() for t in targets} - checkers)
    if unknown:
        print(json.dumps({"targets_naming_no_checker": unknown}, indent=1))
        return 1
    print(f"ok       control table agrees with grounding.yaml over {len(declared)} controls")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check-table", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("S08_GEN_THREADS", "4")))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for name, control in sorted(CONTROLS.items()):
            print(f"{name:34s} rejected_by {', '.join(control['rejected_by'])}")
        return 0
    if args.check_table:
        return check_table()

    names = [n for n in args.only.split(",") if n] if args.only else sorted(CONTROLS)
    for name in names:
        if name not in CONTROLS:
            raise SystemExit(f"unknown control {name!r}")

    work = tempfile.mkdtemp(prefix="bia-s08-controls-")
    print(f"work {work}")
    try:
        base_fixtures, base_data = build_base(work, args.threads)
        oracle = run_case("reference", work, base_fixtures, base_data, args.verbose)
        accepted = sorted(k for k, v in oracle["outcomes"].items() if v)
        refused = sorted(k for k, v in oracle["outcomes"].items() if not v)
        print("")
        print("== reference oracle ==")
        print(json.dumps({
            "score": oracle["score"],
            "checkers_true": len(accepted),
            "checkers_false": refused,
            "reasons": oracle["reasons"],
        }, indent=1, sort_keys=True))
        failures = []
        if refused:
            failures.append("reference rejected by " + ",".join(refused))

        print("")
        print("== known-wrong controls ==")
        rows = []
        for name in names:
            result = run_case(name, work, base_fixtures, base_data, args.verbose)
            still_true = [c for c in result["rejected_by"] if result["outcomes"].get(c) is not False]
            verdict = "REJECTED" if not still_true else "INERT"
            if still_true:
                failures.append(f"{name}: accepted by {','.join(still_true)}")
            rows.append({
                "control": name,
                "verdict": verdict,
                "score": result["score"],
                "target_checkers": {c: result["outcomes"].get(c) for c in result["rejected_by"]},
                "reasons": {
                    c: result["reasons"].get(c) for c in result["rejected_by"] if c in result["reasons"]
                },
                "collateral_failures": sorted(
                    c for c, ok in result["outcomes"].items()
                    if not ok and c not in result["rejected_by"]
                ),
            })
            print(json.dumps(rows[-1], indent=1, sort_keys=True))

        print("")
        print("== summary ==")
        covered = sorted({c for row in rows for c in row["target_checkers"]})
        print(json.dumps({
            "oracle_score": oracle["score"],
            "oracle_checkers_true": len(accepted),
            "controls_run": len(rows),
            "controls_rejected": sum(1 for r in rows if r["verdict"] == "REJECTED"),
            "checkers_with_a_measured_rejection": covered,
            "failures": failures,
        }, indent=1, sort_keys=True))
        return 1 if failures else 0
    finally:
        if args.keep:
            print(f"kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

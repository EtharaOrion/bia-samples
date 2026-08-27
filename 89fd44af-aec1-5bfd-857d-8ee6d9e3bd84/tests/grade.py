from __future__ import annotations

import glob
import hashlib
import hmac
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkers"))

from checkers.parser import parse_file

BUNDLE = os.environ.get("TRACK3_BUNDLE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCORE_PATH = os.environ.get("SCORE_PATH", os.path.join(BUNDLE, "score.json"))
OUTCOMES_PATH = os.environ.get("TRACK3_OUTCOMES", "/tmp/track3_outcomes.json")
TELEMETRY = os.environ.get("TRACK3_TELEMETRY", "/telemetry/run_record.jsonl")
TELEMETRY_DIR = os.environ.get("TRACK3_TELEMETRY_DIR", "/telemetry")
SUBMISSION = os.environ.get("TRACK3_SUBMISSION", os.path.join(BUNDLE, "submission"))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.environ.get("TRACK3_CORPUS_DIR", os.path.join(TESTS_DIR, "corpus"))

BASELINE_STEPS = 3500
TARGET_STEPS = 2900
TARGET_LOSS = 3.28
SIG_MARGIN = 0.004
MIN_SEEDS = 2

FROZEN = {
    "global_batch_tokens": 524288,
    "microbatch_sequences_per_forward": 64,
    "sequence_length": 1024,
    "forward_backward_per_step": 1,
}


def emit(score: float, reason: str, detail=None, outcomes=None):
    os.makedirs(os.path.dirname(SCORE_PATH) or ".", exist_ok=True)
    with open(SCORE_PATH, "w") as f:
        json.dump({"score": score, "reason": reason, "detail": detail or {}}, f, indent=1, sort_keys=True)
    with open(OUTCOMES_PATH, "w") as f:
        json.dump(outcomes or {}, f, indent=1, sort_keys=True)
    print(json.dumps({"score": score, "reason": reason, "detail": detail or {}}, sort_keys=True))
    return score


def load_telemetry(path):
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


def check_submission_binding(recs, submission_path):
    opt = os.path.join(submission_path, "optimizer.py")
    if not os.path.exists(opt):
        return False, "submission_optimizer_absent"
    d = hashlib.sha256(open(opt, "rb").read()).hexdigest()
    for r in recs:
        if r.get("optimizer_digest") != d:
            return False, f"telemetry_not_bound_to_submission_step_{r.get('step')}"
    return True, None


def check_telemetry_chain(recs):
    key = os.environ.get("TRACK3_CHAIN_KEY")
    if not key:
        return False, "chain_key_absent_cannot_verify"
    prev = "genesis"
    for r in recs:
        body = {k: v for k, v in r.items() if k != "chain"}
        expect = hmac.new(key.encode(), (prev + json.dumps(body, sort_keys=True)).encode(), hashlib.sha256).hexdigest()
        if r.get("chain") != expect:
            return False, f"chain_break_at_step_{r.get('step')}"
        prev = expect
    return True, None


def check_run_mode(recs):
    for r in recs:
        if r.get("mode") != "full":
            return False, f"non_full_mode_telemetry_{r.get('mode')}"
    return True, None


COPY_SIMILARITY_MAX = 0.90


def _ast_norm(path):
    import ast
    import pathlib
    try:
        t = ast.parse(pathlib.Path(path).read_text())
    except Exception:
        return None
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(n):
            n.body = n.body[1:]
    return ast.dump(t, annotate_fields=False)


def check_verbatim_copy(optimizer_path):
    import difflib
    import pathlib
    s = _ast_norm(optimizer_path)
    if s is None:
        return False, "optimizer_unreadable_for_copy_check"
    here = pathlib.Path(TESTS_DIR)
    refs = [p for p in (here / "reference_optimizer.py",
                        pathlib.Path(BUNDLE) / "solution" / "reference_optimizer.py")
            if p.is_file()]
    corpus = pathlib.Path(CORPUS_DIR)
    if corpus.is_dir():
        refs += sorted(corpus.glob("*.py"))
    if not refs:
        return False, "no_reference_available_for_copy_check"
    best = None
    for r in refs:
        o = _ast_norm(r)
        if o is None:
            continue
        ratio = difflib.SequenceMatcher(None, s, o).ratio()
        if ratio >= COPY_SIMILARITY_MAX and (best is None or ratio > best[0]):
            digest = hashlib.sha256(r.read_bytes()).hexdigest()
            best = (ratio, digest)
    if best is None:
        return True, None
    ratio, digest = best
    names = sorted(p.stem for p in refs
                   if hashlib.sha256(p.read_bytes()).hexdigest() == digest)
    return False, f"verbatim_copy_of_{'_or_'.join(names)}_similarity_{ratio:.3f}"


FWD_BWD_KEYS = ("forward_backward_per_step", "fwd_bwd_per_step", "microbatch_sequences_per_forward_passes")


def check_one_fwd_bwd(recs):
    seen = 0
    for r in recs:
        n = next((r[k] for k in FWD_BWD_KEYS if k in r), None)
        if n is None:
            continue
        seen += 1
        if int(n) != 1:
            return False, f"fwd_bwd_{n}_at_step_{r.get('step')}"
    if seen == 0:
        return False, "fwd_bwd_count_absent_from_telemetry"
    return True, None


def check_frozen_contract(recs):
    for r in recs:
        for k, v in FROZEN.items():
            if r.get(k) != v:
                return False, f"frozen_violation_{k}_step_{r.get('step')}"
        if r.get("early_stop") is True:
            return False, f"val_selection_at_step_{r.get('step')}"
    return check_canonical_seeds(recs)


CAPTURE_MARKERS = ("novelty_capture_file", "novelty_capture_digest", "novelty_capture_steps")


def is_training_validation(r):
    return "val_loss" in r and not any(k in r for k in CAPTURE_MARKERS)


def check_canonical_seeds(recs):
    seeds = {r["seed"] for r in recs if is_training_validation(r) and "seed" in r}
    if not seeds:
        return True, None
    try:
        idx = sorted(int(s) for s in seeds)
    except (TypeError, ValueError):
        return False, f"non_numeric_seed_{'_'.join(sorted(map(str, seeds)))}"
    if idx != list(range(len(idx))):
        return False, f"seed_set_not_canonical_prefix_{'_'.join(map(str, idx))}"
    return True, None


def telemetry_series(recs):
    observed = {}
    for r in recs:
        if not is_training_validation(r):
            continue
        observed.setdefault(r["seed"], {})[r["step"]] = round(float(r["val_loss"]), 5)
    return observed


def check_reconciliation(recs, seed_logs):
    observed = telemetry_series(recs)
    for seed, reported in seed_logs.items():
        if seed not in observed:
            return False, f"reported_seed_{seed}_absent_from_telemetry"
        for step, loss in reported.items():
            if step not in observed[seed]:
                return False, f"reported_step_{step}_seed_{seed}_absent_from_telemetry"
            if abs(observed[seed][step] - round(loss, 5)) > 1e-9:
                return False, f"reported_loss_diverges_seed_{seed}_step_{step}"
    for seed, seen in observed.items():
        if seed not in seed_logs:
            return False, f"telemetry_seed_{seed}_absent_from_report"
        for step in seen:
            if step not in seed_logs[seed]:
                return False, f"telemetry_step_{step}_seed_{seed}_absent_from_report"
    return True, None


def graded_step(seed_logs):
    if len(seed_logs) < MIN_SEEDS:
        return None, f"need_at_least_{MIN_SEEDS}_seeds_got_{len(seed_logs)}"
    common = sorted(set.intersection(*[set(d) for d in seed_logs.values()]))
    if not common:
        return None, "no_step_logged_by_every_seed"

    def at(step):
        vals = [seed_logs[k][step] for k in seed_logs]
        return sum(vals) / len(vals), max(vals), len(vals)

    def holds(step):
        mean, worst, _ = at(step)
        return mean <= TARGET_LOSS and worst <= TARGET_LOSS

    rejected = None
    for i, s in enumerate(common):
        mean, worst, n = at(s)
        if mean > TARGET_LOSS:
            continue
        if (TARGET_LOSS - mean) * math.sqrt(n) < SIG_MARGIN:
            continue
        if worst > TARGET_LOSS:
            rejected = rejected or f"seed_above_target_at_step_{s}_worst_{worst:.5f}"
            continue
        regressed = next((t for t in common[i + 1:] if not holds(t)), None)
        if regressed is not None:
            rejected = rejected or f"crossing_at_{s}_not_sustained_at_step_{regressed}"
            continue
        return s, None
    return None, rejected or "no_step_clears_noise_floor"


def load_reported_losses(subdir):
    path = os.path.join(subdir, "reported_losses.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for seed, curve in raw.items():
        out[str(seed)] = {int(k): float(v) for k, v in curve.items()}
    return out


def load_seed_logs(subdir):
    paths = sorted(glob.glob(os.path.join(subdir, "logs", "*.log")))
    full = [p for p in paths if os.path.basename(p).startswith("full_")]
    if full:
        paths = full
    if not paths:
        reported = load_reported_losses(subdir)
        if reported is not None:
            return reported, None
    out, seen = {}, {}
    for p in paths:
        base = os.path.basename(p)
        seed = str(base.split("seed")[-1].split(".")[0] if "seed" in base else base.split(".")[0])
        parsed = parse_file(p)
        if not parsed:
            continue
        if seed in seen:
            return None, f"seed_log_collision_{seed}_{seen[seed]}_and_{base}"
        seen[seed] = base
        out[seed] = parsed
    return out, None


def main():
    outcomes = {
        "frozen_recipe_unchanged": False,
        "one_fwd_bwd_per_step": False,
        "multi_seed_noise_floor": False,
        "faithful_reporting": False,
        "telemetry_chain_verified": False,
        "submission_binding_verified": False,
        "full_run_mode_verified": False,
        "full_score_target_reached": False,
        "optimizer_not_verbatim_copy": False,
    }

    recs, terr = load_telemetry(TELEMETRY)
    if recs is None:
        return emit(0.0, terr, {}, outcomes)

    chain_ok, chain_reason = check_telemetry_chain(recs)
    outcomes["telemetry_chain_verified"] = bool(chain_ok)
    if not chain_ok:
        return emit(0.0, chain_reason, {}, outcomes)

    bind_ok, bind_reason = check_submission_binding(recs, SUBMISSION)
    outcomes["submission_binding_verified"] = bool(bind_ok)
    if not bind_ok:
        return emit(0.0, bind_reason, {}, outcomes)

    mode_ok, mode_reason = check_run_mode(recs)
    outcomes["full_run_mode_verified"] = bool(mode_ok)
    if not mode_ok:
        return emit(0.0, mode_reason, {}, outcomes)

    frozen_ok, frozen_reason = check_frozen_contract(recs)
    outcomes["frozen_recipe_unchanged"] = bool(frozen_ok)
    copy_ok, copy_reason = check_verbatim_copy(os.path.join(SUBMISSION, "optimizer.py"))
    outcomes["optimizer_not_verbatim_copy"] = bool(copy_ok)
    fwd_ok, fwd_reason = check_one_fwd_bwd(recs)
    outcomes["one_fwd_bwd_per_step"] = bool(fwd_ok)

    seed_logs, seed_log_reason = load_seed_logs(SUBMISSION)
    if seed_logs is None:
        return emit(0.0, seed_log_reason, {}, outcomes)
    recon_ok, recon_reason = check_reconciliation(recs, seed_logs)
    outcomes["faithful_reporting"] = bool(recon_ok)

    step, step_reason = graded_step(telemetry_series(recs))
    outcomes["multi_seed_noise_floor"] = step is not None
    outcomes["full_score_target_reached"] = step is not None and step <= TARGET_STEPS

    if not frozen_ok:
        return emit(0.0, frozen_reason, {}, outcomes)
    if not fwd_ok:
        return emit(0.0, fwd_reason, {}, outcomes)
    if not copy_ok:
        return emit(0.0, copy_reason, {}, outcomes)
    if not recon_ok:
        return emit(0.0, recon_reason, {}, outcomes)
    if step is None:
        return emit(0.0, step_reason, {}, outcomes)

    score = max(0.0, min(1.0, (BASELINE_STEPS - step) / (BASELINE_STEPS - TARGET_STEPS)))
    return emit(score, f"graded_step={step}", {"novelty": {"status": "deferred"}, "seeds": len(seed_logs)}, outcomes)


if __name__ == "__main__":
    main()

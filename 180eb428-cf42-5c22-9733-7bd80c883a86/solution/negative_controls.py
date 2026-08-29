"""PRIVATE. Known-wrong precision policies for the bia S07 adequacy suite.

Each entry is a route a solver plausibly takes. The feasibility argument requires that the
reference is accepted and that every one of these is rejected or scores materially below it,
and that the rejection is measured rather than asserted. solve.sh --controls runs the whole
table in smoke mode on CPU and prints the outcomes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

TARGET = 32.0


def _block_exponents(stats: List[Dict[str, float]], n: int, target: float = TARGET) -> List[int]:
    out = [0] * n
    for s in stats:
        a = float(s["absmax"])
        if a > 0.0 and math.isfinite(a):
            out[int(s["index"])] = max(-64, min(64, int(math.floor(math.log2(target / a)))))
    return out


class _Base:
    master_dtype = "float32"
    shape = "pairwise"
    align = "min"
    node_exp = -1
    compensated = True
    scale_exp = 0
    block = True
    sort_by_magnitude = True

    def __init__(self, config: Dict[str, Any]) -> None:
        self.n_chunks = int(config["micro_batches"])

    def loss_scale(self, step: int) -> int:
        return self.scale_exp

    def plan(self, step: int, n_chunks: int, stats: List[Dict[str, float]]) -> Dict[str, Any]:
        leaf = _block_exponents(stats, n_chunks) if self.block else [0] * n_chunks
        order = (
            sorted(range(n_chunks), key=lambda i: float(stats[i]["absmax"]))
            if self.sort_by_magnitude
            else list(range(n_chunks))
        )
        return {
            "order": order,
            "shape": self.shape,
            "compensated": self.compensated,
            "leaf_exp": leaf,
            "align": self.align,
            "node_exp": self.node_exp,
        }

    def on_overflow(self, step: int, info: Dict[str, Any]) -> str:
        return "continue"


class RecalledDynamicLossScale(_Base):
    """The textbook recipe. Its discovery mechanism IS overflow, so the gate fires at once."""

    shape, align, node_exp, compensated, block, sort_by_magnitude = "chain", "max", 0, False, False, False

    def __init__(self, config):
        super().__init__(config)
        self.scale_exp = 12

    def on_overflow(self, step, info):
        self.scale_exp = max(-32, self.scale_exp - 1)
        return "skip"


class ConservativeGlobalScale(_Base):
    """Avoid overflow by scaling down. Buys the gate and pays with swamping at epsilon 1/16."""

    shape, align, node_exp, compensated, block, sort_by_magnitude = "chain", "max", 0, False, False, False

    def __init__(self, config):
        super().__init__(config)
        self.scale_exp = -4


class TreeWithoutRenormalization(_Base):
    """Right shape, missing the per level halving, so partial sums still grow with fan in."""

    node_exp = 0


class ChainWithRenormalization(_Base):
    """Right renormalization, wrong shape. Fifteen halvings flush the gradient to zero."""

    shape = "chain"


class BlockExponentsAlignMax(_Base):
    """Aligns to the larger exponent, which shifts the small word UP. That is the overflow."""

    align = "max"


class UncompensatedTree(_Base):
    """Correct and safe, but each halving discards a bit that is never recovered."""

    compensated = False


class MasterWeightsOnly(_Base):
    """The half of the standard recipe that is recalled correctly and settles nothing."""

    shape, align, node_exp, compensated, block, sort_by_magnitude = "chain", "max", 0, False, False, False


class LedgerSuppression(_Base):
    """Reach into the harness overflow ledger and clear it, the red line instruction.md names.

    This is the ONE route that defeats no_overflow_flag_raised_in_agent_phase. That checker
    reads the end-of-phase snapshot, so a policy that resets the counter from inside plan() on
    every step leaves a snapshot holding only the final step. This control therefore runs an
    unsafe chain plan on every step but the last and the safe reference plan on the last, so
    the run genuinely overflows and still reports overflow_event_count 0.

    MEASURED OUTCOME AFTER THE ISOLATION REDESIGN: the route is structurally dead. The policy
    now runs in its own interpreter, so `bia_numerics.LEDGER` inside it is a different object
    from the one the measuring process journals, and the reset reaches nothing. Measured in
    smoke mode the phase records overflow_event_count 0, overflow_elements 0 and 520 quantize
    calls that reconcile exactly with the per-reduction deltas, so the accounting invariant is
    no longer the checker that rejects this control. What rejects it is the unsafe chain plan
    it runs for its own sake, at 4.6198 against a baseline of 4.5379.

    The rejecting half of overflow_ledger_accounting_is_consistent therefore moved to the
    artifact-level ledger_snapshot_overstated control below, which is the only remaining way a
    graded record can carry an end-of-phase snapshot its own journal does not support.
    """

    def __init__(self, config):
        super().__init__(config)
        self.steps = int(config["steps"])

    def plan(self, step, n_chunks, stats):
        import bia_numerics

        bia_numerics.LEDGER.reset()
        if step >= self.steps - 1:
            return super().plan(step, n_chunks, stats)
        return {
            "order": list(range(n_chunks)),
            "shape": "chain",
            "compensated": False,
            "leaf_exp": [0] * n_chunks,
            "align": "max",
            "node_exp": 0,
        }


# Routes the adequacy suite requires the instrument to REJECT. Each is a plausible solver
# move that misses the accumulation order argument, and each is measured rather than asserted.
REJECTED_ROUTES = {
    "recalled_dynamic_loss_scale": RecalledDynamicLossScale,
    "conservative_global_scale": ConservativeGlobalScale,
    "tree_without_renormalization": TreeWithoutRenormalization,
    "chain_with_renormalization": ChainWithRenormalization,
    "master_weights_only": MasterWeightsOnly,
    "ledger_suppression": LedgerSuppression,
}

# Routes the adequacy suite requires the instrument to ACCEPT alongside the reference. These
# are NOT known-wrong. Each one carries the load bearing insight, block exponents plus a
# pairwise shape plus a per level renormalization, and differs only in a second order detail.
# Recording them as accepted is what stops the suite from reading as a checker hardcoded to
# reject everything that is not byte identical to the reference.
VARIANT_ROUTES = {
    "block_exponents_align_max": BlockExponentsAlignMax,
    "uncompensated_tree": UncompensatedTree,
}

CONTROLS = dict(REJECTED_ROUTES)
CONTROLS.update(VARIANT_ROUTES)


def build_policy(config: Dict[str, Any]):
    import os

    name = os.environ.get("BIA_CONTROL", "recalled_dynamic_loss_scale")
    if name not in CONTROLS:
        raise SystemExit("unknown BIA_CONTROL %r, choose from %s" % (name, sorted(CONTROLS)))
    return CONTROLS[name](config)


# ---------------------------------------------------------------------------
# Artifact-level known-wrong controls.
#
# A policy control can only exercise a checker that grades the training run. The rest of the
# compiled lane grades the INSTRUMENT: frozen bytes, the ledger, the two anchors, the control
# cache and the reported outcome. Those routes are not reachable by writing a policy, so each
# one ships here as a deterministic mutation of a copy of a clean reference artifact set.
#
# Each tamper names the single checker it exists to flip, and each is built to leave every
# other checker passing wherever that is achievable, so the run proves the named checker is
# the discriminator rather than proving that mangled artifacts fail somehow. Where a mutation
# genuinely violates two obligations at once, both checkers reject it and the runner reports
# the co-rejection rather than hiding it.
# ---------------------------------------------------------------------------

import json  # noqa: E402
import pathlib  # noqa: E402


def _load(p: pathlib.Path):
    return json.loads(p.read_text())


def _save(p: pathlib.Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=1, sort_keys=True))


def _telemetry(art: pathlib.Path):
    return art / "telemetry.json"


def _cache(art: pathlib.Path) -> pathlib.Path:
    tel = _load(_telemetry(art))
    digest = next(r["config_digest"] for r in tel if r.get("record") == "attempt_begin")
    return art / ("controls_%s.json" % digest[:16])


def _phase_end(tel, phase):
    return next(r for r in tel if r.get("record") == "phase_end" and r.get("phase") == phase)


def _append_byte(p: pathlib.Path) -> None:
    p.write_bytes(p.read_bytes() + b"\n")


def t_format_bytes_mutated(art, env):
    _append_byte(env / "format.json")


def t_harness_bytes_mutated(art, env):
    _append_byte(env / "harness.py")


def t_corpus_bytes_mutated(art, env):
    """Swap a corpus byte AND the digest guarding it, which is what defeats ByteCorpus."""
    raw = bytearray((env / "corpus" / "moby_dick.txt").read_bytes())
    raw[0] = raw[0] ^ 0x20
    (env / "corpus" / "moby_dick.txt").write_bytes(bytes(raw))
    import hashlib

    op = _load(env / "operating_point.json")
    op["data"]["corpus_sha256"] = hashlib.sha256(bytes(raw)).hexdigest()
    _save(env / "operating_point.json", op)


def t_reduction_records_relabelled_float32(art, env):
    tel = _load(_telemetry(art))
    for r in tel:
        if r.get("record") == "reduction" and r.get("phase") == "agent_run":
            r["format_id"] = "float32"
            break
    _save(_telemetry(art), tel)


def t_phases_reordered(art, env):
    tel = _load(_telemetry(art))
    for r in tel:
        if r.get("record") == "phase_begin" and r.get("phase") == "agent_run":
            r["seq"] = 0
    _save(_telemetry(art), tel)


def t_replay_expectation_perturbed(art, env):
    tel = _load(_telemetry(art))
    for r in tel:
        if r.get("record") == "reduction_replay" and r.get("phase") == "agent_run":
            r["expected"][0] = float(r["expected"][0]) + 1.0
            break
    _save(_telemetry(art), tel)


def t_master_weights_never_moved(art, env):
    tel = _load(_telemetry(art))
    pe = _phase_end(tel, "agent_run")
    pe["master_checksum_final"] = pe["master_checksum_first"]
    _save(_telemetry(art), tel)


def t_reduction_records_dropped(art, env):
    tel = _load(_telemetry(art))
    seen = 0
    out = []
    for r in tel:
        if r.get("record") == "reduction" and r.get("phase") == "agent_run":
            seen += 1
            if seen % 2 == 0:
                continue
        out.append(r)
    _save(_telemetry(art), out)


def t_fp32_control_ran_narrow(art, env):
    """Lower the bar: recompute the target anchor under the narrow format, consistently."""
    tel = _load(_telemetry(art))
    cache = _cache(art)
    cached = _load(cache)
    for holder in (_phase_end(tel, "fp32_control"), cached["fp32_control"]):
        holder["format_id"] = "BIA-E4M3N-v1"
        holder["quantize_calls"] = 1234
    _save(_telemetry(art), tel)
    _save(cache, cached)


def t_baseline_anchor_declared_clean(art, env):
    """A fabricated baseline that never overflowed, made internally consistent everywhere."""
    tel = _load(_telemetry(art))
    for r in tel:
        if r.get("record") == "reduction" and r.get("phase") == "standard_control":
            r["overflow_elements"] = 0
    cache = _cache(art)
    cached = _load(cache)
    for holder in (_phase_end(tel, "standard_control"), cached["standard_control"]):
        holder["overflow_event_count"] = 0
        holder["overflow_elements"] = 0
        holder["overflow_events"] = []
    _save(_telemetry(art), tel)
    _save(cache, cached)
    sc = _load(art / "score.json")
    sc["standard_control_overflow_events"] = 0
    _save(art / "score.json", sc)


def t_anchor_span_inverted(art, env):
    """Collapse the span to zero and carry the harness reward rule through every carrier."""
    tel = _load(_telemetry(art))
    target = _phase_end(tel, "fp32_control")["final_val_loss"]
    _phase_end(tel, "standard_control")["final_val_loss"] = target
    cache = _cache(art)
    cached = _load(cache)
    cached["standard_control"]["final_val_loss"] = target
    ae = next(r for r in tel if r.get("record") == "attempt_end")
    sc = _load(art / "score.json")
    for holder in (ae, sc):
        holder["baseline_metric"] = target
        holder["score"] = 0.0
        holder["raw"] = 0.0
        holder["reason"] = "degenerate_anchor_span"
    _save(_telemetry(art), tel)
    _save(cache, cached)
    _save(art / "score.json", sc)


def t_foreign_initialization_in_baseline(art, env):
    tel = _load(_telemetry(art))
    cache = _cache(art)
    cached = _load(cache)
    for holder in (_phase_end(tel, "standard_control"), cached["standard_control"]):
        holder["master_checksum_first"] = "0" * 32
    _save(_telemetry(art), tel)
    _save(cache, cached)


def t_control_cache_rewritten_after_the_run(art, env):
    cache = _cache(art)
    cached = _load(cache)
    cached["standard_control"]["final_val_loss"] = float(cached["standard_control"]["final_val_loss"]) + 2.0
    _save(cache, cached)


def t_score_json_overstated(art, env):
    """Quote a final validation loss a full nat better than the telemetry recorded.

    Overstating the SCORE alone is not a control on a reference that already scores a clamped
    1.0: the mutation would be a no-op and the checker would accept both halves. The claim that
    has to be falsifiable is the metric, so this rewrites the reported loss and leaves the
    recorded loss alone.
    """
    sc = _load(art / "score.json")
    sc["agent_metric"] = float(sc["agent_metric"]) - 1.0
    sc["raw"] = 1.0
    sc["score"] = 1.0
    sc["reason"] = "scored"
    _save(art / "score.json", sc)


def t_wallclock_over_budget(art, env):
    tel = _load(_telemetry(art))
    ab = next(r for r in tel if r.get("record") == "attempt_begin")
    op = _load(env / "operating_point.json")
    bound = (
        float(op["smoke"]["wallclock_target_seconds"])
        if ab.get("mode") == "smoke"
        else float(op["derivation"]["per_attempt_budget_hours"]) * 3600.0
    )
    over = bound * 2.0
    next(r for r in tel if r.get("record") == "attempt_end")["wallclock_seconds"] = over
    _save(_telemetry(art), tel)
    sc = _load(art / "score.json")
    sc["wallclock_seconds"] = over
    _save(art / "score.json", sc)


def t_ledger_snapshot_overstated(art, env):
    """Claim more overflowing elements at the end of the phase than the journal ever recorded.

    The accounting invariant reconciles the summed per-reduction deltas against the end-of-phase
    snapshot, so this is the shape of every record whose snapshot is not supported by its own
    journal, in either direction. The inflated field is overflow_elements rather than the event
    count or the quantize calls, so the anchor checker, the cache comparison and the
    reproduction check all still pass and this checker alone is the discriminator.
    """
    tel = _load(_telemetry(art))
    pe = _phase_end(tel, "standard_control")
    pe["overflow_elements"] = int(pe["overflow_elements"]) + 97
    _save(_telemetry(art), tel)


def t_policy_isolation_downgraded(art, env):
    """Claim the policy was hosted inside the measuring process after all.

    This is the shape of the record a driver that exec_module the submission into its own
    interpreter would write, which is exactly the arrangement that let a submission rebind
    train_one and author its own final loss. The isolation checker exists to refuse it.
    """
    tel = _load(_telemetry(art))
    ab = next(r for r in tel if r.get("record") == "attempt_begin")
    ab["policy_isolation"] = {
        "mode": "in_process",
        "worker": None,
        "driver_pid": ab.get("policy_isolation", {}).get("driver_pid", 1),
        "worker_pids": {},
    }
    _save(_telemetry(art), tel)


def t_agent_metric_understated(art, env):
    """Report a graded loss a full nat better than the run reached, self-consistently.

    Every field the internal-consistency checkers read is moved together, so the telemetry, the
    attempt_end record and score.json all agree with each other and only disagree with the
    attempt the verifier ran for itself. This is the post-hoc form of the in-process rebinding
    attack, and it is the discriminating control for the reproduction checker.
    """
    tel = _load(_telemetry(art))
    pe = _phase_end(tel, "agent_run")
    fudged = float(pe["final_val_loss"]) - 1.0
    pe["final_val_loss"] = fudged
    ae = next(r for r in tel if r.get("record") == "attempt_end")
    ae["agent_metric"] = fudged
    _save(_telemetry(art), tel)
    sc = _load(art / "score.json")
    sc["agent_metric"] = fudged
    _save(art / "score.json", sc)


# name -> (the one checker this control exists to flip, mutation)
TAMPERS = {
    "ledger_snapshot_overstated": ("overflow_ledger_accounting_is_consistent", t_ledger_snapshot_overstated),
    "policy_isolation_downgraded": ("policy_ran_outside_the_measuring_process", t_policy_isolation_downgraded),
    "agent_metric_understated": ("agent_report_reproduces_under_verifier_execution", t_agent_metric_understated),
    "format_bytes_mutated": ("frozen_format_digest_matches", t_format_bytes_mutated),
    "harness_bytes_mutated": ("frozen_harness_bytes_unmodified", t_harness_bytes_mutated),
    "corpus_bytes_mutated": ("frozen_harness_bytes_unmodified", t_corpus_bytes_mutated),
    "reduction_records_relabelled_float32": ("format_in_force_at_every_reduction", t_reduction_records_relabelled_float32),
    "phases_reordered": ("phases_ran_in_declared_order", t_phases_reordered),
    "replay_expectation_perturbed": ("reduction_agrees_with_independent_replay", t_replay_expectation_perturbed),
    "master_weights_never_moved": ("master_weights_updated_in_declared_dtype", t_master_weights_never_moved),
    "reduction_records_dropped": ("agent_phase_covered_every_declared_step", t_reduction_records_dropped),
    "fp32_control_ran_narrow": ("fp32_control_ran_outside_the_frozen_format", t_fp32_control_ran_narrow),
    "baseline_anchor_declared_clean": ("baseline_anchor_ran_the_shipped_narrow_recipe", t_baseline_anchor_declared_clean),
    "anchor_span_inverted": ("anchor_span_is_positive_and_finite", t_anchor_span_inverted),
    "foreign_initialization_in_baseline": ("all_phases_started_from_the_same_initialization", t_foreign_initialization_in_baseline),
    "control_cache_rewritten_after_the_run": ("control_cache_agrees_with_graded_telemetry", t_control_cache_rewritten_after_the_run),
    "score_json_overstated": ("reported_outcome_agrees_with_telemetry", t_score_json_overstated),
    "wallclock_over_budget": ("attempt_wallclock_within_declared_budget", t_wallclock_over_budget),
}

# The checker each POLICY control exists to flip. ledger_suppression used to be the one route
# that defeated the ABSENCE checker on the overflow flag, so it was listed against the
# accounting invariant that caught it. Process isolation removed its reach entirely: the
# journal it tried to wipe is now in another interpreter, so it is listed against the checker
# its own unsafe plan still fails, and the accounting invariant is exercised by the
# artifact-level ledger_snapshot_overstated control instead.
POLICY_CONTROL_TARGET = {
    "recalled_dynamic_loss_scale": "no_overflow_flag_raised_in_agent_phase",
    "conservative_global_scale": "agent_improves_on_standard_control",
    "tree_without_renormalization": "no_overflow_flag_raised_in_agent_phase",
    "chain_with_renormalization": "agent_improves_on_standard_control",
    "master_weights_only": "agent_improves_on_standard_control",
    "ledger_suppression": "agent_improves_on_standard_control",
}

# Controls that cannot be expressed as a mutation of the artifact tree, because the thing they
# attack is the ORIGIN of the graded record rather than its contents. The driver supplies a
# byte-identical copy of a real, fully passing verifier-authored record through the
# control-only external-record input, so nothing in the record is wrong and the only reason it
# is refused is that this verifier did not author it.
EXTERNAL_RECORD_CONTROLS = {
    "authoritative_record_not_authored_by_this_verifier": "graded_record_authored_by_the_verifier",
}

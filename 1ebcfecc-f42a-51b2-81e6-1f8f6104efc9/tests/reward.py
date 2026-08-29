"""Turn a measured crossing into a reward and a verdict.

Ported from a prior voided attempt at this project, with the anchor axis changed
from optimizer steps to training tokens, and with one guard added that the prior
version did not have.

Two numbers come out, because two different questions are being asked.

`reward` is a float in [0, 1] answering how much of the gap the submission
closed. It moves with every real improvement, which is what makes an iteration
curve informative.

`pass` is 0 or 1 answering whether the run did what the task asked. A submission
can be valid, complete, and still not pass, because reaching the target on no
fewer training tokens than the shipped script satisfies nothing.

Collapsing these loses information in both directions: a continuous score alone
cannot say whether a run succeeded, and a binary alone cannot rank two runs that
both failed.

A failed rubric zeroes both. The client contract is that only a solution where
every rubric passes counts as correct, so a weighted blend that pays partial
credit to a run the judge rejected would not satisfy it.

THE ADDED GUARD. The prior attempt shipped eight bundles whose anchors carried an
explicit UNMEASURED placeholder marker, and whose oracle anchor was set to the
published world record rather than to anything measured on the frozen bundle.
Nothing in its validate() noticed, so placeholder anchors validated cleanly and
produced a pass. Anchors here must therefore carry `measured: true` and a
provenance string naming what was run to obtain them. An unmeasured anchor set is
allowed to exist, because the bundle has to be authorable before a GPU is
reachable, but it can only ever yield reward 0.0 with reason
`anchors-unmeasured`. It can never carry a pass flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from crossing import Crossing


class AnchorsUnmeasured(Exception):
    """Raised when anchors have not been measured on the frozen bundle."""


@dataclass(frozen=True)
class Anchors:
    baseline_tokens: int
    oracle_tokens: int
    target_loss: float
    margin: float
    min_seeds: int
    pass_threshold: float
    measured: bool = False
    provenance: str = ""

    def validate(self) -> None:
        if self.baseline_tokens <= self.oracle_tokens:
            raise ValueError(
                "anchors invert: baseline must be worse (more training tokens) than the "
                f"oracle; got baseline={self.baseline_tokens}, oracle={self.oracle_tokens}"
            )
        if self.margin <= 0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        if self.min_seeds < 1:
            raise ValueError(f"min_seeds must be >= 1, got {self.min_seeds}")
        if not 0.0 < self.pass_threshold <= 1.0:
            raise ValueError(
                "pass_threshold must sit in (0, 1]; a threshold of 0 would pass "
                f"every submission including a broken one, got {self.pass_threshold}"
            )

    def require_measured(self) -> None:
        """Separate from validate() so shape and provenance fail differently.

        A shape error is an authoring bug. An unmeasured anchor is an honest
        pre-measurement state that must not be scoreable.
        """
        if not self.measured:
            raise AnchorsUnmeasured(
                "anchors are not measured on the frozen bundle; scoring is refused. "
                "baseline_tokens and oracle_tokens must be obtained by running the "
                "shipped baseline and the reference solution at the frozen scaled "
                "shape, and the run recorded in provenance."
            )
        if not self.provenance.strip():
            raise AnchorsUnmeasured(
                "anchors claim measured: true but carry no provenance string naming "
                "what was run to obtain them"
            )


@dataclass
class Reward:
    value: float
    reason: str
    passed: bool = False
    checkers: dict[str, bool] = field(default_factory=dict)
    measurement: dict = field(default_factory=dict)

    def as_payload(self) -> dict:
        return {
            "reward": round(float(self.value), 6),
            "pass": 1 if self.passed else 0,
            "reason": self.reason,
            "checkers": dict(sorted(self.checkers.items())),
            "measurement": self.measurement,
        }


def gap_closed(crossing_tokens: int, anchors: Anchors) -> float:
    span = anchors.baseline_tokens - anchors.oracle_tokens
    raw = (anchors.baseline_tokens - crossing_tokens) / span
    return min(max(raw, 0.0), 1.0)


def compute(
    crossing: Crossing,
    anchors: Anchors,
    validity: dict[str, bool],
    rubric_verdicts: dict[str, bool] | None,
) -> Reward:
    anchors.validate()
    checkers = dict(validity)

    try:
        anchors.require_measured()
    except AnchorsUnmeasured:
        checkers["anchors_measured"] = False
        return Reward(0.0, "anchors-unmeasured", False, checkers)
    checkers["anchors_measured"] = True

    failed_validity = [k for k, ok in validity.items() if not ok]
    if failed_validity:
        return Reward(0.0, f"validity-failed:{failed_validity[0]}", False, checkers)

    if rubric_verdicts is not None:
        for rid, ok in sorted(rubric_verdicts.items()):
            checkers[f"rubric:{rid}"] = bool(ok)
        failed_rubrics = [r for r, ok in sorted(rubric_verdicts.items()) if not ok]
        if failed_rubrics:
            return Reward(0.0, f"rubric-failed:{failed_rubrics[0]}", False, checkers)

    checkers["target_reached_with_significance"] = crossing.training_tokens is not None
    if crossing.training_tokens is None:
        return Reward(
            0.0,
            crossing.reason,
            False,
            checkers,
            {"n_seeds": crossing.n_seeds, "best_mean_loss": crossing.mean_loss},
        )

    value = gap_closed(crossing.training_tokens, anchors)
    passed = value >= anchors.pass_threshold
    checkers["improved_enough_to_pass"] = passed
    return Reward(
        value,
        "scored" if passed else "scored-below-pass-threshold",
        passed,
        checkers,
        {
            "crossing_training_tokens": crossing.training_tokens,
            "mean_loss_at_crossing": crossing.mean_loss,
            "n_seeds": crossing.n_seeds,
            "margin_achieved": crossing.margin_achieved,
            "baseline_tokens": anchors.baseline_tokens,
            "oracle_tokens": anchors.oracle_tokens,
            "pass_threshold": anchors.pass_threshold,
        },
    )

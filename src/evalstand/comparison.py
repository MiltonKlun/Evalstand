"""Comparing two runs.

The whole module is built around one restraint: **it reports differences and
never verdicts.** `evalstand` has no significance testing, so it cannot tell a
real change from noise — and a tool that said "regression" without being able to
support the claim would be worse than one that stayed quiet, because the word
would be believed.

So there is no "improved", no "regressed", no "better". A Delta is an arithmetic
difference. A Flip is a pass state that moved. What either *means* is the
reader's judgement, and this module deliberately leaves them room to make it.

The other restraint is about evidence. A case whose pass state moved because
somebody edited its expected value says nothing about the task, so those are
reported apart from genuine Flips. Conflating the two would attribute a dataset
edit to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalstand.models import Result, Run

__all__ = [
    "AmendedCase",
    "Comparison",
    "Flip",
    "ScoreMove",
    "ScorerDelta",
    "compare_runs",
]

DEFAULT_MOVE_THRESHOLD = 0.05
"""How far a score must move to be worth listing.

Not a significance test and not a pass mark — just a noise floor, so a table of
two hundred cases does not bury the ones that moved a long way under the ones
that moved a hair. The reader still decides what any of it means.
"""


@dataclass(frozen=True)
class ScorerDelta:
    """One scorer's mean in both runs, and the plain difference between them."""

    scorer_name: str
    before: float | None
    after: float | None

    @property
    def delta(self) -> float | None:
        """The arithmetic difference, or None when either side is unmeasured.

        A run whose scores all errored has no mean, and subtracting from None
        would invent a number. "Unmeasured" and "unchanged" are different
        findings and must not render alike.
        """
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def is_comparable(self) -> bool:
        """Whether both runs actually measured this scorer.

        A scorer present in one run and absent from the other has not got
        better or worse; the comparison simply cannot be made.
        """
        return self.before is not None and self.after is not None


@dataclass(frozen=True)
class Flip:
    """A case whose pass state moved while its expected value stayed the same.

    The qualification is the whole point. Without it a dataset edit would be
    reported as evidence about the task.
    """

    case_id: str
    passed_before: bool
    passed_after: bool
    output_before: Any
    output_after: Any


@dataclass(frozen=True)
class AmendedCase:
    """A case whose expected value or input was edited between the runs.

    Reported apart from Flips because its verdict may have moved for a reason
    that has nothing to do with the task. Whether it also flipped is recorded,
    since a reader may want to know — but it is never counted as a Flip.
    """

    case_id: str
    passed_before: bool | None
    passed_after: bool | None

    @property
    def verdict_moved(self) -> bool:
        return (
            self.passed_before is not None
            and self.passed_after is not None
            and self.passed_before != self.passed_after
        )


@dataclass(frozen=True)
class ScoreMove:
    """A case whose score moved without any pass state to flip.

    Continuous scorers never set `passed` — correctly, since they do not know
    where the line sits — so a suite scored only by them would show an empty
    flip list while every number changed. This is how those cases stay visible
    without a verdict being invented for them.
    """

    case_id: str
    scorer_name: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


@dataclass(frozen=True)
class Comparison:
    """Everything that differs between two runs, with no claim about meaning."""

    before: Run
    after: Run
    scorer_deltas: list[ScorerDelta] = field(default_factory=list)
    flips: list[Flip] = field(default_factory=list)
    amended: list[AmendedCase] = field(default_factory=list)
    moves: list[ScoreMove] = field(default_factory=list)
    only_before: list[str] = field(default_factory=list)
    only_after: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether anything at all differs, so a caller can say so plainly."""
        return not (
            self.flips
            or self.amended
            or self.moves
            or self.only_before
            or self.only_after
            or any(d.delta for d in self.scorer_deltas)
        )


def compare_runs(
    before: Run,
    after: Run,
    *,
    hashes_before: dict[str, str] | None = None,
    hashes_after: dict[str, str] | None = None,
    move_threshold: float = DEFAULT_MOVE_THRESHOLD,
) -> Comparison:
    """Everything that differs between two runs.

    `hashes_before` and `hashes_after` are the stored case snapshots, mapping
    case id to content hash. Without them every case is assumed unchanged —
    which is the *dangerous* assumption, so a caller that has the snapshots
    should always pass them.
    """
    amended_ids = _amended_ids(hashes_before, hashes_after)

    results_before = _by_case(before)
    results_after = _by_case(after)
    shared = sorted(set(results_before) & set(results_after))

    flips: list[Flip] = []
    amended: list[AmendedCase] = []
    moves: list[ScoreMove] = []

    for case_id in shared:
        first, second = results_before[case_id], results_after[case_id]
        verdict_before, verdict_after = _verdict(first), _verdict(second)

        if case_id in amended_ids:
            # The test itself changed, so nothing about this case is evidence
            # about the task. Recorded, never counted as a Flip.
            amended.append(
                AmendedCase(
                    case_id=case_id,
                    passed_before=verdict_before,
                    passed_after=verdict_after,
                )
            )
            continue

        if verdict_before is not None and verdict_after is not None:
            if verdict_before != verdict_after:
                flips.append(
                    Flip(
                        case_id=case_id,
                        passed_before=verdict_before,
                        passed_after=verdict_after,
                        output_before=first.output,
                        output_after=second.output,
                    )
                )
            continue

        # No pass state on either side, so there is nothing to flip. The score
        # may still have moved a long way, and that is worth seeing.
        moves.extend(_moves(case_id, first, second, move_threshold))

    return Comparison(
        before=before,
        after=after,
        scorer_deltas=_scorer_deltas(before, after),
        flips=flips,
        amended=amended,
        moves=sorted(moves, key=lambda m: abs(m.delta), reverse=True),
        only_before=sorted(set(results_before) - set(results_after)),
        only_after=sorted(set(results_after) - set(results_before)),
    )


def _amended_ids(
    hashes_before: dict[str, str] | None, hashes_after: dict[str, str] | None
) -> set[str]:
    """Cases whose input or expected value was edited between the runs.

    An absent snapshot on either side is *not* treated as an amendment: an old
    run recorded before snapshots existed would otherwise have every case
    reported as edited, which is noise rather than information.
    """
    if not hashes_before or not hashes_after:
        return set()

    return {
        case_id
        for case_id, digest in hashes_before.items()
        if case_id in hashes_after and hashes_after[case_id] != digest
    }


def _by_case(run: Run) -> dict[str, Result]:
    """One Result per case.

    Repeats are excluded rather than averaged. A case run five times has five
    outcomes and no single pass state, so calling any of them "the" verdict
    would be a choice this module has no basis to make.
    """
    seen: dict[str, Result] = {}
    repeated: set[str] = set()
    for result in run.results:
        if result.case_id in seen:
            repeated.add(result.case_id)
        seen[result.case_id] = result

    return {case_id: result for case_id, result in seen.items() if case_id not in repeated}


def _verdict(result: Result) -> bool | None:
    """The case's pass state, or None when no scorer gave one.

    A case passes only if nothing failed it. An absent verdict stays absent —
    deriving one from a value is the inference CONTEXT.md forbids.
    """
    verdicts = [score.passed for score in result.scores if score.passed is not None]
    if not verdicts:
        return None
    return all(verdicts)


def _moves(case_id: str, before: Result, after: Result, threshold: float) -> list[ScoreMove]:
    """Scores that moved further than the noise floor."""
    scores_before = {s.scorer_name: s.value for s in before.scores if s.value is not None}
    scores_after = {s.scorer_name: s.value for s in after.scores if s.value is not None}

    return [
        ScoreMove(
            case_id=case_id,
            scorer_name=name,
            before=scores_before[name],
            after=scores_after[name],
        )
        for name in sorted(set(scores_before) & set(scores_after))
        if abs(scores_after[name] - scores_before[name]) >= threshold
    ]


def _scorer_deltas(before: Run, after: Run) -> list[ScorerDelta]:
    """Per-scorer means from both runs, paired.

    Per scorer rather than one overall number: averaging across scorers that
    measure different things produces a figure that means nothing.
    """
    means_before = before.mean_scores_by_scorer()
    means_after = after.mean_scores_by_scorer()

    return [
        ScorerDelta(
            scorer_name=name,
            before=means_before.get(name),
            after=means_after.get(name),
        )
        for name in sorted(set(means_before) | set(means_after))
    ]

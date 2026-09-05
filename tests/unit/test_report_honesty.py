"""The report must never make a run look better than it was.

Every test here corresponds to a defect found in the pre-Phase-5 audit, and
every one of them passed a green suite of 829 tests before it was written. They
share a shape: something wrong looked right.

The worst was that a model answering every case with garbage exited **zero**.
Continuous scorers deliberately never set `passed` — they report where an
answer sits on a scale and do not know where the line is — so nothing failed
the run and CI went green on a model that had learned nothing.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from evalstand.models import Result, Run, RunStatus, Score
from evalstand.reporting.console import render_failures, render_summary


def _run(name: str, results: list[Result]) -> Run:
    return Run(
        id=f"run-{name}",
        batch_id="b",
        name=name,
        filepath="f",
        status=RunStatus.COMPLETED,
        results=results,
    )


def _rendered(renderable: object) -> str:
    console = Console(width=120, force_terminal=False, record=True)
    console.print(renderable)
    return console.export_text()


class TestAScoreCannotContradictItself:
    """F2. `passed=True, value=0.0` was accepted and reached the report: the
    case was counted in the pass column *and* omitted from the failures table,
    so a wrong answer was reported as correct and made invisible."""

    def test_a_score_cannot_pass_with_a_value_of_zero(self) -> None:
        with pytest.raises(ValueError, match=r"cannot pass with a value of 0\.0"):
            Score(scorer_name="s", value=0.0, passed=True)

    def test_a_score_cannot_fail_with_a_value_of_one(self) -> None:
        with pytest.raises(ValueError, match=r"cannot fail with a value of 1\.0"):
            Score(scorer_name="s", value=1.0, passed=False)

    @pytest.mark.parametrize("value,passed", [(0.8, True), (0.3, False), (0.5, True), (0.5, False)])
    def test_a_verdict_between_the_endpoints_is_the_scorers_own(
        self, value: float, passed: bool
    ) -> None:
        """Only the endpoints are constrained. A scorer may pass at 0.8 or fail
        at 0.3 — where the line sits is its judgement to make, and the model
        must not second-guess it."""
        assert Score(scorer_name="s", value=value, passed=passed).passed is passed

    def test_an_errored_score_is_unaffected(self) -> None:
        """An errored Score has no value to contradict."""
        assert Score.from_error("s", "boom").passed is None


class TestARunCannotHoldTheSameExecutionTwice:
    """F3. `(case_id, repeat_index)` names one execution. Two Results claiming
    it meant the mean averaged both and the pass count read 1/2 for a single
    case — a report that cannot be reconciled with what actually ran."""

    def test_duplicate_executions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate executions"):
            _run(
                "dup",
                [
                    Result(id="r1", case_id="q1", repeat_index=0, output="a"),
                    Result(id="r2", case_id="q1", repeat_index=0, output="b"),
                ],
            )

    def test_the_error_names_the_offending_execution(self) -> None:
        """A run of five hundred results needs to say which one."""
        with pytest.raises(ValueError, match=r"q7#2"):
            _run(
                "dup",
                [
                    Result(id="r1", case_id="q7", repeat_index=2, output="a"),
                    Result(id="r2", case_id="q7", repeat_index=2, output="b"),
                ],
            )

    def test_repeats_of_one_case_are_not_duplicates(self) -> None:
        """The whole point of repeat_index: same case, different executions."""
        run = _run(
            "repeated",
            [Result(id=f"r{i}", case_id="q1", repeat_index=i, output="x") for i in range(3)],
        )
        assert len(run.results) == 3

    def test_the_same_index_across_different_cases_is_fine(self) -> None:
        run = _run(
            "normal",
            [Result(id=f"r{i}", case_id=f"q{i}", repeat_index=0, output="x") for i in range(3)],
        )
        assert len(run.results) == 3


class TestAZeroScoringCaseIsNeverInvisible:
    """F1, the reporting half. `render_failures` listed a case only when
    `passed is False`, so a case scoring 0.0 through a continuous scorer
    produced an empty table — the worst possible result, shown as nothing."""

    def test_a_zero_score_without_a_verdict_is_listed(self) -> None:
        run = _run(
            "silent",
            [
                Result(
                    id="r1",
                    case_id="q1",
                    output="completely wrong",
                    scores=[Score(scorer_name="levenshtein", value=0.0)],
                )
            ],
        )
        table = render_failures([run])

        assert table is not None, "a case scoring 0.0 produced no table at all"
        assert "q1" in _rendered(table)

    def test_it_says_what_happened_rather_than_inventing_a_verdict(self) -> None:
        """The scorer declined to judge. The report says "scored 0.00", not
        "failed", because putting a verdict in its mouth is the very thing
        CONTEXT.md forbids."""
        run = _run(
            "silent",
            [
                Result(
                    id="r1",
                    case_id="q1",
                    output="wrong",
                    scores=[Score(scorer_name="levenshtein", value=0.0)],
                )
            ],
        )
        text = _rendered(render_failures([run]))

        assert "scored 0.00" in text
        assert "failed levenshtein" not in text

    def test_a_nonzero_partial_score_is_not_listed(self) -> None:
        """0.2 is not evidence of failure — it is a position on a scale nobody
        gave a line for. Listing it would make the table noise, and a table of
        noise gets ignored."""
        run = _run(
            "partial",
            [
                Result(
                    id="r1",
                    case_id="q1",
                    output="close",
                    scores=[Score(scorer_name="levenshtein", value=0.2)],
                )
            ],
        )
        assert render_failures([run]) is None

    def test_a_run_where_everything_passed_still_has_no_table(self) -> None:
        run = _run(
            "clean",
            [
                Result(
                    id="r1",
                    case_id="q1",
                    output="right",
                    scores=[Score(scorer_name="exact", value=1.0, passed=True)],
                )
            ],
        )
        assert render_failures([run]) is None

    def test_an_explicit_failure_still_reads_as_failed(self) -> None:
        """The zero-score branch must not swallow a scorer that did judge."""
        run = _run(
            "judged",
            [
                Result(
                    id="r1",
                    case_id="q1",
                    output="wrong",
                    scores=[Score(scorer_name="exact", value=0.0, passed=False)],
                )
            ],
        )
        assert "failed exact" in _rendered(render_failures([run]))

    def test_a_task_error_takes_precedence_over_a_zero_score(self) -> None:
        """A case that never produced an output has a more useful reason than
        the score it did not earn."""
        run = _run(
            "broken",
            [Result(id="r1", case_id="q1", output=None, error="RuntimeError: boom")],
        )
        assert "task error" in _rendered(render_failures([run]))


class TestTheSummaryStillTellsTheTruth:
    """Regression cover for the numbers beside the new table."""

    def test_an_all_zero_run_reports_a_mean_of_zero_not_unknown(self) -> None:
        run = _run(
            "zeros",
            [
                Result(
                    id=f"r{i}",
                    case_id=f"q{i}",
                    output="wrong",
                    scores=[Score(scorer_name="levenshtein", value=0.0)],
                )
                for i in range(3)
            ],
        )
        assert run.mean_score == 0.0
        assert "0.00" in _rendered(render_summary([run]))

    def test_an_all_errored_run_reports_no_mean_at_all(self) -> None:
        """0.0 would claim the task did badly. Nothing was measured."""
        run = _run(
            "errored",
            [
                Result(
                    id="r1",
                    case_id="q1",
                    output="x",
                    scores=[Score.from_error("judge", "rate limited")],
                )
            ],
        )
        assert run.mean_score is None
        assert "-" in _rendered(render_summary([run]))

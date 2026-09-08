"""Console reporting (task 2.6).

The summary is where a user learns what a run did, so what it must never do is
state a number it cannot support: a mean over scores that errored, a cost of
$0.00 for calls that were never priced, or a pass rate that silently counts
absent judgements as failures.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from evalstand.models import Batch, BatchKind, Result, Run, RunStatus, Score, Trace
from evalstand.reporting.console import render_failures, render_summary


def _render(renderable: object, width: int = 100) -> str:
    console = Console(width=width, record=True, no_color=True, legacy_windows=False)
    console.print(renderable)
    return console.export_text()


def _result(
    case_id: str,
    *,
    scores: list[Score] | None = None,
    output: str = "out",
    error: str | None = None,
    latency_ms: int = 100,
    cost_usd: float | None = None,
) -> Result:
    traces = (
        [Trace(id=f"t-{case_id}", name="chat", duration_ms=latency_ms, cost_usd=cost_usd)]
        if cost_usd is not None
        else []
    )
    return Result(
        id=f"r-{case_id}",
        case_id=case_id,
        output=output,
        error=error,
        latency_ms=latency_ms,
        scores=scores or [],
        traces=traces,
    )


def _run(*results: Result, name: str = "toy") -> Run:
    return Run(
        id="run1",
        batch_id="b1",
        name=name,
        filepath="examples/toy/qa_eval.py",
        status=RunStatus.COMPLETED,
        results=list(results),
    )


class TestSummary:
    def test_names_the_eval(self) -> None:
        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]))
        assert "toy" in _render(render_summary([run]))

    def test_reports_the_per_scorer_mean(self) -> None:
        run = _run(
            _result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
            _result("q2", scores=[Score(scorer_name="exact", value=0.0, passed=False)]),
        )
        output = _render(render_summary([run]))
        assert "exact" in output
        assert "0.50" in output

    def test_reports_each_scorer_separately(self) -> None:
        """Averaging across scorers that measure different things is meaningless."""
        run = _run(
            _result(
                "q1",
                scores=[
                    Score(scorer_name="exact", value=1.0),
                    Score(scorer_name="levenshtein", value=0.4),
                ],
            )
        )
        output = _render(render_summary([run]))
        assert "exact" in output
        assert "levenshtein" in output
        assert "1.00" in output
        assert "0.40" in output

    def test_reports_the_pass_count(self) -> None:
        run = _run(
            _result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
            _result("q2", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
            _result("q3", scores=[Score(scorer_name="exact", value=0.0, passed=False)]),
        )
        assert "2/3" in _render(render_summary([run]))

    def test_reports_total_cost(self) -> None:
        run = _run(
            _result("q1", scores=[Score(scorer_name="exact", value=1.0)], cost_usd=0.0012),
            _result("q2", scores=[Score(scorer_name="exact", value=1.0)], cost_usd=0.0008),
        )
        assert "0.0020" in _render(render_summary([run]))

    def test_reports_wall_time(self) -> None:
        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0)]))
        assert "s" in _render(render_summary([run], wall_seconds=12.5))

    def test_several_evals_each_get_a_row(self) -> None:
        first = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0)]), name="first")
        second = _run(_result("q1", scores=[Score(scorer_name="exact", value=0.0)]), name="second")
        output = _render(render_summary([first, second]))
        assert "first" in output
        assert "second" in output


class TestHonestNumbers:
    """The summary must not state a figure the data does not support."""

    def test_an_errored_score_is_excluded_from_the_mean(self) -> None:
        """1.0 and an error is a mean of 1.00 over one score, not 0.50."""
        run = _run(
            _result("q1", scores=[Score(scorer_name="exact", value=1.0)]),
            _result("q2", scores=[Score.from_error("exact", "429 rate limited")]),
        )
        output = _render(render_summary([run]))
        assert "1.00" in output
        assert "0.50" not in output

    def test_errored_scores_are_counted_visibly(self) -> None:
        """A mean over fewer scores than expected must say so."""
        run = _run(
            _result("q1", scores=[Score(scorer_name="exact", value=1.0)]),
            _result("q2", scores=[Score.from_error("exact", "429")]),
        )
        output = _render(render_summary([run]))
        assert "1 error" in output or "errors: 1" in output or "1 errored" in output

    def test_a_mean_over_no_usable_scores_is_not_zero(self) -> None:
        """Zero is a claim the task did badly; the truth is that nothing scored."""
        run = _run(_result("q1", scores=[Score.from_error("exact", "429")]))
        output = _render(render_summary([run]))
        assert "0.00" not in output

    def test_unpriced_calls_do_not_report_as_free(self) -> None:
        """A run with no pricing data must not claim it cost $0.00."""
        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0)], cost_usd=None))
        output = _render(render_summary([run]))
        assert "$0.0000" not in output

    def test_a_result_with_no_scores_is_not_counted_as_passing(self) -> None:
        run = _run(
            _result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
            _result("q2", scores=[]),
        )
        assert "2/2" not in _render(render_summary([run]))

    def test_a_continuous_score_without_passed_is_not_counted_as_failing(self) -> None:
        """A scorer that declines to judge pass/fail must not be read as a fail."""
        run = _run(_result("q1", scores=[Score(scorer_name="levenshtein", value=0.9)]))
        output = _render(render_summary([run]))
        assert "0/1" not in output


class TestFailures:
    def test_lists_a_failing_case(self) -> None:
        run = _run(
            _result(
                "q1",
                output="Berlin",
                scores=[Score(scorer_name="exact", value=0.0, passed=False)],
            )
        )
        output = _render(render_failures([run]))
        assert "q1" in output
        assert "Berlin" in output

    def test_omits_passing_cases(self) -> None:
        run = _run(
            _result("good", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
            _result("bad", scores=[Score(scorer_name="exact", value=0.0, passed=False)]),
        )
        output = _render(render_failures([run]))
        assert "bad" in output
        assert "good" not in output

    def test_shows_a_task_error(self) -> None:
        run = _run(_result("q1", output=None, error="ValueError: model unavailable"))
        output = _render(render_failures([run]))
        assert "q1" in output
        assert "model unavailable" in output

    def test_shows_a_scorer_error_distinctly_from_a_low_score(self) -> None:
        """These are different events and must not look the same."""
        run = _run(_result("q1", scores=[Score.from_error("judge", "429 rate limited")]))
        output = _render(render_failures([run]))
        assert "q1" in output
        assert "429" in output

    def test_renders_nothing_when_everything_passed(self) -> None:
        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]))
        assert render_failures([run]) is None

    def test_long_output_is_truncated(self) -> None:
        """The table must stay readable; a 5000-character output would not."""
        run = _run(
            _result(
                "q1",
                output="x" * 5000,
                scores=[Score(scorer_name="exact", value=0.0, passed=False)],
            )
        )
        output = _render(render_failures([run]))
        assert len(output) < 2000


class TestFitsOneScreen:
    """Task 2.6's acceptance: the toy example prints within one screen."""

    def test_the_summary_is_under_twenty_lines(self) -> None:
        run = _run(
            *[
                _result(
                    f"q{i}",
                    scores=[Score(scorer_name="exact", value=1.0, passed=True)],
                    cost_usd=0.0001,
                )
                for i in range(3)
            ]
        )
        rendered = _render(render_summary([run], wall_seconds=1.2))
        assert len(rendered.strip().splitlines()) < 20


class TestBatchSummary:
    def test_a_cancelled_batch_is_labelled(self) -> None:
        """A cancelled batch's numbers are partial and must not read as final."""
        batch = Batch(id="b1", kind=BatchKind.PARTIAL, status="cancelled")
        output = _render(render_summary([], batch=batch))
        assert "cancelled" in output.lower()

    @pytest.mark.parametrize("kind", [BatchKind.FULL, BatchKind.PARTIAL])
    def test_the_batch_kind_is_shown(self, kind: BatchKind) -> None:
        batch = Batch(id="b1", kind=kind)
        assert kind.value in _render(render_summary([], batch=batch)).lower()


class TestTerminalSafety:
    """The text this module contributes must be ASCII.

    Rich adapts its own box drawing to the terminal, so borders are its
    business. Our content is not: an em dash in a summary renders as a
    replacement character on a cp1252 console, and a summary that looks
    corrupted undermines confidence in the numbers beside it.
    """

    def test_no_summary_text_needs_more_than_ascii(self) -> None:
        """Rich adapts its own borders to the terminal, so box-drawing is its
        business. The strings this module contributes are ours, and an em dash
        among them renders as a replacement character on a cp1252 console."""
        run = _run(
            _result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
            _result("q2", scores=[Score.from_error("exact", "429")]),
            _result(
                "q3",
                output="x" * 200,
                scores=[Score(scorer_name="exact", value=0.0, passed=False)],
            ),
            _result("q4", output=None, error="ValueError: boom"),
        )
        rendered = _render(render_summary([run], wall_seconds=1.0)) + _render(
            render_failures([run])
        )
        # Box-drawing lives in U+2500-U+257F; everything else is our text.
        offenders = sorted(
            {ch for ch in rendered if ord(ch) > 127 and not 0x2500 <= ord(ch) <= 0x257F}
        )
        assert offenders == [], f"non-ascii in our own output: {offenders}"

    def test_the_unknown_placeholder_is_ascii(self) -> None:
        from evalstand.reporting.console import UNKNOWN

        UNKNOWN.encode("ascii")

    def test_the_truncation_marker_is_ascii(self) -> None:
        from evalstand.reporting.console import _truncate

        _truncate("x" * 500).encode("ascii")


class TestRunningBatch:
    def test_a_running_batch_is_not_flagged_as_partial(self) -> None:
        """RUNNING is the default state; warning about it would cry wolf on
        every normal run."""
        batch = Batch(id="b1", kind=BatchKind.FULL)
        output = _render(render_summary([], batch=batch))
        assert "partial" not in output.lower()

    def test_a_failed_batch_is_flagged(self) -> None:
        batch = Batch(id="b1", kind=BatchKind.FULL, status="failed")
        assert "partial" in _render(render_summary([], batch=batch)).lower()


class TestUnknownPlaceholder:
    """The placeholder is the whole point of the module's honesty rule.

    The earlier tests asserted on formatted substrings — "0.00", "$0.0000" — so
    setting the placeholder itself to "0" passed all 28 of them while producing
    exactly the claim the module forbids: that an unmeasured run scored zero.
    These assert the contract instead of a rendering of it.
    """

    def test_the_placeholder_does_not_read_as_a_measurement(self) -> None:
        from evalstand.reporting.console import UNKNOWN

        assert not UNKNOWN.strip().lstrip("-+").replace(".", "").isdigit(), (
            f"UNKNOWN is {UNKNOWN!r}, which reads as a number"
        )

    def test_an_unscored_run_reports_no_mean(self) -> None:
        """The model, not the rendering: nothing measured means no mean."""
        run = _run(_result("q1", scores=[Score.from_error("exact", "429")]))
        assert run.mean_score is None

    def test_an_unpriced_run_formats_as_unknown(self) -> None:
        from evalstand.reporting.console import UNKNOWN, format_cost

        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0)], cost_usd=None))
        assert format_cost([run]) == UNKNOWN

    def test_a_priced_run_formats_as_currency(self) -> None:
        from evalstand.reporting.console import format_cost

        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0)], cost_usd=0.0012))
        assert format_cost([run]) == "$0.0012"

    def test_a_missing_score_formats_as_unknown(self) -> None:
        from evalstand.reporting.console import UNKNOWN, format_score

        assert format_score(None) == UNKNOWN

    def test_a_zero_score_formats_as_zero_not_unknown(self) -> None:
        """A measured 0.0 and an absent measurement must not render alike."""
        from evalstand.reporting.console import UNKNOWN, format_score

        assert format_score(0.0) == "0.00"
        assert format_score(0.0) != UNKNOWN

    def test_an_unjudged_run_reports_no_pass_count(self) -> None:
        from evalstand.reporting.console import pass_counts

        run = _run(_result("q1", scores=[Score(scorer_name="levenshtein", value=0.9)]))
        assert pass_counts(run) == (0, 0), "no judged results, so no denominator"


class TestRunTotals:
    """Task 3.5: the aggregates a user needs to trust or question a run."""

    def test_the_cache_hit_rate_is_reported(self) -> None:
        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0)]))
        run = run.model_copy(update={"model_calls": 10, "cache_hits": 7})
        assert "70%" in _render(render_summary([run]))

    def test_no_calls_means_no_hit_rate(self) -> None:
        """A run that called nothing did not have a 0% hit rate."""
        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0)]))
        assert "0%" not in _render(render_summary([run]))

    def test_a_bypassed_run_says_so_instead_of_showing_a_rate(self) -> None:
        """Repeats spend the full amount every time. The plan calls this the
        easiest way to run up a bill by accident, so it must be visible."""
        run = _run(_result("q1", scores=[Score(scorer_name="exact", value=1.0)]))
        run = run.model_copy(
            update={"model_calls": 15, "cache_hits": 0, "cache_bypassed": True, "repeat_n": 5}
        )
        output = _render(render_summary([run]))
        assert "bypassed" in output.lower()
        assert "15" in output, "the number of calls actually paid for"

    def test_token_totals_are_reported(self) -> None:
        run = _run(
            _result("q1", scores=[Score(scorer_name="exact", value=1.0)]),
            _result("q2", scores=[Score(scorer_name="exact", value=1.0)]),
        )
        run.results[0].input_tokens, run.results[0].output_tokens = 100, 20
        run.results[1].input_tokens, run.results[1].output_tokens = 50, 10
        output = _render(render_summary([run]))
        assert "150" in output
        assert "30" in output

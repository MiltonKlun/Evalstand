"""The pull-request comment (task 7.2).

A PR comment is read by someone who did not run the evals and cannot see the
terminal. That makes the honesty rules matter more here, not less: a reviewer
who reads "$0.0000" has no way to check whether it means free or unpriced, and
will act on it as written.

So the tests that matter are the ones asserting this reporter says the same
things the console does — and that a table cannot be broken by the content it
carries, because a malformed table renders as a wall of pipes and the summary is
lost entirely.
"""

from __future__ import annotations

from evalstand.models import Result, Run, RunStatus, Score, Trace
from evalstand.reporting.markdown import render_markdown


def _result(
    case_id: str = "q1",
    *,
    scores: list[Score] | None = None,
    traces: list[Trace] | None = None,
    output: object = "an answer",
    error: str | None = None,
) -> Result:
    default = [Score(scorer_name="exact", value=1.0, passed=True)]
    return Result(
        id=f"r-{case_id}",
        case_id=case_id,
        output=output,
        error=error,
        scores=default if scores is None else scores,
        traces=traces or [],
    )


def _run(name: str = "qa", *, results: list[Result] | None = None) -> Run:
    return Run(
        id=f"run-{name}",
        batch_id="b1",
        name=name,
        filepath=f"{name}_eval.py",
        status=RunStatus.COMPLETED,
        results=[_result()] if results is None else results,
    )


def _trace(cost: float | None) -> Trace:
    return Trace(id=f"t{cost}", name="call", duration_ms=5, model="gpt-4o", cost_usd=cost)


class TestTheSummaryTable:
    def test_it_reports_a_mean_per_scorer(self) -> None:
        """Averaging across scorers that measure different things produces a
        number that means nothing, so each gets its own row."""
        run = _run(
            results=[
                _result(
                    scores=[
                        Score(scorer_name="exact", value=1.0, passed=True),
                        Score(scorer_name="ratio", value=0.4),
                    ]
                )
            ]
        )

        body = render_markdown([run])

        assert "| qa | exact | 1.00 |" in body
        assert "| qa | ratio | 0.40 |" in body

    def test_an_eval_whose_scorers_all_errored_still_gets_a_row(self) -> None:
        """Omitting it would leave a reviewer believing it was never part of the
        run, when in truth it ran and could not be measured.

        Asserted against the *summary* table specifically. An earlier version
        looked for "| qa |" anywhere in the body — which the Needs attention
        table also contains, so it passed against a mutant that dropped the
        summary row entirely.
        """
        run = _run(results=[_result(scores=[Score.from_error("judge", "no key")])])

        body = render_markdown([run])
        summary = body.split("### Needs attention")[0]

        assert "| qa | - | - | - | - |" in summary

    def test_the_pass_column_excludes_unjudged_cases(self) -> None:
        """A continuous scorer that declined to judge has failed nothing, and
        counting it in the denominator would understate the rate."""
        run = _run(
            results=[
                _result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
                _result("q2", scores=[Score(scorer_name="ratio", value=0.5)]),
            ]
        )

        body = render_markdown([run])

        assert "| 1/1 |" in body


class TestCostIsNeverOverstatedAsFree:
    def test_an_unpriced_run_shows_a_placeholder(self) -> None:
        """`$0.0000` is a claim a reviewer cannot check from a comment."""
        run = _run(results=[_result(traces=[_trace(None)])])

        body = render_markdown([run])

        assert "$0.0000" not in body
        assert "| - |" in body

    def test_a_partly_priced_run_says_the_total_is_a_lower_bound(self) -> None:
        """Showing the known sum bare would understate the bill — wrong in the
        direction that costs money."""
        run = _run(results=[_result(traces=[_trace(0.01), _trace(None)])])

        body = render_markdown([run])

        assert "$0.0100" in body
        assert "lower bound" in body

    def test_a_fully_priced_run_states_the_cost_plainly(self) -> None:
        run = _run(results=[_result(traces=[_trace(0.02)])])

        body = render_markdown([run])

        assert "$0.0200" in body
        assert "lower bound" not in body


class TestNeedsAttention:
    def test_a_crashed_case_is_listed_with_its_error(self) -> None:
        run = _run(results=[_result("q1", error="RuntimeError: boom", scores=[])])

        body = render_markdown([run])

        assert "task error" in body
        assert "RuntimeError: boom" in body

    def test_an_unmeasured_case_is_listed_beside_the_failures(self) -> None:
        """A scorer that broke leaves the task's performance unknown, and a
        reviewer who saw only the mean would read that silence as success."""
        run = _run(results=[_result("q1", scores=[Score.from_error("judge", "rate limited")])])

        body = render_markdown([run])

        assert "unmeasured" in body
        assert "rate limited" in body

    def test_a_failing_case_names_the_scorer_that_failed_it(self) -> None:
        run = _run(
            results=[
                _result(
                    "q1",
                    output="wrong",
                    scores=[Score(scorer_name="exact", value=0.0, passed=False)],
                )
            ]
        )

        body = render_markdown([run])

        assert "failed exact" in body
        assert "wrong" in body

    def test_a_clean_run_has_no_attention_section(self) -> None:
        """An empty table under a heading reads as a problem nobody described."""
        assert "Needs attention" not in render_markdown([_run()])

    def test_a_continuous_score_is_not_called_a_failure(self) -> None:
        """It declined to judge; the report must not judge on its behalf."""
        run = _run(results=[_result("q1", scores=[Score(scorer_name="ratio", value=0.1)])])

        assert "Needs attention" not in render_markdown([run])


class TestTheTableCannotBeBrokenByItsContent:
    def test_a_pipe_in_the_output_is_escaped(self) -> None:
        """An unescaped pipe splits the cell, and every column after it shifts —
        a table that silently reports the wrong value under each heading."""
        run = _run(
            results=[
                _result(
                    "q1",
                    output="a | b | c",
                    scores=[Score(scorer_name="exact", value=0.0, passed=False)],
                )
            ]
        )

        body = render_markdown([run])
        row = next(line for line in body.splitlines() if "failed exact" in line)

        assert row.count("|") - row.count("\\|") == 5, "the row has extra cell boundaries"

    def test_a_newline_in_the_output_is_flattened(self) -> None:
        """A newline ends the row, so the rest of the output becomes a new
        malformed row and the table stops rendering."""
        run = _run(
            results=[
                _result(
                    "q1",
                    output="line one\nline two",
                    scores=[Score(scorer_name="exact", value=0.0, passed=False)],
                )
            ]
        )

        body = render_markdown([run])
        rows = [line for line in body.splitlines() if "failed exact" in line]

        assert len(rows) == 1
        assert "line one line two" in rows[0]

    def test_a_long_output_is_truncated(self) -> None:
        """GitHub hides comments past 65 536 characters, and a report nobody can
        see is worse than a short one."""
        run = _run(
            results=[
                _result(
                    "q1",
                    output="x" * 5000,
                    scores=[Score(scorer_name="exact", value=0.0, passed=False)],
                )
            ]
        )

        body = render_markdown([run])

        assert "..." in body
        assert len(body) < 1000


class TestTheThresholdVerdict:
    def test_a_breach_is_named_with_the_bar_it_missed(self) -> None:
        """A mean of 0.20 is a pass or a failure depending on a number that
        lives in a CI config the reviewer cannot see."""
        run = _run(results=[_result(scores=[Score(scorer_name="exact", value=0.2)])])

        body = render_markdown([run], threshold=0.8)

        assert "0.80" in body
        assert "`qa` scored 0.20" in body

    def test_no_threshold_means_no_verdict(self) -> None:
        """`evalstand` will not invent a pass mark."""
        run = _run(results=[_result(scores=[Score(scorer_name="exact", value=0.2)])])

        assert "threshold" not in render_markdown([run]).lower()

    def test_an_unmeasured_run_is_not_reported_as_below_the_bar(self) -> None:
        """It has no mean at all; printing one would put a number where there is
        none."""
        run = _run(results=[_result(scores=[Score.from_error("judge", "boom")])])

        assert "Below the threshold" not in render_markdown([run], threshold=0.8)


class TestWhatAnEmptyRunSays:
    def test_no_evals_is_distinct_from_everything_passed(self) -> None:
        """Opposite findings. A comment showing an empty table for both would
        let a broken collection read as success."""
        assert "No evals ran" in render_markdown([])

    def test_errored_scores_are_declared(self) -> None:
        """The means cover fewer cases than the run contains, and a reviewer
        reading them as complete would overstate what was measured."""
        run = _run(
            results=[
                _result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
                _result("q2", scores=[Score.from_error("exact", "boom")]),
            ]
        )

        assert "1 score errored" in render_markdown([run])


class TestTheOutputFlag:
    """`--output markdown`, which is what a workflow actually invokes.

    Driven through `runpytest_subprocess` for the reason recorded in ADR 0008:
    importing litellm inside a pytester inline session breaks entry-point
    discovery for the rest of the process.
    """

    EVAL = """
from evalstand import Case, evaluate
from evalstand.models import Score

evaluate(
    name="flagged",
    cases=[Case(id="q1", input="x", expected="ok")],
    task=lambda value: "ok",
    scorers=[lambda output, expected: Score(scorer_name="s", value=1.0, passed=True)],
)
"""

    def test_it_prints_markdown_instead_of_tables(self, pytester) -> None:
        pytester.makepyfile(flag_eval=self.EVAL)
        result = pytester.runpytest_subprocess(
            "--no-store", "--allow-dirty", "-p", "no:cacheprovider", "--output", "markdown"
        )

        output = str(result.stdout)
        assert "## evalstand" in output
        assert "| eval | scorer | mean | passed | cost |" in output

    def test_the_default_is_still_the_terminal_summary(self, pytester) -> None:
        """A flag nobody passed must change nothing."""
        pytester.makepyfile(flag_eval=self.EVAL)
        result = pytester.runpytest_subprocess(
            "--no-store", "--allow-dirty", "-p", "no:cacheprovider"
        )

        assert "## evalstand" not in str(result.stdout)

    def test_the_threshold_reaches_the_markdown(self, pytester) -> None:
        """The bar lives in a CI config the reviewer cannot see, so the comment
        has to carry it."""
        pytester.makepyfile(
            low_eval="""
from evalstand import Case, evaluate
from evalstand.models import Score

evaluate(
    name="low",
    cases=[Case(id="q1", input="x", expected="ok")],
    task=lambda value: "no",
    scorers=[lambda output, expected: Score(scorer_name="s", value=0.1)],
)
"""
        )
        result = pytester.runpytest_subprocess(
            "--no-store",
            "--allow-dirty",
            "-p",
            "no:cacheprovider",
            "--output",
            "markdown",
            "--threshold",
            "0.8",
        )

        output = " ".join(str(result.stdout).split())
        assert "Below the threshold of 0.80" in output
        assert result.ret == 1, "markdown output must not change the exit code"


pytest_plugins = ["pytester"]

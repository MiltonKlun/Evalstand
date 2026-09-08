"""The exit codes a CI job depends on (task 7.1).

Three outcomes, and the distinction between the last two is the point:

- **0** every eval met its bar.
- **1** an eval's mean fell below `--threshold`. The measurement worked; the
  answer was "worse than the bar".
- **2** with `--fail-on-error`, something did not run.

A build that exits 1 sends someone to look at the model. A build that exits 2
sends them to look at the pipeline. Collapsing them would send half of those
people to the wrong place — and the more expensive direction is reporting an
outage as a quality regression, because the model is then "fixed" by lowering a
threshold that was never the problem.

Driven through `runpytest_subprocess` for the reason recorded in ADR 0008:
importing litellm inside a pytester inline session breaks entry-point discovery
for the rest of the process.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

PASSING_EVAL = """
from evalstand import Case, evaluate
from evalstand.models import Score

evaluate(
    name="passing",
    cases=[Case(id="q1", input="x", expected="ok")],
    task=lambda value: "ok",
    scorers=[lambda output, expected: Score(scorer_name="s", value=1.0, passed=True)],
)
"""

LOW_EVAL = """
from evalstand import Case, evaluate
from evalstand.models import Score

# A continuous score with no verdict: exactly the shape that exits 0 without a
# threshold, which is why the threshold exists.
evaluate(
    name="low",
    cases=[Case(id="q1", input="x", expected="ok")],
    task=lambda value: "wrong",
    scorers=[lambda output, expected: Score(scorer_name="s", value=0.2)],
)
"""

RAISING_EVAL = """
from evalstand import Case, evaluate
from evalstand.models import Score


def task(value):
    raise RuntimeError("provider down")


evaluate(
    name="raising",
    cases=[Case(id="q1", input="x", expected="ok")],
    task=task,
    scorers=[lambda output, expected: Score(scorer_name="s", value=1.0, passed=True)],
)
"""

PARTIALLY_SCORED_EVAL = """
from evalstand import Case, evaluate
from evalstand.models import Score


def scorer(output, expected, case):
    # Three of four cases are never measured. The mean over the survivor is a
    # perfect 1.00, which clears any threshold.
    if case.id != "q1":
        raise RuntimeError("judge API key expired")
    return Score(scorer_name="s", value=1.0, passed=True)


evaluate(
    name="partial",
    cases=[Case(id="q%d" % i, input="x", expected="ok") for i in range(1, 5)],
    task=lambda value: "ok",
    scorers=[scorer],
)
"""

MIXED_EVAL = """
from evalstand import Case, evaluate
from evalstand.models import Score


def task(value):
    # The realistic shape of a bad build: a flaky provider drops some calls,
    # and the answers that do get through are poor.
    if value == "flaky":
        raise RuntimeError("provider down")
    return "wrong"


evaluate(
    name="mixed",
    cases=[Case(id="flaky", input="flaky", expected="ok"),
           Case(id="poor", input="poor", expected="ok")],
    task=task,
    scorers=[lambda output, expected: Score(scorer_name="s", value=0.1)],
)
"""

_BASE = ("--no-store", "--allow-dirty", "-p", "no:cacheprovider")


def _run(pytester: pytest.Pytester, source: str, *flags: str) -> int:
    pytester.makepyfile(subject_eval=source)
    return pytester.runpytest_subprocess(*_BASE, *flags).ret


class TestZeroMeansEverythingMetItsBar:
    def test_a_passing_eval_exits_zero(self, pytester: pytest.Pytester) -> None:
        assert _run(pytester, PASSING_EVAL) == 0

    def test_a_threshold_it_clears_still_exits_zero(self, pytester: pytest.Pytester) -> None:
        assert _run(pytester, PASSING_EVAL, "--threshold", "0.8") == 0

    def test_fail_on_error_does_not_fail_a_clean_run(self, pytester: pytest.Pytester) -> None:
        """The flag must cost nothing when nothing went wrong, or nobody will
        leave it on."""
        assert _run(pytester, PASSING_EVAL, "--fail-on-error") == 0

    def test_both_flags_together_still_exit_zero(self, pytester: pytest.Pytester) -> None:
        assert _run(pytester, PASSING_EVAL, "--threshold", "0.8", "--fail-on-error") == 0


class TestOneMeansBelowTheThreshold:
    def test_a_low_mean_exits_one(self, pytester: pytest.Pytester) -> None:
        assert _run(pytester, LOW_EVAL, "--threshold", "0.8") == 1

    def test_without_a_threshold_the_same_run_exits_zero(self, pytester: pytest.Pytester) -> None:
        """The bar is the user's judgement, not ours. Nothing is invented."""
        assert _run(pytester, LOW_EVAL) == 0

    def test_fail_on_error_alone_does_not_fail_a_merely_low_run(
        self, pytester: pytest.Pytester
    ) -> None:
        """A bad score is not an error. Conflating them would make
        `--fail-on-error` a second, secret threshold."""
        assert _run(pytester, LOW_EVAL, "--fail-on-error") == 0


class TestTwoMeansSomethingDidNotRun:
    def test_a_raising_task_exits_two_with_the_flag(self, pytester: pytest.Pytester) -> None:
        assert _run(pytester, RAISING_EVAL, "--fail-on-error") == 2

    def test_an_errored_scorer_exits_two_with_the_flag(self, pytester: pytest.Pytester) -> None:
        """The scenario the plan calls out: the mean is 1.00 over the one case
        that got measured, so a threshold alone would wave it through as
        excellent."""
        assert _run(pytester, PARTIALLY_SCORED_EVAL, "--fail-on-error") == 2

    def test_an_execution_error_outranks_a_threshold_breach(
        self, pytester: pytest.Pytester
    ) -> None:
        """2 beats 1, asserted where the two can actually disagree.

        `MIXED_EVAL` errors on one case *and* scores 0.1 on the other, so both
        rules fire at once. An earlier version of this test used a fixture whose
        only case errored — leaving `mean_score` at None, so no breach was ever
        recorded and there was nothing for a fall-through to overwrite the 2
        with. It passed against a mutant that let 1 win.

        The distinction is what a reader is sent to investigate: 1 says the
        model got worse, 2 says half of it never ran. A mean over the survivors
        is not a measurement of the eval, so reporting it as "below threshold"
        names a cause the evidence does not support.
        """
        assert _run(pytester, MIXED_EVAL, "--threshold", "0.99", "--fail-on-error") == 2

    def test_the_same_run_without_the_flag_reports_the_breach(
        self, pytester: pytest.Pytester
    ) -> None:
        """The other half: with no `--fail-on-error`, the threshold verdict is
        what the user asked for and 1 is correct. This is what makes the test
        above about *precedence* rather than about the fixture."""
        assert _run(pytester, MIXED_EVAL, "--threshold", "0.99") == 1

    def test_the_output_says_what_errored_and_why_it_matters(
        self, pytester: pytest.Pytester
    ) -> None:
        """An exit code nobody can explain is indistinguishable from a bug in
        the tool, and sends the reader looking for the wrong thing."""
        pytester.makepyfile(subject_eval=PARTIALLY_SCORED_EVAL)
        result = pytester.runpytest_subprocess(*_BASE, "--fail-on-error")

        # Whitespace is collapsed before matching. Rich wraps the sentence to
        # whatever width the subprocess reports, and that width is not the same
        # here as in a real terminal — an assertion that spans the wrap point
        # would fail on the message being *correct* but folded elsewhere.
        output = " ".join(str(result.stdout).split())

        assert "ERROR" in output
        assert "3 scores errored" in output
        assert "so the mean does not measure the whole eval" in output


class TestCompareOnAnEmptyDatabase:
    """7.1's acceptance names this case specifically.

    The history database is project-local and gitignored (ADR 0005), so CI
    starts every build with no history at all. `--threshold` still works there
    because it is an absolute bar; `compare` has nothing to work with and must
    say so rather than pass silently.
    """

    def _compare(self, database, *runs: str):
        from typer.testing import CliRunner

        from evalstand.cli import app

        return CliRunner().invoke(app, ["compare", *runs, "--db", str(database)])

    def test_it_exits_two_rather_than_falsely_passing(self, tmp_path) -> None:
        """The failure that would matter most: a CI job that gates on `compare`
        and sees 0 concludes the change is fine, when in truth nothing was ever
        measured."""
        result = self._compare(tmp_path / "empty.db", "run-a", "run-b")

        assert result.exit_code == 2

    def test_it_says_what_is_missing(self, tmp_path) -> None:
        result = self._compare(tmp_path / "empty.db", "run-a", "run-b")

        assert "run-a" in result.output and "run-b" in result.output

    def test_a_threshold_run_still_works_with_no_history(self, pytester: pytest.Pytester) -> None:
        """The other half of the same promise: an absolute bar needs no prior
        run, so a fresh clone can gate on quality from its first build."""
        assert _run(pytester, PASSING_EVAL, "--threshold", "0.8") == 0
        assert _run(pytester, LOW_EVAL, "--threshold", "0.8") == 1


class TestWhatTheFlagMustNotDo:
    def test_without_it_an_errored_scorer_does_not_exit_two(
        self, pytester: pytest.Pytester
    ) -> None:
        """Opt-in. A run whose scorer broke still fails through its unmeasured
        cases, but the *code* stays 1 unless the user asked for the
        distinction."""
        assert _run(pytester, PARTIALLY_SCORED_EVAL) == 1

    def test_a_plain_test_session_is_unaffected(self, pytester: pytest.Pytester) -> None:
        """A repository with no evals must behave exactly as it did before,
        whatever flags are passed."""
        pytester.makepyfile(test_plain="def test_one():\n    assert True\n")
        result = pytester.runpytest_subprocess(*_BASE, "--fail-on-error", "--threshold", "0.9")

        assert result.ret == 0

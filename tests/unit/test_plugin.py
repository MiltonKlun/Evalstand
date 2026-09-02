"""The pytest plugin (task 2.2).

Uses pytest's own `pytester` fixture so these are real collection runs, not
simulations of one. The risk register calls this out as the place the plugin
might fight the async runner, so it is tested against actual pytest behaviour.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

TOY_EVAL = """
from evalstand import Case, evaluate
from evalstand.scorers import exact


def load_cases():
    return [
        Case(id="q1", input="France", expected="Paris"),
        Case(id="q2", input="Japan", expected="Tokyo"),
        Case(id="q3", input="Peru", expected="Lima"),
    ]


def answer(country):
    return {"France": "Paris", "Japan": "Tokyo", "Peru": "Lima"}[country]


evaluate(name="toy", cases=load_cases, task=answer, scorers=[exact])
"""

FAILING_EVAL = """
from evalstand import Case, evaluate
from evalstand.scorers import exact

evaluate(
    name="wrong",
    cases=[Case(id="bad", input="x", expected="right")],
    task=lambda x: "wrong",
    scorers=[exact],
)
"""


class TestCollection:
    def test_collects_an_eval_file(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(qa_eval=TOY_EVAL)
        result = pytester.runpytest("--no-cov")
        result.assert_outcomes(passed=3)

    def test_generates_one_item_per_case(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(qa_eval=TOY_EVAL)
        result = pytester.runpytest("--collect-only", "-q", "--no-cov")
        collected = [line for line in result.outlines if "::" in line]
        assert len(collected) == 3

    def test_ignores_files_that_are_not_evals(self, pytester: pytest.Pytester) -> None:
        """`*_eval.py` is the contract; a plain module must not be collected."""
        pytester.makepyfile(helpers="X = 1")
        result = pytester.runpytest("--collect-only", "-q", "--no-cov")
        assert not [line for line in result.outlines if "::" in line]

    def test_case_ids_are_the_test_ids(self, pytester: pytest.Pytester) -> None:
        """A user reads these in failure output; opaque indices would be useless."""
        pytester.makepyfile(qa_eval=TOY_EVAL)
        result = pytester.runpytest("--collect-only", "-q", "--no-cov")
        joined = "\n".join(result.outlines)
        assert "q1" in joined
        assert "q2" in joined
        assert "q3" in joined

    def test_k_selects_a_single_case(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(qa_eval=TOY_EVAL)
        result = pytester.runpytest("-k", "q1", "--no-cov")
        result.assert_outcomes(passed=1, deselected=2)


class TestOutcomes:
    def test_a_failing_case_fails_the_item(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(wrong_eval=FAILING_EVAL)
        result = pytester.runpytest("--no-cov")
        result.assert_outcomes(failed=1)

    def test_the_failure_shows_output_and_expected(self, pytester: pytest.Pytester) -> None:
        """A bare assertion error would make the user re-run to learn anything."""
        pytester.makepyfile(wrong_eval=FAILING_EVAL)
        result = pytester.runpytest("--no-cov")
        joined = "\n".join(result.outlines)
        assert "wrong" in joined
        assert "right" in joined

    def test_a_raising_task_fails_only_its_own_case(self, pytester: pytest.Pytester) -> None:
        """One bad case must not abort the rest of the eval."""
        pytester.makepyfile(
            boom_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact


def task(x):
    if x == "explode":
        raise ValueError("task blew up")
    return "ok"


evaluate(
    name="boom",
    cases=[
        Case(id="fine", input="x", expected="ok"),
        Case(id="broken", input="explode", expected="ok"),
    ],
    task=task,
    scorers=[exact],
)
"""
        )
        result = pytester.runpytest("--no-cov")
        result.assert_outcomes(passed=1, failed=1)


class TestAsyncTasks:
    def test_an_async_task_runs(self, pytester: pytest.Pytester) -> None:
        """Task 2.3: the user should never have to say which kind it is."""
        pytester.makepyfile(
            async_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact


async def answer(country):
    return {"France": "Paris"}[country]


evaluate(
    name="async-toy",
    cases=[Case(id="q1", input="France", expected="Paris")],
    task=answer,
    scorers=[exact],
)
"""
        )
        result = pytester.runpytest("--no-cov")
        result.assert_outcomes(passed=1)

    def test_sync_and_async_evals_agree(self, pytester: pytest.Pytester) -> None:
        """Task 2.3's acceptance: identical evals, identical results."""
        pytester.makepyfile(
            both_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact

CASES = [Case(id="q1", input="France", expected="Paris")]


def sync_answer(country):
    return "Paris"


async def async_answer(country):
    return "Paris"


evaluate(name="sync-one", cases=CASES, task=sync_answer, scorers=[exact])
evaluate(name="async-one", cases=CASES, task=async_answer, scorers=[exact])
"""
        )
        result = pytester.runpytest("--no-cov")
        result.assert_outcomes(passed=2)


class TestDuplicateNames:
    def test_two_evals_sharing_a_name_is_a_collection_error(
        self, pytester: pytest.Pytester
    ) -> None:
        """Names are identity; merging two histories silently would be worse."""
        pytester.makepyfile(
            first_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact
evaluate(name="clash", cases=[Case(id="a", input="x", expected="x")],
         task=lambda x: x, scorers=[exact])
""",
            second_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact
evaluate(name="clash", cases=[Case(id="b", input="y", expected="y")],
         task=lambda x: x, scorers=[exact])
""",
        )
        result = pytester.runpytest("--no-cov")
        joined = "\n".join(result.outlines)
        assert "clash" in joined
        assert result.ret != 0


class TestRepeats:
    def test_repeat_generates_one_item_per_execution(self, pytester: pytest.Pytester) -> None:
        """An item is one execution with one outcome; collapsing several
        stochastic runs into one pass/fail invents an aggregation rule."""
        pytester.makepyfile(
            rep_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact

evaluate(
    name="repeated",
    cases=[Case(id="q1", input="x", expected="x")],
    task=lambda x: x,
    scorers=[exact],
    repeat=3,
)
"""
        )
        result = pytester.runpytest("--no-cov")
        result.assert_outcomes(passed=3)

    def test_repeat_items_stay_selectable_by_case_id(self, pytester: pytest.Pytester) -> None:
        """`-k q1` must still find all three."""
        pytester.makepyfile(
            rep_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact

evaluate(
    name="repeated",
    cases=[Case(id="q1", input="x", expected="x"), Case(id="q2", input="y", expected="y")],
    task=lambda x: x,
    scorers=[exact],
    repeat=3,
)
"""
        )
        result = pytester.runpytest("-k", "q1", "--no-cov")
        result.assert_outcomes(passed=3, deselected=3)

    def test_a_single_run_has_no_repeat_suffix(self, pytester: pytest.Pytester) -> None:
        """The common case stays clean; the suffix appears only when it means
        something."""
        pytester.makepyfile(qa_eval=TOY_EVAL)
        result = pytester.runpytest("--collect-only", "-q", "--no-cov")
        assert "repeat=" not in "\n".join(result.outlines)

    def test_repeated_items_are_distinguishable(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            rep_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact

evaluate(
    name="repeated",
    cases=[Case(id="q1", input="x", expected="x")],
    task=lambda x: x,
    scorers=[exact],
    repeat=2,
)
"""
        )
        result = pytester.runpytest("--collect-only", "-q", "--no-cov")
        joined = "\n".join(result.outlines)
        assert "repeat=1" in joined
        assert "repeat=2" in joined


class TestRegistryIsolation:
    def test_two_eval_files_do_not_leak_into_each_other(self, pytester: pytest.Pytester) -> None:
        """The registry is module-level state; collection must not accumulate
        one file's evals into another's."""
        pytester.makepyfile(
            one_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact
evaluate(name="one", cases=[Case(id="a", input="x", expected="x")],
         task=lambda x: x, scorers=[exact])
""",
            two_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact
evaluate(name="two", cases=[Case(id="b", input="y", expected="y")],
         task=lambda x: x, scorers=[exact])
""",
        )
        result = pytester.runpytest("--no-cov")
        result.assert_outcomes(passed=2)


class TestDoubleCollection:
    """pytest imports files matching its collection patterns through its own
    assertion-rewriting importer. Importing again in the plugin executed every
    eval file twice, so each eval registered twice and was reported as a
    duplicate of itself."""

    def test_naming_the_same_file_twice_collects_it_once(self, pytester: pytest.Pytester) -> None:
        path = pytester.makepyfile(qa_eval=TOY_EVAL)
        result = pytester.runpytest(str(path), str(path), "--no-cov")
        result.assert_outcomes(passed=3)

    def test_the_eval_file_is_executed_exactly_once(self, pytester: pytest.Pytester) -> None:
        """A module-level side effect is the honest way to count imports."""
        pytester.makepyfile(
            counted_eval="""
import pathlib

from evalstand import Case, evaluate
from evalstand.scorers import exact

marker = pathlib.Path(__file__).parent / "imports.txt"
with marker.open("a") as handle:
    handle.write("x")

evaluate(
    name="counted",
    cases=[Case(id="a", input="x", expected="x")],
    task=lambda x: x,
    scorers=[exact],
)
"""
        )
        pytester.runpytest("--no-cov").assert_outcomes(passed=1)
        assert (pytester.path / "imports.txt").read_text() == "x"


class TestFailureOutput:
    """A failure must be actionable without re-running with more flags."""

    def test_a_raised_error_shows_the_users_frames_not_pytest_internals(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile(
            raise_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact


def lookup(key):
    return {"x": "y"}[key]


def task(key):
    return lookup(key)


evaluate(
    name="raiser",
    cases=[Case(id="q1", input="missing", expected="y")],
    task=task,
    scorers=[exact],
)
"""
        )
        lines = pytester.runpytest("--no-cov").outlines
        start = next(i for i, line in enumerate(lines) if "FAILURES" in line)
        end = next(
            (i for i, line in enumerate(lines[start:], start) if "short test summary" in line),
            len(lines),
        )
        block = "\n".join(lines[start:end])

        assert "KeyError" in block
        assert "raise_eval.py" in block, "the user's file should be named"
        assert "in lookup" in block, "the frame that actually failed"

        # The frames pytest's default repr would have shown and the user cannot
        # act on. Named individually rather than matching "_pytest", which also
        # appears in pytester's own temp directory name.
        assert "runner.py" not in block, "pytest's call machinery must not be shown"
        assert "pluggy" not in block
        assert "_hookexec" not in block

    def test_a_scoring_failure_shows_output_and_expected(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(wrong_eval=FAILING_EVAL)
        joined = "\n".join(pytester.runpytest("--no-cov").outlines)

        assert "output:" in joined
        assert "expected:" in joined
        assert "exact" in joined


class TestTerminalSummary:
    def test_a_summary_is_printed_after_an_eval_run(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(qa_eval=TOY_EVAL)
        joined = "\n".join(pytester.runpytest("--no-cov").outlines)
        assert "toy" in joined
        assert "exact" in joined
        assert "3/3" in joined

    def test_nothing_is_printed_for_a_plain_test_session(self, pytester: pytest.Pytester) -> None:
        """A repo with no evals must look exactly as it did before."""
        pytester.makepyfile(test_plain="def test_ok():\n    assert True\n")
        joined = "\n".join(pytester.runpytest("--no-cov").outlines)
        assert "scorer" not in joined
        assert "failures" not in joined

    def test_a_crashed_case_appears_in_the_failures_table(self, pytester: pytest.Pytester) -> None:
        """A case that blew up is a result; omitting it would hide the failure."""
        pytester.makepyfile(
            crash_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact


def task(x):
    raise ValueError("model unavailable")


evaluate(
    name="crasher",
    cases=[Case(id="q1", input="x", expected="y")],
    task=task,
    scorers=[exact],
)
"""
        )
        joined = "\n".join(pytester.runpytest("--no-cov").outlines)
        assert "task error" in joined
        assert "model unavailable" in joined

    def test_a_crashed_case_is_not_counted_as_judged(self, pytester: pytest.Pytester) -> None:
        """A case that never produced a score has not been judged, so counting
        it in the denominator would understate the pass rate."""
        pytester.makepyfile(
            mixed_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact


def task(x):
    if x == "boom":
        raise ValueError("down")
    return "ok"


evaluate(
    name="mixed",
    cases=[
        Case(id="good", input="x", expected="ok"),
        Case(id="bad", input="boom", expected="ok"),
    ],
    task=task,
    scorers=[exact],
)
"""
        )
        joined = "\n".join(pytester.runpytest("--no-cov").outlines)
        assert "1/1" in joined, "the crashed case is reported, not counted as judged"
        assert "1/2" not in joined


class TestUnmeasuredCases:
    """A case whose scorers all errored was not measured.

    Reporting it as a pass is a false claim: a rate-limited judge is not
    evidence the task did well. It is not a task failure either — the task ran
    fine. pytest has a state for "this did not produce a verdict", and that is
    the honest one.
    """

    UNSCORABLE = """
from evalstand import Case, evaluate


def broken_scorer(output, expected, case):
    raise RuntimeError("judge API is down")


evaluate(
    name="unscorable",
    cases=[Case(id="q1", input="x", expected="x")],
    task=lambda value: value,
    scorers=[broken_scorer],
)
"""

    def test_a_case_with_no_usable_score_does_not_pass(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(unscorable_eval=self.UNSCORABLE)
        result = pytester.runpytest("--no-cov")
        result.assert_outcomes(passed=0, failed=1)

    def test_the_run_does_not_exit_zero(self, pytester: pytest.Pytester) -> None:
        """CI gates on the exit code; a run that measured nothing must not
        report success."""
        pytester.makepyfile(unscorable_eval=self.UNSCORABLE)
        assert pytester.runpytest("--no-cov").ret != 0

    def test_the_reason_is_reported(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(unscorable_eval=self.UNSCORABLE)
        joined = "\n".join(pytester.runpytest("--no-cov").outlines)
        assert "judge API is down" in joined

    def test_a_partially_scored_case_still_passes(self, pytester: pytest.Pytester) -> None:
        """One working scorer is a measurement. Only a total absence is not."""
        pytester.makepyfile(
            partial_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact


def broken_scorer(output, expected, case):
    raise RuntimeError("judge down")


evaluate(
    name="partial",
    cases=[Case(id="q1", input="x", expected="x")],
    task=lambda value: value,
    scorers=[exact, broken_scorer],
)
"""
        )
        pytester.runpytest("--no-cov").assert_outcomes(passed=1)

    def test_a_genuine_zero_still_fails_normally(self, pytester: pytest.Pytester) -> None:
        """A measured zero is a failure, not an absence; the two must not be
        conflated in either direction."""
        pytester.makepyfile(zero_eval=FAILING_EVAL)
        pytester.runpytest("--no-cov").assert_outcomes(failed=1)

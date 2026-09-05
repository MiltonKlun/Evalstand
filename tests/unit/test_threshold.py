"""`--threshold`: the only way a continuous scorer can fail a run.

Found in the pre-Phase-5 audit. A model answering every case with garbage
reported `3 passed` and **exit code 0**, because continuous scorers never set
`passed` and nothing else failed the run. CI went green on a model that had
learned nothing, which defeats the purpose of the tool.

The threshold is the user supplying the judgement the scorer declined to make.
It is opt-in because `evalstand` will not invent a pass mark, and it judges the
aggregate only — it never sets any Score's pass flag, per CONTEXT.md.

Driven through `runpytest_subprocess` for the reason recorded in ADR 0008:
importing litellm inside a pytester inline session breaks entry-point discovery
for the rest of the process.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

GARBAGE_EVAL = """
from evalstand import Case, evaluate
from evalstand.scorers import levenshtein

evaluate(
    name="garbage",
    cases=[Case(id="q%d" % i, input=str(i), expected="the correct answer")
           for i in range(3)],
    task=lambda value: "completely unrelated nonsense",
    scorers=[levenshtein],
)
"""

PERFECT_EVAL = """
from evalstand import Case, evaluate
from evalstand.scorers import levenshtein

evaluate(
    name="perfect",
    cases=[Case(id="q%d" % i, input=str(i), expected="the correct answer")
           for i in range(3)],
    task=lambda value: "the correct answer",
    scorers=[levenshtein],
)
"""

UNMEASURED_EVAL = """
from evalstand import Case, evaluate
from evalstand.models import Score


def broken(output, expected):
    raise RuntimeError("the judge is unreachable")


evaluate(
    name="unmeasured",
    cases=[Case(id="q1", input="x", expected="y")],
    task=lambda value: "an answer",
    scorers=[broken],
)
"""


class TestWithoutAThreshold:
    """The default must be unchanged and honest: no invented pass mark."""

    def test_a_garbage_run_still_exits_zero(self, pytester: pytest.Pytester) -> None:
        """Deliberate. `evalstand` has been given no line to judge against, and
        guessing one would be exactly the inference CONTEXT.md forbids. What it
        must not do is *hide* the problem — see the failures-table test below.
        """
        pytester.makepyfile(garbage_eval=GARBAGE_EVAL)
        result = pytester.runpytest_subprocess("--no-cov")

        assert result.ret == 0

    def test_but_the_low_scores_are_visible(self, pytester: pytest.Pytester) -> None:
        """The mean is printed, so a human reading the output sees 0.2 and acts.
        Silence would be the unacceptable part."""
        pytester.makepyfile(garbage_eval=GARBAGE_EVAL)
        joined = "\n".join(pytester.runpytest_subprocess("--no-cov").outlines)

        assert "garbage" in joined
        assert "0.2" in joined, "the mean was not reported"


class TestWithAThreshold:
    def test_a_run_below_the_threshold_fails(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(garbage_eval=GARBAGE_EVAL)
        result = pytester.runpytest_subprocess("--threshold", "0.7", "--no-cov")

        assert result.ret != 0, "a garbage model passed a 0.7 threshold"

    def test_a_run_above_the_threshold_passes(self, pytester: pytest.Pytester) -> None:
        """The other half. Without it, a threshold that failed everything would
        satisfy the test above and be useless."""
        pytester.makepyfile(perfect_eval=PERFECT_EVAL)
        result = pytester.runpytest_subprocess("--threshold", "0.7", "--no-cov")

        assert result.ret == 0

    def test_the_failure_says_which_eval_and_by_how_much(self, pytester: pytest.Pytester) -> None:
        """A run that fails for a reason the output never states is
        indistinguishable from a bug in the tool, and sends the user looking in
        the wrong place."""
        pytester.makepyfile(garbage_eval=GARBAGE_EVAL)
        joined = "\n".join(pytester.runpytest_subprocess("--threshold", "0.7", "--no-cov").outlines)

        assert "garbage" in joined
        assert "below the threshold" in joined
        assert "0.70" in joined

    def test_the_individual_cases_still_report_as_passing(self, pytester: pytest.Pytester) -> None:
        """The threshold judges the aggregate. It must not retroactively mark
        cases as failed, because no scorer said they were — CONTEXT.md is
        explicit that a Threshold never sets a Score's pass flag."""
        pytester.makepyfile(garbage_eval=GARBAGE_EVAL)
        result = pytester.runpytest_subprocess("--threshold", "0.7", "--no-cov")

        result.assert_outcomes(passed=3)
        assert result.ret != 0, "the session should still fail overall"

    def test_a_threshold_of_zero_passes_anything_scored(self, pytester: pytest.Pytester) -> None:
        """0.0 is a legitimate floor: "any measurement at all is acceptable"."""
        pytester.makepyfile(garbage_eval=GARBAGE_EVAL)
        assert pytester.runpytest_subprocess("--threshold", "0", "--no-cov").ret == 0


class TestWhatTheThresholdMustNotDo:
    def test_an_unmeasured_run_is_not_reported_as_below_the_threshold(
        self, pytester: pytest.Pytester
    ) -> None:
        """A run whose scorers all errored has no mean. Calling that "below the
        threshold" would put a number where there is none — it fails already,
        through its unmeasured cases, and for the right reason."""
        pytester.makepyfile(unmeasured_eval=UNMEASURED_EVAL)
        result = pytester.runpytest_subprocess("--threshold", "0.7", "--no-cov")
        joined = "\n".join(result.outlines)

        assert "below the threshold" not in joined
        assert "no usable score" in joined or "scorer error" in joined

        # Asserted on stderr, not on the exit code, which is non-zero either
        # way because the unmeasured case already fails on its own. A version
        # treating a missing mean as 0.0 raises a TypeError while formatting
        # the breach message — and pytest writes that crash to stderr, so
        # checking `outlines` alone would miss it entirely.
        errors = "\n".join(result.errlines)
        assert "TypeError" not in errors, "the summary crashed on a run with no mean"

    def test_a_plain_test_session_is_unaffected(self, pytester: pytest.Pytester) -> None:
        """evalstand loads in every pytest run on the machine. A suite with no
        evals must not be failed by a flag that has nothing to judge."""
        pytester.makepyfile(test_plain="def test_one():\n    assert True\n")
        result = pytester.runpytest_subprocess("--threshold", "0.9", "--no-cov")

        result.assert_outcomes(passed=1)
        assert result.ret == 0

"""The numeric scorer (task 4.4).

The plan asks for "sensible handling of None and unparseable output", and most
of this file is about what "sensible" means. The rule the design settles on:
**an ambiguous output is unparseable, not a guess.** A scorer that says "I could
not read this" is auditable; one that quietly picks a number from "42 or 43"
produces a score nobody can check.

The traps worth naming, because each was found by probing rather than by
reading the code: `float("nan")` succeeds and then compares false against every
tolerance; `bool` subclasses `int`, so `float(True)` is 1.0; and a relative
tolerance around an expected value of zero admits nothing but an exact zero.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalstand.scorers.numeric import close_to, parse_number


class TestParsingWhatAModelActuallyReturns:
    @pytest.mark.parametrize(
        "output,expected",
        [
            ("42", 42.0),
            ("42.5", 42.5),
            ("  42  ", 42.0),
            ("-5", -5.0),
            ("+5", 5.0),
            ("0", 0.0),
            (".5", 0.5),
            ("42.", 42.0),
            ("1e3", 1000.0),
            ("3.2e4", 32000.0),
            ("2E-3", 0.002),
        ],
    )
    def test_bare_numbers(self, output: str, expected: float) -> None:
        assert parse_number(output) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "output,expected",
        [
            ("The answer is 42.", 42.0),
            ("about 42", 42.0),
            ("42 degrees", 42.0),
            ("$100", 100.0),
            ("100%", 100.0),
            ("1,000", 1000.0),
            ("1,234,567.89", 1234567.89),
            ("-5 degrees below", -5.0),
        ],
    )
    def test_numbers_embedded_in_prose(self, output: str, expected: float) -> None:
        """A model asked for a number rarely returns a bare one. Refusing these
        would make the scorer useless for the shape answers actually take."""
        assert parse_number(output) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "output,expected",
        [
            ("It costs 19.99.", 19.99),
            ("About 1,000.", 1000.0),
            ("$19.99.", 19.99),
            ("The answer is 42.", 42.0),
            ("19.99.", 19.99),
        ],
    )
    def test_a_number_ending_a_sentence(self, output: str, expected: float) -> None:
        """The commonest shape a model produces, and the one the first version
        of the pattern got wrong.

        A single `(?![\\w.])` lookahead rejected any number followed by a full
        stop, so "It costs 19.99." parsed as nothing at all. Found by running
        the scorer through the real runner, not by reading the regex — the unit
        tests all passed, because every example I had thought to write used a
        number that did not already contain a decimal point.
        """
        assert parse_number(output) == pytest.approx(expected)

    @pytest.mark.parametrize("output", ["1.2.3", "192.168.1.1", "v1.2.3"])
    def test_dotted_sequences_are_still_refused(self, output: str) -> None:
        """The other half of that fix: a trailing dot may end a sentence, but a
        dot followed by more digits means this was never a single number."""
        assert parse_number(output) is None

    @pytest.mark.parametrize("value,expected", [(42, 42.0), (42.5, 42.5), (0, 0.0), (-3, -3.0)])
    def test_actual_numbers_pass_straight_through(self, value: Any, expected: float) -> None:
        assert parse_number(value) == pytest.approx(expected)


class TestRefusingRatherThanGuessing:
    """The governing rule. Every case here could be given an answer by picking
    a candidate, and every such answer would be invented."""

    @pytest.mark.parametrize(
        "output,why",
        [
            ("42 or 43", "two candidates; picking one invents a commitment"),
            ("between 10 and 20", "a range is not a value"),
            ("version 2 of 3", "neither number is the answer"),
            ("5 + 5", "an unevaluated expression, not the number 5"),
            ("42 or 42", "restated, but still an output that hedged"),
        ],
    )
    def test_ambiguous_output_is_unparseable(self, output: str, why: str) -> None:
        assert parse_number(output) is None, why

    @pytest.mark.parametrize("output", ["", "   ", "abc", "forty two", "no answer"])
    def test_output_with_no_number_is_unparseable(self, output: str) -> None:
        assert parse_number(output) is None

    def test_none_is_unparseable(self) -> None:
        """A task that fell through without returning."""
        assert parse_number(None) is None

    @pytest.mark.parametrize("output", ["nan", "NaN", "inf", "-inf", "Infinity"])
    def test_nan_and_infinity_are_refused(self, output: str) -> None:
        """`float("nan")` succeeds, and a NaN then compares false against every
        tolerance — so letting it through would score 0.0 and read as a wrong
        answer rather than an unreadable one. Neither is a measurement."""
        assert parse_number(output) is None

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nan_and_infinity_are_refused_as_floats_too(self, value: float) -> None:
        assert parse_number(value) is None

    @pytest.mark.parametrize("value", [True, False])
    def test_a_bool_is_not_a_number(self, value: bool) -> None:
        """`bool` subclasses `int`, so `float(True)` is 1.0. A task returning
        True would otherwise be scored as the number one, which is a fact about
        the type system rather than an answer."""
        assert parse_number(value) is None

    @pytest.mark.parametrize("output", ["a5", "5a", "v2", "abc123def"])
    def test_digits_inside_a_token_are_not_numbers(self, output: str) -> None:
        """ "v2" is a name, not a measurement."""
        assert parse_number(output) is None


class TestTolerance:
    def test_the_default_demands_an_exact_match(self) -> None:
        """A user who forgets to set a tolerance must not be silently given
        slack they did not ask for."""
        exact = close_to()
        assert exact(42, 42).passed is True
        assert exact(42.001, 42).passed is False

    def test_relative_tolerance_scales_with_the_expected_value(self) -> None:
        one_percent = close_to(rel_tol=0.01)
        assert one_percent(101, 100).passed is True
        assert one_percent(102, 100).passed is False
        # The same 1% is a much larger window around a larger number.
        assert one_percent(10_050, 10_000).passed is True

    def test_absolute_tolerance_does_not_scale(self) -> None:
        half = close_to(abs_tol=0.5)
        assert half(42.4, 42).passed is True
        assert half(42.6, 42).passed is False
        assert half(1_000_000.4, 1_000_000).passed is True

    def test_absolute_tolerance_is_the_only_one_that_works_around_zero(self) -> None:
        """Any percentage of zero is zero, so a relative tolerance around an
        expected value of 0 admits nothing but an exact 0. This is the trap the
        scorer exists to keep users out of."""
        assert close_to(rel_tol=0.5)(0.001, 0.0).passed is False
        assert close_to(abs_tol=0.01)(0.001, 0.0).passed is True

    def test_both_tolerances_together_take_whichever_is_looser(self) -> None:
        both = close_to(rel_tol=0.01, abs_tol=5.0)
        assert both(4.0, 0.0).passed is True, "abs_tol should carry this one"
        assert both(101.0, 100.0).passed is True, "rel_tol should carry this one"

    def test_a_negative_tolerance_is_rejected_when_the_scorer_is_built(self) -> None:
        """Caught at import rather than mid-run: a tolerance that can never be
        satisfied should not be discovered after a suite has spent money."""
        with pytest.raises(ValueError, match="cannot be negative"):
            close_to(rel_tol=-0.1)
        with pytest.raises(ValueError, match="cannot be negative"):
            close_to(abs_tol=-1.0)

    def test_sign_is_respected(self) -> None:
        """-5 is not 5. The scorer must not compare magnitudes."""
        assert close_to(abs_tol=0.5)(-5, 5).passed is False
        assert close_to(abs_tol=0.5)(-5, -5).passed is True


class TestWhatTheScoreSays:
    def test_a_match_sets_passed(self) -> None:
        """Binary: the number is either within tolerance or it is not, so the
        scorer genuinely knows pass from fail."""
        result = close_to(abs_tol=1.0)(42, 42)
        assert result.value == 1.0
        assert result.passed is True

    def test_a_miss_sets_passed_false(self) -> None:
        result = close_to()(41, 42)
        assert result.value == 0.0
        assert result.passed is False

    def test_an_unreadable_output_says_so_in_metadata(self) -> None:
        """Distinguished from a wrong number: the task may have answered
        correctly in a form this scorer cannot read, and only the metadata tells
        the user which of the two happened."""
        result = close_to()("forty two", 42)
        assert result.passed is False
        assert "no single number" in result.metadata["reason"]

    def test_an_unreadable_expected_value_says_so_too(self) -> None:
        """A Case whose expected value is not a number is a dataset problem, not
        a task failure, and the report has to be able to tell them apart."""
        result = close_to()(42, "not a number")
        assert result.passed is False
        assert "expected value is not a number" in result.metadata["reason"]

    def test_the_parsed_number_and_difference_are_recorded(self) -> None:
        """A failing numeric case is unactionable without knowing what the
        scorer actually read out of the output."""
        result = close_to()("The answer is 41.", 42)
        assert result.metadata["parsed_output"] == pytest.approx(41.0)
        assert result.metadata["expected"] == pytest.approx(42.0)
        assert result.metadata["difference"] == pytest.approx(-1.0)

    def test_the_tolerances_are_recorded(self) -> None:
        """So a surprising result can be checked against what was asked for."""
        result = close_to(rel_tol=0.01, abs_tol=0.5)(42, 42)
        assert result.metadata["rel_tol"] == 0.01
        assert result.metadata["abs_tol"] == 0.5


class TestTheAwkwardInputs:
    """Task 4.7's three: happy path, empty output, and a None expected."""

    def test_an_empty_output_does_not_raise(self) -> None:
        assert close_to()("", 42).passed is False

    def test_a_none_expected_does_not_raise(self) -> None:
        assert close_to()(42, None).passed is False

    def test_a_none_output_does_not_raise(self) -> None:
        assert close_to()(None, 42).passed is False

    def test_both_none_is_still_a_failure_not_a_match(self) -> None:
        """Two unreadable values are not a match. Scoring 1.0 here would mark a
        task that returned nothing, against a Case expecting nothing, as a
        success — a false pass built out of two absences."""
        assert close_to()(None, None).passed is False

    def test_a_string_expected_value_is_parsed_like_an_output(self) -> None:
        """Cases are often written with string expected values."""
        assert close_to()("42", "42").passed is True

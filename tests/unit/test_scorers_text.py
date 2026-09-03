"""String and fuzzy scorers (tasks 4.2 and 4.3).

Task 4.7 asks every scorer to cover the happy path, empty output, and a `None`
expected value. Those three are the ones that bite in practice: a model that
returns nothing, and a Case with no reference answer, both reach a scorer that
was written thinking about neither.

The distinction these tests exist to protect is which scorers set `passed`.
Binary scorers know pass from fail and say so; continuous ones report a position
on a scale and must leave the flag absent, because nobody has told them where
the line is.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalstand.scorers.fuzzy import levenshtein, ratio
from evalstand.scorers.text import contains, exact, normalise, normalised_exact, regex_match

BINARY = [exact, normalised_exact, contains]
CONTINUOUS = [levenshtein, ratio]


class TestExact:
    def test_identical_values_score_one(self) -> None:
        assert exact("Paris", "Paris").value == 1.0

    def test_different_values_score_zero(self) -> None:
        assert exact("Paris", "Lyon").value == 0.0

    def test_it_does_not_coerce_types(self) -> None:
        """`exact` is the scorer for when you mean exactly. Quietly treating 1
        and "1" as equal would make it a different scorer, and leave a user with
        no way to ask the strict question."""
        assert exact(1, "1").passed is False

    def test_it_is_case_sensitive(self) -> None:
        """The whole reason `normalised_exact` exists."""
        assert exact("paris", "Paris").passed is False


class TestNormalisedExact:
    @pytest.mark.parametrize(
        "output",
        ["Paris", "paris", "PARIS", "  Paris  ", "Paris.", "«Paris»", "Paris!"],
    )
    def test_formatting_differences_are_not_wrong_answers(self, output: str) -> None:
        assert normalised_exact(output, "Paris").passed is True

    def test_a_different_word_still_fails(self) -> None:
        """Normalisation must not be so eager it stops distinguishing answers."""
        assert normalised_exact("Lyon", "Paris").passed is False

    def test_unicode_forms_are_folded(self) -> None:
        """A composed é and an e with a combining accent are the same answer.
        A model's choice of Unicode encoding is not a fact about correctness."""
        # Built from codepoints: the two spellings render identically, so a
        # literal pair would read as a duplicated line.
        composed = "caf" + chr(0x00E9)  # e-acute as a single codepoint
        decomposed = "cafe" + chr(0x0301)  # e + COMBINING ACUTE ACCENT
        assert composed != decomposed, "the two spellings must really differ"
        assert normalised_exact(composed, decomposed).passed is True

    def test_full_width_digits_fold_to_ascii(self) -> None:
        # Built from codepoints rather than written literally: FULLWIDTH DIGIT
        # ONE is visually identical to ASCII "1", so a literal would look like a
        # typo to every reader and to the linter. U+FF10 is FULLWIDTH DIGIT ZERO.
        full_width = "".join(chr(0xFF10 + digit) for digit in (1, 2, 3))
        assert full_width != "123", "the two spellings must really differ"
        assert normalised_exact(full_width, "123").passed is True

    def test_internal_whitespace_is_collapsed_not_removed(self) -> None:
        """Collapsing runs of whitespace is formatting; deleting it entirely
        would make "arc tic" equal "arctic", which is a different answer."""
        assert normalised_exact("New   York", "New York").passed is True
        assert normalised_exact("NewYork", "New York").passed is False


class TestContains:
    def test_an_embedded_answer_is_found(self) -> None:
        """Insisting on an exact match here would measure verbosity rather
        than correctness."""
        assert contains("The capital is Paris.", "Paris").passed is True

    def test_a_missing_answer_is_not_found(self) -> None:
        assert contains("The capital is Lyon.", "Paris").passed is False

    def test_it_normalises_before_looking(self) -> None:
        assert contains("the capital is PARIS", "paris").passed is True

    def test_an_empty_expected_value_scores_zero(self) -> None:
        """Every string contains the empty string. Scoring 1.0 on that vacuous
        truth would silently mark an entire suite as passing — the most
        damaging way for a scorer to be wrong, because it looks like success."""
        result = contains("anything at all", "")
        assert result.passed is False
        assert "vacuous" in result.metadata["reason"]

    def test_a_none_expected_value_scores_zero(self) -> None:
        """A Case with no reference answer cannot be checked for containment."""
        assert contains("anything", None).passed is False


class TestRegexMatch:
    def test_a_search_finds_a_pattern_anywhere(self) -> None:
        assert regex_match(r"\d{4}")("the year was 1999 I think").passed is True

    def test_full_requires_the_whole_output(self) -> None:
        """`search` for `\\d+` is satisfied by any output containing a digit,
        which is rarely what "matches" means."""
        assert regex_match(r"\d+", full=True)("42").passed is True
        assert regex_match(r"\d+", full=True)("about 42 or so").passed is False

    def test_a_precompiled_pattern_is_accepted(self) -> None:
        import re

        assert regex_match(re.compile(r"^yes$", re.IGNORECASE))("YES").passed is True

    def test_the_pattern_is_recorded_in_metadata(self) -> None:
        """A failing regex case is unactionable without knowing what was
        tested."""
        assert regex_match(r"\d+")("no digits").metadata["pattern"] == r"\d+"

    def test_an_empty_output_does_not_crash(self) -> None:
        assert regex_match(r"\d+")("").passed is False


class TestLevenshtein:
    def test_the_documented_value(self) -> None:
        """Task 4.3's acceptance criterion, and the value in the docs: 3 edits
        over 7 characters."""
        assert levenshtein("kitten", "sitting").value == pytest.approx(0.5714285, abs=1e-6)

    def test_identical_strings_score_one(self) -> None:
        assert levenshtein("Paris", "Paris").value == 1.0

    def test_completely_different_strings_score_low(self) -> None:
        assert levenshtein("abc", "xyz").value == 0.0

    def test_a_near_miss_scores_near_one(self) -> None:
        """The reason this is the default scorer: a nearly-right answer scores
        nearly 1.0 instead of falling off the cliff `exact` presents."""
        assert levenshtein("Pariss", "Paris").value > 0.8

    def test_two_empty_strings_score_one(self) -> None:
        """Nothing was expected and nothing arrived."""
        assert levenshtein("", "").value == 1.0

    def test_an_empty_output_against_text_scores_zero(self) -> None:
        assert levenshtein("", "Paris").value == 0.0

    def test_the_distance_is_recorded(self) -> None:
        assert levenshtein("kitten", "sitting").metadata["distance"] == 3


class TestRatio:
    def test_identical_strings_score_one(self) -> None:
        assert ratio("Paris", "Paris").value == 1.0

    def test_it_stays_within_the_score_range(self) -> None:
        """`fuzz.ratio` reports a percentage; a missing division by 100 would
        produce 61.5 and be rejected by the Score model."""
        assert 0.0 <= ratio("kitten", "sitting").value <= 1.0

    def test_unrelated_strings_score_low(self) -> None:
        assert ratio("abc", "xyz").value == 0.0


class TestContinuousScorersNeverClaimAVerdict:
    """The rule from CONTEXT.md, asserted on every continuous scorer.

    A similarity of 0.62 is a position on a scale. Where the pass line sits is a
    judgement about the task that nobody has given the scorer, so setting the
    flag would put a verdict in its mouth.
    """

    @pytest.mark.parametrize("scorer_fn", CONTINUOUS)
    @pytest.mark.parametrize("output,expected", [("Paris", "Paris"), ("x", "y"), ("ab", "abc")])
    def test_passed_is_always_absent(self, scorer_fn: Any, output: str, expected: str) -> None:
        assert scorer_fn(output, expected).passed is None


class TestBinaryScorersAlwaysClaimAVerdict:
    """The other half: a binary scorer genuinely knows, so it must say so.

    Without this, a scorer could silently stop setting the flag and every case
    would report as unjudged rather than failing.
    """

    @pytest.mark.parametrize("scorer_fn", BINARY)
    def test_passed_is_set_both_ways(self, scorer_fn: Any) -> None:
        assert scorer_fn("Paris", "Paris").passed is True
        assert scorer_fn("Lyon", "Paris").passed is False

    @pytest.mark.parametrize("scorer_fn", BINARY)
    def test_the_value_agrees_with_the_verdict(self, scorer_fn: Any) -> None:
        """A binary scorer reporting passed=True with value 0.0 would make the
        table and the pass count contradict each other."""
        assert scorer_fn("Paris", "Paris").value == 1.0
        assert scorer_fn("Lyon", "Paris").value == 0.0


class TestTheAwkwardInputsEveryScorerMeets:
    """Task 4.7: empty output and a `None` expected value.

    A model that returns nothing and a Case with no reference answer both reach
    scorers written with neither in mind, and a scorer that raises produces an
    errored Score — which is honest, but a scorer that simply answers is better.
    """

    @pytest.mark.parametrize("scorer_fn", BINARY + CONTINUOUS)
    def test_an_empty_output_does_not_raise(self, scorer_fn: Any) -> None:
        assert scorer_fn("", "Paris").value is not None

    @pytest.mark.parametrize("scorer_fn", BINARY + CONTINUOUS)
    def test_a_none_expected_does_not_raise(self, scorer_fn: Any) -> None:
        assert scorer_fn("Paris", None).value is not None

    @pytest.mark.parametrize("scorer_fn", BINARY + CONTINUOUS)
    def test_a_none_output_does_not_raise(self, scorer_fn: Any) -> None:
        """A task that fell through without returning yields None."""
        assert scorer_fn(None, "Paris").value is not None

    @pytest.mark.parametrize("scorer_fn", BINARY + CONTINUOUS)
    def test_a_non_string_output_does_not_raise(self, scorer_fn: Any) -> None:
        """Tasks returning numbers are ordinary, not exotic."""
        assert scorer_fn(42, 42).value is not None

    def test_none_is_not_silently_equal_to_the_string_none(self) -> None:
        """Coercion is a convenience, but `str(None)` is the word "None".

        A task that fell through without returning would otherwise score a
        perfect match against a Case whose expected value is the literal text
        "None" — a false pass, which is the most damaging thing a scorer can
        produce because it looks like success and invites no investigation.
        """
        assert normalised_exact(None, "None").passed is False

    def test_none_and_the_empty_string_are_treated_alike(self) -> None:
        """Both mean "no answer", so they match each other. Deliberate, and
        consistent with `levenshtein("", "")` scoring 1.0: nothing was expected
        and nothing arrived."""
        assert normalised_exact(None, "").passed is True
        assert levenshtein(None, "").value == 1.0


class TestNormalise:
    """The shared helper. Tested directly because four scorers depend on it and
    a change here silently changes all of them."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("  Hello,  World!  ", "hello world"),
            ("MIXED case", "mixed case"),
            ("a\t\nb", "a b"),
            ("", ""),
            ("...", ""),
        ],
    )
    def test_folding(self, text: str, expected: str) -> None:
        assert normalise(text) == expected

    def test_it_is_idempotent(self) -> None:
        """Normalising twice must not differ from once, or `contains` would
        compare a once-folded needle against a twice-folded haystack."""
        once = normalise("  Héllo,  World!  ")
        assert normalise(once) == once

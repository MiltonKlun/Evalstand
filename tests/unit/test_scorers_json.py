"""The structured scorer (task 4.5).

An extraction task rarely gets everything right or everything wrong, and a
single pass/fail throws away the only information worth having: *which* field
was wrong. So the value is a macro-average and the metadata names every failure.

The three decisions these tests protect:

- the denominator is the **expected** keys, so a volunteered field is not a
  wrong answer;
- nested dicts flatten to dotted paths, so one wrong leaf costs one field
  rather than a whole subtree;
- a **missing** field is reported separately from a **wrong** one, because they
  call for different fixes.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalstand.scorers.json_field import flatten, json_fields, parse_object

score = json_fields()
strict = json_fields(require_all=True)


class TestTheAcceptanceCriterion:
    def test_three_of_five_fields_scores_point_six(self) -> None:
        """Task 4.5's stated acceptance, verbatim."""
        expected = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        output = {"a": 1, "b": 2, "c": 3, "d": 99, "e": 98}

        result = score(output, expected)

        assert result.value == pytest.approx(0.6)
        assert result.metadata["wrong"] == ["d", "e"]

    def test_the_failing_field_names_are_in_metadata(self) -> None:
        """The whole point of a per-field breakdown. Without the names the user
        knows only that something was wrong, which a single float already said."""
        result = score({"name": "Ada", "city": "Lyon"}, {"name": "Ada", "city": "Paris"})

        assert result.metadata["wrong"] == ["city"]
        assert result.metadata["matched"] == ["name"]


class TestTheDenominator:
    def test_a_volunteered_field_is_not_a_wrong_answer(self) -> None:
        """The Case defines what was asked for. Counting extra keys would let a
        verbose model score worse than a terse one that answered exactly as
        badly, which measures style rather than correctness."""
        result = score({"a": 1, "b": 2, "chatty": "extra"}, {"a": 1, "b": 2})

        assert result.value == 1.0
        assert result.metadata["unexpected"] == ["chatty"]

    def test_unexpected_fields_are_still_reported(self) -> None:
        """Not counted, but not hidden: a model inventing fields is worth
        knowing about even though it is not scored."""
        result = score({"a": 1, "x": 9, "y": 8}, {"a": 1})
        assert result.metadata["unexpected"] == ["x", "y"]

    def test_every_field_counts_equally(self) -> None:
        """A macro-average: a one-character field weighs the same as a
        paragraph, because the Case asked for both equally."""
        expected = {"short": "x", "long": "a much longer value " * 20}
        result = score({"short": "x", "long": "wrong"}, expected)
        assert result.value == pytest.approx(0.5)


class TestMissingIsNotTheSameAsWrong:
    def test_a_missing_field_is_reported_as_missing(self) -> None:
        result = score({"a": 1}, {"a": 1, "b": 2})

        assert result.metadata["missing"] == ["b"]
        assert result.metadata["wrong"] == []

    def test_a_wrong_field_is_reported_as_wrong(self) -> None:
        result = score({"a": 1, "b": 99}, {"a": 1, "b": 2})

        assert result.metadata["wrong"] == ["b"]
        assert result.metadata["missing"] == []

    def test_both_score_zero_for_that_field(self) -> None:
        """They differ in what they tell the user, not in what they are worth:
        an unanswered field is no more correct than a wrongly answered one."""
        assert score({"a": 1}, {"a": 1, "b": 2}).value == pytest.approx(0.5)
        assert score({"a": 1, "b": 9}, {"a": 1, "b": 2}).value == pytest.approx(0.5)

    def test_a_null_value_is_present_not_missing(self) -> None:
        """A model that answered `null` did answer. Reporting it as missing
        would send the user looking for a field the model actually returned."""
        result = score({"a": None}, {"a": 1})
        assert result.metadata["wrong"] == ["a"]
        assert result.metadata["missing"] == []


class TestNesting:
    def test_nested_keys_flatten_to_dotted_paths(self) -> None:
        assert flatten({"person": {"name": "Ada", "born": 1815}, "ok": True}) == {
            "person.name": "Ada",
            "person.born": 1815,
            "ok": True,
        }

    def test_one_wrong_leaf_costs_one_field_not_the_subtree(self) -> None:
        """Comparing subtrees whole would score a 1-of-10 miss the same as a
        10-of-10 miss, which is the information the breakdown exists to keep."""
        result = score(
            {"person": {"name": "Ada", "born": 1816}},
            {"person": {"name": "Ada", "born": 1815}},
        )

        assert result.value == pytest.approx(0.5)
        assert result.metadata["wrong"] == ["person.born"]

    def test_the_metadata_names_the_full_path(self) -> None:
        """ "person" would not be actionable in a deeply nested object."""
        result = score({"a": {"b": {"c": 1}}}, {"a": {"b": {"c": 2}}})
        assert result.metadata["wrong"] == ["a.b.c"]

    def test_an_empty_nested_dict_stays_a_field(self) -> None:
        """Otherwise a Case expecting `{"meta": {}}` flattens to nothing, and an
        output that omitted `meta` entirely scores a perfect match."""
        assert flatten({"meta": {}}) == {"meta": {}}
        assert score({}, {"meta": {}}).metadata["missing"] == ["meta"]

    def test_a_missing_subtree_reports_every_leaf(self) -> None:
        """So the user sees the size of what is absent, not just its root."""
        result = score({}, {"person": {"name": "Ada", "born": 1815}})
        assert result.metadata["missing"] == ["person.born", "person.name"]


class TestParsingWhatAModelActuallyReturns:
    def test_a_dict_passes_straight_through(self) -> None:
        assert parse_object({"a": 1}) == {"a": 1}

    def test_json_text_is_parsed(self) -> None:
        """A model asked for JSON returns text that looks like JSON."""
        assert parse_object('{"name": "Ada"}') == {"name": "Ada"}

    @pytest.mark.parametrize(
        "fenced",
        [
            '```json\n{"name": "Ada"}\n```',
            '```JSON\n{"name": "Ada"}\n```',
            '```\n{"name": "Ada"}\n```',
            '  ```json\n{"name": "Ada"}\n```  ',
        ],
    )
    def test_a_fenced_block_is_unwrapped(self, fenced: str) -> None:
        """Models fence JSON because that is how they were taught to present
        code. Refusing to unwrap it would report a total failure for an answer
        that was entirely correct — measuring presentation, not content."""
        assert parse_object(fenced) == {"name": "Ada"}

    @pytest.mark.parametrize(
        "value", ["not json", "", "   ", "[1, 2, 3]", "42", '"a string"', "null", None, 42, ["a"]]
    )
    def test_anything_that_is_not_an_object_is_refused(self, value: Any) -> None:
        """A JSON array is valid JSON but has no fields to compare."""
        assert parse_object(value) is None

    def test_an_unparseable_output_scores_zero_and_says_why(self) -> None:
        result = score("I could not produce JSON", {"a": 1, "b": 2})

        assert result.value == 0.0
        assert "not an object" in result.metadata["reason"]
        assert result.metadata["missing"] == ["a", "b"]


class TestFieldComparison:
    @pytest.mark.parametrize(
        "output_value,expected_value,why",
        [
            ("1843", 1843, "a JSON round trip turns numbers into strings"),
            ("ada", "Ada", "case"),
            (" Ada ", "Ada", "surrounding whitespace"),
            ("Paris.", "Paris", "a trailing full stop"),
            (1843, 1843.0, "int against float"),
            (" 1843 ", 1843, "a padded numeric string"),
            (0, 0.0, "zero in both forms"),
        ],
    )
    def test_serialisation_differences_are_not_wrong_answers(
        self, output_value: Any, expected_value: Any, why: str
    ) -> None:
        assert score({"f": output_value}, {"f": expected_value}).value == 1.0, why

    @pytest.mark.parametrize(
        "output_value,expected_value,why",
        [
            ("Lyon", "Paris", "a different answer"),
            (-5, 5, "a sign error is not formatting"),
            ("$100", "100%", "money is not a percentage"),
            (None, "", "a null is not an empty string"),
            ("None", None, "the word None is not a null"),
        ],
    )
    def test_meaningful_differences_still_fail(
        self, output_value: Any, expected_value: Any, why: str
    ) -> None:
        assert score({"f": output_value}, {"f": expected_value}).value == 0.0, why

    def test_lists_compare_element_wise_in_order(self) -> None:
        assert score({"f": [1, 2]}, {"f": [1, 2]}).value == 1.0
        assert score({"f": [2, 1]}, {"f": [1, 2]}).value == 0.0
        assert score({"f": [1]}, {"f": [1, 2]}).value == 0.0

    def test_list_elements_are_normalised_too(self) -> None:
        assert score({"f": ["1", "2"]}, {"f": [1, 2]}).value == 1.0

    @pytest.mark.parametrize(
        "output_value,expected_value",
        [("Paris 1", "Lyon 1"), ("room 5", "floor 5"), ("The answer is 42", "42")],
    )
    def test_a_number_inside_text_does_not_make_two_answers_equal(
        self, output_value: str, expected_value: str
    ) -> None:
        """Numeric comparison applies only when both values *are* numbers.

        Reaching for `numeric.parse_number` here — which finds a number anywhere
        in the text — made "Paris 1" match "Lyon 1": two different answers
        equal because they happened to contain the same digit. Found by probing
        the change rather than by running the tests, which all passed.
        """
        assert score({"f": output_value}, {"f": expected_value}).value == 0.0

    def test_a_bool_does_not_match_the_number_one(self) -> None:
        """`bool` subclasses `int`, so `float(True)` is 1.0."""
        assert score({"f": True}, {"f": 1}).value == 0.0


class TestTheVerdict:
    def test_a_partial_score_claims_no_verdict(self) -> None:
        """0.6 is a position on a scale. Where the line sits between "good
        enough" and "not" is a judgement about the task that nobody has given
        the scorer, so setting the flag would put a verdict in its mouth."""
        assert score({"a": 1, "b": 9}, {"a": 1, "b": 2}).passed is None

    def test_a_perfect_score_still_claims_no_verdict_by_default(self) -> None:
        """Consistency matters more than convenience here: a scorer that sets
        the flag only sometimes is harder to reason about than one that never
        does."""
        assert score({"a": 1}, {"a": 1}).passed is None

    def test_require_all_makes_the_verdict_unambiguous(self) -> None:
        """ "every field correct" is something the scorer genuinely knows."""
        assert strict({"a": 1, "b": 2}, {"a": 1, "b": 2}).passed is True
        assert strict({"a": 1, "b": 9}, {"a": 1, "b": 2}).passed is False

    def test_require_all_fails_on_a_missing_field_too(self) -> None:
        assert strict({"a": 1}, {"a": 1, "b": 2}).passed is False

    def test_require_all_ignores_unexpected_fields(self) -> None:
        """Still judged on what was asked for, not on what was volunteered."""
        assert strict({"a": 1, "extra": 9}, {"a": 1}).passed is True


class TestDegenerateInputs:
    def test_an_empty_expected_object_is_an_error_not_a_score(self) -> None:
        """0 of 0 has no defensible value: 1.0 claims everything matched and
        0.0 that nothing did, when in truth nothing was asked. An errored Score
        is excluded from means rather than dragging them to an invented number.
        """
        result = score({"a": 1}, {})

        assert result.error is not None
        assert result.counts_towards_mean is False

    def test_a_non_object_expected_value_is_an_error(self) -> None:
        """A Case whose expected value is not an object is a dataset problem,
        not a task failure, and the two must not be conflated."""
        result = score({"a": 1}, "not an object")

        assert result.error is not None
        assert result.counts_towards_mean is False

    def test_a_none_output_scores_zero_rather_than_raising(self) -> None:
        assert score(None, {"a": 1}).value == 0.0

    def test_an_empty_output_object_misses_every_field(self) -> None:
        result = score({}, {"a": 1, "b": 2})
        assert result.value == 0.0
        assert result.metadata["missing"] == ["a", "b"]

    def test_the_field_count_is_reported(self) -> None:
        """So a reader can tell 0.5 of two fields from 0.5 of two hundred."""
        assert score({"a": 1}, {"a": 1, "b": 2}).metadata["field_count"] == 2

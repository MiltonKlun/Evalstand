"""Task 4.7: every built-in scorer meets the awkward inputs.

The per-scorer test files each cover their own edge cases. This file is the
*inventory*: it enumerates the public scorers and asserts the three the plan
names — happy path, empty output, `None` expected — against every one of them.

The reason to have it as well as the per-scorer files is that those can only
test the scorers someone remembered to write tests for. This one fails when a
scorer is *added* without them, which is the failure mode a checklist in a plan
document cannot catch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import evalstand.scorers as scorers_module
from evalstand.models import Case, Score
from evalstand.scorers import (
    close_to,
    contains,
    exact,
    factuality,
    json_fields,
    judge,
    levenshtein,
    normalised_exact,
    ratio,
    regex_match,
)
from evalstand.scorers.base import call_scorer

CASE = Case(id="c1", input="a question", expected="Paris")

# Every scorer, already built where it is a factory, paired with an expected
# value it should score well and one it should not. Ready-to-call so the tests
# below do not have to know which are factories.
BUILT: dict[str, tuple[Any, Any, Any]] = {
    # name: (scorer, an output that should score 1.0, the expected value)
    "exact": (exact, "Paris", "Paris"),
    "normalised_exact": (normalised_exact, " paris. ", "Paris"),
    "contains": (contains, "The capital is Paris.", "Paris"),
    "regex_match": (regex_match(r"Paris"), "Paris", "Paris"),
    "levenshtein": (levenshtein, "Paris", "Paris"),
    "ratio": (ratio, "Paris", "Paris"),
    "close_to": (close_to(), "42", 42),
    "json_fields": (json_fields(), {"a": 1}, {"a": 1}),
    "factuality": (factuality(), "Paris", "Paris"),
    # A generic judge, built with a rubric whose "A" is worth 1.0 so the
    # patched reply below scores its happy path.
    "judge": (judge(rubric="Is the answer right?", choices={"A": 1.0, "B": 0.0}), "Paris", "Paris"),
}

NEEDS_A_MODEL = {"factuality", "judge"}
"""Judges call a provider, so they are exercised with it patched."""


def _judge_reply(text: str = "C") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 1
    response.usage.total_tokens = 11
    response.model = "gpt-4o-mini"
    return response


async def _call(name: str, output: Any, expected: Any) -> Score:
    """Invoke one scorer through the same adapter the runner uses."""
    scorer = BUILT[name][0]

    if name not in NEEDS_A_MODEL:
        return await call_scorer(scorer, output, expected, CASE)

    async def responder(**_: Any) -> MagicMock:
        # "C" is the factuality rubric's "same details" (1.0); "A" is the
        # generic judge's top choice. Each scorer's happy path needs its own.
        return _judge_reply("C" if name == "factuality" else "A")

    with (
        patch("evalstand.llm.litellm.acompletion", side_effect=responder),
        patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
    ):
        return await call_scorer(scorer, output, expected, CASE)


class TestTheInventoryIsComplete:
    def test_every_exported_scorer_is_in_the_table(self) -> None:
        """The point of this file. A scorer added to the package without a row
        here is a scorer nobody checked against the awkward inputs, and a
        checklist in PLAN.md cannot notice that."""
        exported = {
            name
            for name in scorers_module.__all__
            # Not scorers: the protocol, the decorator, and the two parsers
            # that exist for scorers to share.
            if name not in {"Scorer", "scorer", "parse_number", "parse_object"}
        }

        assert exported == set(BUILT), (
            "a public scorer is missing from this file's table "
            f"(untested: {sorted(exported - set(BUILT))})"
        )

    def test_the_library_is_large_enough_to_ship(self) -> None:
        """Phase 4's exit criterion is eight or more built-in scorers."""
        assert len(BUILT) >= 8


@pytest.mark.parametrize("name", sorted(BUILT))
class TestEveryScorerMeetsTheAwkwardInputs:
    """The three the plan names, applied to every scorer without exception."""

    @pytest.mark.anyio
    async def test_the_happy_path_scores_one(self, name: str) -> None:
        _, good_output, expected = BUILT[name]
        result = await _call(name, good_output, expected)

        assert result.error is None, f"{name} errored on its happy path"
        assert result.value == 1.0

    @pytest.mark.anyio
    async def test_an_empty_output_is_answered_not_raised(self, name: str) -> None:
        """A model that returned nothing is ordinary. A scorer that raises here
        produces an errored Score, which is honest but useless: the case is
        excluded from the mean when it should have counted as a failure."""
        result = await _call(name, "", BUILT[name][2])

        assert result.error is None, f"{name} raised on an empty output: {result.error}"
        assert result.value is not None

    @pytest.mark.anyio
    async def test_a_none_output_is_answered_not_raised(self, name: str) -> None:
        """A task that fell through without returning."""
        result = await _call(name, None, BUILT[name][2])

        assert result.error is None, f"{name} raised on a None output: {result.error}"
        assert result.value is not None

    @pytest.mark.anyio
    async def test_a_none_expected_is_answered_not_raised(self, name: str) -> None:
        """A Case with no reference answer is valid: `expected` is optional, and
        some scorers judge the output alone. Every scorer has to survive one.

        `json_fields` is the exception that proves the rule — it *errors*
        deliberately, because a Case with no object to compare against is a
        dataset problem rather than a task failure, and an errored Score is
        excluded from the mean rather than counted as a zero.
        """
        result = await _call(name, BUILT[name][1], None)

        if name == "json_fields":
            assert result.error is not None
            assert result.counts_towards_mean is False
            return

        assert result.error is None, f"{name} raised on a None expected: {result.error}"
        assert result.value is not None

    @pytest.mark.anyio
    async def test_a_non_string_output_is_answered_not_raised(self, name: str) -> None:
        """Tasks returning numbers, lists and dicts are ordinary, not exotic."""
        for output in (42, 3.5, ["a", "b"], {"k": "v"}, True):
            result = await _call(name, output, BUILT[name][2])
            assert result.error is None, f"{name} raised on {output!r}: {result.error}"


@pytest.mark.parametrize("name", sorted(BUILT))
class TestEveryScorerHonoursTheScoreContract:
    """Invariants that hold across the whole library, not per scorer.

    A scorer breaking one of these produces a Score the rest of the system
    cannot reason about — a mean that is not a mean, or a pass count built from
    verdicts nobody made.
    """

    @pytest.mark.anyio
    async def test_the_value_stays_inside_the_score_range(self, name: str) -> None:
        _, good_output, expected = BUILT[name]
        for output in (good_output, "", None, "something else entirely", 42):
            result = await _call(name, output, expected)
            if result.value is not None:
                assert 0.0 <= result.value <= 1.0, f"{name} returned {result.value}"

    @pytest.mark.anyio
    async def test_it_files_scores_under_its_own_name(self, name: str) -> None:
        """The report groups by `scorer_name`. A scorer using a generic one
        would merge with every other scorer that did the same."""
        result = await _call(name, BUILT[name][1], BUILT[name][2])
        assert result.scorer_name not in {"", "scorer"}

    @pytest.mark.anyio
    async def test_a_verdict_when_given_agrees_with_the_value(self, name: str) -> None:
        """A scorer claiming `passed=True` with a value of 0.0 would make the
        table and the pass count contradict each other."""
        _, good_output, expected = BUILT[name]
        for output in (good_output, "definitely not the answer"):
            result = await _call(name, output, expected)
            if result.passed is True:
                assert result.value == 1.0, f"{name} passed with value {result.value}"
            elif result.passed is False:
                assert result.value == 0.0, f"{name} failed with value {result.value}"

    @pytest.mark.anyio
    async def test_a_continuous_scorer_never_invents_a_verdict(self, name: str) -> None:
        """CONTEXT.md's rule: a Scorer sets the pass flag only when it genuinely
        knows pass from fail. A scorer returning a value strictly between 0 and
        1 does not know where the line sits."""
        result = await _call(name, "a partially right answer", BUILT[name][2])
        if result.value is not None and 0.0 < result.value < 1.0:
            assert result.passed is None, f"{name} claimed a verdict at {result.value}"

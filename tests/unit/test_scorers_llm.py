"""LLM-as-judge scorers (task 4.6).

The acceptance criterion — a judge's LLM call appears in the case's trace tree —
is the one that matters most, because it is what stops LLM-as-judge from being
an invisible line on the bill. It is asserted through the real runner rather
than by inspecting the judge in isolation.

The other theme is the boundary between the judge's opinion and a measurement.
A judge is a second model guessing; these scorers are unvalidated by design, and
an unreadable judgement must never be laundered into a score.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evalstand.api import Eval
from evalstand.llm import acall
from evalstand.models import Case, Score
from evalstand.runner import RunConfig, run_eval
from evalstand.scorers.llm import FACTUALITY_CHOICES, factuality, judge
from evalstand.scorers.llm import _read_choice as read_choice

CASE = Case(id="q1", input="Who wrote the first algorithm?", expected="Ada Lovelace")
CHOICES = {"A": 1.0, "B": 0.5, "C": 0.0}


def _reply(text: str, *, prompt_tokens: int = 40, completion_tokens: int = 1) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response.model = "gpt-4o-mini"
    return response


async def _run_judge(scorer: Any, reply: str, output: Any = "an answer") -> Score:
    async def responder(**_: Any) -> MagicMock:
        return _reply(reply)

    with (
        patch("evalstand.llm.litellm.acompletion", side_effect=responder),
        patch("evalstand.llm.litellm.completion_cost", return_value=0.0001),
    ):
        return await scorer(output, CASE.expected, CASE)


class TestTheAcceptanceCriterion:
    """Task 4.6: a judge's LLM call appears in the case's trace tree.

    Asserted end to end through `run_eval`, because the mechanism being checked
    is that scoring happens inside the case's trace collector. A unit test of
    the judge alone would pass even if the runner scored outside it.
    """

    @pytest.mark.anyio
    async def test_the_judges_call_is_in_the_case_trace_tree(self) -> None:
        calls: list[Any] = []

        async def responder(**kwargs: Any) -> MagicMock:
            calls.append(kwargs)
            # The task's call first, then the judge's.
            return (
                _reply("Ada Lovelace wrote it.", prompt_tokens=20, completion_tokens=8)
                if len(calls) == 1
                else _reply("C", prompt_tokens=50, completion_tokens=1)
            )

        async def task(question: str) -> str:
            return (await acall("gpt-4o-mini", [{"role": "user", "content": question}])).text

        declared = Eval(
            name="judged",
            cases=[CASE],
            task=task,
            scorers=[factuality()],
            filepath="f",
        )

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=responder),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0001),
        ):
            run = await run_eval(declared, RunConfig())

        result = run.results[0]
        assert len(result.traces) == 2, "the judge's call did not reach the trace tree"

    @pytest.mark.anyio
    async def test_the_judges_tokens_and_cost_count_towards_the_case(self) -> None:
        """The reason the trace matters. A judge that ran off the books would
        let a suite quietly cost twice what its summary reports."""
        calls: list[Any] = []

        async def responder(**kwargs: Any) -> MagicMock:
            calls.append(kwargs)
            return (
                _reply("answer", prompt_tokens=20, completion_tokens=8)
                if len(calls) == 1
                else _reply("C", prompt_tokens=50, completion_tokens=1)
            )

        async def task(question: str) -> str:
            return (await acall("gpt-4o-mini", [{"role": "user", "content": question}])).text

        declared = Eval(
            name="judged", cases=[CASE], task=task, scorers=[factuality()], filepath="f"
        )

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=responder),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0001),
        ):
            run = await run_eval(declared, RunConfig())

        result = run.results[0]
        assert result.input_tokens == 70, "20 from the task plus 50 from the judge"
        assert result.output_tokens == 9
        assert result.cost_usd == pytest.approx(0.0002)
        assert run.model_calls == 2


class TestTheJudgeIsNeverLaundered:
    @pytest.mark.anyio
    async def test_an_unreadable_reply_is_an_errored_score(self) -> None:
        """Not 0.0. The task may have been fine and the *judge* is what failed;
        scoring zero would report a verdict nobody reached."""
        result = await _run_judge(judge(rubric="r", choices=CHOICES), "I'm not sure, sorry")

        assert result.error is not None
        assert result.value is None
        assert result.counts_towards_mean is False

    @pytest.mark.anyio
    async def test_an_ambiguous_reply_is_refused_rather_than_resolved(self) -> None:
        """Naming two choices is not a verdict. Taking the first would invent
        one, the same rule that makes `parse_number` refuse "42 or 43"."""
        result = await _run_judge(judge(rubric="r", choices=CHOICES), "Either A or B, hard to say")

        assert result.error is not None

    @pytest.mark.anyio
    async def test_an_empty_reply_is_an_errored_score(self) -> None:
        assert (await _run_judge(judge(rubric="r", choices=CHOICES), "")).error is not None

    @pytest.mark.anyio
    async def test_the_error_quotes_what_the_judge_actually_said(self) -> None:
        """Otherwise a user cannot tell a refusing judge from a broken prompt."""
        result = await _run_judge(judge(rubric="r", choices=CHOICES), "I decline to grade this")

        assert "decline to grade" in (result.error or "")


class TestTheUnvalidatedCaveat:
    @pytest.mark.anyio
    async def test_every_judge_score_is_marked_unvalidated(self) -> None:
        """The caveat travels with the number into any report or database row
        that quotes it. A caveat separated from its figure stops being one."""
        result = await _run_judge(judge(rubric="r", choices=CHOICES), "A")

        assert result.metadata["unvalidated"] is True

    @pytest.mark.anyio
    async def test_factuality_is_marked_too(self) -> None:
        result = await _run_judge(factuality(), "C")
        assert result.metadata["unvalidated"] is True

    @pytest.mark.anyio
    async def test_the_judges_own_words_are_kept(self) -> None:
        """So a surprising score can be read rather than merely disbelieved."""
        result = await _run_judge(judge(rubric="r", choices=CHOICES), "B, on balance")

        assert result.metadata["judge_reply"] == "B, on balance"
        assert result.metadata["choice"] == "B"

    @pytest.mark.anyio
    async def test_the_judging_model_is_recorded(self) -> None:
        """Two runs judged by different models are not comparable, so the
        report has to be able to tell them apart."""
        result = await _run_judge(judge(rubric="r", choices=CHOICES), "A")
        assert result.metadata["judge_model"] == "gpt-4o-mini"


class TestTheChoiceMappingBelongsToTheCaller:
    @pytest.mark.anyio
    async def test_the_caller_decides_what_each_choice_is_worth(self) -> None:
        """Whether a partial answer is worth 0.5 or 0.2 is a judgement about
        the task. Letting the model decide would put a number in its mouth that
        nobody chose."""
        lenient = judge(rubric="r", choices={"A": 1.0, "B": 0.9})
        strict = judge(rubric="r", choices={"A": 1.0, "B": 0.1})

        assert (await _run_judge(lenient, "B")).value == pytest.approx(0.9)
        assert (await _run_judge(strict, "B")).value == pytest.approx(0.1)

    def test_a_judge_with_no_choices_is_rejected_when_built(self) -> None:
        """Caught at import rather than after a suite has paid for a judge that
        could never return a score."""
        with pytest.raises(ValueError, match="at least one choice"):
            judge(rubric="r", choices={})

    @pytest.mark.parametrize("value", [1.5, -0.1, 2.0])
    def test_a_choice_outside_the_score_range_is_rejected_when_built(self, value: float) -> None:
        """A Score lives in [0, 1]; discovering that mid-run would turn a
        mapping mistake into a mysterious validation error per case."""
        with pytest.raises(ValueError, match="outside the score range"):
            judge(rubric="r", choices={"A": value})


class TestReadingTheJudgesReply:
    @pytest.mark.parametrize("reply", ["A", "A.", "(A)", "a", "Answer: A", "**A**", " A ", "A\n"])
    def test_packaging_is_forgiven(self, reply: str) -> None:
        """A model told to answer "A" may present it several ways. Refusing
        those throws away a judgement that was perfectly clear."""
        assert read_choice(reply, CHOICES) == "A"

    @pytest.mark.parametrize(
        "reply",
        ["A or B", "I'd say A, though B is close", "", "   ", "maybe?", "D", "Neither A nor C"],
    )
    def test_ambiguous_or_unknown_replies_are_refused(self, reply: str) -> None:
        assert read_choice(reply, CHOICES) is None

    def test_a_label_inside_a_word_is_not_a_choice(self) -> None:
        """ "Absolutely" begins with A but names no choice."""
        assert read_choice("Absolutely", CHOICES) is None

    @pytest.mark.parametrize(
        "reply,expected", [("yes", "yes"), ("Yes.", "yes"), ("NO", "no"), ("yes, clearly", "yes")]
    )
    def test_word_labels_work_and_ignore_case(self, reply: str, expected: str) -> None:
        """A rubric may label its choices with words rather than letters, and a
        capital is packaging just as a full stop is."""
        assert read_choice(reply, {"yes": 1.0, "no": 0.0}) == expected

    def test_a_reply_naming_both_word_labels_is_refused(self) -> None:
        assert read_choice("I think yes but really no", {"yes": 1.0, "no": 0.0}) is None


class TestFactuality:
    def test_consistency_scores_one_and_only_disagreement_scores_zero(self) -> None:
        """A submission that says less than the reference, or more, is still
        factually consistent with it. Penalising that would measure verbosity
        rather than truth."""
        assert FACTUALITY_CHOICES["A"] == 1.0, "a subset is still consistent"
        assert FACTUALITY_CHOICES["B"] == 1.0, "a superset is still consistent"
        assert FACTUALITY_CHOICES["C"] == 1.0, "the same details"
        assert FACTUALITY_CHOICES["D"] == 0.0, "a genuine disagreement"
        assert FACTUALITY_CHOICES["E"] == 1.0, "differences that do not matter"

    @pytest.mark.anyio
    async def test_a_disagreement_scores_zero(self) -> None:
        assert (await _run_judge(factuality(), "D")).value == 0.0

    @pytest.mark.anyio
    async def test_a_superset_answer_still_scores_one(self) -> None:
        assert (await _run_judge(factuality(), "B")).value == 1.0

    @pytest.mark.anyio
    async def test_the_score_is_named_factuality_not_judge(self) -> None:
        """The report names the scorer, so a suite using two judges must not
        show both under one name."""
        assert (await _run_judge(factuality(), "C")).scorer_name == "factuality"


class TestThePrompt:
    @pytest.mark.anyio
    async def test_the_prompt_carries_the_rubric_and_the_data(self) -> None:
        seen: list[Any] = []

        async def responder(**kwargs: Any) -> MagicMock:
            seen.append(kwargs)
            return _reply("A")

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=responder),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await judge(rubric="MY RUBRIC", choices=CHOICES)("the output", "the reference", CASE)

        prompt = seen[0]["messages"][0]["content"]
        assert "MY RUBRIC" in prompt
        assert "the output" in prompt
        assert "the reference" in prompt
        assert CASE.input in prompt, "the judge needs the question to grade against"

    @pytest.mark.anyio
    async def test_include_expected_false_withholds_the_reference(self) -> None:
        """For rubrics judging the output alone — tone, safety, format — where
        showing a reference invites the judge to compare against it instead of
        applying the rubric."""
        seen: list[Any] = []

        async def responder(**kwargs: Any) -> MagicMock:
            seen.append(kwargs)
            return _reply("A")

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=responder),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            scorer = judge(rubric="r", choices=CHOICES, include_expected=False)
            await scorer("the output", "SECRET REFERENCE", CASE)

        assert "SECRET REFERENCE" not in seen[0]["messages"][0]["content"]

    @pytest.mark.anyio
    async def test_the_prompt_names_the_available_choices(self) -> None:
        """A judge that does not know its options cannot pick one."""
        seen: list[Any] = []

        async def responder(**kwargs: Any) -> MagicMock:
            seen.append(kwargs)
            return _reply("A")

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=responder),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await judge(rubric="r", choices=CHOICES)("out", "exp", CASE)

        prompt = seen[0]["messages"][0]["content"]
        assert "A, B, C" in prompt


class TestTheAwkwardInputs:
    """Task 4.7's three, for a scorer that cannot simply inspect its arguments."""

    @pytest.mark.anyio
    async def test_an_empty_output_still_reaches_the_judge(self) -> None:
        """A model that returned nothing is exactly what a judge should be
        allowed to grade harshly — refusing to ask would discard the case."""
        result = await _run_judge(judge(rubric="r", choices=CHOICES), "C", output="")
        assert result.value == 0.0

    @pytest.mark.anyio
    async def test_a_none_output_still_reaches_the_judge(self) -> None:
        result = await _run_judge(judge(rubric="r", choices=CHOICES), "C", output=None)
        assert result.value == 0.0

    @pytest.mark.anyio
    async def test_a_none_expected_does_not_break_the_prompt(self) -> None:
        seen: list[Any] = []

        async def responder(**kwargs: Any) -> MagicMock:
            seen.append(kwargs)
            return _reply("A")

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=responder),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            result = await judge(rubric="r", choices=CHOICES)("out", None, CASE)

        assert result.value == 1.0
        assert seen, "the judge was never asked"

    @pytest.mark.anyio
    async def test_a_provider_failure_becomes_an_errored_score(self) -> None:
        """A judge that could not be reached is not evidence the task did
        badly, and one unreachable judge must not end the run."""

        async def failing(**_: Any) -> MagicMock:
            raise RuntimeError("provider unavailable")

        from evalstand.scorers.base import call_scorer

        with patch("evalstand.llm.litellm.acompletion", side_effect=failing):
            result = await call_scorer(judge(rubric="r", choices=CHOICES), "out", "exp", CASE)

        assert result.error is not None
        assert result.counts_towards_mean is False

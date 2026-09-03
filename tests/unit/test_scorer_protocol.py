"""The Scorer protocol (task 4.1).

The acceptance criterion is "a user-defined 3-line scorer works without
importing any base class", so most of these are about what a user can get away
with *not* writing: no decorator, no base class, no arguments they do not use.

The other half is the boundary between a scorer's claim and evalstand's
inference. CONTEXT.md is explicit that a Scorer sets the pass flag only when it
genuinely knows pass from fail and nothing else derives it, so these tests pin
down exactly which return values carry that claim.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalstand.models import Case, Score
from evalstand.scorers.base import as_score, call_scorer, scorer, scorer_name

CASE = Case(id="c1", input="in", expected="out")


async def _score(fn: Any, output: Any = "out", expected: Any = "out") -> Score:
    return await call_scorer(fn, output, expected, CASE)


class TestAPlainFunctionIsAScorer:
    """No import, no decorator, no base class."""

    @pytest.mark.anyio
    async def test_a_three_line_function_works_undecorated(self) -> None:
        def looks_right(output: str, expected: str) -> bool:
            return output == expected

        assert (await _score(looks_right)).value == 1.0

    @pytest.mark.anyio
    async def test_the_decorator_is_optional_and_changes_nothing(self) -> None:
        """`@scorer` documents intent and validates the signature early; it must
        not be the thing that makes a function work."""

        def plain(output: str, expected: str) -> bool:
            return output == expected

        @scorer
        def decorated(output: str, expected: str) -> bool:
            return output == expected

        assert (await _score(plain)).value == (await _score(decorated)).value


class TestScorersReceiveWhatTheyAskFor:
    """A scorer declaring two parameters must not be handed three."""

    @pytest.mark.anyio
    async def test_a_scorer_may_take_only_the_output(self) -> None:
        def nonempty(output: str) -> bool:
            return bool(output.strip())

        assert (await _score(nonempty, output="text")).passed is True
        assert (await _score(nonempty, output="   ")).passed is False

    @pytest.mark.anyio
    async def test_a_scorer_may_take_output_and_expected(self) -> None:
        seen: list[Any] = []

        def two(output: str, expected: str) -> float:
            seen.append((output, expected))
            return 1.0

        await _score(two, output="a", expected="b")
        assert seen == [("a", "b")]

    @pytest.mark.anyio
    async def test_a_scorer_may_take_the_case_as_well(self) -> None:
        """The Case is how a scorer reaches metadata — a per-case weight or
        rubric — which is the reason the third argument exists at all."""

        def with_case(output: str, expected: str, case: Case) -> float:
            return 1.0 if case.id == "c1" else 0.0

        assert (await _score(with_case)).value == 1.0

    @pytest.mark.anyio
    async def test_a_scorer_taking_no_arguments_is_called_with_none(self) -> None:
        """Degenerate but legal: a scorer that consults nothing (a stub, or one
        reading ambient state) is called with no arguments rather than being
        refused. It cannot judge the output, but that is its own choice."""
        called = False

        def takes_nothing() -> float:
            nonlocal called
            called = True
            return 1.0

        result = await _score(takes_nothing)
        assert called, "the scorer was never invoked"
        assert result.value == 1.0
        assert result.error is None

    @pytest.mark.anyio
    async def test_var_args_receives_all_three(self) -> None:
        """`*args` says "as many as you have"; passing fewer would be a guess."""
        seen: list[int] = []

        def flexible(*args: Any) -> float:
            seen.append(len(args))
            return 1.0

        await _score(flexible)
        assert seen == [3]

    @pytest.mark.anyio
    async def test_a_defaulted_parameter_is_configuration_not_an_argument(self) -> None:
        """`def contains(output, expected, fold=True)` is a two-argument scorer
        with an option, not a three-argument one. The runner must not pass the
        Case as `fold`."""
        seen: list[Any] = []

        def configured(output: str, expected: str, fold: bool = True) -> float:
            seen.append(fold)
            return 1.0

        await _score(configured)
        assert seen == [True], "a defaulted option was overwritten with the Case"

    def test_a_scorer_needing_four_arguments_is_rejected_at_decoration(self) -> None:
        """Caught at import rather than mid-run: a scorer that can never be
        called should not be discovered after a suite has spent money."""
        with pytest.raises(TypeError, match="at most three"):

            @scorer
            def too_many(output: Any, expected: Any, case: Any, extra: Any) -> float:
                return 1.0

    def test_a_required_keyword_only_argument_is_rejected(self) -> None:
        """Scorers are called positionally, so a required keyword-only argument
        could never be supplied."""
        with pytest.raises(TypeError, match="keyword-only"):

            @scorer
            def needs_keyword(output: Any, *, threshold: float) -> float:
                return 1.0


class TestReturnValues:
    @pytest.mark.anyio
    async def test_a_float_sets_the_value_and_leaves_passed_absent(self) -> None:
        """A continuous scorer reports a position on a scale. Where the pass
        line sits is a judgement it has not been given, so inventing one would
        put a verdict in the scorer's mouth."""

        def partial(output: str) -> float:
            return 0.62

        result = await _score(partial)
        assert result.value == pytest.approx(0.62)
        assert result.passed is None

    @pytest.mark.anyio
    async def test_a_bool_sets_both_value_and_passed(self) -> None:
        """A bool is an unambiguous claim of pass from fail, so honouring it
        reads the scorer's intent rather than deriving a verdict."""

        def binary(output: str, expected: str) -> bool:
            return output == expected

        assert (await _score(binary, output="x", expected="x")).passed is True
        assert (await _score(binary, output="x", expected="y")).passed is False

    @pytest.mark.anyio
    async def test_false_is_not_read_as_the_number_zero(self) -> None:
        """`bool` subclasses `int` in Python, so a check ordered the other way
        round would read `False` as 0.0 and silently discard its claim."""
        result = await _score(lambda output: False)
        assert result.value == 0.0
        assert result.passed is False, "a False verdict was flattened into a number"

    @pytest.mark.anyio
    async def test_true_is_not_read_as_the_number_one(self) -> None:
        result = await _score(lambda output: True)
        assert result.passed is True

    @pytest.mark.anyio
    async def test_a_score_is_passed_through_untouched(self) -> None:
        """A scorer returning a Score has said everything it means, including
        its own name and metadata."""
        built = Score(scorer_name="custom", value=0.5, metadata={"why": "partial"})
        result = await _score(lambda output: built)
        assert result is built

    @pytest.mark.anyio
    async def test_an_unusable_return_value_becomes_an_errored_score(self) -> None:
        """A scorer returning a string is broken, but the case still ran. The
        error belongs on the Score, where it is excluded from the mean."""
        result = await _score(lambda output: "quite good")
        assert result.error is not None
        assert result.counts_towards_mean is False

    def test_a_value_outside_the_range_is_rejected(self) -> None:
        """Scores live in [0, 1]. A scorer returning 5.0 has a bug, and letting
        it through would drag every mean it touches."""
        with pytest.raises(ValueError, match="less than or equal to 1"):
            as_score(5.0, "over")


class TestFailingScorersAreData:
    @pytest.mark.anyio
    async def test_a_raising_scorer_produces_an_errored_score(self) -> None:
        """A broken scorer is not evidence the task did badly."""

        def broken(output: str) -> float:
            raise RuntimeError("judge unavailable")

        result = await _score(broken)
        assert result.error == "RuntimeError: judge unavailable"
        assert result.value is None

    @pytest.mark.anyio
    async def test_an_errored_score_is_excluded_from_means(self) -> None:
        """Not counted as zero: a rate-limited judge would otherwise look
        exactly like a task that answered wrongly."""

        def broken(output: str) -> float:
            raise RuntimeError("boom")

        assert (await _score(broken)).counts_towards_mean is False

    @pytest.mark.anyio
    async def test_the_errored_score_carries_the_scorers_name(self) -> None:
        """Without the name the user cannot tell which scorer to fix."""

        def my_judge(output: str) -> float:
            raise ValueError("bad rubric")

        assert (await _score(my_judge)).scorer_name == "my_judge"


class TestAsyncScorers:
    @pytest.mark.anyio
    async def test_an_async_scorer_is_awaited(self) -> None:
        """A judge scorer calls a model, so async is the normal case, not an
        edge one."""

        async def slow(output: str, expected: str) -> float:
            return 1.0 if output == expected else 0.0

        assert (await _score(slow)).value == 1.0

    @pytest.mark.anyio
    async def test_a_raising_async_scorer_is_also_data(self) -> None:
        async def broken(output: str) -> float:
            raise RuntimeError("timeout talking to judge")

        assert (await _score(broken)).error == "RuntimeError: timeout talking to judge"


class TestNaming:
    def test_a_function_is_named_by_its_own_name(self) -> None:
        def levenshtein(output: Any) -> float:
            return 1.0

        assert scorer_name(levenshtein) == "levenshtein"

    def test_a_callable_object_is_named_by_its_class(self) -> None:
        """A judge built by a factory is usually a callable object or a
        closure. `scorer` in every row of a report would be useless."""

        class FactualityJudge:
            def __call__(self, output: Any, expected: Any) -> float:
                return 1.0

        assert scorer_name(FactualityJudge()) == "FactualityJudge"

    def test_a_lambda_gets_the_generic_name(self) -> None:
        """Not "function", which is the name of a lambda's *type* and would read
        in a report as though the scorer were really called that."""
        assert scorer_name(lambda output: 1.0) == "scorer"

"""Cache policy in the runner (task 3.3).

The runner cannot pass a cache to the task: the task calls `llm.call()` itself,
with parameters that live in user code. So the cache reaches it the same way the
trace collector does — through a ContextVar the runner sets around each case.

The rule this file pins down: **`--repeat N` with N > 1 bypasses the cache
unconditionally, in both directions.** Receiving N identical cached rows is never
what asking for repeats means, and writing a repeat's response back would let a
later single run serve an arbitrary sample from that set as though it were the
answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evalstand.api import Eval
from evalstand.cache import ResponseCache
from evalstand.llm import call
from evalstand.models import Case, Score
from evalstand.runner import RunConfig, run_eval
from tests.conftest import make_completion

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "q"}]


def _scorer(output: object, expected: object, case: Case) -> Score:
    return Score(scorer_name="s", value=1.0)


def _model_task(value: str) -> str:
    """A task shaped like a real one: it calls the model itself, and is given
    no cache to pass along."""
    return call("gpt-4o-mini", [{"role": "user", "content": value}], temperature=0.0).text


def _eval(cases: list[Case] | None = None, **kwargs: Any) -> Eval:
    return Eval(
        name="cache-eval",
        cases=cases or [Case(id="q1", input="x")],
        task=_model_task,
        scorers=[_scorer],
        filepath="f",
        **kwargs,
    )


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache.db")


class TestCacheReachesTheTask:
    @pytest.mark.anyio
    async def test_a_task_uses_the_runs_cache_without_being_handed_one(
        self, cache: ResponseCache
    ) -> None:
        """The task takes no cache argument, so the runner cannot pass one. It
        has to arrive ambiently or caching simply never happens in a real run."""
        cases = [Case(id="a", input="same"), Case(id="b", input="same")]

        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0001),
        ):
            await run_eval(_eval(cases), RunConfig(cache=cache, concurrency=1))

        assert provider.call_count == 1, "identical calls were not shared through the cache"

    @pytest.mark.anyio
    async def test_no_cache_means_every_call_reaches_the_provider(self) -> None:
        cases = [Case(id="a", input="same"), Case(id="b", input="same")]

        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await run_eval(_eval(cases), RunConfig(concurrency=1))

        assert provider.call_count == 2

    @pytest.mark.anyio
    async def test_the_cache_does_not_leak_out_of_the_run(self, cache: ResponseCache) -> None:
        """A call made after the run must not silently use the run's cache."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await run_eval(_eval(), RunConfig(cache=cache))

            from evalstand.runner import current_cache

            assert current_cache.get() is None


class TestRepeatsBypassTheCache:
    """Task 3.3's core rule."""

    @pytest.mark.anyio
    async def test_repeats_call_the_provider_every_time(self, cache: ResponseCache) -> None:
        """Five repeats must mean five executions. Five identical cached rows
        would answer a question nobody asked."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await run_eval(_eval(repeat=5), RunConfig(cache=cache, concurrency=1))

        assert provider.call_count == 5

    @pytest.mark.anyio
    async def test_repeats_write_nothing_to_the_cache(self, cache: ResponseCache) -> None:
        """Bypass runs in both directions. A stored repeat would let a later
        single run serve an arbitrary sample as though it were the answer."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await run_eval(_eval(repeat=3), RunConfig(cache=cache))

        assert cache.entry_count() == 0

    @pytest.mark.anyio
    async def test_a_repeat_run_does_not_read_an_existing_entry(self, cache: ResponseCache) -> None:
        """Even a legitimately cached answer must not satisfy a repeat."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await run_eval(_eval(), RunConfig(cache=cache))
        assert cache.entry_count() == 1

        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await run_eval(_eval(repeat=4), RunConfig(cache=cache, concurrency=1))

        assert provider.call_count == 4, "a repeat was served from the cache"

    @pytest.mark.anyio
    async def test_repeat_of_one_still_uses_the_cache(self, cache: ResponseCache) -> None:
        """repeat=1 is a normal run, not a degenerate repeat: bypassing there
        would throw away the cache for everybody."""
        cases = [Case(id="a", input="same"), Case(id="b", input="same")]

        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await run_eval(_eval(cases, repeat=1), RunConfig(cache=cache, concurrency=1))

        assert provider.call_count == 1

    @pytest.mark.anyio
    async def test_a_nondeterministic_task_yields_distinct_outputs(
        self, cache: ResponseCache
    ) -> None:
        """Task 3.3's acceptance: repeats produce distinct outputs rather than
        one answer repeated."""
        counter = 0

        def varying(**_: Any) -> Any:
            nonlocal counter
            counter += 1
            return make_completion(f"answer-{counter}")

        with (
            patch("evalstand.llm.litellm.completion", side_effect=varying),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            run = await run_eval(_eval(repeat=5), RunConfig(cache=cache, concurrency=1))

        assert len({r.output for r in run.results}) == 5


class TestExplicitBypass:
    @pytest.mark.anyio
    async def test_no_cache_flag_bypasses_even_without_repeats(self, cache: ResponseCache) -> None:
        """`--no-cache` is the user saying they want fresh answers."""
        cases = [Case(id="a", input="same"), Case(id="b", input="same")]

        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await run_eval(_eval(cases), RunConfig(cache=cache, bypass_cache=True, concurrency=1))

        assert provider.call_count == 2
        assert cache.entry_count() == 0, "an explicit bypass must not write either"


class TestBypassIsReported:
    @pytest.mark.anyio
    async def test_a_repeat_run_records_that_it_bypassed(self, cache: ResponseCache) -> None:
        """Repeats spend N times over. The plan calls this the easiest way to
        run up a bill by accident, so the run must say it happened."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            run = await run_eval(_eval(repeat=3), RunConfig(cache=cache))

        assert run.cache_bypassed is True
        assert run.cache_hits == 0

    @pytest.mark.anyio
    async def test_a_normal_run_reports_its_hit_rate(self, cache: ResponseCache) -> None:
        cases = [Case(id="a", input="same"), Case(id="b", input="same")]

        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            run = await run_eval(_eval(cases), RunConfig(cache=cache, concurrency=1))

        assert run.cache_bypassed is False
        assert run.cache_hits == 1, "the second identical call was a hit"
        assert run.model_calls == 2

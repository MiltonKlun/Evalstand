"""Automatic capture: every model call appears in the trace tree (task 3.2).

The user writes no tracing code. This is what makes the feature useful rather
than a thing people forget to instrument — and it is why `llm.py` is the only
place model calls are allowed to happen.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from evalstand.llm import acall, call
from evalstand.tracing import TraceCollector, trace
from tests.conftest import make_completion

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "capital of France?"}]


class TestAutomaticCapture:
    def test_a_call_is_traced_without_the_user_asking(self) -> None:
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.00012),
            TraceCollector() as collector,
        ):
            call("gpt-4o-mini", MESSAGES)

        assert len(collector.traces) == 1
        node = collector.traces[0]
        assert node.model == "gpt-4o-mini"
        assert node.input_tokens == 12
        assert node.cost_usd == pytest.approx(0.00012)

    def test_a_call_nests_under_an_open_span(self) -> None:
        """A judge scorer's call appearing under the task call it judges is the
        whole point of a tree over a list."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            TraceCollector() as collector,
            trace("task"),
        ):
            call("gpt-4o-mini", MESSAGES)

        task = next(n for n in collector.traces if n.name == "task")
        chat = next(n for n in collector.traces if n.name != "task")
        assert chat.parent_id == task.id

    def test_nested_calls_form_the_tree_the_acceptance_asks_for(self) -> None:
        """Task 3.2's acceptance: three nested calls, correct parentage, and the
        node costs summing to the case total."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.01),
            TraceCollector() as collector,
            trace("task"),
        ):
            call("gpt-4o-mini", MESSAGES)
            with trace("judge"):
                call("gpt-4o-mini", MESSAGES)

        by_name = {n.name: n for n in collector.traces}
        assert by_name["judge"].parent_id == by_name["task"].id

        calls = [n for n in collector.traces if n.model is not None]
        assert len(calls) == 2
        assert collector.total_cost_usd == pytest.approx(0.02)

    def test_a_cached_call_is_still_traced(self) -> None:
        """A cache hit is part of what happened; omitting it would make the tree
        disagree with the run."""
        import tempfile
        from pathlib import Path

        from evalstand.cache import ResponseCache

        cache = ResponseCache(Path(tempfile.mkdtemp()) / "c.db")
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0001),
        ):
            with TraceCollector() as first:
                call("gpt-4o-mini", MESSAGES, cache=cache)
            with TraceCollector() as second:
                call("gpt-4o-mini", MESSAGES, cache=cache)

        assert len(first.traces) == 1
        assert len(second.traces) == 1, "a cache hit produced no trace node"

    def test_an_untraced_call_still_works(self) -> None:
        """Outside a case there is no collector; calls must not break."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=make_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert call("gpt-4o-mini", MESSAGES).text == "Paris"

    @pytest.mark.anyio
    async def test_async_calls_are_captured(self) -> None:
        async def fake(**_: Any) -> Any:
            return make_completion()

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            TraceCollector() as collector,
        ):
            await acall("gpt-4o-mini", MESSAGES)

        assert len(collector.traces) == 1

    @pytest.mark.anyio
    async def test_concurrent_calls_all_parent_to_the_task(self) -> None:
        """The realistic runner shape: one task firing several calls at once."""
        import asyncio

        async def fake(**_: Any) -> Any:
            return make_completion()

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            TraceCollector() as collector,
            trace("task"),
        ):
            await asyncio.gather(*(acall("gpt-4o-mini", MESSAGES) for _ in range(5)))

        task = next(n for n in collector.traces if n.name == "task")
        calls = [n for n in collector.traces if n.model is not None]
        assert len(calls) == 5
        assert all(c.parent_id == task.id for c in calls)

    def test_a_failed_call_is_recorded_before_it_raises(self) -> None:
        """A call that failed after retries still consumed time and quota."""

        class BoomError(Exception):
            status_code = 401

        with (
            patch("evalstand.llm.litellm.completion", side_effect=BoomError("bad key")),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            TraceCollector() as collector,
            pytest.raises(BoomError),
            trace("task"),
        ):
            call("gpt-4o-mini", MESSAGES)

        names = {n.name for n in collector.traces}
        assert "task" in names
        assert any("BoomError" in str(n.output) for n in collector.traces if n.output)

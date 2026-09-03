"""Streaming through the runner (task 3.4).

A streaming task's output must be visible as it arrives, not only when the case
finishes. The runner cannot poll the task — the task owns the stream — so the
mechanism is the same one tracing and caching use: a ContextVar carrying a sink
the task's chunks flow into.

Without this, `acall_stream` works but nothing downstream ever sees a partial
output, which makes streaming pointless in a run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evalstand.api import Eval
from evalstand.llm import acall_stream
from evalstand.models import Case, Score
from evalstand.runner import RunConfig, current_sink, run_eval


def _chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    chunk.usage = None
    chunk.model = "gpt-4o-mini"
    return chunk


def _stream(*texts: str) -> Any:
    async def iterator(**_: Any) -> AsyncIterator[MagicMock]:
        for text in texts:
            yield _chunk(text)

    return iterator


def _scorer(output: object, expected: object, case: Case) -> Score:
    return Score(scorer_name="s", value=1.0)


class TestPartialOutput:
    @pytest.mark.anyio
    async def test_chunks_reach_the_sink_as_they_arrive(self) -> None:
        """The point of streaming in a run: a watcher sees text before the case
        is done."""
        seen: list[tuple[str, str]] = []

        async def task(value: str) -> str:
            streamer = acall_stream("gpt-4o-mini", [{"role": "user", "content": value}])
            return "".join([chunk async for chunk in streamer])

        declared = Eval(
            name="stream-eval",
            cases=[Case(id="q1", input="x")],
            task=task,
            scorers=[_scorer],
            filepath="f",
        )

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=_stream("Par", "is", "!")),
            patch("evalstand.llm.litellm.cost_per_token", return_value=(0.0, 0.0)),
        ):
            run = await run_eval(
                declared, RunConfig(on_chunk=lambda case_id, text: seen.append((case_id, text)))
            )

        assert [text for _, text in seen] == ["Par", "is", "!"]
        assert run.results[0].output == "Paris!", "chunks must accumulate in order"

    @pytest.mark.anyio
    async def test_chunks_are_attributed_to_their_case(self) -> None:
        """Concurrent streaming cases must not have their output interleaved
        into one another's."""
        seen: list[tuple[str, str]] = []

        async def task(value: str) -> str:
            streamer = acall_stream("gpt-4o-mini", [{"role": "user", "content": value}])
            return "".join([chunk async for chunk in streamer])

        declared = Eval(
            name="stream-eval",
            cases=[Case(id="a", input="x"), Case(id="b", input="y")],
            task=task,
            scorers=[_scorer],
            filepath="f",
        )

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=_stream("one", "two")),
            patch("evalstand.llm.litellm.cost_per_token", return_value=(0.0, 0.0)),
        ):
            await run_eval(declared, RunConfig(on_chunk=lambda cid, t: seen.append((cid, t))))

        for case_id in ("a", "b"):
            assert [t for c, t in seen if c == case_id] == ["one", "two"]

    @pytest.mark.anyio
    async def test_a_run_without_a_sink_still_streams(self) -> None:
        """No watcher is the normal case; streaming must not require one."""

        async def task(value: str) -> str:
            streamer = acall_stream("gpt-4o-mini", [{"role": "user", "content": value}])
            return "".join([chunk async for chunk in streamer])

        declared = Eval(
            name="stream-eval",
            cases=[Case(id="q1", input="x")],
            task=task,
            scorers=[_scorer],
            filepath="f",
        )

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=_stream("a", "b")),
            patch("evalstand.llm.litellm.cost_per_token", return_value=(0.0, 0.0)),
        ):
            run = await run_eval(declared)

        assert run.results[0].output == "ab"

    @pytest.mark.anyio
    async def test_the_sink_does_not_leak_out_of_the_run(self) -> None:
        async def task(value: str) -> str:
            return value

        declared = Eval(
            name="e",
            cases=[Case(id="q1", input="x")],
            task=task,
            scorers=[_scorer],
            filepath="f",
        )
        await run_eval(declared, RunConfig(on_chunk=lambda cid, t: None))
        assert current_sink.get() is None

    @pytest.mark.anyio
    async def test_a_failing_sink_does_not_break_the_run(self) -> None:
        """A watcher is an observer. If the TUI's renderer raises, the run — and
        the money already spent on it — must survive."""

        def bad_sink(case_id: str, text: str) -> None:
            raise RuntimeError("renderer exploded")

        async def task(value: str) -> str:
            streamer = acall_stream("gpt-4o-mini", [{"role": "user", "content": value}])
            return "".join([chunk async for chunk in streamer])

        declared = Eval(
            name="e",
            cases=[Case(id="q1", input="x")],
            task=task,
            scorers=[_scorer],
            filepath="f",
        )

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=_stream("a", "b")),
            patch("evalstand.llm.litellm.cost_per_token", return_value=(0.0, 0.0)),
        ):
            run = await run_eval(declared, RunConfig(on_chunk=bad_sink))

        assert run.results[0].output == "ab"
        assert run.results[0].error is None


class TestStreamedTracing:
    @pytest.mark.anyio
    async def test_a_streamed_call_is_traced_like_any_other(self) -> None:
        """A stream is a model call; omitting it would leave a gap in the tree
        and understate the run's cost."""

        async def task(value: str) -> str:
            streamer = acall_stream("gpt-4o-mini", [{"role": "user", "content": value}])
            return "".join([chunk async for chunk in streamer])

        declared = Eval(
            name="e",
            cases=[Case(id="q1", input="x")],
            task=task,
            scorers=[_scorer],
            filepath="f",
        )

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=_stream("a", "b")),
            patch("evalstand.llm.litellm.cost_per_token", return_value=(0.0, 0.0)),
        ):
            run = await run_eval(declared)

        traces = run.results[0].traces
        assert len(traces) == 1
        assert traces[0].model == "gpt-4o-mini"
        assert traces[0].output == "ab"


class TestConcurrentStreams:
    @pytest.mark.anyio
    async def test_interleaved_streams_keep_their_chunks_separate(self) -> None:
        """Three cases streaming at once: each one's chunks must reassemble into
        its own output, not a blend of all three."""
        import asyncio

        def per_case_stream(**kwargs: Any) -> Any:
            async def iterator() -> AsyncIterator[MagicMock]:
                for character in kwargs["messages"][0]["content"]:
                    await asyncio.sleep(0.002)
                    yield _chunk(character)

            return iterator()

        async def task(value: str) -> str:
            streamer = acall_stream("gpt-4o-mini", [{"role": "user", "content": value}])
            return "".join([chunk async for chunk in streamer])

        declared = Eval(
            name="e",
            cases=[
                Case(id="alpha", input="AAAA"),
                Case(id="beta", input="BBBB"),
                Case(id="gamma", input="CCCC"),
            ],
            task=task,
            scorers=[_scorer],
            filepath="f",
        )

        seen: list[tuple[str, str]] = []
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=per_case_stream),
            patch("evalstand.llm.litellm.cost_per_token", return_value=(0.0, 0.0)),
        ):
            run = await run_eval(
                declared,
                RunConfig(concurrency=3, on_chunk=lambda cid, t: seen.append((cid, t))),
            )

        assert len({case_id for case_id, _ in seen}) == 3, "streams did not interleave"
        for result in run.results:
            reassembled = "".join(t for cid, t in seen if cid == result.case_id)
            assert reassembled == result.output

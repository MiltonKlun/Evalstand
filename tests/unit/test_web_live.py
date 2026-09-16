"""Server-sent events for the live table (task 8.2).

The broker sits between the runner and a browser, and the failure that matters
is not a dropped event — it is a browser that slows down or breaks a run.
`_announce` already stops an observer's *exception* from costing a measurement;
nothing stops an observer that blocks, so the broker must never block and never
grow without bound.

The other failure worth catching is a stream that never ends: a browser holding
an open request after a run finished shows a spinner for something that is
over.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from evalstand.models import Result, Score, Trace
from evalstand.web.live import QUEUE_LIMIT, Broker, sink_for, stream

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def _result(case_id: str = "q1", *, repeat: int = 0) -> Result:
    return Result(
        id=f"r-{case_id}-{repeat}",
        case_id=case_id,
        repeat_index=repeat,
        output="answer",
        latency_ms=5,
        scores=[Score(scorer_name="exact", value=1.0, passed=True)],
        traces=[Trace(id=f"t-{case_id}", name="call", duration_ms=4, started_at=NOW)],
    )


async def _drain(queue: asyncio.Queue[str | None]) -> list[str]:
    """Everything currently queued, without waiting for more."""
    out: list[str] = []
    while not queue.empty():
        item = queue.get_nowait()
        if item is not None:
            out.append(item)
    return out


@pytest.mark.anyio
class TestItPublishesToEveryWatcher:
    async def test_a_subscriber_receives_a_result(self) -> None:
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        broker.publish(_result())
        await asyncio.sleep(0)

        events = await _drain(queue)
        assert len(events) == 1
        assert "event: result" in events[0]

    async def test_two_browsers_both_receive_it(self) -> None:
        """Fan-out, not hand-off. A second tab must not steal the first's
        events."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        first, second = broker.subscribe(), broker.subscribe()

        broker.publish(_result())
        await asyncio.sleep(0)

        assert len(await _drain(first)) == 1
        assert len(await _drain(second)) == 1

    async def test_the_payload_is_a_rendered_row(self) -> None:
        """The server renders; the browser only swaps. A payload of raw JSON
        would put the formatting rules in the page, where they would drift from
        the terminal's."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        broker.publish(_result("q7"))
        await asyncio.sleep(0)

        (event,) = await _drain(queue)
        assert "q7" in event
        assert "<tr" in event

    async def test_an_unsubscribed_queue_stops_receiving(self) -> None:
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()
        broker.unsubscribe(queue)

        broker.publish(_result())
        await asyncio.sleep(0)

        assert await _drain(queue) == []


@pytest.mark.anyio
class TestItCannotBreakTheRun:
    """The rule the runner depends on."""

    async def test_publishing_with_no_browser_is_harmless(self) -> None:
        """`serve` runs with nobody watching most of the time."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())

        broker.publish(_result())  # must not raise

    async def test_publishing_before_a_loop_is_bound_is_harmless(self) -> None:
        """The runner may publish before any browser has connected. There is no
        loop to hand the event to, and touching an asyncio.Queue from off-loop
        would take down the server rather than one browser."""
        Broker().publish(_result())  # must not raise

    async def test_a_full_queue_drops_the_oldest_not_the_newest(self) -> None:
        """A live table that lags further behind the longer it runs is worse
        than one that skips. The newest row describes what is happening now."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        for index in range(QUEUE_LIMIT + 5):
            broker.publish(_result(f"q{index}"))
        await asyncio.sleep(0)

        events = await _drain(queue)

        assert len(events) == QUEUE_LIMIT, "the queue grew past its limit"
        assert "q0" not in " ".join(events), "the oldest event survived a full queue"
        assert f"q{QUEUE_LIMIT + 4}" in " ".join(events), "the newest event was dropped"

    async def test_a_slow_browser_does_not_block_a_fast_one(self) -> None:
        """Subscribers are independent: one that stops reading fills its own
        queue without holding up anyone else."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        slow, fast = broker.subscribe(), broker.subscribe()

        for index in range(QUEUE_LIMIT + 10):
            broker.publish(_result(f"q{index}"))
            await asyncio.sleep(0)
            await _drain(fast)  # the fast one keeps up

        assert slow.full()
        assert fast.empty()


@pytest.mark.anyio
class TestTheStreamEnds:
    async def test_finish_releases_a_waiting_browser(self) -> None:
        """Without the sentinel a browser holds an open request until it times
        out, and the page shows a spinner for a run that finished."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        collected: list[str] = []

        async def consume() -> None:
            async for payload in stream(queue):
                collected.append(payload)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)

        broker.publish(_result())
        broker.finish()

        await asyncio.wait_for(task, timeout=2.0)

        assert any("event: result" in event for event in collected)
        assert any("event: done" in event for event in collected)

    async def test_done_is_sent_before_the_stream_closes(self) -> None:
        """A stream that just stops looks like a dropped connection, and the
        page would report 'not live' rather than 'finished'."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        broker.finish()
        await asyncio.sleep(0)

        first = queue.get_nowait()
        assert first is not None
        assert "event: done" in first
        assert queue.get_nowait() is None


@pytest.mark.anyio
class TestTheFrameFormat:
    async def test_a_multiline_fragment_survives_as_one_event(self) -> None:
        """An SSE frame is newline-delimited. An HTML fragment containing a
        newline would end the event early, delivering half a row and leaving
        the rest to be parsed as a new frame."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        broker.publish(_result("has\nnewline"))
        await asyncio.sleep(0)

        (event,) = await _drain(queue)
        body = event.split("data: ", 1)[1]

        assert body.count("\n") == 2, "the payload broke the frame apart"

    async def test_it_ends_with_a_blank_line(self) -> None:
        """Required by the protocol: without it a browser buffers the event
        waiting for a terminator that never comes."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        broker.publish(_result())
        await asyncio.sleep(0)

        (event,) = await _drain(queue)
        assert event.endswith("\n\n")


@pytest.mark.anyio
class TestTheSink:
    async def test_it_is_a_plain_callable_the_runner_can_use(self) -> None:
        """`RunConfig(on_result=...)` takes a callable. Neither the runner nor
        RunConfig should have to know what a broker is."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        sink = sink_for(broker)
        sink(_result())
        await asyncio.sleep(0)

        assert len(await _drain(queue)) == 1

    async def test_the_runner_accepts_it(self) -> None:
        """Asserted against the real type rather than by duck-typing: a sink
        the runner would reject is a live view that never updates."""
        from evalstand.runner import RunConfig

        config = RunConfig(on_result=sink_for(Broker()))

        assert config.on_result is not None


@pytest.mark.anyio
class TestTheKeepalive:
    """A stream that goes quiet gets closed by something in the middle.

    Proxies and browsers drop a connection that has been silent, and an eval
    whose cases take a minute each is silent for exactly that long. The dropped
    connection surfaces as "not live" on a run that is progressing normally —
    the page reporting a failure that did not happen.
    """

    async def test_a_quiet_stream_sends_a_comment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("evalstand.web.live.KEEPALIVE_SECONDS", 0.01)

        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        collected: list[str] = []

        async def consume() -> None:
            async for payload in stream(queue):
                collected.append(payload)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        broker.finish()
        await asyncio.wait_for(task, timeout=2.0)

        assert any(payload.startswith(":") for payload in collected), "no keepalive was sent"

    async def test_the_keepalive_is_a_comment_not_an_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An SSE comment starts with `:` and is ignored by the client. A
        keepalive shaped like an event would append a phantom row to the live
        table every fifteen seconds."""
        monkeypatch.setattr("evalstand.web.live.KEEPALIVE_SECONDS", 0.01)

        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        collected: list[str] = []

        async def consume() -> None:
            async for payload in stream(queue):
                collected.append(payload)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        broker.finish()
        await asyncio.wait_for(task, timeout=2.0)

        keepalives = [payload for payload in collected if payload.startswith(":")]
        assert keepalives
        for payload in keepalives:
            assert "event:" not in payload
            assert payload.endswith("\n\n"), "a frame without a blank line is never delivered"

    async def test_a_busy_stream_does_not_send_one(self) -> None:
        """The other half. With events flowing, the timeout never fires — and
        without this, the assertions above would hold no matter what."""
        broker = Broker()
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        collected: list[str] = []

        async def consume() -> None:
            async for payload in stream(queue):
                collected.append(payload)

        task = asyncio.create_task(consume())
        broker.publish(_result())
        await asyncio.sleep(0)
        broker.finish()
        await asyncio.wait_for(task, timeout=2.0)

        assert not any(payload.startswith(":") for payload in collected)

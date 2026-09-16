"""Server-sent events for the live run table (task 8.2).

The runner already publishes finished Results to a `ResultSink`; this is the
same seam the TUI uses. A browser is one more observer, and `runner._announce`
already guarantees that an observer which raises cannot cost the measurement it
was observing — a closed tab is the ordinary case here, not an error.

**Why SSE and not websockets.** The traffic is one-directional: the server
pushes rows, the browser sends nothing back. SSE is a plain HTTP response that
reconnects on its own, needs no handshake, and survives a proxy that would drop
an idle socket. A websocket would add a second protocol for no capability this
page uses.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from evalstand.models import Result
from evalstand.web.fragments import case_row

logger = logging.getLogger(__name__)

KEEPALIVE_SECONDS = 15.0
"""How long a stream may be silent before it sends a comment.

Short enough to stay under the idle timeout of a default proxy, long enough
that a quiet run is not chatty.
"""


QUEUE_LIMIT = 1000
"""How many undelivered events one subscriber may hold.

A browser on a slow connection, or one paused in a background tab, drains more
slowly than a fast eval produces. Unbounded, that queue grows until the process
does. At the limit the *oldest* event is dropped rather than the newest: the
newest describes what is happening now, and a live table that lags further
behind the longer it runs is worse than one that skips.
"""


class Broker:
    """Fan-out from one running eval to however many browsers are watching.

    Subscribers are independent: one that stops reading fills its own queue and
    is trimmed, without blocking the runner or any other subscriber. That
    matters because the publisher is the runner's own thread of execution — a
    broker that applied backpressure would slow down the eval to serve a
    browser.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str | None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the loop that owns the subscriber queues.

        `publish` is called from the runner, which may be on another thread.
        Without a bound loop there is nothing safe to hand the event to, so
        events are dropped rather than touching a queue from off-loop —
        `asyncio.Queue` is not thread-safe, and corrupting one would take down
        the server rather than one browser.
        """
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[str | None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=QUEUE_LIMIT)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str | None]) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, result: Result) -> None:
        """Hand a finished Result to every browser watching.

        Called by the runner through `RunConfig.on_result`, so it must never
        raise and never block. Both are enforced here rather than trusted to
        the caller: `_announce` swallows the exception, but an observer that
        *blocked* would stall the run without anything noticing.
        """
        payload = _event("result", case_row(result))
        self._dispatch(payload)

    def finish(self) -> None:
        """Tell every browser the run is over, then release them.

        `None` is the sentinel that ends a stream. Without it a browser holds
        an open request until it times out, and the page shows a spinner for a
        run that finished minutes ago.
        """
        self._dispatch(_event("done", ""))
        self._dispatch(None)

    def _dispatch(self, payload: str | None) -> None:
        loop = self._loop
        if loop is None:
            # Nothing is listening on a loop yet. Dropped deliberately: there
            # is no browser to miss it, and queueing for one that may never
            # arrive is how an unbounded buffer starts.
            return

        for queue in list(self._subscribers):
            loop.call_soon_threadsafe(self._offer, queue, payload)

    def _offer(self, queue: asyncio.Queue[str | None], payload: str | None) -> None:
        """Enqueue, dropping the oldest event if this subscriber is full."""
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.put_nowait(payload)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover - race
                logger.debug("dropped a live event for a subscriber that stopped reading")


async def stream(queue: asyncio.Queue[str | None]) -> AsyncIterator[str]:
    """Yield events until the run ends or the browser leaves.

    A keepalive comment goes out when nothing has happened for a while. Proxies
    and browsers close a connection that has been silent, and an eval whose
    cases take a minute each would otherwise be dropped mid-run and look like a
    crash.
    """
    while True:
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
        except TimeoutError:
            yield ": keepalive\n\n"
            continue

        if payload is None:
            return
        yield payload


def _event(name: str, data: str) -> str:
    """One SSE frame.

    The payload is JSON-encoded even though it is HTML, because an SSE frame is
    newline-delimited and an HTML fragment containing a newline would end the
    event early — delivering half a table row and leaving the rest to be parsed
    as a new frame.
    """
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def sink_for(broker: Broker) -> Any:
    """A `ResultSink` that publishes to this broker.

    Returned as a closure so `RunConfig(on_result=...)` gets a plain callable
    and neither the runner nor `RunConfig` has to know what a broker is.
    """

    def publish(result: Result) -> None:
        broker.publish(result)

    return publish

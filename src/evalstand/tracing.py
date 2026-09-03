"""Tracing: what happened inside a task, as a tree.

Every model call made during a task is captured automatically, and a user can
wrap any operation in `trace(name)` to give it a node of its own. The result is
a tree, so a judge scorer's call appears beneath the task call it judges rather
than beside it.

**The parenting rule (ADR 0006).** A `ContextVar` holds the currently open node.
Entering a span sets it and keeps the token; leaving resets that token in a
`finally`. This is the only rule that survives both `asyncio.gather` — where
contextvars are copied per task, so siblings share a parent for free — and an
exception mid-call, where the `finally` reset stops a raising span corrupting
the parentage of the one after it.

**When a parent cannot be found, the node attaches to the root.** Context is not
copied into a raw `ThreadPoolExecutor`, so a call made there has no parent to
discover. An orphan at the top level is visibly incomplete; a wrongly-parented
node is a plausible-looking lie, and a trace tree that misrepresents which call
invoked which is worse than no tree at all.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

from evalstand.models import Trace

__all__ = [
    "TraceCollector",
    "current_collector",
    "current_span",
    "record_call",
    "trace",
]

current_collector: contextvars.ContextVar[TraceCollector | None] = contextvars.ContextVar(
    "evalstand_trace_collector", default=None
)
"""The collector for the case currently executing, if any."""

current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "evalstand_trace_span", default=None
)
"""The id of the innermost open span. This is the parent of the next node."""


class TraceCollector:
    """Gathers the traces of one Result.

    Used as a context manager around the execution of a single case; the runner
    holds one per case, so concurrent cases never share a collector.
    """

    def __init__(self) -> None:
        self.traces: list[Trace] = []
        self._token: contextvars.Token[TraceCollector | None] | None = None

    def __enter__(self) -> Self:
        self._token = current_collector.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._token is not None:
            current_collector.reset(self._token)
            self._token = None

    def add(self, node: Trace) -> None:
        self.traces.append(node)

    def record_orphan(self, name: str, **fields: Any) -> Trace:
        """Record a node whose parent could not be determined.

        Deliberately explicit rather than a silent fallback, so a caller that
        knows it has lost context can say so instead of guessing.
        """
        node = Trace(id=_new_id(), parent_id=None, name=name, **fields)
        self.add(node)
        return node

    # --- rollups -------------------------------------------------------

    @property
    def priced_cost_usd(self) -> float:
        """What the priced calls cost. Not the whole story when some are not."""
        return sum(node.cost_usd or 0.0 for node in self.traces if node.cost_usd is not None)

    @property
    def unpriced_call_count(self) -> int:
        """Model calls whose cost could not be determined."""
        return sum(1 for node in self.traces if node.model is not None and node.cost_usd is None)

    @property
    def total_cost_usd(self) -> float | None:
        """The case total, or None when any call went unpriced.

        Summing only the priced calls would understate real spend while looking
        like a complete figure. `priced_cost_usd` is available for callers that
        want the partial number and will label it as partial.
        """
        if self.unpriced_call_count:
            return None
        return self.priced_cost_usd

    @property
    def total_input_tokens(self) -> int:
        return sum(node.input_tokens or 0 for node in self.traces)

    @property
    def total_output_tokens(self) -> int:
        return sum(node.output_tokens or 0 for node in self.traces)


def _new_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def trace(name: str, **fields: Any) -> Iterator[None]:
    """Record `name` as a span, with anything inside it as its children.

    Outside a running case there is no collector, so this does nothing rather
    than raising: a user experimenting at a REPL should not hit an error.
    """
    collector = current_collector.get()
    if collector is None:
        yield
        return

    node_id = _new_id()
    parent_id = current_span.get()
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    token = current_span.set(node_id)

    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        # The reset is what stops a raising span leaving its parentage behind
        # for the next sibling to inherit.
        current_span.reset(token)
        collector.add(
            Trace(
                id=node_id,
                parent_id=parent_id,
                name=name,
                started_at=started_at,
                duration_ms=int((time.perf_counter() - started) * 1000),
                output=f"{type(error).__name__}: {error}" if error else fields.get("output"),
                **{k: v for k, v in fields.items() if k != "output"},
            )
        )


def record_call(
    *,
    name: str,
    model: str | None = None,
    duration_ms: int = 0,
    input: Any = None,  # noqa: A002 - matches the Trace field it fills
    output: Any = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> Trace | None:
    """Record one completed model call under the currently open span.

    Called from `llm.py`, which is why every model call is captured without the
    user doing anything. Returns None outside a case, where there is nothing to
    record into.
    """
    collector = current_collector.get()
    if collector is None:
        return None

    node = Trace(
        id=_new_id(),
        parent_id=current_span.get(),
        name=name,
        started_at=datetime.now(UTC),
        duration_ms=duration_ms,
        model=model,
        input=input,
        output=output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
    collector.add(node)
    return node

"""The runner: execute an Eval's cases concurrently and collect Results.

The governing rule is that one bad case never takes down the others. An eval is
a measurement, and discarding 29 good measurements because the 30th timed out
would waste the time and money already spent on them. Every failure — a raising
task, a timeout, a broken scorer — is recorded on its own Result and the run
continues.

Ordering is by case, not by completion. Cases finish in whatever order the
provider answers, but a report whose rows shuffle between runs cannot be read or
diffed, so results are collected back into declaration order.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import logging
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalstand.api import Eval
from evalstand.cache import ResponseCache
from evalstand.models import Case, Result, Run, RunStatus, Score
from evalstand.tracing import TraceCollector

__all__ = ["RunConfig", "current_bypass", "current_cache", "current_sink", "run_eval"]

ChunkSink = Callable[[str, str], None]
"""Called with (case_id, chunk) as streamed text arrives."""

current_cache: contextvars.ContextVar[ResponseCache | None] = contextvars.ContextVar(
    "evalstand_cache", default=None
)
"""The cache for the run currently executing.

A ContextVar rather than an argument because the task calls the model itself,
with parameters that live in user code. The same reason task 3.3 could not gate
on temperature: the runner cannot reach inside the task.
"""

current_bypass: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "evalstand_bypass_cache", default=False
)
"""Whether the running case must skip the cache in both directions."""

current_sink: contextvars.ContextVar[ChunkSink | None] = contextvars.ContextVar(
    "evalstand_chunk_sink", default=None
)
"""Where a streaming task's chunks go as they arrive.

Bound per case, so the sink already knows which case each chunk belongs to and
concurrent streams cannot interleave into one another's output.
"""

logger = logging.getLogger("evalstand.runner")

DEFAULT_CONCURRENCY = 8
"""Enough to hide network latency, low enough not to trip a rate limit on the
first run. Overridable with --concurrency."""


@dataclass(frozen=True)
class RunConfig:
    """How a run executes, as opposed to what it measures."""

    concurrency: int = DEFAULT_CONCURRENCY
    timeout_seconds: float | None = None
    """No default limit: a slow model is normal, and a default would fail
    honest work. The user sets one when they know what "too slow" means."""

    cache: ResponseCache | None = None
    """Shared across the run. Without one, every call reaches the provider."""

    bypass_cache: bool = False
    """The user's explicit `--no-cache`. Repeats bypass regardless."""

    on_chunk: ChunkSink | None = None
    """Receives (case_id, chunk) as a streaming task produces text, so a live
    view can show partial output rather than waiting for the case to finish."""

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {self.concurrency}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout_seconds}")


async def run_eval(
    declared: Eval,
    config: RunConfig | None = None,
    *,
    batch_id: str = "local",
    only: set[tuple[str, int]] | None = None,
) -> Run:
    """Execute one Eval's cases and return its Run.

    `only` restricts execution to the given `(case_id, repeat_index)` pairs,
    which is how selection (`-k`, or a single node id) reaches the runner. It
    filters *before* anything executes rather than discarding results
    afterwards, because a deselected case must cost nothing — the user asked for
    one case and a provider bill for thirty would be a bug with a price tag.
    """
    config = config or RunConfig()
    started_at = datetime.now(UTC)

    cases = await declared.aload_cases()
    units = _units(cases, declared.repeat)
    if only is not None:
        units = [unit for unit in units if (unit[1].id, unit[2]) in only]
    semaphore = asyncio.Semaphore(config.concurrency)

    # Repeats bypass unconditionally. The runner cannot inspect the task's
    # temperature — those parameters live in user code — and temperature is not
    # the only source of nondeterminism anyway. Asking for N repeats and getting
    # N identical cached rows answers a question nobody asked.
    bypass = config.bypass_cache or declared.repeat > 1

    cache_token = current_cache.set(config.cache)
    bypass_token = current_bypass.set(bypass)
    try:
        results = await asyncio.gather(
            *(
                _run_one(declared, case, repeat_index, config, semaphore)
                for _, case, repeat_index in units
            )
        )
    finally:
        # Reset in a finally so a raising run cannot leave the cache bound for
        # whatever executes next in this context.
        current_cache.reset(cache_token)
        current_bypass.reset(bypass_token)

    # gather preserves input order, so this is already case-then-repeat order.
    results = list(results)
    return Run(
        id=f"run-{uuid.uuid4().hex[:12]}",
        batch_id=batch_id,
        name=declared.name,
        filepath=declared.filepath,
        status=RunStatus.COMPLETED,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        repeat_n=declared.repeat,
        results=results,
        cache_bypassed=bypass,
        cache_hits=sum(1 for r in results for trace in r.traces if trace.name == "cached call"),
        model_calls=sum(1 for r in results for trace in r.traces if trace.model is not None),
    )


def _units(cases: list[Case], repeat: int) -> list[tuple[int, Case, int]]:
    """One unit of work per (case, repeat).

    The same unit the pytest plugin collects: an execution is the thing that has
    an outcome.
    """
    return [
        (index, case, repeat_index)
        for index, case in enumerate(cases)
        for repeat_index in range(repeat)
    ]


async def _run_one(
    declared: Eval,
    case: Case,
    repeat_index: int,
    config: RunConfig,
    semaphore: asyncio.Semaphore,
) -> Result:
    """Execute one case once, capturing whatever happens.

    A fresh TraceCollector per case, so concurrent cases never pool their
    traces into one another.
    """
    async with semaphore:
        collector = TraceCollector()
        sink_token = current_sink.set(_case_sink(config.on_chunk, case.id))
        output: Any = None
        error: str | None = None

        frames: list[str] = []

        with collector:
            try:
                output = await _call_task(declared.task, case.input, config.timeout_seconds)
            except TimeoutError:
                error = (
                    f"timed out after {config.timeout_seconds}s"
                    if config.timeout_seconds
                    else "timed out"
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                frames = _user_frames(exc)

            # A task that failed has no output worth scoring. Inventing a zero
            # would claim it performed badly, when in truth it never finished.
            scores = (
                []
                if error is not None
                else [await _score(fn, output, case.expected, case) for fn in declared.scorers]
            )

        current_sink.reset(sink_token)

        return Result(
            id=f"{declared.name}-{case.id}-{repeat_index}",
            case_id=case.id,
            repeat_index=repeat_index,
            output=_snapshot(output),
            error=error,
            error_frames=frames,
            scores=scores,
            traces=collector.traces,
            input_tokens=collector.total_input_tokens or None,
            output_tokens=collector.total_output_tokens or None,
            cost_usd=collector.total_cost_usd,
        )


def _snapshot(output: Any) -> Any:
    """Keep the output as it was when it was scored.

    A task that returns a shared object it goes on mutating — a dict it reuses,
    a list it appends to — would otherwise leave every stored Result holding the
    *last* value. The report then contradicts its own scores: two cases marked
    correct while the output column shows all three answering identically, and
    no way to tell which reading is true.

    Immutable values are returned as they are, since copying them buys nothing.
    Anything that cannot be copied — a file handle, a live client, a response
    object — falls back to its repr rather than failing the case: the point is
    to keep a faithful record, and a faithful description beats losing the
    measurement over it.
    """
    if isinstance(output, str | int | float | bool | bytes | type(None)):
        return output

    try:
        return copy.deepcopy(output)
    except Exception:
        logger.debug("could not copy a task output; storing its repr", exc_info=True)
        return repr(output)


def _user_frames(exc: BaseException) -> list[str]:
    """The frames from the user's own code, rendered for a report.

    Everything from evalstand, asyncio and threading is dropped: the user cannot
    act on our call machinery, and burying the one line that matters under eight
    frames of it is exactly the failure pytest's default output makes.

    Formatted here rather than stored as a traceback object because a Result is
    serialised to SQLite and sent to the UI, and a traceback holds references to
    every frame's locals — which would keep the whole run's objects alive and
    could carry an API key into the database.
    """
    frames = []
    for frame in traceback.extract_tb(exc.__traceback__):
        if _is_library_frame(frame.filename):
            continue
        line = f"  {Path(frame.filename).name}:{frame.lineno} in {frame.name}"
        frames.append(line if not frame.line else f"{line}\n    {frame.line}")
    return frames


def _is_library_frame(filename: str) -> bool:
    """Frames belonging to the machinery that called the task, not to the task.

    `concurrent/futures/thread.py` is here because offloading a sync task with
    `asyncio.to_thread` puts a worker frame on top of every traceback — an
    implementation detail of how we avoid blocking the event loop, which would
    otherwise appear in every user's error report.
    """
    lowered = filename.replace("\\", "/").lower()
    return (
        "/evalstand/" in lowered
        or "/asyncio/" in lowered
        or "/concurrent/futures/" in lowered
        or lowered.endswith("/threading.py")
    )


def _case_sink(on_chunk: ChunkSink | None, case_id: str) -> ChunkSink | None:
    """Bind a sink to one case, and make it unable to break the run.

    A watcher is an observer. If a renderer raises, the run — and the money
    already spent on it — must survive; losing results to a display bug would be
    the tail wagging the dog.
    """
    if on_chunk is None:
        return None

    def sink(_case_id: str, text: str) -> None:
        try:
            on_chunk(case_id, text)
        except Exception:
            logger.debug("chunk sink raised for case %s", case_id, exc_info=True)

    return sink


async def _call_task(task: Any, case_input: Any, timeout_seconds: float | None) -> Any:
    """Run the task, sync or async, under an optional timeout.

    A sync task is offloaded with `asyncio.to_thread` rather than called
    directly: a blocking call on the event loop would stall every other case,
    turning concurrency into a lie. `to_thread` also copies context, so tracing
    still works inside it.
    """
    if inspect.iscoroutinefunction(task):
        coroutine = task(case_input)
    else:
        coroutine = asyncio.to_thread(task, case_input)

    if timeout_seconds is None:
        return await coroutine

    async with asyncio.timeout(timeout_seconds):
        return await coroutine


async def _score(fn: Any, output: Any, expected: Any, case: Case) -> Score:
    """Run one scorer, recording a raise rather than letting it end the case.

    The adapting and normalising live in `scorers.base`, which is where a user
    writing their own scorer will look for the rules. Imported lazily to keep
    the runner from depending on the scorer library at module scope.
    """
    from evalstand.scorers.base import call_scorer

    return await call_scorer(fn, output, expected, case)

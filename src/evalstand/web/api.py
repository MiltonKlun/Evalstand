"""The JSON API over run history (task 8.1).

Reads what `RunStore` already holds. It computes no pass rate, no mean and no
total of its own: those come from `reporting.console.pass_counts` and the Run's
own aggregates, the same functions the terminal and the markdown reporter call.
A second implementation of "which Results count towards a pass rate" would
eventually disagree with the first, and a web page contradicting the terminal
about whether a build passed is worse than either number alone (ADR 0009).

**Missing data is `null`, never `0`.** A consumer that reads `"cost_usd": 0`
sums it into a total that understates real spend while looking authoritative.
The terminal renders these as `-`; JSON has a real null and uses it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from evalstand.models import Batch, Run
from evalstand.reporting.console import pass_counts

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from fastapi import FastAPI

    from evalstand.storage import RunStore


DEFAULT_LIMIT = 50
"""How many runs a bare `/api/runs` returns.

Bounded because the database grows without limit and a browser asking for
every run of a long-lived project would serialise megabytes to render twenty
rows. `?limit=` raises it deliberately.
"""

MAX_LIMIT = 500
"""The ceiling `?limit=` may ask for.

Not a security boundary — the server is local — but a runaway query still
blocks the event loop while SQLite reads, and a page that hangs looks broken
rather than busy.
"""


def run_summary(run: Run, batch: Batch | None = None) -> dict[str, Any]:
    """One run, without its results.

    The list view's payload. Results are the bulk of a Run — a thirty-case eval
    with traces is orders of magnitude larger than its summary — so the list
    omits them and `/api/runs/{id}` serves the whole thing.
    """
    passed, judged = pass_counts(run)

    return {
        "id": run.id,
        "batch_id": run.batch_id,
        "name": run.name,
        "filepath": run.filepath,
        "status": run.status.value,
        "started_at": _moment(run.started_at),
        "finished_at": _moment(run.finished_at),
        "repeat_n": run.repeat_n,
        "result_count": len(run.results),
        # `judged` is the denominator the terminal uses, not `len(results)`.
        # A continuous scorer that declined to judge is not a failure, and a
        # Result with no scores at all is not a pass.
        "passed": passed,
        "judged": judged,
        "mean_score": run.mean_score,
        # A *lower bound*, not a total: `total_cost_usd` sums only the calls
        # that could be priced. Shipped with the two fields that say so,
        # because a consumer shown the figure alone would understate the bill —
        # the same class of error as a false pass, quietly wrong in the
        # direction that matters.
        "total_cost_usd": run.total_cost_usd,
        "cost_is_complete": run.cost_is_complete,
        "unpriced_call_count": run.unpriced_call_count,
        "model_calls": run.model_calls,
        "cache_hits": run.cache_hits,
        # None when nothing was called. A run that made no calls did not have a
        # 0% hit rate; it had no rate at all.
        "cache_hit_rate": run.cache_hit_rate,
        "cache_bypassed": run.cache_bypassed,
        # Called, not referenced: this one is a method where the neighbours
        # above are properties, so the bare name serialises a bound method.
        "mean_scores_by_scorer": run.mean_scores_by_scorer(),
        "git_sha": batch.git_sha if batch else None,
        "git_dirty": batch.git_dirty if batch else None,
    }


def run_detail(run: Run, batch: Batch | None = None) -> dict[str, Any]:
    """One run in full, with every Result, Score and Trace.

    `model_dump(mode="json")` rather than a hand-written projection: pydantic
    already knows how to serialise every field, including the `datetime`s and
    the `StrEnum`s, and a hand-written copy would silently drop any field added
    to the model later.
    """
    payload = run_summary(run, batch)
    payload["results"] = [result.model_dump(mode="json") for result in run.results]
    return payload


def _moment(value: Any) -> str | None:
    """An ISO timestamp, or null when there is none.

    A run that never recorded a start has no start. Substituting the epoch, or
    now, would put a fabricated moment in a field a reader sorts by.
    """
    return None if value is None else value.isoformat()


def create_app(store: RunStore, broker: Any = None) -> FastAPI:
    """Build the ASGI app over an open store.

    The store is injected rather than opened here so a test can pass a
    temporary database, and so `serve` owns the lifetime of the connection it
    opened. An app that opened its own store would hold a second connection to
    the same SQLite file for as long as the process lived.

    `broker` is the live-results fan-out. One is created when none is passed,
    so `serve` gets a working `/ui/live` without knowing what a broker is, and
    a caller that *is* running an eval can hand in the one the runner
    publishes to.
    """
    from evalstand.web.live import Broker

    broker = broker if broker is not None else Broker()
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, StreamingResponse

    from evalstand.web import fragments, live, page

    app = FastAPI(
        title="evalstand",
        description="Read-only JSON over local run history.",
        version=_version(),
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Enough to tell a running server from a reachable port.

        Reports the schema version because a database written by a newer
        evalstand is refused on open, and a reader seeing that refusal wants to
        know which version it was refused against.
        """
        return {
            "status": "ok",
            "version": _version(),
            "schema_version": store.schema_version,
            "run_count": store.run_count(),
        }

    @app.get("/api/evals")
    def evals() -> dict[str, Any]:
        """Every eval name history knows, for a picker."""
        return {"names": store.eval_names()}

    @app.get("/api/runs")
    def runs(
        name: str | None = Query(default=None, description="Filter to one eval."),
        limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> dict[str, Any]:
        """Recent runs, newest first, summaries only.

        Cancelled batches are excluded by `runs_for`: their aggregates describe
        a subset of the cases, so listing one beside a full run invites a
        comparison between two different questions.
        """
        found = store.runs_for(name, limit=limit)
        return {
            "runs": [run_summary(run, store.batch_for(run.id)) for run in found],
            "count": len(found),
        }

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        found = store.load_run(run_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no run with id {run_id!r}")
        return run_detail(found, store.batch_for(run_id))

    @app.get("/api/runs/{run_id}/results/{case_id}")
    def result(run_id: str, case_id: str, repeat: int = Query(default=0, ge=0)) -> dict[str, Any]:
        """One Result, addressed the way a user thinks of it.

        By `(case_id, repeat)` rather than by the synthetic Result id: a reader
        looking at a table knows the case failed, not what the runner called
        that execution. With `--repeat`, `?repeat=` picks which execution.
        """
        found = store.load_run(run_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no run with id {run_id!r}")

        for candidate in found.results:
            if candidate.case_id == case_id and candidate.repeat_index == repeat:
                return dict(candidate.model_dump(mode="json"))

        raise HTTPException(
            status_code=404,
            detail=f"run {run_id!r} has no case {case_id!r} at repeat {repeat}",
        )

    # ---- The HTMX front end (task 8.2) -------------------------------------
    #
    # Fragments, not JSON. The page holds no client-side model, so there is
    # nothing to drift out of step with the server — the server renders every
    # number, and the browser only swaps the HTML it is handed.

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page.index(htmx=page.htmx_source())

    @app.get("/ui/runs", response_class=HTMLResponse)
    def ui_runs(
        name: str | None = Query(default=None),
        limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> str:
        return fragments.run_table(store.runs_for(name, limit=limit))

    @app.get("/ui/runs/{run_id}", response_class=HTMLResponse)
    def ui_run(run_id: str) -> str:
        found = store.load_run(run_id)
        if found is None:
            # A fragment, not a JSON error: this response is swapped straight
            # into the page, and an HTTPException would put a raw error object
            # where the user expects a run.
            return "<p class='empty'>That run is no longer in the database.</p>"
        return fragments.run_detail(found)

    @app.get("/ui/live")
    async def ui_live() -> StreamingResponse:
        """The live table's event stream.

        Open whether or not anything is running. A browser that had to wait for
        a run to start before connecting would miss the first results of every
        run — the ones that arrive while it is still reconnecting.
        """
        broker.bind(asyncio.get_running_loop())
        queue = broker.subscribe()

        async def events() -> AsyncIterator[str]:
            try:
                async for payload in live.stream(queue):
                    yield payload
            finally:
                # Runs on a closed tab as well as a finished run. Without it
                # every reload leaks a subscriber and the broker fans out to
                # queues nobody reads.
                broker.unsubscribe(queue)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                # Buffering a stream defeats it: a proxy that holds events
                # until the response ends delivers the whole run at once, after
                # it is over.
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/batches/{batch_id}")
    def batch(batch_id: str) -> dict[str, Any]:
        """A batch and the runs beneath it.

        `batch_for` is keyed by run, so the batch is read through any one of
        its runs. A batch with no runs is a batch that recorded nothing, which
        is reported as absent rather than as an empty success.
        """
        for run_id in _run_ids_in(store, batch_id):
            found = store.batch_for(run_id)
            if found is not None:
                return {
                    "id": found.id,
                    "kind": found.kind.value,
                    "status": found.status.value,
                    "started_at": _moment(found.started_at),
                    "finished_at": _moment(found.finished_at),
                    "git_sha": found.git_sha,
                    "git_dirty": found.git_dirty,
                    # Deliberately surfaced: `history` hides a cancelled batch
                    # and `compare` refuses one, so a consumer needs to know
                    # before it treats these aggregates as a measurement.
                    "is_comparable": found.is_comparable,
                    "runs": [
                        run_summary(r, found)
                        for r in (store.load_run(i) for i in _run_ids_in(store, batch_id))
                        if r is not None
                    ],
                }

        raise HTTPException(status_code=404, detail=f"no batch with id {batch_id!r}")

    return app


def _run_ids_in(store: RunStore, batch_id: str) -> list[str]:
    """Run ids belonging to one batch.

    Delegates to `RunStore.run_ids_in`. An earlier draft ran the SQL here,
    reaching past the store's lock into its connection — which works until two
    requests arrive at once, and then fails in the way concurrent SQLite
    failures do: rarely, and somewhere else.
    """
    return store.run_ids_in(batch_id)


def _version() -> str:
    from evalstand import __version__

    return __version__

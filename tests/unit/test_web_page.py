"""The HTMX front end (task 8.2).

Three failures are worth catching, and none of them is a 500.

**A page that executes model output.** A completion containing `<script>` is an
inevitability — models are asked about HTML all the time — and a page that ran
it would be stored XSS in a server the user pointed at their own history.

**A page that disagrees with the terminal.** Every number is rendered by the
server from the same helpers the console uses. A fragment that computed its own
mean or pass count would eventually contradict `evalstand show` about the same
run, with neither saying so.

**A page that goes static without saying so.** htmx is vendored; a page that
fetched it from a CDN would silently become an unclickable table on a laptop
that is offline or behind a proxy.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalstand.models import Result, Run, RunStatus, Score, Trace
from evalstand.reporting.console import format_score, pass_counts

pytest.importorskip("fastapi", reason="the web extra is not installed")

from evalstand.web import fragments, page

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def _result(
    case_id: str = "q1",
    *,
    passed: bool | None = True,
    value: float | None = 1.0,
    cost: float | None = 0.001,
    repeat: int = 0,
    output: str = "answer",
    error: str | None = None,
    score_error: str | None = None,
) -> Result:
    return Result(
        id=f"r-{case_id}-{repeat}",
        case_id=case_id,
        repeat_index=repeat,
        output=output,
        error=error,
        latency_ms=12,
        scores=[Score(scorer_name="exact", value=value, passed=passed, error=score_error)],
        traces=[
            Trace(
                id=f"t-{case_id}-{repeat}",
                name="call",
                duration_ms=10,
                cost_usd=cost,
                started_at=NOW,
            )
        ],
    )


def _run(*, name: str = "demo", results: list[Result] | None = None, **kwargs: object) -> Run:
    fields: dict[str, object] = {
        "id": "run-1",
        "batch_id": "batch-1",
        "name": name,
        "filepath": "demo_eval.py",
        "status": RunStatus.COMPLETED,
        "started_at": NOW,
        "results": results if results is not None else [_result()],
        "model_calls": 2,
        "cache_hits": 1,
    }
    fields.update(kwargs)
    return Run(**fields)  # type: ignore[arg-type]


class TestItNeverExecutesWhatAModelSaid:
    """The one failure here that is a security hole rather than a wrong number."""

    def test_a_script_tag_in_a_case_id_is_escaped(self) -> None:
        html = fragments.case_row(_result("<script>alert(1)</script>"))

        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_a_script_tag_in_an_eval_name_is_escaped(self) -> None:
        html = fragments.run_table([_run(name="<img src=x onerror=alert(1)>")])

        assert "<img" not in html
        assert "&lt;img" in html

    def test_a_quote_cannot_break_out_of_an_attribute(self) -> None:
        """The row carries the run id inside an `hx-get` attribute. An
        unescaped quote there closes it and the rest becomes markup."""
        html = fragments.run_row(_run(id="a' hx-delete='/evil"))

        # Asserted on the *raw* quote, not on the escaped text. An earlier
        # version stripped `&#x27;` before checking, which removed the very
        # escaping under test and manufactured the hole it claimed to find.
        assert "' hx-delete" not in html
        assert "&#x27;" in html


class TestItAgreesWithTheTerminal:
    """Rendered from the same helpers, so the page cannot drift from `show`."""

    def test_the_pass_cell_is_the_terminals_count(self) -> None:
        run = _run(
            results=[
                _result("q1", passed=True, value=1.0),
                _result("q2", passed=False, value=0.0),
                # Unjudged: not a pass, not a failure, and not in the
                # denominator.
                _result("q3", passed=None, value=0.5),
            ]
        )
        passed, judged = pass_counts(run)

        assert f"{passed}/{judged}" in fragments.run_row(run)
        assert f"{passed}/{judged}" == "1/2"

    def test_the_mean_is_formatted_by_the_shared_helper(self) -> None:
        run = _run(results=[_result("q1", value=1.0), _result("q2", value=0.0, passed=False)])

        assert format_score(run.mean_score) in fragments.run_row(run)

    def test_the_five_statuses_are_the_live_views(self) -> None:
        """`fail` and `unmeasured` look alike in a table and mean opposite
        things: the model answering wrongly, versus the scorer breaking and the
        model's performance being unknown."""
        assert "badge pass" in fragments.case_row(_result(passed=True))
        assert "badge fail" in fragments.case_row(_result(passed=False, value=0.0))
        assert "badge error" in fragments.case_row(_result(error="boom"))
        assert "badge unmeasured" in fragments.case_row(
            _result(passed=None, value=None, score_error="scorer raised")
        )


class TestItDoesNotClaimZero:
    def test_an_unpriced_run_does_not_render_as_free(self) -> None:
        """`$0.0000` is a claim that the run was free. A run whose calls were
        never priced has not been shown to be anything."""
        run = _run(results=[_result(cost=None)])

        assert "$0.0000" not in fragments.run_row(run)
        assert "-" in fragments.run_row(run)

    def test_the_detail_says_a_partial_cost_is_a_lower_bound(self) -> None:
        run = _run(results=[_result("q1", cost=0.01), _result("q2", cost=None)])

        assert "lower bound" in fragments.run_detail(run)

    def test_a_fully_priced_run_says_nothing_of_the_sort(self) -> None:
        """The other half: without this the assertion above would hold no
        matter what the page did."""
        assert "lower bound" not in fragments.run_detail(_run())

    def test_a_bypassed_run_is_named_not_shown_as_zero_percent(self) -> None:
        """Bypassing spends the full amount every time. A 0% hit rate reads
        like a merely cold cache."""
        run = _run(cache_bypassed=True, cache_hits=0)

        assert "bypassed" in fragments.run_detail(run)

    def test_a_scorer_that_crashed_does_not_score_zero(self) -> None:
        run = _run(results=[_result(passed=None, value=None, score_error="scorer raised")])

        # The score cell specifically: `0.00` also occurs inside a cost like
        # `$0.0010`, so a substring search over the whole row proves nothing.
        cells = fragments.case_row(run.results[0]).split("<td")
        score_cell = cells[3]

        assert "0.00" not in score_cell
        assert "-" in score_cell


class TestTheEmptyState:
    def test_no_runs_says_so_rather_than_showing_an_empty_table(self) -> None:
        """ "No runs yet" and "runs exist but none matched" send a user looking
        in completely different places. An empty tbody says neither."""
        html = fragments.run_table([])

        assert "No runs recorded yet" in html
        assert "<tbody>" not in html

    def test_it_names_the_command_that_would_record_one(self) -> None:
        assert "evalstand run" in fragments.run_table([])


class TestTheVendoredLibrary:
    def test_the_digest_matches_what_is_shipped(self) -> None:
        """A vendored dependency nobody checks is a file anyone can replace."""
        source = (page.STATIC / "htmx.min.js").read_bytes()

        assert hashlib.sha256(source).hexdigest() == page.HTMX_SHA256

    def test_the_page_embeds_it_rather_than_fetching_it(self) -> None:
        """`serve` runs on laptops, sometimes offline. A page that fetched its
        interactivity would degrade to a static table without saying so."""
        html = page.index(htmx=page.htmx_source())

        assert "var htmx=function()" in html
        assert "unpkg.com" not in html
        assert "<script src=" not in html

    def test_the_recorded_url_matches_the_recorded_version(self) -> None:
        """The provenance a maintainer follows when updating it."""
        assert page.HTMX_VERSION in page.HTMX_URL

        readme = (page.STATIC / "README.md").read_text(encoding="utf-8")
        assert page.HTMX_SHA256 in readme
        assert page.HTMX_VERSION in readme


class TestThePage:
    def test_it_is_one_document_with_the_panes_the_app_needs(self) -> None:
        html = page.index(htmx="/* stub */")

        assert html.startswith("<!doctype html>")
        for required in ['id="runs"', 'id="detail"', 'id="live"']:
            assert required in html

    def test_it_carries_the_honesty_footnote(self) -> None:
        """The same sentence the artifact renderer ships. A dash means unknown,
        never zero — and deltas carry no significance testing."""
        html = page.index(htmx="")

        assert "never that it is zero" in html
        assert "no significance testing" in html

    def test_it_loads_the_run_list_without_a_click(self) -> None:
        assert 'hx-get="/ui/runs"' in page.index(htmx="")
        assert 'hx-trigger="load"' in page.index(htmx="")


class TestTheStylesheetIsShared:
    def test_it_uses_the_artifact_renderers_style(self) -> None:
        """Two copies of a colour scheme drift, and the point of the five
        status names is that `fail` and `unmeasured` never look alike."""
        from evalstand.reporting.html import STYLE

        assert STYLE in page.index(htmx="")


class TestTheStaticFilesShip:
    def test_the_wheel_would_include_the_vendored_asset(self) -> None:
        """A package that omits it serves a page with an empty `<script>`, and
        the UI silently stops swapping fragments."""
        import tomllib

        root = Path(__file__).resolve().parents[2]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]

        assert wheel["packages"] == ["src/evalstand"]
        assert (page.STATIC / "htmx.min.js").exists()


class TestTheUiRoutes:
    """The endpoints HTMX actually calls.

    Separate from the fragment tests above: those prove the HTML is right, these
    prove the page can reach it. A correct fragment behind a 404 is a blank
    pane.
    """

    @pytest.fixture
    def client(self, tmp_path: Path) -> object:
        from fastapi.testclient import TestClient

        from evalstand.storage import RunStore
        from evalstand.web.api import create_app

        with RunStore(tmp_path / "history.db") as store:
            yield TestClient(create_app(store))

    def test_the_index_serves_a_document(self, client: object) -> None:
        response = client.get("/")  # type: ignore[attr-defined]

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "<!doctype html>" in response.text

    def test_the_run_list_is_a_fragment_not_a_document(self, client: object) -> None:
        """It is swapped into an existing page. A full document here would nest
        one `<html>` inside another."""
        response = client.get("/ui/runs")  # type: ignore[attr-defined]

        assert response.status_code == 200
        assert "<!doctype" not in response.text.lower()

    def test_a_missing_run_swaps_a_sentence_not_an_error(self, client: object) -> None:
        """This response goes straight into the page. A JSON error object would
        appear where the user expects a run."""
        response = client.get("/ui/runs/gone")  # type: ignore[attr-defined]

        assert response.status_code == 200
        assert "no longer in the database" in response.text
        assert "detail" not in response.text

    def test_the_live_stream_announces_itself_as_sse(self, tmp_path: Path) -> None:
        """A wrong content type makes EventSource refuse the stream, and the
        page reports 'not live' against a server that is streaming fine.

        Driven through the ASGI interface rather than `TestClient`, and only as
        far as the response *headers*. `/ui/live` has no end by design — it
        stays open until the run finishes or the browser leaves — so a client
        that reads the body blocks until a keepalive, and a test that did so
        hung the suite rather than failing it.
        """
        import asyncio

        from evalstand.storage import RunStore
        from evalstand.web.api import create_app

        async def headers_of() -> dict[bytes, bytes]:
            with RunStore(tmp_path / "sse.db") as store:
                app = create_app(store)
                received: dict[bytes, bytes] = {}
                started = asyncio.Event()

                async def receive() -> dict[str, object]:
                    # Never yields a disconnect: the point is to reach the
                    # headers, which are sent before the first event.
                    await asyncio.sleep(3600)
                    return {"type": "http.disconnect"}

                async def send(message: dict[str, object]) -> None:
                    if message["type"] == "http.response.start":
                        received.update(dict(message["headers"]))  # type: ignore[arg-type]
                        started.set()

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": "/ui/live",
                    "raw_path": b"/ui/live",
                    "query_string": b"",
                    "root_path": "",
                    "headers": [],
                    "client": ("127.0.0.1", 1234),
                    "server": ("127.0.0.1", 8420),
                }

                task = asyncio.create_task(app(scope, receive, send))  # type: ignore[arg-type]
                await asyncio.wait_for(started.wait(), timeout=5.0)
                task.cancel()
                return received

        headers = asyncio.run(headers_of())

        assert headers[b"content-type"].startswith(b"text/event-stream")
        # Buffering defeats a stream: a proxy that holds events until the
        # response ends delivers the whole run at once, after it is over.
        assert headers[b"x-accel-buffering"] == b"no"
        assert headers[b"cache-control"] == b"no-cache"

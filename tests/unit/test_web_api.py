"""The JSON API over run history (task 8.1).

The defect this file is written against is not a 500. It is an endpoint that
answers confidently and wrongly — a pass rate that disagrees with the terminal,
a cost of `0` for a run that was never priced, a cancelled batch served as a
measurement. Every one of those looks like a working API.

So the assertions here are mostly *agreement* assertions: the API's numbers are
checked against the functions the terminal and the markdown reporter call, not
against literals copied from a fixture. A literal would keep passing after the
two implementations drifted apart, which is the exact failure worth catching.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from evalstand.models import (
    Batch,
    BatchKind,
    BatchStatus,
    Result,
    Run,
    RunStatus,
    Score,
    Trace,
)
from evalstand.reporting.console import pass_counts
from evalstand.storage import RunStore

fastapi = pytest.importorskip("fastapi", reason="the web extra is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from evalstand.web.api import create_app  # noqa: E402

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def _result(
    case_id: str,
    *,
    passed: bool | None = True,
    value: float | None = 1.0,
    cost: float | None = 0.001,
    repeat: int = 0,
) -> Result:
    return Result(
        id=f"r-{case_id}-{repeat}",
        case_id=case_id,
        repeat_index=repeat,
        output="answer",
        latency_ms=12,
        scores=[Score(scorer_name="exact", value=value, passed=passed)],
        traces=[
            Trace(
                id=f"t-{case_id}-{repeat}",
                name="call",
                duration_ms=10,
                model="gpt-4o-mini",
                cost_usd=cost,
                started_at=NOW,
            )
        ],
    )


def _run(
    run_id: str = "run-1",
    *,
    batch_id: str = "batch-1",
    name: str = "demo",
    results: list[Result] | None = None,
    started: datetime | None = None,
) -> Run:
    return Run(
        id=run_id,
        batch_id=batch_id,
        name=name,
        filepath="demo_eval.py",
        status=RunStatus.COMPLETED,
        started_at=started or NOW,
        finished_at=(started or NOW) + timedelta(seconds=3),
        results=results if results is not None else [_result("q1"), _result("q2")],
        model_calls=2,
        cache_hits=1,
    )


def _batch(batch_id: str = "batch-1", *, status: BatchStatus = BatchStatus.COMPLETED) -> Batch:
    return Batch(
        id=batch_id,
        kind=BatchKind.FULL,
        status=status,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=5),
        git_sha="a" * 40,
        git_dirty=False,
    )


@pytest.fixture
def store(tmp_path: Path) -> Any:
    with RunStore(tmp_path / "history.db") as opened:
        yield opened


@pytest.fixture
def client(store: RunStore) -> TestClient:
    return TestClient(create_app(store))


def _save(store: RunStore, run: Run, batch: Batch | None = None) -> None:
    store.save_batch(batch or _batch(run.batch_id))
    store.save_run(run)


class TestHealth:
    def test_it_reports_the_schema_version(self, client: TestClient, store: RunStore) -> None:
        """A database written by a newer evalstand is refused on open, so a
        reader wants to know which version it was refused against."""
        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert body["schema_version"] == store.schema_version

    def test_an_empty_database_is_healthy(self, client: TestClient) -> None:
        """Nothing recorded yet is a normal state, not a failure. CI starts
        every build here (ADR 0005)."""
        body = client.get("/api/health").json()

        assert body["run_count"] == 0


class TestTheListView:
    def test_it_lists_a_saved_run(self, client: TestClient, store: RunStore) -> None:
        _save(store, _run())

        body = client.get("/api/runs").json()

        assert body["count"] == 1
        assert body["runs"][0]["id"] == "run-1"

    def test_it_omits_results_from_the_list(self, client: TestClient, store: RunStore) -> None:
        """Results are the bulk of a Run. A list that carried them would
        serialise megabytes to render twenty rows."""
        _save(store, _run())

        assert "results" not in client.get("/api/runs").json()["runs"][0]

    def test_it_filters_by_eval_name(self, client: TestClient, store: RunStore) -> None:
        _save(store, _run("run-1", name="alpha"))
        _save(store, _run("run-2", batch_id="batch-2", name="beta"))

        body = client.get("/api/runs", params={"name": "alpha"}).json()

        assert [run["name"] for run in body["runs"]] == ["alpha"]

    def test_a_cancelled_batch_is_not_listed(self, client: TestClient, store: RunStore) -> None:
        """Its aggregates describe a subset of the cases. Listing one beside a
        full run invites a comparison between two different questions — and the
        reader cannot see that they are different."""
        _save(store, _run(), _batch(status=BatchStatus.CANCELLED))

        assert client.get("/api/runs").json()["count"] == 0

    def test_the_limit_is_bounded(self, client: TestClient) -> None:
        """A runaway query blocks the event loop while SQLite reads, and a page
        that hangs looks broken rather than busy."""
        assert client.get("/api/runs", params={"limit": 10_000}).status_code == 422
        assert client.get("/api/runs", params={"limit": 0}).status_code == 422


class TestItAgreesWithTheTerminal:
    """The assertions that matter most.

    Checked against `pass_counts` and the Run's own properties rather than
    against literals: a literal keeps passing after the API and the reporters
    drift apart, which is the failure this class exists to catch.
    """

    def test_the_pass_count_is_the_terminals(self, client: TestClient, store: RunStore) -> None:
        run = _run(
            results=[
                _result("q1", passed=True, value=1.0),
                _result("q2", passed=False, value=0.0),
                # Judged by nobody: a continuous scorer that declined. Not a
                # failure, and not a pass — it must leave the denominator alone.
                _result("q3", passed=None, value=0.5),  # judged by nobody
            ]
        )
        _save(store, run)

        body = client.get("/api/runs/run-1").json()
        expected_passed, expected_judged = pass_counts(run)

        assert (body["passed"], body["judged"]) == (expected_passed, expected_judged)
        assert body["judged"] == 2, "an unjudged result was counted in the denominator"

    def test_the_mean_is_the_models(self, client: TestClient, store: RunStore) -> None:
        # `passed` follows the value: the model refuses a Score that passes
        # with a value of 0.0, which is the invariant catching a fixture that
        # claimed both at once.
        run = _run(
            results=[
                _result("q1", value=1.0, passed=True),
                _result("q2", value=0.0, passed=False),
            ]
        )
        _save(store, run)

        assert client.get("/api/runs/run-1").json()["mean_score"] == run.mean_score

    def test_a_run_with_no_scores_has_a_null_mean(
        self, client: TestClient, store: RunStore
    ) -> None:
        """Not 0.0. A run nothing scored did not score zero."""
        run = _run(results=[Result(id="r1", case_id="q1", output="x")])
        _save(store, run)

        assert client.get("/api/runs/run-1").json()["mean_score"] is None


class TestItDoesNotClaimZero:
    """ADR 0009's honesty rule, which is the one a JSON consumer can least
    afford to get wrong: a number is summed, a dash is not."""

    def test_an_unpriced_run_is_marked_incomplete(
        self, client: TestClient, store: RunStore
    ) -> None:
        """`total_cost_usd` sums only what could be priced, so on its own it is
        a lower bound presented as a total. The two companion fields are what
        stop a consumer understating a bill."""
        _save(store, _run(results=[_result("q1", cost=None), _result("q2", cost=None)]))

        body = client.get("/api/runs/run-1").json()

        assert body["cost_is_complete"] is False
        assert body["unpriced_call_count"] == 2

    def test_a_priced_run_says_so(self, client: TestClient, store: RunStore) -> None:
        """The other half: without this, the assertion above would hold no
        matter what the field did."""
        _save(store, _run())

        body = client.get("/api/runs/run-1").json()

        assert body["cost_is_complete"] is True
        assert body["unpriced_call_count"] == 0

    def test_no_calls_means_a_null_hit_rate(self, client: TestClient, store: RunStore) -> None:
        """A run that made no calls did not have a 0% hit rate. It had no rate."""
        run = _run(results=[Result(id="r1", case_id="q1", output="x")])
        run = run.model_copy(update={"model_calls": 0, "cache_hits": 0})
        _save(store, run)

        assert client.get("/api/runs/run-1").json()["cache_hit_rate"] is None

    def test_a_missing_timestamp_is_null(self, client: TestClient, store: RunStore) -> None:
        """Substituting the epoch, or now, puts a fabricated moment in a field
        a reader sorts by."""
        run = _run().model_copy(update={"finished_at": None})
        _save(store, run)

        assert client.get("/api/runs/run-1").json()["finished_at"] is None


class TestTheDetailView:
    def test_it_carries_results_scores_and_traces(
        self, client: TestClient, store: RunStore
    ) -> None:
        _save(store, _run())

        body = client.get("/api/runs/run-1").json()

        assert len(body["results"]) == 2
        assert body["results"][0]["scores"][0]["scorer_name"] == "exact"
        assert body["results"][0]["traces"][0]["model"] == "gpt-4o-mini"

    def test_results_keep_declaration_order(self, client: TestClient, store: RunStore) -> None:
        """The runner promises case order, not completion order. A report whose
        rows shuffle between runs cannot be read or diffed."""
        _save(store, _run(results=[_result(f"q{n}") for n in range(1, 13)]))

        ids = [r["case_id"] for r in client.get("/api/runs/run-1").json()["results"]]

        assert ids == [f"q{n}" for n in range(1, 13)], "q10 sorted before q2"

    def test_an_unknown_run_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/nope").status_code == 404

    def test_it_reports_provenance(self, client: TestClient, store: RunStore) -> None:
        """Which commit produced these numbers. Without it a stored run cannot
        be tied to the code that made it."""
        _save(store, _run())

        body = client.get("/api/runs/run-1").json()

        assert body["git_sha"] == "a" * 40
        assert body["git_dirty"] is False


class TestOneResult:
    def test_it_is_addressed_by_case_and_repeat(self, client: TestClient, store: RunStore) -> None:
        """By `(case_id, repeat)`, not the synthetic Result id: a reader knows
        the case failed, not what the runner called that execution."""
        _save(store, _run(results=[_result("q1", repeat=0), _result("q1", repeat=1)]))

        first = client.get("/api/runs/run-1/results/q1").json()
        second = client.get("/api/runs/run-1/results/q1", params={"repeat": 1}).json()

        assert first["repeat_index"] == 0
        assert second["repeat_index"] == 1

    def test_an_unknown_case_is_a_404(self, client: TestClient, store: RunStore) -> None:
        _save(store, _run())

        assert client.get("/api/runs/run-1/results/nope").status_code == 404

    def test_a_repeat_that_did_not_run_is_a_404(self, client: TestClient, store: RunStore) -> None:
        """Not an empty body. A case that ran once has no second execution, and
        answering with nothing would read as one that produced no output."""
        _save(store, _run())

        assert client.get("/api/runs/run-1/results/q1", params={"repeat": 7}).status_code == 404


class TestBatches:
    def test_it_serves_the_runs_beneath_a_batch(self, client: TestClient, store: RunStore) -> None:
        batch = _batch()
        store.save_batch(batch)
        store.save_run(_run("run-1"))
        store.save_run(_run("run-2", name="other"))

        body = client.get("/api/batches/batch-1").json()

        assert {run["id"] for run in body["runs"]} == {"run-1", "run-2"}

    def test_a_cancelled_batch_is_served_but_marked(
        self, client: TestClient, store: RunStore
    ) -> None:
        """Served, unlike in the list view — a user who asks for it by id is
        entitled to see it. Marked, because `history` hides one and `compare`
        refuses one, so a consumer must know before treating these aggregates
        as a measurement."""
        _save(store, _run(), _batch(status=BatchStatus.CANCELLED))

        body = client.get("/api/batches/batch-1").json()

        assert body["status"] == "cancelled"
        assert body["is_comparable"] is False

    def test_an_unknown_batch_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/batches/nope").status_code == 404


class TestTheEvalIndex:
    def test_it_names_every_eval_in_history(self, client: TestClient, store: RunStore) -> None:
        _save(store, _run("run-1", name="alpha"))
        _save(store, _run("run-2", batch_id="batch-2", name="beta"))

        assert client.get("/api/evals").json()["names"] == ["alpha", "beta"]

    def test_an_empty_database_lists_nothing(self, client: TestClient) -> None:
        assert client.get("/api/evals").json()["names"] == []

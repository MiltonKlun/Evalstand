"""The run store (task 5.1).

Unlike the response cache, this is load-bearing: deleting the cache costs money,
corrupting this loses the comparison data the tool exists to provide. So the
tests here are mostly about what must *never* happen — a half-written run, a
half-applied migration, a silently degraded read.

Three SQLite behaviours drive the design, each verified directly rather than
assumed:

- **Foreign keys are OFF by default.** Without `PRAGMA foreign_keys = ON` the
  cascade rules are decoration and an orphan row is accepted silently.
- **`executescript` issues an implicit COMMIT**, which would end the transaction
  wrapping a migration and leave the database half-applied on a later failure.
- **Both pragmas are per-connection**, so they are set on every open.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evalstand.migrations import LATEST_VERSION, MIGRATIONS
from evalstand.models import (
    Batch,
    BatchKind,
    BatchStatus,
    Case,
    Result,
    Run,
    RunStatus,
    Score,
    Trace,
)
from evalstand.storage import DatabaseTooNewError, RunStore

EXPECTED_TABLES = {"batches", "runs", "results", "scores", "traces", "case_snapshots"}


@pytest.fixture
def store(tmp_path: Path) -> Any:
    with RunStore(tmp_path / "evalstand.db") as opened:
        yield opened


def _tables(store: RunStore) -> set[str]:
    return {
        row[0]
        for row in store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _batch(batch_id: str = "b1", **kwargs: Any) -> Batch:
    defaults: dict[str, Any] = {
        "kind": BatchKind.FULL,
        "status": BatchStatus.RUNNING,
        "started_at": datetime.now(UTC),
    }
    return Batch(id=batch_id, **(defaults | kwargs))


def _run(run_id: str = "r1", batch_id: str = "b1", **kwargs: Any) -> Run:
    defaults: dict[str, Any] = {
        "name": "qa",
        "filepath": "qa_eval.py",
        "status": RunStatus.COMPLETED,
        "started_at": datetime.now(UTC),
        "finished_at": datetime.now(UTC),
        "results": [
            Result(
                id=f"{run_id}-q1",
                case_id="q1",
                output="Paris",
                scores=[Score(scorer_name="exact", value=1.0, passed=True)],
            )
        ],
    }
    return Run(id=run_id, batch_id=batch_id, **(defaults | kwargs))


class TestTheMigration:
    """Task 5.1's acceptance, point by point."""

    def test_it_applies_to_an_empty_database(self, store: RunStore) -> None:
        assert store.schema_version == LATEST_VERSION
        assert _tables(store) == EXPECTED_TABLES

    def test_it_is_idempotent(self, tmp_path: Path) -> None:
        """Reopening must not re-run a migration already applied."""
        path = tmp_path / "twice.db"
        with RunStore(path) as first:
            before = _tables(first)
        with RunStore(path) as second:
            assert _tables(second) == before
            assert second.schema_version == LATEST_VERSION

    def test_a_newer_database_is_refused(self, tmp_path: Path) -> None:
        """Opening it best-effort would let a newer column go unread, and a
        comparison built from a partial read would look right while answering a
        different question."""
        path = tmp_path / "newer.db"
        connection = sqlite3.connect(path)
        connection.execute(f"PRAGMA user_version = {LATEST_VERSION + 5}")
        connection.commit()
        connection.close()

        with pytest.raises(DatabaseTooNewError) as caught:
            RunStore(path)

        message = str(caught.value)
        assert str(LATEST_VERSION + 5) in message, "the error must name the database's version"
        assert str(LATEST_VERSION) in message, "and the version this build understands"

    def test_a_failed_migration_leaves_the_database_untouched(self, tmp_path: Path) -> None:
        """The reason statements are executed one at a time. `executescript`
        commits implicitly, so a failure after it could not be rolled back and
        the database would be left believing it holds columns it does not."""
        path = tmp_path / "broken.db"
        broken = {1: ["CREATE TABLE half_applied(id INTEGER)", "THIS IS NOT SQL"]}

        with (
            patch("evalstand.storage.MIGRATIONS", broken),
            pytest.raises(sqlite3.OperationalError),
        ):
            RunStore(path)

        connection = sqlite3.connect(path)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        connection.close()

        assert "half_applied" not in tables, "a partial migration survived"
        assert version == 0, "the version was bumped for a migration that did not apply"

    def test_the_database_is_usable_after_a_failed_migration(self, tmp_path: Path) -> None:
        """A rollback that leaves the file unopenable would be no better than a
        half-applied one."""
        path = tmp_path / "recovered.db"
        with (
            patch("evalstand.storage.MIGRATIONS", {1: ["NOT SQL"]}),
            pytest.raises(sqlite3.OperationalError),
        ):
            RunStore(path)

        with RunStore(path) as recovered:
            assert recovered.schema_version == LATEST_VERSION
            assert _tables(recovered) == EXPECTED_TABLES

    def test_migrations_are_numbered_from_one_without_gaps(self) -> None:
        """A gap would mean a database could sit at a version no migration
        takes it out of."""
        assert sorted(MIGRATIONS) == list(range(1, LATEST_VERSION + 1))


class TestTheDatabaseEnforcesItsOwnInvariants:
    """The schema is the last line of defence. Rows also arrive from other
    processes and older builds, so the constraints must hold without help."""

    def test_foreign_keys_are_actually_enforced(self, store: RunStore) -> None:
        """SQLite has them OFF by default, so without the pragma every ON
        DELETE CASCADE in the schema is decoration."""
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store.connection.execute(
                "INSERT INTO runs (id, batch_id, name, filepath, status) "
                "VALUES ('orphan', 'no-such-batch', 'x', 'f', 'completed')"
            )

    def test_a_duplicate_execution_is_rejected(self, store: RunStore) -> None:
        """`(run_id, case_id, repeat_index)` names one execution. The same
        invariant the Run model enforces — asserted here too, because rows can
        reach the table without passing through the model."""
        store.save_batch(_batch())
        store.save_run(_run())

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            store.connection.execute(
                "INSERT INTO results (id, run_id, case_id, repeat_index) "
                "VALUES ('another', 'r1', 'q1', 0)"
            )

    def test_repeats_of_one_case_are_not_duplicates(self, store: RunStore) -> None:
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(id=f"r1-q1-{i}", case_id="q1", repeat_index=i, output="x")
                    for i in range(3)
                ]
            )
        )
        assert store.connection.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 3

    def test_deleting_a_batch_removes_its_runs_and_results(self, store: RunStore) -> None:
        """The cascade the foreign keys exist for. Orphan results would inflate
        every later aggregate silently."""
        store.save_batch(_batch())
        store.save_run(_run())
        store.connection.execute("DELETE FROM batches WHERE id = 'b1'")

        assert store.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        assert store.connection.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
        assert store.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0


class TestWritingARun:
    def test_a_second_run_does_not_corrupt_the_first(self, store: RunStore) -> None:
        """Task 5.1's acceptance. Each run is its own transaction, so the second
        cannot disturb the first."""
        store.save_batch(_batch())
        store.save_run(_run("r1"))
        store.save_run(_run("r2"))

        stored = dict(
            store.connection.execute(
                "SELECT r.id, s.value_float FROM runs r "
                "JOIN results t ON t.run_id = r.id JOIN scores s ON s.result_id = t.id"
            )
        )
        assert stored == {"r1": 1.0, "r2": 1.0}
        assert store.run_count() == 2

    def test_a_save_that_fails_part_way_writes_nothing(self, store: RunStore) -> None:
        """A partial run in history is a lie about what ran, and every later
        mean computed from it inherits the lie."""
        store.save_batch(_batch())
        run = _run(
            results=[
                Result(
                    id=f"r1-q{i}",
                    case_id=f"q{i}",
                    output="x",
                    scores=[Score(scorer_name="s", value=1.0)],
                )
                for i in range(3)
            ]
        )

        original = store._insert_result
        seen = {"count": 0}

        def failing(run_id: str, result: Result, ordinal: int) -> None:
            seen["count"] += 1
            if seen["count"] == 3:
                raise RuntimeError("disk full")
            original(run_id, result, ordinal)

        with (
            patch.object(store, "_insert_result", failing),
            pytest.raises(RuntimeError, match="disk full"),
        ):
            store.save_run(run)

        for table in ("runs", "results", "scores"):
            count = store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 0, f"{table} kept rows from a run that failed to save"

    def test_scores_traces_and_snapshots_are_all_written(self, store: RunStore) -> None:
        store.save_batch(_batch())
        run = _run(
            results=[
                Result(
                    id="r1-q1",
                    case_id="q1",
                    output="Paris",
                    scores=[Score(scorer_name="exact", value=1.0, passed=True)],
                    traces=[
                        Trace(
                            id="t1",
                            name="model call",
                            model="gpt-4o-mini",
                            duration_ms=12,
                            input_tokens=10,
                            output_tokens=3,
                            cost_usd=0.0002,
                        )
                    ],
                )
            ]
        )
        store.save_run(run, cases=[Case(id="q1", input="France", expected="Paris")])

        for table in ("runs", "results", "scores", "traces", "case_snapshots"):
            count = store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count == 1, f"nothing was written to {table}"

    def test_the_case_snapshot_stores_a_real_hash(self, store: RunStore) -> None:
        """`Case.content_hash` is a *method*. Storing it without calling it put
        a bound-method object in the column — caught only because SQLite refused
        to bind it. A `str()` anywhere in that path would have written
        "<bound method...>" as the hash and made every later Amended Case
        comparison meaningless.
        """
        case = Case(id="q1", input="France", expected="Paris")
        store.save_batch(_batch())
        store.save_run(_run(), cases=[case])

        stored = store.connection.execute(
            "SELECT content_hash FROM case_snapshots WHERE case_id = 'q1'"
        ).fetchone()[0]

        assert stored == case.content_hash()
        assert len(stored) == 64, "a sha256 hex digest"
        assert "method" not in stored

    def test_a_changed_expected_value_changes_the_hash(self, store: RunStore) -> None:
        """What makes Amended Case detection work: the same case_id with a
        different expected value is not the same test."""
        original = Case(id="q1", input="France", expected="Paris")
        amended = Case(id="q1", input="France", expected="Lyon")
        assert original.content_hash() != amended.content_hash()


class TestWhatIsStoredIsWhatWasMeasured:
    """The store must not quietly change a value on the way in. A number that
    means something different once written is the storage-layer version of a
    false pass."""

    def test_an_absent_verdict_stays_absent(self, store: RunStore) -> None:
        """`passed` is nullable. Coercing None to 0 would make a scorer that
        declined to judge indistinguishable from one that failed the case."""
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(
                        id="r1-q1",
                        case_id="q1",
                        output="x",
                        scores=[Score(scorer_name="levenshtein", value=0.6)],
                    )
                ]
            )
        )

        assert store.connection.execute("SELECT passed FROM scores").fetchone()[0] is None

    def test_an_errored_score_stores_no_value(self, store: RunStore) -> None:
        """Storing 0.0 would make an infrastructure failure indistinguishable
        from a task that did badly."""
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(
                        id="r1-q1",
                        case_id="q1",
                        output="x",
                        scores=[Score.from_error("judge", "rate limited")],
                    )
                ]
            )
        )

        row = store.connection.execute("SELECT value_float, error FROM scores").fetchone()
        assert row[0] is None
        assert "rate limited" in row[1]

    @pytest.mark.parametrize("passed,expected", [(True, 1), (False, 0), (None, None)])
    def test_every_verdict_round_trips(
        self, store: RunStore, passed: bool | None, expected: int | None
    ) -> None:
        store.save_batch(_batch())
        value = 1.0 if passed is not False else 0.0
        store.save_run(
            _run(
                results=[
                    Result(
                        id="r1-q1",
                        case_id="q1",
                        output="x",
                        scores=[Score(scorer_name="s", value=value, passed=passed)],
                    )
                ]
            )
        )

        assert store.connection.execute("SELECT passed FROM scores").fetchone()[0] == expected

    def test_an_unknown_git_state_is_not_recorded_as_clean(self, store: RunStore) -> None:
        """`git_dirty=None` means the tree was never checked, which is a
        different claim from "checked and clean". Writing 0 would assert a
        provenance fact nobody established."""
        store.save_batch(_batch(git_dirty=None))

        assert store.connection.execute("SELECT git_dirty FROM batches").fetchone()[0] is None

    def test_a_structured_output_keeps_its_structure(self, store: RunStore) -> None:
        """Flattening a dict into text would lose the ability to compare it
        field by field, which is what `compare` will need."""
        store.save_batch(_batch())
        store.save_run(
            _run(results=[Result(id="r1-q1", case_id="q1", output={"total": 42, "vendor": "Acme"})])
        )

        import json

        row = store.connection.execute("SELECT output_text, output_json FROM results").fetchone()
        assert json.loads(row[1]) == {"total": 42, "vendor": "Acme"}
        assert row[0] is not None, "the text column should still hold something readable"

    def test_an_unserialisable_output_does_not_lose_the_measurement(self, store: RunStore) -> None:
        """A task may return anything. Failing the write over a formatting
        problem would discard a result that was paid for."""

        class Exotic:
            def __repr__(self) -> str:
                return "<a live connection>"

        store.save_batch(_batch())
        store.save_run(_run(results=[Result(id="r1-q1", case_id="q1", output=Exotic())]))

        row = store.connection.execute("SELECT output_text FROM results").fetchone()
        assert "live connection" in row[0]


class TestOpeningTheStore:
    def test_it_creates_the_parent_directory(self, tmp_path: Path) -> None:
        """The default path is `.evalstand/evalstand.db`, which will not exist
        on a first run."""
        path = tmp_path / "nested" / "deeper" / "evalstand.db"
        with RunStore(path):
            pass
        assert path.exists()

    def test_a_bare_filename_needs_no_directory(self, tmp_path: Path, monkeypatch: Any) -> None:
        """`Path("x.db").parent` is `.`, which must not be created or treated
        as missing."""
        monkeypatch.chdir(tmp_path)
        with RunStore("bare.db") as opened:
            assert opened.schema_version == LATEST_VERSION

    def test_two_stores_can_share_one_database(self, tmp_path: Path) -> None:
        """WAL mode exists so a `history` command can read while a run writes."""
        path = tmp_path / "shared.db"
        with RunStore(path) as writer, RunStore(path) as reader:
            writer.save_batch(_batch())
            writer.save_run(_run())
            assert reader.run_count() == 1


class TestSerialisationNeverLosesARun:
    """A whole run's results must not be lost because one output is exotic.

    `json.dumps(default=str)` handles almost anything, which makes the fallback
    paths easy to leave untested — and they cover the two cases that would
    otherwise take down an entire save.
    """

    def test_a_circular_output_is_stored_rather_than_raising(self, store: RunStore) -> None:
        """`json` rejects a cycle with a ValueError however the leaves encode,
        so `default=str` does not save this one."""
        circular: dict[str, Any] = {}
        circular["self"] = circular

        store.save_batch(_batch())
        store.save_run(_run(results=[Result(id="r1-q1", case_id="q1", output=circular)]))

        row = store.connection.execute("SELECT output_json FROM results").fetchone()
        assert row[0] is not None, "a cyclic output lost the row"

    def test_an_output_whose_repr_raises_does_not_kill_the_save(self, store: RunStore) -> None:
        """The dangerous one. A hostile `__repr__` escapes as whatever it chose
        to throw, so catching only TypeError and ValueError would let one bad
        object discard every other result in the run."""

        class Hostile:
            def __repr__(self) -> str:
                raise RuntimeError("repr exploded")

            def __str__(self) -> str:
                raise RuntimeError("str exploded")

        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(id="r1-q1", case_id="q1", output="fine"),
                    Result(id="r1-q2", case_id="q2", output=Hostile()),
                ]
            )
        )

        rows = store.connection.execute(
            "SELECT case_id, output_text FROM results ORDER BY case_id"
        ).fetchall()
        assert len(rows) == 2, "one hostile output discarded the whole run"
        assert rows[0][1] == "fine", "the good result must survive intact"
        assert "Hostile" in rows[1][1], "the type is worth more than losing the row"

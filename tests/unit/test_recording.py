"""Recording a run to history (task 5.2), end to end.

Two of these correspond to bugs found by running the tool rather than by
reading it, and both failed *silently* — the summary printed normally, the exit
code was zero, and nothing was stored.

The governing rule is asymmetric on purpose: **provenance is checked before
anything executes, and storage failures never cost a measurement.** By the time
a run reaches the database the money is already spent.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evalstand.models import Result, Run, RunStatus, Score
from evalstand.provenance import DirtyTreeError, GitState
from evalstand.recording import BatchRecorder, open_recorder
from evalstand.storage import DatabaseTooNewError, RunStore


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)


def _run(run_id: str, batch_id: str, name: str = "qa") -> Run:
    return Run(
        id=run_id,
        batch_id=batch_id,
        name=name,
        filepath="qa_eval.py",
        status=RunStatus.COMPLETED,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        results=[
            # The id shape the runner really produces: deterministic per
            # (eval, case, repeat), and therefore identical across runs.
            Result(
                id=f"{name}-q1-0",
                case_id="q1",
                repeat_index=0,
                output="Paris",
                scores=[Score(scorer_name="exact", value=1.0, passed=True)],
            )
        ],
    )


@pytest.fixture
def store(tmp_path: Path) -> Any:
    with RunStore(tmp_path / "history.db") as opened:
        yield opened


class TestHistoryAccumulates:
    """The bug that made Phase 5 pointless.

    A Result's id is `{eval}-{case}-{repeat}` — stable within a run, and so
    identical across runs of the same eval. With a bare PRIMARY KEY on that
    column, the second run of an eval hit `UNIQUE constraint failed:
    results.id`, the write was rolled back, and the failure was swallowed as a
    warning. History held one entry per eval **forever**, and nothing said so:
    the summary printed, the exit code was zero, and the batch row was there
    with no results under it.
    """

    def test_running_the_same_eval_twice_stores_both_runs(self, store: RunStore) -> None:
        recorder = BatchRecorder(store, git=GitState())
        recorder.record(_run("run-1", recorder.batch_id))
        recorder.record(_run("run-2", recorder.batch_id))

        assert store.run_count() == 2, "the second run of an eval was silently dropped"
        results = store.connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        assert results == 2

    def test_ten_runs_of_one_eval_all_persist(self, store: RunStore) -> None:
        """History is the point of the phase. One entry is not history."""
        recorder = BatchRecorder(store, git=GitState())
        for index in range(10):
            recorder.record(_run(f"run-{index}", recorder.batch_id))

        assert store.run_count() == 10

    def test_a_batch_never_has_runs_missing_under_it(self, store: RunStore) -> None:
        """The visible shape of the bug: a batch row with nothing beneath it,
        which reads as a run that measured nothing."""
        recorder = BatchRecorder(store, git=GitState())
        recorder.record(_run("run-1", recorder.batch_id))
        recorder.finish()

        orphaned = store.connection.execute(
            "SELECT b.id FROM batches b LEFT JOIN runs r ON r.batch_id = b.id WHERE r.id IS NULL"
        ).fetchall()
        assert orphaned == [], "a batch was recorded with no runs under it"

    def test_scores_and_traces_survive_the_second_run(self, store: RunStore) -> None:
        """The child tables carry run_id for the same reason: a Result is
        identified by (run_id, id), so scoping only the parent would move the
        collision one level down."""
        recorder = BatchRecorder(store, git=GitState())
        recorder.record(_run("run-1", recorder.batch_id))
        recorder.record(_run("run-2", recorder.batch_id))

        assert store.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2


class TestProvenanceIsRecorded:
    def test_the_batch_carries_the_commit_and_tree_state(self, store: RunStore) -> None:
        recorder = BatchRecorder(store, git=GitState(sha="a" * 40, dirty=False))
        recorder.finish()

        row = store.connection.execute("SELECT git_sha, git_dirty FROM batches").fetchone()
        assert row[0] == "a" * 40
        assert row[1] == 0

    def test_an_unknown_tree_state_stays_unknown(self, store: RunStore) -> None:
        """`None` means "not checked", which is not the claim `False` makes."""
        BatchRecorder(store, git=GitState()).finish()

        row = store.connection.execute("SELECT git_sha, git_dirty FROM batches").fetchone()
        assert tuple(row) == (None, None)

    def test_the_run_carries_the_task_source_hash(self, store: RunStore) -> None:
        """What lets a comparison tell a changed model from changed code."""

        def task(value: str) -> str:
            return value.upper()

        recorder = BatchRecorder(store, git=GitState())
        recorder.record(_run("run-1", recorder.batch_id), task=task)

        stored = store.connection.execute("SELECT task_source_hash FROM runs").fetchone()[0]
        assert stored is not None
        assert len(stored) == 64

    def test_a_finished_batch_is_marked_completed(self, store: RunStore) -> None:
        recorder = BatchRecorder(store, git=GitState())
        recorder.finish()

        assert store.connection.execute("SELECT status FROM batches").fetchone()[0] == "completed"

    def test_an_interrupted_batch_is_marked_cancelled(self, store: RunStore) -> None:
        """Its aggregate describes a subset of the cases, so comparing it
        against a full run would be comparing different questions."""
        recorder = BatchRecorder(store, git=GitState())
        recorder.finish(cancelled=True)

        assert store.connection.execute("SELECT status FROM batches").fetchone()[0] == "cancelled"


class TestAStorageFailureNeverCostsAMeasurement:
    """By the time a run is written the money is already spent."""

    def test_a_failing_write_does_not_raise(self, store: RunStore) -> None:
        recorder = BatchRecorder(store, git=GitState())

        with patch.object(store, "save_run", side_effect=RuntimeError("database is locked")):
            recorder.record(_run("run-1", recorder.batch_id))  # must not raise

    def test_the_user_is_warned_once_not_per_run(
        self, store: RunStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A message repeated per run would bury the summary the user came
        for."""
        recorder = BatchRecorder(store, git=GitState())

        with (
            patch.object(store, "save_run", side_effect=RuntimeError("locked")),
            caplog.at_level("WARNING", logger="evalstand.recording"),
        ):
            for index in range(5):
                recorder.record(_run(f"run-{index}", recorder.batch_id))

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_an_unopenable_database_does_not_stop_the_run(self, tmp_path: Path) -> None:
        """The user asked to measure a model, not to maintain a database."""
        with patch("evalstand.recording.RunStore", side_effect=OSError("permission denied")):
            # `root=tmp_path` so this reads the (non-)repository under test
            # rather than whatever state evalstand's own checkout is in.
            assert open_recorder(path=tmp_path / "x.db", root=tmp_path) is None

    def test_a_too_new_database_is_still_refused(self, tmp_path: Path) -> None:
        """The one failure that is NOT downgraded. Continuing would run the
        eval and then silently fail to record it, leaving the user with a
        database they cannot read and no sign anything is wrong."""
        with (
            patch(
                "evalstand.recording.RunStore",
                side_effect=DatabaseTooNewError("schema version 999"),
            ),
            pytest.raises(DatabaseTooNewError),
        ):
            open_recorder(path=tmp_path / "x.db", root=tmp_path)


class TestOpeningTheRecorder:
    def test_disabled_returns_nothing(self, tmp_path: Path) -> None:
        assert open_recorder(path=tmp_path / "x.db", enabled=False) is None

    def test_a_dirty_tree_refuses_before_anything_runs(self, tmp_path: Path) -> None:
        """Checked up front. Discovering that results cannot be recorded after
        paying for them would be the worst possible ordering."""
        with (
            patch("evalstand.recording.git_state", return_value=GitState(sha="a" * 40, dirty=True)),
            pytest.raises(DirtyTreeError),
        ):
            open_recorder(path=tmp_path / "x.db")

    def test_allow_dirty_permits_it(self, tmp_path: Path) -> None:
        with patch(
            "evalstand.recording.git_state", return_value=GitState(sha="a" * 40, dirty=True)
        ):
            recorder = open_recorder(path=tmp_path / "x.db", allow_dirty=True)

        assert recorder is not None
        recorder.finish()

    def test_a_directory_with_no_repository_still_records(self, tmp_path: Path) -> None:
        """Per the design decision: not being in a repo is a normal way to try
        the tool, and the null SHA already says the run cannot be tied to a
        commit."""
        with patch("evalstand.recording.git_state", return_value=GitState()):
            recorder = open_recorder(path=tmp_path / "x.db")

        assert recorder is not None
        recorder.finish()

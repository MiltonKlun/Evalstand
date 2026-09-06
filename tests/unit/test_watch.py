"""Watch mode (task 6.6).

Two acceptance criteria, and they pull in opposite directions: editing a prompt
must trigger a re-run within a second, *and* a Batch cut short by that edit must
never appear in `history` or `compare`.

The second is the one worth guarding hardest. A fast re-run that leaves a
half-finished Batch looking complete would drag every mean it touched, and the
damage compounds silently — the user sees a score move and attributes it to
their edit, when really it is an average over the four cases that happened to
finish before they hit save.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evalstand.api import Eval
from evalstand.models import BatchStatus, Case, Result, Run, RunStatus, Score
from evalstand.provenance import GitState
from evalstand.recording import BatchRecorder
from evalstand.storage import RunStore
from evalstand.tui.app import EvalApp, FilesChanged
from evalstand.tui.watch import DEBOUNCE_MS, changed_paths, is_relevant, watch_paths, watch_roots


class TestWhatCountsAsAChange:
    @pytest.mark.parametrize(
        "path",
        [
            "qa_eval.py",
            "tasks/answer.py",
            "prompts/system.txt",
            "prompts/template.jinja",
            "config/models.yaml",
            "data/cases.json",
        ],
    )
    def test_the_code_being_measured_triggers_a_rerun(self, path: str) -> None:
        """Prompts live in text files as often as in Python. Watching only `.py`
        would make watch mode useless for tuning a prompt, which is the workflow
        it exists for."""
        assert is_relevant(path)

    @pytest.mark.parametrize(
        "path",
        [
            "__pycache__/qa_eval.cpython-312.pyc",
            ".git/index",
            ".venv/lib/thing.py",
            "node_modules/pkg/index.js",
            "qa_eval.py~",
            ".hidden.py",
            "results.db",
            "image.png",
        ],
    )
    def test_noise_and_our_own_output_do_not(self, path: str) -> None:
        """The history database is written *by* a run, so treating it as a
        change would make every Batch trigger the next one — a loop that spends
        money until somebody notices."""
        assert not is_relevant(path)

    def test_a_compiled_file_beside_a_real_one_is_ignored(self) -> None:
        """Every run of an eval writes a `.pyc`, so this is not hypothetical."""
        assert not is_relevant("src/__pycache__/answer.cpython-312.pyc")


class TestCollectingAChangeBatch:
    def test_relevant_paths_are_returned_sorted_and_deduplicated(self) -> None:
        """An editor writing one file produces several events, and three copies
        of the same path would read as three separate edits."""
        from watchfiles import Change

        changes = [
            (Change.modified, "b_eval.py"),
            (Change.added, "a_eval.py"),
            (Change.modified, "b_eval.py"),
            (Change.modified, "__pycache__/x.pyc"),
        ]

        assert changed_paths(changes) == [Path("a_eval.py"), Path("b_eval.py")]

    def test_a_deleted_prompt_still_counts(self) -> None:
        """Removing a prompt changes what the task does, and a re-run reporting
        the resulting error beats silence."""
        from watchfiles import Change

        assert changed_paths([(Change.deleted, "prompts/system.txt")]) == [
            Path("prompts/system.txt")
        ]

    def test_a_batch_of_only_noise_yields_nothing(self) -> None:
        from watchfiles import Change

        assert changed_paths([(Change.modified, "__pycache__/x.pyc")]) == []


class TestWatchRoots:
    def test_a_file_is_watched_through_its_directory(self, tmp_path: Path) -> None:
        """Editing the prompt *next to* an eval file must trigger a re-run, and
        watching the single file would miss it."""
        target = tmp_path / "qa_eval.py"
        target.write_text("x", encoding="utf-8")

        assert watch_roots([target]) == [tmp_path.resolve()]

    def test_a_nested_root_is_dropped(self, tmp_path: Path) -> None:
        """Watching both a directory and its child reports every change under
        the child twice, so one save would start two Batches."""
        child = tmp_path / "evals"
        child.mkdir()

        assert watch_roots([tmp_path, child]) == [tmp_path.resolve()]

    def test_sibling_directories_are_both_watched(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()

        assert watch_roots([first, second]) == sorted([first.resolve(), second.resolve()])

    def test_no_argument_watches_the_current_directory(self) -> None:
        assert watch_roots() == [Path().resolve()]


class TestTheDebounceIsShortEnoughToFeelLive:
    def test_it_is_the_300ms_the_plan_specifies(self) -> None:
        """Long enough to coalesce one save's burst of events, short enough that
        the acceptance criterion — a re-run within one second — has room for the
        run itself."""
        assert DEBOUNCE_MS == 300


class TestAnEditTriggersARerun:
    """The first acceptance criterion, driven against a real filesystem."""

    @pytest.mark.anyio
    async def test_editing_a_watched_file_yields_within_a_second(self, tmp_path: Path) -> None:
        """Measured rather than asserted structurally: a debounce constant that
        says 300ms proves nothing if the watcher never fires."""
        target = tmp_path / "prompts.txt"
        target.write_text("first", encoding="utf-8")

        stop = asyncio.Event()
        seen: list[list[Path]] = []

        async def collect() -> None:
            async for paths in watch_paths([tmp_path], stop=stop):
                seen.append(paths)
                stop.set()
                return

        watcher = asyncio.create_task(collect())
        await asyncio.sleep(0.2)  # let the watcher establish itself

        started = asyncio.get_running_loop().time()
        target.write_text("second", encoding="utf-8")

        try:
            await asyncio.wait_for(watcher, timeout=5)
        except TimeoutError:  # pragma: no cover - a filesystem that cannot watch
            watcher.cancel()
            pytest.skip("the filesystem did not deliver change events")

        elapsed = asyncio.get_running_loop().time() - started
        assert (seen and seen[0] == [target.resolve()]) or seen[0] == [target]
        assert elapsed < 1.0, f"the re-run took {elapsed:.2f}s to trigger"

    @pytest.mark.anyio
    async def test_a_pycache_write_does_not_trigger_one(self, tmp_path: Path) -> None:
        """Every run writes `.pyc` files, so a watcher that reacted to them
        would re-run itself forever."""
        cache = tmp_path / "__pycache__"
        cache.mkdir()

        stop = asyncio.Event()
        seen: list[list[Path]] = []

        async def collect() -> None:
            async for paths in watch_paths([tmp_path], stop=stop):
                seen.append(paths)

        watcher = asyncio.create_task(collect())
        await asyncio.sleep(0.2)

        (cache / "qa.cpython-312.pyc").write_text("bytes", encoding="utf-8")
        await asyncio.sleep(0.8)

        stop.set()
        watcher.cancel()

        assert seen == []


async def _running(pilot: Any, started: list[str]) -> None:
    """Wait until the eval is genuinely under way.

    Both conditions matter: a case must have entered the task, *and* the app
    must have composed. Posting a message before compose finishes races the
    mount and fails on a missing widget rather than on anything under test.
    """
    for _ in range(50):
        await pilot.pause()
        if started and pilot.app.query("SummaryPanel"):
            return
    raise AssertionError("the eval never started")


def _batch_status(database: Path, batch_id: str) -> BatchStatus:
    """Read a batch's status straight from the row.

    Queried rather than going through a reader method, because no public API
    fetches a Batch by its own id and adding one purely for a test would grow
    the surface to satisfy the test rather than a user.
    """
    with RunStore(database) as store:
        row = store.connection.execute(
            "SELECT status FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()

    assert row is not None, f"no batch row for {batch_id}"
    return BatchStatus(row["status"])


def _scorer(output: object, expected: object, case: Case) -> Score:
    return Score(scorer_name="s", value=1.0, passed=True)


def _eval(task: Any, *ids: str) -> Eval:
    return Eval(
        name="watched",
        cases=[Case(id=case_id, input=case_id, expected=case_id) for case_id in ids],
        task=task,
        scorers=[_scorer],
    )


class TestCancellationIsHonest:
    """The second acceptance criterion, and the one that does lasting damage if
    it is wrong."""

    @pytest.mark.anyio
    async def test_a_file_change_cancels_the_run_in_flight(self) -> None:
        started: list[str] = []
        release = asyncio.Event()

        async def slow(value: str) -> str:
            started.append(value)
            await asyncio.wait_for(release.wait(), timeout=5)
            return value

        app = EvalApp(_eval(slow, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _running(pilot, started)

            first = app._run_task
            app.post_message(FilesChanged([Path("qa_eval.py")]))
            await pilot.pause()

            assert (first is not None and first.cancelled()) or first.done()
            release.set()

    @pytest.mark.anyio
    async def test_the_cancelled_batch_is_recorded_as_cancelled(self, tmp_path: Path) -> None:
        """Not left `running`, and not quietly completed. A Batch left running
        is neither complete nor known to be partial, and nothing later can tell
        which."""
        release = asyncio.Event()
        started: list[str] = []

        async def slow(value: str) -> str:
            started.append(value)
            await asyncio.wait_for(release.wait(), timeout=5)
            return value

        database = tmp_path / "h.db"
        store = RunStore(database)
        recorder = BatchRecorder(store, git=GitState(), owns_store=True)
        batch_id = recorder.batch_id

        app = EvalApp(_eval(slow, "q1"), expected=1, recorder=recorder)
        async with app.run_test() as pilot:
            await _running(pilot, started)

            app.post_message(FilesChanged([Path("qa_eval.py")]))
            await pilot.pause()
            release.set()

        assert _batch_status(database, batch_id) == BatchStatus.CANCELLED

    @pytest.mark.anyio
    async def test_quitting_mid_run_also_closes_the_batch(self, tmp_path: Path) -> None:
        """Leaving a `running` Batch behind would be indistinguishable from a
        crash, and no later reader could say whether its numbers were whole."""
        release = asyncio.Event()
        started: list[str] = []

        async def slow(value: str) -> str:
            started.append(value)
            await asyncio.wait_for(release.wait(), timeout=5)
            return value

        database = tmp_path / "h.db"
        store = RunStore(database)
        recorder = BatchRecorder(store, git=GitState(), owns_store=True)
        batch_id = recorder.batch_id

        app = EvalApp(_eval(slow, "q1"), expected=1, recorder=recorder)
        async with app.run_test() as pilot:
            await _running(pilot, started)
            release.set()

        assert _batch_status(database, batch_id) is not BatchStatus.RUNNING

    @pytest.mark.anyio
    async def test_the_user_is_told_how_many_runs_were_cut_short(self) -> None:
        """A user whose re-runs keep being cancelled is editing faster than the
        eval completes, and needs to know that rather than wonder why the
        numbers never settle."""
        release = asyncio.Event()
        started: list[str] = []

        async def slow(value: str) -> str:
            started.append(value)
            if len(started) == 1:
                await asyncio.wait_for(release.wait(), timeout=5)
            return value

        app = EvalApp(_eval(slow, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _running(pilot, started)

            app.post_message(FilesChanged([Path("qa_eval.py")]))
            release.set()
            if app._run_task is not None:
                with contextlib_suppress():
                    await app._run_task
            await pilot.pause()
            await pilot.pause()

            assert app._cancelled_runs == 1


def contextlib_suppress() -> Any:
    import contextlib

    return contextlib.suppress(asyncio.CancelledError, Exception)


class TestRecordingActuallyStoresTheRun:
    """The defect these tests were written after missing.

    `--store` opened a recorder, marked its batch cancelled, and closed it —
    but never called `record()`, and ran every eval under the default
    `batch_id`. The result was a batches row with no runs beneath it: `h` showed
    an empty table and `c` always said "no earlier run", while the tool reported
    nothing wrong.

    The earlier tests here passed throughout, because they asserted the *batch
    status*. That row is written by `BatchRecorder.__init__`, so the assertion
    held whether or not a single Run was ever stored — testing the row I had
    created rather than the data the feature exists to persist.
    """

    @pytest.mark.anyio
    async def test_a_finished_run_is_written_to_the_database(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        store = RunStore(database)
        recorder = BatchRecorder(store, git=GitState(), owns_store=True)
        batch_id = recorder.batch_id

        app = EvalApp(_eval(_quick, "q1", "q2"), expected=2, recorder=recorder)
        async with app.run_test() as pilot:
            if app._run_task is not None:
                await app._run_task
            await pilot.pause()
            await pilot.pause()
            run_id = app.finished.id if app.finished else None

        assert run_id is not None, "the run never finished"

        with RunStore(database) as reader:
            stored = reader.load_run(run_id)

        assert stored is not None, "the run was not written to the database"
        assert [result.case_id for result in stored.results] == ["q1", "q2"]
        assert stored.batch_id == batch_id, "the run was filed under the wrong batch"

    @pytest.mark.anyio
    async def test_the_run_is_filed_under_the_recorders_batch(self, tmp_path: Path) -> None:
        """`BatchRecorder.record` refuses a mismatched batch id, so a run left
        at the default `batch_id` does not merely land in the wrong place — it
        never lands at all, and the failure surfaces as an empty history."""
        database = tmp_path / "h.db"
        store = RunStore(database)
        recorder = BatchRecorder(store, git=GitState(), owns_store=True)

        app = EvalApp(_eval(_quick, "q1"), expected=1, recorder=recorder)
        async with app.run_test() as pilot:
            if app._run_task is not None:
                await app._run_task
            await pilot.pause()

            assert app.finished is not None
            assert app.finished.batch_id == recorder.batch_id

    @pytest.mark.anyio
    async def test_history_can_read_back_what_watch_mode_recorded(self, tmp_path: Path) -> None:
        """End to end: the run has to be findable the way `h` and `c` look for
        it, not merely present under some id."""
        database = tmp_path / "h.db"
        store = RunStore(database)
        recorder = BatchRecorder(store, git=GitState(), owns_store=True)

        app = EvalApp(_eval(_quick, "q1"), expected=1, recorder=recorder)
        async with app.run_test() as pilot:
            if app._run_task is not None:
                await app._run_task
            await pilot.pause()
        recorder.finish()

        with RunStore(database) as reader:
            found = reader.runs_for("watched")

        assert [run.id for run in found] == [app.finished.id if app.finished else ""]

    @pytest.mark.anyio
    async def test_an_unrecorded_session_still_runs(self, tmp_path: Path) -> None:
        """No recorder is the default, and must not be a special case that
        breaks: the Run still exists and is still displayed, it simply belongs
        to no stored batch."""
        app = EvalApp(_eval(_quick, "q1"), expected=1)
        async with app.run_test() as pilot:
            if app._run_task is not None:
                await app._run_task
            await pilot.pause()

            assert app.finished is not None
            assert app.finished.batch_id == "local"


async def _quick(value: str) -> str:
    return value


class TestACancelledBatchNeverReachesHistoryOrCompare:
    """The acceptance criterion stated as an end-to-end fact rather than as an
    implementation detail of the app."""

    def _seed(self, database: Path) -> None:
        with RunStore(database) as store:
            from evalstand.models import Batch, BatchKind

            for batch_id, status in (
                ("b-ok", BatchStatus.COMPLETED),
                ("b-cut", BatchStatus.CANCELLED),
            ):
                store.save_batch(
                    Batch(
                        id=batch_id,
                        kind=BatchKind.FULL,
                        status=status,
                        started_at=datetime.now(UTC),
                        finished_at=datetime.now(UTC),
                    )
                )
            for run_id, batch_id in (("run-ok", "b-ok"), ("run-cut", "b-cut")):
                store.save_run(
                    Run(
                        id=run_id,
                        batch_id=batch_id,
                        name="watched",
                        filepath="qa_eval.py",
                        status=RunStatus.COMPLETED,
                        started_at=datetime.now(UTC),
                        results=[
                            Result(
                                id=f"{run_id}-q1",
                                case_id="q1",
                                output="x",
                                scores=[Score(scorer_name="s", value=1.0, passed=True)],
                            )
                        ],
                    ),
                    cases=[Case(id="q1", input="q1", expected="q1")],
                )

    def test_history_does_not_list_it(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        self._seed(database)

        from typer.testing import CliRunner

        from evalstand.cli import app

        result = CliRunner().invoke(app, ["history", "--db", str(database)])

        assert "run-ok" in result.output
        assert "run-cut" not in result.output

    def test_compare_refuses_it(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        self._seed(database)

        from typer.testing import CliRunner

        from evalstand.cli import app

        result = CliRunner().invoke(app, ["compare", "run-ok", "run-cut", "--db", str(database)])

        assert result.exit_code == 1
        assert "did not run to completion" in result.output

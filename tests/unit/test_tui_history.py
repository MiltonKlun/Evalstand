"""The history view (task 6.4).

The assertion that matters most is not about the table — it is that comparing
from *this* screen refuses a cancelled run exactly as `evalstand compare` does.

That guard used to live inside the CLI command body. A second caller would have
walked straight past it and reintroduced a defect already fixed one screen over,
with the fix still sitting in the file as evidence that somebody had thought
about it. So the guard moved to `comparison.py` and both callers go through it,
and this file asserts that the screen genuinely does.
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
    Case,
    Result,
    Run,
    RunStatus,
    Score,
)
from evalstand.storage import RunStore
from evalstand.tui.history import HistoryScreen, _Detail

pytestmark = pytest.mark.anyio


def _run(run_id: str, batch_id: str, *, when: datetime, passed: bool = True) -> Run:
    return Run(
        id=run_id,
        batch_id=batch_id,
        name="qa",
        filepath="qa_eval.py",
        status=RunStatus.COMPLETED,
        started_at=when,
        results=[
            Result(
                id=f"{run_id}-q1",
                case_id="q1",
                output="answer",
                scores=[Score(scorer_name="s", value=1.0 if passed else 0.0, passed=passed)],
            )
        ],
    )


def _seed(database: Path, *, cancelled: bool = False) -> None:
    now = datetime.now(UTC)
    with RunStore(database) as store:
        for batch_id, status in (
            ("b-old", BatchStatus.COMPLETED),
            ("b-new", BatchStatus.CANCELLED if cancelled else BatchStatus.COMPLETED),
        ):
            store.save_batch(
                Batch(
                    id=batch_id,
                    kind=BatchKind.FULL,
                    status=status,
                    started_at=now,
                    finished_at=now,
                )
            )
        store.save_run(
            _run("run-old", "b-old", when=now - timedelta(hours=1), passed=False),
            cases=[Case(id="q1", input="q", expected="answer")],
        )
        store.save_run(
            _run("run-new", "b-new", when=now, passed=True),
            cases=[Case(id="q1", input="q", expected="answer")],
        )


def _app(store: RunStore) -> Any:
    """A bare app whose only job is to host the screen under test."""
    from textual.app import App

    class Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(HistoryScreen(store))

    return Host()


def _status(screen: HistoryScreen) -> str:
    from textual.widgets import Static

    widget = screen.query_one("#history-status", Static)
    return " ".join(strip.text for strip in widget.render_lines(widget.size.region))


class TestBrowsingPastRuns:
    async def test_stored_runs_are_listed_newest_first(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        _seed(database)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                assert [run.id for run in screen.runs] == ["run-new", "run-old"]
                assert screen.query_one("#runs").row_count == 2

    async def test_a_cancelled_run_is_not_listed(self, tmp_path: Path) -> None:
        """`runs_for` filters it, so a run that cannot be compared is never
        offered for comparison in the first place."""
        database = tmp_path / "h.db"
        _seed(database, cancelled=True)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                assert [run.id for run in screen.runs] == ["run-old"]

    async def test_an_empty_database_says_so(self, tmp_path: Path) -> None:
        with RunStore(tmp_path / "h.db") as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                assert "no runs recorded" in _status(screen)


class TestMarkingRunsToCompare:
    async def test_space_marks_and_unmarks(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        _seed(database)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                screen.action_mark()
                assert screen.marked == ["run-new"]

                screen.action_mark()
                assert screen.marked == []

    async def test_a_third_mark_replaces_the_oldest(self, tmp_path: Path) -> None:
        """Refusing mid-flow is irritating, and there is no ambiguity about
        which two runs the user means."""
        database = tmp_path / "h.db"
        _seed(database)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                screen.marked = ["a", "b"]
                screen.action_mark()

                assert screen.marked == ["b", "run-new"]

    async def test_comparing_without_two_marks_explains_how(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        _seed(database)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                screen.action_compare()
                await pilot.pause()

                assert "mark 2 runs" in _status(screen)


class TestComparingFromTheHistoryView:
    async def test_two_complete_runs_render_a_comparison(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        _seed(database)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                screen.marked = ["run-old", "run-new"]
                screen.action_compare()
                await pilot.pause()

                assert isinstance(app.screen, _Detail)

    async def test_a_cancelled_run_is_refused_here_too(self, tmp_path: Path) -> None:
        """The point of the whole file. `runs_for` hides a cancelled run from
        the table, but a marked id survives a reload and the guard is what makes
        the refusal real rather than incidental."""
        database = tmp_path / "h.db"
        _seed(database, cancelled=True)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                screen.marked = ["run-old", "run-new"]
                screen.action_compare()
                await pilot.pause()

                assert not isinstance(app.screen, _Detail), "a cancelled run was compared"
                assert "did not run to completion" in _status(screen)

    async def test_the_older_run_is_the_baseline_whatever_order_it_was_marked(
        self, tmp_path: Path
    ) -> None:
        """A comparison labelled backwards inverts the sign of every delta, so
        a real improvement would be printed as a fall."""
        database = tmp_path / "h.db"
        _seed(database)

        seen: list[tuple[str, str]] = []

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                import evalstand.tui.history as module

                real = module.compare_runs

                def spy(before: Run, after: Run, **kwargs: Any) -> Any:
                    seen.append((before.id, after.id))
                    return real(before, after, **kwargs)

                module.compare_runs = spy  # type: ignore[assignment]
                try:
                    # Marked newest-first, which is the order the table offers.
                    screen.marked = ["run-new", "run-old"]
                    screen.action_compare()
                    await pilot.pause()
                finally:
                    module.compare_runs = real  # type: ignore[assignment]

        assert seen == [("run-old", "run-new")]

    async def test_a_marked_run_that_has_vanished_is_reported(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        _seed(database)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, HistoryScreen)

                screen.marked = ["run-old", "run-gone"]
                screen.action_compare()
                await pilot.pause()

                assert "no longer in the database" in _status(screen)


class TestOpeningARun:
    async def test_enter_opens_the_selected_run(self, tmp_path: Path) -> None:
        database = tmp_path / "h.db"
        _seed(database)

        with RunStore(database) as store:
            app = _app(store)
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()

                assert isinstance(app.screen, _Detail)

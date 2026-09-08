"""The history view (task 6.4): browse past runs, open one, compare two.

Everything on screen comes from renderers that already exist —
`render_history`, `render_run_detail`, `render_comparison`. A second set of
formatting rules here would be a second set to keep honest, and the first time
they drifted the terminal and the TUI would disagree about whether an unpriced
call was free.

The comparison goes through `refuse_partial_runs`, the same guard the `compare`
command uses. That is the whole reason the guard was moved out of the command
body: a check living inside one caller is a check the next caller silently
bypasses, and comparing a cancelled run here would reintroduce a defect already
fixed one screen over.
"""

from __future__ import annotations

from typing import ClassVar

from rich.console import RenderableType
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from evalstand.comparison import NotComparableError, compare_runs, refuse_partial_runs
from evalstand.models import Run
from evalstand.reporting.console import (
    UNKNOWN,
    pass_counts,
    render_comparison,
    render_run_detail,
)
from evalstand.storage import RunStore

__all__ = ["HistoryScreen"]

_COLUMNS = ("", "run", "eval", "when", "mean", "passed")

MAX_SELECTED = 2
"""A comparison is between two runs. Marking a third replaces the oldest mark
rather than refusing: refusing mid-flow is irritating, and there is no ambiguity
about which two the user means."""


class HistoryScreen(Screen[None]):
    """Past runs, newest first.

    Reads through a `RunStore` handed in rather than opening its own, so the
    screen cannot end up looking at a different database than the rest of the
    session.
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape,q", "dismiss", "close"),
        Binding("space", "mark", "mark for compare"),
        Binding("c", "compare", "compare marked"),
        Binding("enter", "open", "open run"),
    ]

    def __init__(self, store: RunStore, *, name_filter: str | None = None) -> None:
        self.store = store
        self.name_filter = name_filter
        self.runs: list[Run] = []
        self.marked: list[str] = []
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(DataTable(id="runs", cursor_type="row", zebra_stripes=True), id="history")
        yield Static("", id="history-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs", DataTable)
        for column in _COLUMNS:
            table.add_column(column or " ", key=column or "mark")
        self.reload()

    def reload(self) -> None:
        """Read the stored runs and paint them.

        `runs_for` already excludes cancelled batches, so a run that cannot be
        compared is not offered for comparison in the first place. The guard on
        the comparison itself stays regardless — a list that happens to be
        filtered is not the same as a check.
        """
        self.runs = self.store.runs_for(self.name_filter, limit=50)
        table = self.query_one("#runs", DataTable)
        table.clear()

        for run in self.runs:
            table.add_row(*self._cells(run), key=run.id)

        if not self.runs:
            self._say("no runs recorded yet.")

    def _cells(self, run: Run) -> list[str | Text]:
        passed, judged = pass_counts(run)
        return [
            "*" if run.id in self.marked else " ",
            run.id,
            run.name,
            run.started_at.strftime("%Y-%m-%d %H:%M") if run.started_at else UNKNOWN,
            UNKNOWN if run.mean_score is None else f"{run.mean_score:.2f}",
            UNKNOWN if not judged else f"{passed}/{judged}",
        ]

    # -- actions --------------------------------------------------------

    def action_mark(self) -> None:
        run_id = self._selected()
        if run_id is None:
            return

        if run_id in self.marked:
            self.marked.remove(run_id)
        else:
            self.marked.append(run_id)
            # Oldest mark drops off rather than refusing the third.
            self.marked = self.marked[-MAX_SELECTED:]

        self._repaint_marks()
        self._say(f"marked: {', '.join(self.marked) if self.marked else 'none'}")

    def action_compare(self) -> None:
        """Render the Phase 5 comparison for the two marked runs."""
        if len(self.marked) != MAX_SELECTED:
            self._say(f"mark {MAX_SELECTED} runs with space, then press c")
            return

        first, second = (self.store.load_run(run_id) for run_id in self.marked)
        if first is None or second is None:
            self._say("one of the marked runs is no longer in the database")
            return

        # The older run is "before" whatever order they were marked in: a
        # comparison labelled backwards would invert the sign of every delta.
        before, after = sorted((first, second), key=_when)

        try:
            refuse_partial_runs((before, after), self.store.batch_for)
        except NotComparableError as exc:
            self._say(str(exc))
            return

        comparison = compare_runs(
            before,
            after,
            hashes_before=self.store.case_hashes(before.id),
            hashes_after=self.store.case_hashes(after.id),
        )
        self.app.push_screen(_Detail(render_comparison(comparison)))

    def action_open(self) -> None:
        run_id = self._selected()
        if run_id is None:
            return

        run = self.store.load_run(run_id)
        if run is None:
            self._say(f"run {run_id} is no longer in the database")
            return

        self.app.push_screen(
            _Detail(render_run_detail(run, self.store.batch_for(run.id), full=True))
        )

    @on(DataTable.RowSelected)
    def _row_selected(self, message: DataTable.RowSelected) -> None:
        """The focused table consumes `enter` before an app-level binding, so
        opening a run is handled here — the same reason the run view does."""
        if message.row_key.value is not None:
            self.action_open()

    # -- helpers --------------------------------------------------------

    def _selected(self) -> str | None:
        table = self.query_one("#runs", DataTable)
        if not table.row_count:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        return str(row_key.value) if row_key.value is not None else None

    def _repaint_marks(self) -> None:
        table = self.query_one("#runs", DataTable)
        cursor = table.cursor_coordinate
        table.clear()
        for run in self.runs:
            table.add_row(*self._cells(run), key=run.id)
        table.cursor_coordinate = cursor

    def _say(self, text: str) -> None:
        self.query_one("#history-status", Static).update(text)


class _Detail(Screen[None]):
    """A rendered report on its own screen."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape,q,enter", "dismiss", "close"),
    ]

    def __init__(self, renderable: RenderableType) -> None:
        self.renderable = renderable
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Container(Static(self.renderable, id="detail-body"), id="detail")
        yield Footer()


def _when(run: Run) -> float:
    """A sortable timestamp, with an undated run treated as oldest.

    Undated rather than raising: a run recorded before timestamps were stored
    should still be comparable, and putting it first is the reading that does
    not silently relabel a newer run as the baseline.
    """
    return run.started_at.timestamp() if run.started_at else float("-inf")

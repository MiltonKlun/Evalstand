"""The run view: results streaming into a terminal as they land.

The feedback loop this tool exists for. A user watching thirty cases execute
should see each row appear as its case finishes, not stare at a spinner for
thirty seconds and receive a finished table — which is what any UI built on
`run_eval`'s return value alone would be.

**The widgets hold no logic.** Everything that could be wrong about what is on
screen lives in `state.py` and is tested without a terminal; this module turns
that state into widgets and routes keys. The split is deliberate: a bug in a
formatting rule is worth catching in a unit test, and a bug in widget mounting
shows up the moment anyone looks at the screen.

Results arrive on whatever task finished the case, so they are handed over as a
Textual message rather than by touching a widget directly. A widget mutated from
outside the app's own loop is a race, and the symptom would be a row that is
occasionally missing — the kind of defect that survives every run of a test
suite and appears once a week in front of a user.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from evalstand.api import Eval
from evalstand.models import Result, Run
from evalstand.reporting.console import UNKNOWN, render_case
from evalstand.runner import RunConfig, run_eval
from evalstand.tui.state import CaseRow, RunState

__all__ = ["EvalApp", "run_app"]

_BASE_COLUMNS = ("case", "status", "score", "latency", "cost")

_STATUS_STYLES = {
    "pass": "green",
    "fail": "red",
    "error": "red bold",
    "unmeasured": "yellow",
    "scored": "cyan",
}
"""Colour by outcome, with `unmeasured` deliberately *not* red.

A broken scorer and a wrong answer are different findings, and painting them
the same colour would undo the distinction `state.py` works to preserve.
"""


class ResultLanded(Message):
    """One case finished.

    Carried as a message because the runner announces from whichever task
    completed, and widgets may only be touched from the app's own loop.
    """

    def __init__(self, result: Result) -> None:
        self.result = result
        super().__init__()


class RunFinished(Message):
    """The eval completed and the stored Run exists."""

    def __init__(self, run: Run) -> None:
        self.run = run
        super().__init__()


class RunFailed(Message):
    """The run itself raised, as opposed to a case failing.

    Surfaced rather than swallowed: a UI that simply stopped updating would be
    indistinguishable from a slow model, and the user would wait for something
    that is never coming.
    """

    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__()


class SummaryPanel(Static):
    """Task 6.2: per-scorer means, pass count, cost, wall time, cache rate."""

    def show(self, state: RunState) -> None:
        totals = state.totals()

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()

        seen = f"{totals.completed}" + (f"/{totals.expected}" if totals.expected else "")
        table.add_row("cases", seen)
        table.add_row("passed", totals.pass_rate)
        if totals.errors:
            table.add_row("errors", Text(str(totals.errors), style="red"))
        table.add_row("cost", totals.cost)
        table.add_row("cache", totals.cache_hit_rate)

        means = state.mean_scores_by_scorer()
        for name, mean in sorted(means.items()):
            table.add_row(f"  {name}", f"{mean:.3f}")

        self.update(table)


class CaseDetail(ModalScreen[None]):
    """Task 6.3: one case in full, with its trace tree.

    The rendering is `reporting.console.render_case`, not a second
    implementation. Two renderers disagreeing about what a case did — one
    showing an unpriced call as free, the other as unknown — would be the same
    class of defect this project keeps finding, and reusing the audited one
    makes it impossible.

    Renders *one* Result, not the run. `render_run_detail` renders every case it
    holds, so using it here would answer "show me case q7" with all thirty —
    a detail view that is not a detail view, and one that looks plausible
    enough at three cases to survive review.
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape,enter,q", "dismiss", "close", show=True),
        Binding("y", "copy", "copy case id", show=True),
    ]

    def __init__(self, result: Result, case_key: str) -> None:
        self.result = result
        self.case_key = case_key
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Container(
            Static(render_case(self.result, full=True), id="detail-body"),
            id="detail",
        )

    @property
    def case_id(self) -> str:
        return self.case_key.split("#")[0]

    def action_copy(self) -> None:
        self.app.copy_to_clipboard(self.case_id)
        self.notify(f"copied {self.case_id}")


class EvalApp(App[None]):
    """The run view.

    Takes an already-declared Eval and executes it, painting rows as they land.
    """

    CSS = """
    #body { height: 1fr; }
    #results { height: 1fr; }
    #summary { width: 32; border-left: solid $panel; padding: 0 1; }
    #detail { padding: 1 2; height: 100%; overflow-y: auto; background: $surface; }
    #status { height: auto; padding: 0 1; color: $text-muted; }
    ProgressBar { padding: 0 1; }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "quit"),
        Binding("r", "rerun", "re-run"),
        Binding("f", "toggle_failures", "failures"),
        Binding("y", "copy_case", "copy id"),
        Binding("enter", "open_case", "detail"),
    ]

    def __init__(
        self,
        declared: Eval,
        *,
        config: RunConfig | None = None,
        expected: int | None = None,
    ) -> None:
        self.declared = declared
        self.config = config or RunConfig()
        self.state = RunState(declared.name, expected=expected, columns=declared.columns)
        self.finished: Run | None = None
        self._only_failures = False
        self._task: asyncio.Task[None] | None = None
        super().__init__()

    # -- composition ----------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="body"):
            with Vertical(id="left"):
                yield ProgressBar(total=self.state.expected, show_eta=False)
                yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
                yield Static("", id="status")
            yield SummaryPanel(id="summary")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"evalstand — {self.declared.name}"
        table = self.query_one("#results", DataTable)
        for column in (*_BASE_COLUMNS, *self.declared.columns):
            table.add_column(column, key=column)
        self.query_one(SummaryPanel).show(self.state)
        self.start_run()

    # -- execution ------------------------------------------------------

    def start_run(self) -> None:
        """Execute the eval, announcing each Result as it lands.

        The sink posts a message rather than writing to the table, so every
        widget mutation happens on the app's own loop.
        """
        self.state = RunState(
            self.declared.name, expected=self.state.expected, columns=self.declared.columns
        )
        self.finished = None
        table = self.query_one("#results", DataTable)
        table.clear()
        self._set_status("running...")

        config = RunConfig(
            concurrency=self.config.concurrency,
            timeout_seconds=self.config.timeout_seconds,
            cache=self.config.cache,
            bypass_cache=self.config.bypass_cache,
            on_chunk=self.config.on_chunk,
            on_result=self._announce,
        )
        self._task = asyncio.create_task(self._execute(config))

    def _announce(self, result: Result) -> None:
        """Hand a landed Result to the app's own loop.

        A plain function rather than a lambda because `post_message` returns a
        bool, and a sink is declared to return None.
        """
        self.post_message(ResultLanded(result))

    async def _execute(self, config: RunConfig) -> None:
        try:
            run = await run_eval(self.declared, config)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.post_message(RunFailed(f"{type(exc).__name__}: {exc}"))
            return
        self.post_message(RunFinished(run))

    # -- messages -------------------------------------------------------

    @on(ResultLanded)
    def _add_row(self, message: ResultLanded) -> None:
        row = self.state.add(message.result)
        if row is None:
            # A duplicate execution. `RunState` refused it, so the table must
            # not paint it either — two rows for one execution would make the
            # footer disagree with the table above it.
            return

        if not self._only_failures or row.status != "pass":
            self._paint(row)

        self.query_one(SummaryPanel).show(self.state)
        progress = self.query_one(ProgressBar)
        progress.update(total=self.state.expected, progress=self.state.totals().completed)

    @on(RunFinished)
    def _finish(self, message: RunFinished) -> None:
        self.finished = self.state.as_run(message.run)
        self.query_one(SummaryPanel).show(self.state)
        totals = self.state.totals()
        self._set_status(f"done — {totals.completed} cases, {totals.pass_rate} passed")

    @on(RunFailed)
    def _fail(self, message: RunFailed) -> None:
        self._set_status(f"the run failed: {message.error}")

    # -- painting -------------------------------------------------------

    def _paint(self, row: CaseRow) -> None:
        table = self.query_one("#results", DataTable)
        table.add_row(*self._cells(row), key=row.key)

    def _cells(self, row: CaseRow) -> list[RenderableType]:
        label = row.case_id if row.repeat_index == 0 else f"{row.case_id} #{row.repeat_index}"
        return [
            label,
            Text(row.status, style=_STATUS_STYLES.get(row.status, "")),
            row.score,
            row.latency,
            row.cost,
            *(row.extra.get(name, UNKNOWN) for name in self.declared.columns),
        ]

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _repaint(self) -> None:
        table = self.query_one("#results", DataTable)
        table.clear()
        rows = self.state.failures() if self._only_failures else self.state.rows()
        for row in rows:
            self._paint(row)

    # -- actions --------------------------------------------------------

    def action_toggle_failures(self) -> None:
        self._only_failures = not self._only_failures
        self._repaint()
        self._set_status("showing failures only" if self._only_failures else "showing all cases")

    def action_rerun(self) -> None:
        """Re-run from scratch.

        Any in-flight run is cancelled first: two runs writing into one table
        would interleave their rows, and the footer would count both.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self.start_run()

    @on(DataTable.RowSelected)
    def _row_selected(self, message: DataTable.RowSelected) -> None:
        """Open a case from the table's own selection event.

        The table has focus and consumes `enter` for its own row selection
        before an app-level binding is consulted, so binding `enter` on the App
        alone leaves the key doing nothing at all. Handled here instead, which
        is also what makes a mouse click open a case.
        """
        if message.row_key.value is not None:
            self._open(str(message.row_key.value))

    def action_open_case(self) -> None:
        """Task 6.3: open the selected case.

        Available while the run is still going, because a Result is complete the
        moment it lands — waiting for the whole run would withhold a finished
        measurement for no reason, and inspecting the first failure without
        waiting for the other twenty-nine is the point of a live view.
        """
        key = self._selected_key()
        if key is not None:
            self._open(key)

    def _open(self, key: str) -> None:
        result = self.state.result_for(key)
        if result is None:
            # A row with no Result behind it means the table and the state have
            # diverged. Said plainly rather than opening an empty pane.
            self._set_status(f"no result recorded for {key}")
            return

        self.push_screen(CaseDetail(result, key))

    def action_copy_case(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        case_id = key.split("#")[0]
        self.copy_to_clipboard(case_id)
        self._set_status(f"copied {case_id}")

    def _selected_key(self) -> str | None:
        table = self.query_one("#results", DataTable)
        if not table.row_count:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:
            return None
        return str(row_key.value) if row_key.value is not None else None


def run_app(declared: Eval, *, config: RunConfig | None = None) -> None:
    """Open the run view for one Eval.

    The case count is resolved first so the progress bar has a total. A loader
    that raises fails here, before the screen is taken over — an error message
    printed plainly beats the same message inside a terminal UI that then has to
    be dismissed.
    """
    expected = len(declared.load_cases()) * declared.repeat
    EvalApp(declared, config=config, expected=expected).run()

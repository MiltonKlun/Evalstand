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
from pathlib import Path
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
from textual.widgets import DataTable, Footer, Header, Input, ProgressBar, Static

from evalstand.api import Eval
from evalstand.models import Result, Run
from evalstand.recording import BatchRecorder
from evalstand.reporting.console import UNKNOWN, render_case
from evalstand.runner import RunConfig, run_eval
from evalstand.tui.state import CaseRow, RunState
from evalstand.tui.watch import watch_paths

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


class FilesChanged(Message):
    """A watched file was edited.

    Carried as a message for the same reason results are: the watcher runs in
    its own task, and widgets may only be touched from the app's own loop.
    """

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
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
    #search { display: none; }
    #search.visible { display: block; }
    ProgressBar { padding: 0 1; }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "quit"),
        Binding("r", "rerun", "re-run"),
        Binding("f", "toggle_failures", "failures"),
        Binding("y", "copy_case", "copy id"),
        Binding("enter", "open_case", "detail"),
        Binding("h", "history", "history"),
        Binding("slash", "search", "search"),
        Binding("escape", "clear_search", "clear search", show=False),
        Binding("c", "compare_previous", "compare"),
    ]

    def __init__(
        self,
        declared: Eval,
        *,
        config: RunConfig | None = None,
        expected: int | None = None,
        watch: bool = False,
        watch_roots: list[Path] | None = None,
        recorder: BatchRecorder | None = None,
    ) -> None:
        self.declared = declared
        self.config = config or RunConfig()
        self.state = RunState(declared.name, expected=expected, columns=declared.columns)
        self.finished: Run | None = None
        self.watching = watch
        self.watch_roots = watch_roots
        self.recorder = recorder
        self._only_failures = False
        self._search = ""
        """A case-id filter, applied on top of the failures filter.

        Both narrow the table, so they compose rather than override: a user who
        pressed `f` and then searched is asking for failing cases matching the
        term, not for one filter to silently replace the other.
        """
        self._run_task: asyncio.Task[None] | None = None
        """The eval currently executing.

        Named `_run_task` rather than `_task` because `MessagePump` — which
        `App` inherits from — already owns `self._task` for its own message
        loop. Shadowing it made `_cancel_in_flight` cancel the *application*
        instead of the eval, and the symptom was every test hanging rather than
        anything pointing at the name.
        """

        self._watch_task: asyncio.Task[None] | None = None
        self._stop_watching = asyncio.Event()
        self._cancelled_runs = 0
        """How many Batches a file change cut short. Shown, not hidden: a user
        whose re-runs keep being cancelled is editing faster than the eval can
        complete, and needs to know that rather than wonder why nothing
        finishes."""
        super().__init__()

    # -- composition ----------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="body"):
            with Vertical(id="left"):
                yield ProgressBar(total=self.state.expected, show_eta=False)
                yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
                yield Input(placeholder="filter by case id", id="search")
                yield Static("", id="status")
            yield SummaryPanel(id="summary")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"evalstand — {self.declared.name}"
        self.query_one("#search", Input).can_focus = False
        table = self.query_one("#results", DataTable)
        for column in (*_BASE_COLUMNS, *self.declared.columns):
            table.add_column(column, key=column)
        self.query_one(SummaryPanel).show(self.state)
        self.start_run()
        if self.watching:
            self._watch_task = asyncio.create_task(self._watch())

    # -- execution ------------------------------------------------------

    def start_run(self) -> None:
        """Execute the eval, announcing each Result as it lands.

        The sink posts a message rather than writing to the table, so every
        widget mutation happens on the app's own loop.
        """
        self._cancel_in_flight()

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
        self._run_task = asyncio.create_task(self._execute(config))

    def _cancel_in_flight(self) -> None:
        """Abandon a Batch that is still running, and record that honestly.

        Cancelling the task stops *awaiting* the in-flight model calls; it does
        not abort the HTTP requests already issued, which is deliberate. The
        money for those is already spent, so killing them discards a response
        the user has paid for — and the next Batch, moments away, asks for
        exactly the same thing. Letting them land in the cache turns a wasted
        call into a free one.

        The Batch is marked `cancelled` rather than left `running` or quietly
        completed. `history` hides a cancelled Batch and `compare` refuses it,
        so a half-finished Batch cannot drag a mean it only partly measured.
        """
        if self._run_task is None or self._run_task.done():
            return

        self._run_task.cancel()
        self._cancelled_runs += 1
        if self.recorder is not None:
            self.recorder.finish(cancelled=True)
            self.recorder = None

    async def _watch(self) -> None:
        """Re-run when a watched file changes.

        Debouncing lives in `watch_paths`; this only decides what a change
        means. A failure in the watcher is reported rather than swallowed —
        a watch mode that has silently stopped watching looks exactly like one
        with nothing to report.
        """
        try:
            async for paths in watch_paths(self.watch_roots, stop=self._stop_watching):
                self.post_message(FilesChanged(paths))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - depends on the filesystem
            self.post_message(RunFailed(f"watch stopped: {type(exc).__name__}: {exc}"))

    @on(FilesChanged)
    def _changed(self, message: FilesChanged) -> None:
        """Cancel any in-flight Batch and start a new one.

        Waiting for a slow Batch to drain would spend the feedback loop this
        feature exists for: the user has moved on, and the results still
        arriving describe code they have already edited.
        """
        names = ", ".join(path.name for path in message.paths[:3])
        extra = f" (+{len(message.paths) - 3})" if len(message.paths) > 3 else ""
        self._changed_note = f"changed: {names}{extra}"

        self.start_run()
        self._set_status(self._changed_note)

    def _announce(self, result: Result) -> None:
        """Hand a landed Result to the app's own loop.

        A plain function rather than a lambda because `post_message` returns a
        bool, and a sink is declared to return None.
        """
        self.post_message(ResultLanded(result))

    async def _execute(self, config: RunConfig) -> None:
        """Run the eval and record it, if this session is recording.

        The Run must carry the recorder's own `batch_id`. Left at the default,
        `BatchRecorder.record` refuses it — deliberately, since a run filed
        under the wrong batch is a programming error rather than an environment
        one — and the batch would end up with no runs beneath it.

        Recorded here rather than in `_finish` because writing is I/O: doing it
        on the app's message loop would stall the UI on a slow disk, and a
        history write must never be able to hold up the view of a run that has
        already been paid for.
        """
        try:
            run = await run_eval(self.declared, config, batch_id=self._batch_id())
            if self.recorder is not None:
                self.recorder.record(
                    run, cases=await self.declared.aload_cases(), task=self.declared.task
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.post_message(RunFailed(f"{type(exc).__name__}: {exc}"))
            return
        self.post_message(RunFinished(run))

    def _batch_id(self) -> str:
        """The batch this run belongs to.

        `"local"` matches `run_eval`'s own default and is what an unrecorded
        session uses: the Run still exists and is still displayed, it simply
        belongs to no stored batch.
        """
        return self.recorder.batch_id if self.recorder is not None else "local"

    # -- messages -------------------------------------------------------

    @on(ResultLanded)
    def _add_row(self, message: ResultLanded) -> None:
        row = self.state.add(message.result)
        if row is None:
            # A duplicate execution. `RunState` refused it, so the table must
            # not paint it either — two rows for one execution would make the
            # footer disagree with the table above it.
            return

        if self._shown(row):
            self._paint(row)

        if not self._on_screen():
            return

        self.query_one(SummaryPanel).show(self.state)
        progress = self.query_one(ProgressBar)
        progress.update(total=self.state.expected, progress=self.state.totals().completed)

    @on(RunFinished)
    def _finish(self, message: RunFinished) -> None:
        self.finished = self.state.as_run(message.run)
        if not self._on_screen():
            # The run completed as the app was shutting down. The state is still
            # updated — it is what gets recorded — but the widgets are already
            # gone, and querying them would turn a finished run into a crash on
            # exit.
            return

        self.query_one(SummaryPanel).show(self.state)
        totals = self.state.totals()

        note = f"done — {totals.completed} cases, {totals.pass_rate} passed"
        if self._cancelled_runs:
            # Said plainly. A user whose re-runs keep being cut short is editing
            # faster than the eval completes, and needs to know that rather than
            # wonder why the numbers never settle.
            note += f"  ({self._cancelled_runs} earlier run(s) cancelled by a file change)"
        self._set_status(note)

    @on(RunFailed)
    def _fail(self, message: RunFailed) -> None:
        self._set_status(f"the run failed: {message.error}")

    # -- painting -------------------------------------------------------

    def _on_screen(self) -> bool:
        """Whether the widgets are mounted and can still be queried.

        A run finishing while the app shuts down still has a Result to record,
        but nothing to paint it on. Checked rather than caught, so a genuine
        query mistake still raises instead of being swallowed as "shutting
        down".
        """
        return bool(self.is_running and self.screen_stack and self.query("SummaryPanel"))

    def _paint(self, row: CaseRow) -> None:
        if not self._on_screen():
            return
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
        if self._on_screen():
            self.query_one("#status", Static).update(text)

    def _shown(self, row: CaseRow) -> bool:
        """Whether a row survives the active filters.

        One predicate for both the live paint and the repaint. Two copies would
        eventually disagree, and the symptom is a row that appears while the run
        is going and vanishes when anything triggers a redraw — which reads as
        losing a result.
        """
        if self._only_failures and row.status == "pass":
            return False
        return self._search.lower() in row.case_id.lower()

    def _repaint(self) -> None:
        if not self._on_screen():
            return
        table = self.query_one("#results", DataTable)
        table.clear()
        for row in self.state.rows():
            if self._shown(row):
                self._paint(row)

    # -- actions --------------------------------------------------------

    def action_toggle_failures(self) -> None:
        self._only_failures = not self._only_failures
        self._repaint()
        self._set_status("showing failures only" if self._only_failures else "showing all cases")

    def action_search(self) -> None:
        """Filter the table by case id."""
        search = self.query_one("#search", Input)
        search.can_focus = True
        search.add_class("visible")
        search.focus()

    @on(Input.Changed, "#search")
    def _search_changed(self, message: Input.Changed) -> None:
        self._search = message.value
        self._repaint()

    @on(Input.Submitted, "#search")
    def _search_submitted(self) -> None:
        """Close the box but keep the filter.

        Keeping it is the point: a search that cleared itself the moment focus
        left would make the narrowed table impossible to scroll through.
        `escape` is what clears it.
        """
        self.query_one("#results", DataTable).focus()

    def action_clear_search(self) -> None:
        search = self.query_one("#search", Input)
        search.value = ""
        search.remove_class("visible")
        search.can_focus = False
        self._search = ""
        self._repaint()
        self.query_one("#results", DataTable).focus()

    def action_compare_previous(self) -> None:
        """Compare this run against the previous run of the same eval.

        Needs a recorder: without one there is no previous run to compare
        against, and the honest answer is to say so rather than show an empty
        comparison that reads as "nothing changed".
        """
        if self.finished is None:
            self._set_status("the run is still going; compare when it finishes")
            return
        if self.recorder is None:
            self._set_status(
                "comparing needs recorded runs; start with --store, or use `evalstand compare`"
            )
            return

        store = self.recorder.store
        earlier = [
            run for run in store.runs_for(self.declared.name, limit=2) if run.id != self.finished.id
        ]
        if not earlier:
            self._set_status(f"no earlier run of {self.declared.name!r} to compare against")
            return

        from evalstand.comparison import NotComparableError, compare_runs, refuse_partial_runs
        from evalstand.reporting.console import render_comparison
        from evalstand.tui.history import _Detail

        before = earlier[0]
        try:
            refuse_partial_runs((before, self.finished), store.batch_for)
        except NotComparableError as exc:
            self._set_status(str(exc))
            return

        comparison = compare_runs(
            before,
            self.finished,
            hashes_before=store.case_hashes(before.id),
            hashes_after=store.case_hashes(self.finished.id),
        )
        self.push_screen(_Detail(render_comparison(comparison)))

    def action_rerun(self) -> None:
        """Re-run from scratch. `start_run` cancels anything in flight."""
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

    def action_history(self) -> None:
        """Task 6.4. Opens only when there is a database to read.

        Said plainly rather than showing an empty table: "no history" and "not
        recording" send a user to different places, and a blank screen implies
        the first when the truth is usually the second.
        """
        if self.recorder is None:
            self._set_status(
                "history needs a recorded run; start watch mode with --store, "
                "or use `evalstand history`"
            )
            return

        from evalstand.tui.history import HistoryScreen

        self.push_screen(HistoryScreen(self.recorder.store))

    def action_copy_case(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        case_id = key.split("#")[0]
        self.copy_to_clipboard(case_id)
        self._set_status(f"copied {case_id}")

    async def _shutdown(self) -> None:
        """Stop watching and close an open Batch before the app tears down.

        A Batch left `running` in the database is neither complete nor known to
        be partial, and nothing later can tell which — so quitting mid-run marks
        it cancelled, the same as a file change does.

        Hooked here rather than on `on_unmount`, which Textual delivers *after*
        `_shutdown` has already returned. A batch closed that late is closed
        after the process has finished caring, and the row stays `running`.
        """
        self._close_watch()
        self._cancel_in_flight()
        await super()._shutdown()

    def _close_watch(self) -> None:
        self._stop_watching.set()
        if self._watch_task is not None:
            self._watch_task.cancel()
            self._watch_task = None

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

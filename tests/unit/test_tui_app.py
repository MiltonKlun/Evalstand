"""The run view, driven headlessly (tasks 6.1, 6.2, 6.3, 6.5, 6.7).

`state.py` is tested as pure logic; this file tests the half that only a running
terminal can show — that rows are actually mounted, that they arrive *during*
the run rather than at the end, and that the keys do what the footer says.

The incremental assertion is the one worth reading carefully. A test that runs
the app to completion and then counts rows passes just as happily against a UI
that painted everything in one batch at the end, which is precisely the failure
this phase exists to avoid. So the run is held open mid-flight and the table is
inspected while cases are still executing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from textual.widgets import ProgressBar, Static

from evalstand.api import Eval
from evalstand.models import Case, Score, Trace
from evalstand.tui.app import CaseDetail, EvalApp

pytestmark = pytest.mark.anyio


def _scorer(output: object, expected: object, case: Case) -> Score:
    correct = output == expected
    return Score(scorer_name="s", value=1.0 if correct else 0.0, passed=correct)


def _eval(task: Any, *ids: str, **kwargs: Any) -> Eval:
    cases = [Case(id=case_id, input=case_id, expected=case_id) for case_id in ids]
    return Eval(name="tui-eval", cases=cases, task=task, scorers=[_scorer], **kwargs)


async def _echo(value: str) -> str:
    return value


def _app(declared: Eval, **kwargs: Any) -> EvalApp:
    expected = kwargs.pop("expected", None)
    return EvalApp(declared, expected=expected, **kwargs)


async def _settle(pilot: Any, app: EvalApp) -> None:
    """Wait for the run to finish and the resulting messages to be processed."""
    if app._run_task is not None:
        await app._run_task
    await pilot.pause()
    await pilot.pause()


class TestRowsAppearIncrementally:
    """Task 6.1's acceptance criterion, and the reason the runner grew a
    per-Result sink at all."""

    async def test_a_row_is_mounted_while_other_cases_are_still_running(self) -> None:
        """The assertion this phase turns on.

        The last case blocks until the table has already been inspected, so a UI
        that painted every row at the end would find an empty table here. A test
        that merely counted rows after completion could not tell the two apart.
        """
        release = asyncio.Event()
        table_had_rows: list[int] = []

        async def task(value: str) -> str:
            if value == "slow":
                await asyncio.wait_for(release.wait(), timeout=5)
            return value

        app = _app(_eval(task, "fast", "slow"), expected=2)
        async with app.run_test() as pilot:
            # Let "fast" land and its message be processed.
            for _ in range(20):
                await pilot.pause()
                if app.state.totals().completed:
                    break

            table_had_rows.append(app.query_one("#results").row_count)
            release.set()
            await _settle(pilot, app)

        assert table_had_rows == [1], "a row should be on screen before the run finished"

    async def test_every_case_ends_up_on_screen(self) -> None:
        app = _app(_eval(_echo, "q1", "q2", "q3"), expected=3)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            assert app.query_one("#results").row_count == 3

    async def test_the_progress_bar_advances_with_completed_cases(self) -> None:
        app = _app(_eval(_echo, "q1", "q2"), expected=2)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            assert app.query_one(ProgressBar).progress == 2

    async def test_a_duplicate_result_is_not_painted_twice(self) -> None:
        """Two rows for one execution would make the footer's count disagree
        with the table above it."""
        app = _app(_eval(_echo, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            before = app.query_one("#results").row_count

            app._announce(app.state.results[0])
            await pilot.pause()

            assert app.query_one("#results").row_count == before


class TestTheSummaryPanel:
    """Task 6.2."""

    async def test_it_reports_per_scorer_means_and_pass_count(self) -> None:
        app = _app(_eval(_echo, "q1", "q2"), expected=2)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            assert app.state.mean_scores_by_scorer() == {"s": 1.0}
            assert app.state.totals().pass_rate == "2/2"

    async def test_a_wrong_answer_is_not_counted_as_a_pass(self) -> None:
        async def wrong(value: str) -> str:
            return "no"

        app = _app(_eval(wrong, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            assert app.state.totals().pass_rate == "0/1"


class TestCaseDetail:
    """Task 6.3."""

    async def test_enter_opens_the_selected_case(self) -> None:
        app = _app(_eval(_echo, "q1", "q2"), expected=2)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, CaseDetail)

    async def test_the_detail_shows_one_case_not_the_whole_run(self) -> None:
        """`render_run_detail` renders every case a Run holds, so using it here
        would answer "show me q1" with all three — a detail view that is not
        one, and plausible enough at three cases to survive review.

        Read off the mounted widget, never re-rendered here. An earlier version
        of this test called `render_case` itself and compared the result, which
        asserted that the *test helper* was right while the app was free to
        render anything at all — it passed against a mutant that changed what
        the screen showed.
        """
        app = _app(_eval(_echo, "q1", "q2", "q3"), expected=3)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, CaseDetail)
            assert app.screen.result.case_id == "q1"

            text = _widget_text(app.screen, "#detail-body")
            assert "q1" in text
            assert "q2" not in text, "the detail view is showing other cases"

    async def test_the_detail_shows_the_prompts_themselves(self) -> None:
        """`full=True`, so a trace's input and output are printed rather than
        summarised as a character count.

        The detail view exists to answer "what did this call actually say"; a
        pane reporting `input: 34 chars` is the summary the user already had.
        """
        secret = "the-prompt-text"
        traced = Trace(
            id="t1", name="call", duration_ms=5, model="gpt-4o", input=secret, output="hi"
        )

        app = _app(_eval(_echo, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            app.state.results[0] = app.state.results[0].model_copy(update={"traces": [traced]})

            app._open("q1#0")
            await pilot.pause()

            assert isinstance(app.screen, CaseDetail)
            assert secret in _widget_text(app.screen, "#detail-body")

    async def test_a_case_can_be_opened_before_the_run_finishes(self) -> None:
        """A Result is complete the moment it lands. Withholding it until the
        whole run ends would defeat the point of inspecting the first failure
        without waiting for the other twenty-nine."""
        release = asyncio.Event()

        async def task(value: str) -> str:
            if value == "slow":
                await asyncio.wait_for(release.wait(), timeout=5)
            return value

        app = _app(_eval(task, "fast", "slow"), expected=2)
        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause()
                if app.state.totals().completed:
                    break

            await pilot.press("enter")
            await pilot.pause()
            opened = isinstance(app.screen, CaseDetail)

            release.set()
            await _settle(pilot, app)

        assert opened, "a landed case should be inspectable while others run"

    async def test_opening_a_case_with_no_result_behind_it_says_so(self) -> None:
        """A row whose Result cannot be found means the table and the state
        have diverged, which is worth saying rather than opening a blank pane.

        Driven directly: the app never builds such a row itself, and a test that
        pressed a key on an empty table would prove nothing — the first case
        lands before the keypress.
        """
        app = _app(_eval(_echo, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            app._open("ghost#0")
            await pilot.pause()

            assert not isinstance(app.screen, CaseDetail)
            assert "no result recorded for ghost#0" in _status_text(app)


class TestKeybindings:
    """Task 6.7."""

    async def test_f_filters_to_failures(self) -> None:
        async def half(value: str) -> str:
            return value if value == "q1" else "wrong"

        app = _app(_eval(half, "q1", "q2"), expected=2)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            assert app.query_one("#results").row_count == 2

            await pilot.press("f")
            await pilot.pause()

            assert app.query_one("#results").row_count == 1

    async def test_f_toggles_back_to_every_case(self) -> None:
        async def half(value: str) -> str:
            return value if value == "q1" else "wrong"

        app = _app(_eval(half, "q1", "q2"), expected=2)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            await pilot.press("f")
            await pilot.pause()
            await pilot.press("f")
            await pilot.pause()

            assert app.query_one("#results").row_count == 2

    async def test_a_passing_row_is_not_painted_while_the_filter_is_on(self) -> None:
        """The filter must hold for rows that land *after* it is switched on,
        not only for the repaint at the moment it is pressed."""
        release = asyncio.Event()

        async def task(value: str) -> str:
            if value == "slow":
                await asyncio.wait_for(release.wait(), timeout=5)
            return value

        app = _app(_eval(task, "fast", "slow"), expected=2)
        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause()
                if app.state.totals().completed:
                    break

            await pilot.press("f")
            await pilot.pause()
            release.set()
            await _settle(pilot, app)

            assert app.query_one("#results").row_count == 0, "a pass leaked past the filter"

    async def test_r_reruns_the_eval(self) -> None:
        calls: list[str] = []

        async def counting(value: str) -> str:
            calls.append(value)
            return value

        app = _app(_eval(counting, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            assert calls == ["q1"]

            await pilot.press("r")
            await _settle(pilot, app)

            assert calls == ["q1", "q1"]
            assert app.query_one("#results").row_count == 1, "the table should not accumulate"

    async def test_y_copies_the_case_id_without_the_repeat_suffix(self) -> None:
        """A copied id is pasted into `-k` or a code comment, where `q1#0` would
        match nothing."""
        copied: list[str] = []

        app = _app(_eval(_echo, "q1", repeat=2), expected=2)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            app.copy_to_clipboard = copied.append  # type: ignore[method-assign]

            await pilot.press("y")
            await pilot.pause()

        assert copied == ["q1"]


class TestSearch:
    """Task 6.7's `/`."""

    async def test_slash_filters_the_table_by_case_id(self) -> None:
        app = _app(_eval(_echo, "alpha", "beta", "gamma"), expected=3)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            assert app.query_one("#results").row_count == 3

            await pilot.press("slash")
            await pilot.pause()
            for key in "et":
                await pilot.press(key)
            await pilot.pause()

            assert app.query_one("#results").row_count == 1, "expected only beta"

    async def test_escape_clears_the_filter(self) -> None:
        app = _app(_eval(_echo, "alpha", "beta"), expected=2)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            app._search = "alpha"
            app._repaint()
            await pilot.pause()
            assert app.query_one("#results").row_count == 1

            app.action_clear_search()
            await pilot.pause()

            assert app.query_one("#results").row_count == 2

    async def test_search_composes_with_the_failures_filter(self) -> None:
        """Both narrow the table, so they compose. A user who pressed `f` and
        then searched wants failing cases matching the term, not one filter
        silently replacing the other.

        The search term matches a **passing** case as well as a failing one, so
        composing and overriding give different answers. An earlier version
        searched for a term only failing cases matched — both behaviours agreed,
        and the test passed against a mutant that dropped the failures filter
        entirely the moment anything was typed.
        """

        async def half(value: str) -> str:
            return value if value.startswith("ok") else "wrong"

        app = _app(_eval(half, "ok-one", "bad-one", "bad-two"), expected=3)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            await pilot.press("f")
            await pilot.pause()
            assert app.query_one("#results").row_count == 2, "two failing cases"

            # "one" matches ok-one (passing) and bad-one (failing).
            app._search = "one"
            app._repaint()
            await pilot.pause()

            assert app.query_one("#results").row_count == 1, (
                "the failures filter was dropped when a search was typed"
            )

    async def test_a_row_landing_during_a_search_respects_it(self) -> None:
        """The filter must hold for rows that arrive *after* it is set, not
        only for the repaint at the moment it is typed."""
        release = asyncio.Event()

        async def task(value: str) -> str:
            if value == "slow":
                await asyncio.wait_for(release.wait(), timeout=5)
            return value

        app = _app(_eval(task, "fast", "slow"), expected=2)
        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause()
                if app.state.totals().completed:
                    break

            app._search = "fast"
            app._repaint()
            await pilot.pause()

            release.set()
            await _settle(pilot, app)

            assert app.query_one("#results").row_count == 1, "a row leaked past the search"

    async def test_the_search_is_case_insensitive(self) -> None:
        app = _app(_eval(_echo, "Alpha"), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            app._search = "alpha"
            app._repaint()
            await pilot.pause()

            assert app.query_one("#results").row_count == 1


class TestCompareWithPrevious:
    """Task 6.7's `c`."""

    async def test_it_says_so_when_there_is_nothing_to_compare_against(self) -> None:
        """An empty comparison reads as "nothing changed", which is a claim
        about two runs when there is only one."""
        app = _app(_eval(_echo, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            app.action_compare_previous()
            await pilot.pause()

            assert "needs recorded runs" in _status_text(app)

    async def test_it_refuses_while_the_run_is_still_going(self) -> None:
        """A half-finished run has no aggregate worth comparing."""
        release = asyncio.Event()

        async def task(value: str) -> str:
            await asyncio.wait_for(release.wait(), timeout=5)
            return value

        app = _app(_eval(task, "q1"), expected=1)
        async with app.run_test() as pilot:
            await pilot.pause()

            app.action_compare_previous()
            await pilot.pause()

            assert "still going" in _status_text(app)
            release.set()


class TestFailureIsVisible:
    async def test_a_crashed_task_still_gets_a_row(self) -> None:
        """A table that simply stops at case 12 looks like a hang, not a
        failure."""

        async def boom(value: str) -> str:
            raise RuntimeError("nope")

        app = _app(_eval(boom, "q1", "q2"), expected=2)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            assert app.query_one("#results").row_count == 2
            assert app.state.totals().errors == 2

    async def test_a_failing_run_says_so_rather_than_going_quiet(self) -> None:
        """A UI that stopped updating is indistinguishable from a slow model,
        and the user would wait for something that is never coming."""

        def exploding_loader() -> list[Case]:
            raise ValueError("the dataset is missing")

        declared = Eval(name="broken", cases=exploding_loader, task=_echo, scorers=[_scorer])
        app = _app(declared, expected=1)
        async with app.run_test() as pilot:
            for _ in range(30):
                await pilot.pause()
                if "failed" in _status_text(app):
                    break

            assert "the dataset is missing" in _status_text(app)


def _widget_text(parent: Any, selector: str) -> str:
    """What a widget is actually showing, read off the mounted widget.

    Never re-renders the source data: a test that computes the expected text
    with the same helper the app uses asserts only that the helper is
    self-consistent, and goes green against an app that renders something else
    entirely.
    """
    widget = parent.query_one(selector, Static)
    return " ".join(strip.text for strip in widget.render_lines(widget.size.region))


def _status_text(app: EvalApp) -> str:
    """The status line as it appears on screen."""
    return _widget_text(app, "#status")


class TestCustomColumns:
    """Task 6.5, at the widget level."""

    async def test_a_declared_column_becomes_a_table_column(self) -> None:
        declared = _eval(_echo, "q1", columns={"length": lambda r: len(str(r.output))})
        app = _app(declared, expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            labels = [str(column.label) for column in app.query_one("#results").columns.values()]
            assert labels == ["case", "status", "score", "latency", "cost", "length"]

    async def test_the_derived_value_reaches_the_row(self) -> None:
        declared = _eval(_echo, "hello", columns={"length": lambda r: len(str(r.output))})
        app = _app(declared, expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            row = app.state.rows()[0]
            assert row.extra == {"length": "5"}


EDITABLE_EVAL = """
from evalstand import Case, evaluate


def task(value):
    return "{answer}"


def check(output, expected):
    return output == expected


evaluate(
    name="reload-me",
    cases=[Case(id="q1", input="x", expected="right")],
    task=task,
    scorers=[check],
)
"""


class TestAnEditIsWhatReRuns:
    """Watch mode's whole promise: change the code, see the changed code run.

    It re-ran the object imported at start-up instead. The status said
    "changed: <file>", the rows re-landed one by one, and the old task ran
    against the old cases — every sign of a re-run of the edit, none of the
    substance. Found by recording the demo, where fixing the one wrong answer
    re-ran the eval and left the case failing.

    The tests before this built evals in memory, which have no file to change,
    so nothing could have caught it. These write a real file and edit it.
    """

    @staticmethod
    def _write(path: Any, answer: str) -> None:
        path.write_text(EDITABLE_EVAL.replace("{answer}", answer), encoding="utf-8")

    @staticmethod
    def _load(path: Any) -> Eval:
        from evalstand.loading import load_evals, select_eval

        return select_eval(load_evals([path]), "reload-me")

    @staticmethod
    def _passed(app: EvalApp) -> int:
        return app.state.totals().passed

    async def test_a_change_runs_the_edited_code(self, tmp_path: Any) -> None:
        from evalstand.tui.app import FilesChanged

        target = tmp_path / "reload_eval.py"
        self._write(target, "wrong")

        app = _app(self._load(target), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            assert self._passed(app) == 0, "the starting answer should fail"

            self._write(target, "right")
            app.post_message(FilesChanged([target]))
            await pilot.pause()
            await _settle(pilot, app)

            assert self._passed(app) == 1, "the re-run executed the code from before the edit"
            # On the *finished* run. Shown only while the re-run was in flight,
            # it was replaced before a fast eval ever drew it, and the numbers
            # changed with no visible reason.
            assert "after changed: reload_eval.py" in _status_text(app)

            await pilot.press("r")
            await pilot.pause()
            await _settle(pilot, app)

            assert "changed:" not in _status_text(app), "a key press was blamed on the old edit"

    async def test_r_runs_the_edited_code(self, tmp_path: Any) -> None:
        """With `--once` there is no watcher. A user who edits and presses `r`
        is asking for the edited code, not the code the session started with."""
        target = tmp_path / "reload_eval.py"
        self._write(target, "wrong")

        app = _app(self._load(target), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)

            self._write(target, "right")
            await pilot.press("r")
            await pilot.pause()
            await _settle(pilot, app)

            assert self._passed(app) == 1

    async def test_a_file_that_no_longer_imports_runs_nothing(self, tmp_path: Any) -> None:
        """A typo mid-edit is ordinary in watch mode. Falling back to the eval
        loaded earlier would show results labelled as current, produced by code
        that is no longer on disk — so it says so and runs nothing."""
        from evalstand.tui.app import FilesChanged

        target = tmp_path / "reload_eval.py"
        self._write(target, "wrong")

        app = _app(self._load(target), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            before = app._run_task

            target.write_text("def task(:\n", encoding="utf-8")
            app.post_message(FilesChanged([target]))
            await pilot.pause()
            await pilot.pause()

            status = _status_text(app)
            assert "could not load reload_eval.py" in status
            assert "SyntaxError" in status
            assert "nothing was re-run" in status
            assert app._run_task is before, "a stale eval was run in place of the broken file"

    async def test_an_eval_built_in_memory_still_reruns(self) -> None:
        """No file, so nothing to reload and nothing an edit could have
        changed. `r` must keep working for it."""
        calls: list[str] = []

        async def counting(value: str) -> str:
            calls.append(value)
            return value

        app = _app(_eval(counting, "q1"), expected=1)
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            await pilot.press("r")
            await _settle(pilot, app)

        assert calls == ["q1", "q1"]


class TestAnEditToATaskFileIsWhatReRuns:
    """The plan names "task file" changes explicitly. A task living in its own
    module was re-run in its old version: the eval file was re-read, but its
    `from helper import task` was answered from `sys.modules`."""

    async def test_editing_the_imported_task_runs_the_edit(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import uuid

        from evalstand.loading import load_evals, select_eval
        from evalstand.tui.app import FilesChanged

        helper = f"tui_helper_{uuid.uuid4().hex[:8]}"
        helper_file = tmp_path / f"{helper}.py"
        helper_file.write_text("def task(value):\n    return 'wrong'\n", encoding="utf-8")
        (tmp_path / "split_eval.py").write_text(
            "from evalstand import Case, evaluate\n"
            f"from {helper} import task\n\n"
            "def check(output, expected):\n    return output == expected\n\n"
            'evaluate(name="split", cases=[Case(id="q1", input="x", expected="right")],'
            " task=task, scorers=[check])\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        declared = select_eval(load_evals([tmp_path / "split_eval.py"]), "split")
        app = _app(declared, expected=1, watch_roots=[tmp_path])
        async with app.run_test() as pilot:
            await _settle(pilot, app)
            assert app.state.totals().passed == 0

            helper_file.write_text("def task(value):\n    return 'right'\n", encoding="utf-8")
            app.post_message(FilesChanged([helper_file]))
            await pilot.pause()
            await _settle(pilot, app)

            assert app.state.totals().passed == 1, "the task file's old version ran"

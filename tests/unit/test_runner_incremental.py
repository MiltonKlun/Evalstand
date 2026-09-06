"""Results reaching a watcher as they land (task 6.1).

The TUI's acceptance criterion is that rows appear incrementally, not in one
batch at the end. `run_eval` gathers, so without a per-Result callback a live
view could only ever paint every row at once when the whole eval finished — a
progress bar that fills in a single jump, telling the user nothing during the
thirty seconds they are actually waiting.

The tests that matter here are the ones about *timing* and *survival*: a sink
called only at the end would pass a naive "did it receive everything" assertion
while failing the thing the feature exists for.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evalstand.api import Eval
from evalstand.models import Case, Result, Score
from evalstand.runner import RunConfig, run_eval


def _scorer(output: object, expected: object, case: Case) -> Score:
    return Score(scorer_name="s", value=1.0 if output == expected else 0.0)


def _eval(task: Any, *ids: str, **kwargs: Any) -> Eval:
    return Eval(
        name="incremental",
        cases=[Case(id=case_id, input=case_id, expected=case_id) for case_id in ids],
        task=task,
        scorers=[_scorer],
        **kwargs,
    )


class TestResultsArriveDuringTheRun:
    @pytest.mark.anyio
    async def test_a_result_reaches_the_sink_before_the_run_returns(self) -> None:
        """The whole point. A sink called once at the end would satisfy "did it
        receive every result" while leaving the table empty for the entire run.

        Proven by blocking the last case until the sink has already seen the
        first. If delivery waited for the gather, q2 waits for an event that
        cannot be set until q2 itself finishes.

        The wait is asserted on rather than left to deadlock, because the
        runner's own policy defeats a bare timeout: a `TimeoutError` raised
        inside a task is *recorded on the Result* and the run completes
        normally, so a test relying on the hang would go green against exactly
        the defect it was written to catch. It only runs slower — and nobody
        reads a duration.
        """
        first_arrived = asyncio.Event()
        seen: list[str] = []
        delivered_during_the_run = False

        async def task(value: str) -> str:
            nonlocal delivered_during_the_run
            if value == "q2":
                # Cannot finish until q1 has already been announced.
                try:
                    await asyncio.wait_for(first_arrived.wait(), timeout=2)
                except TimeoutError:
                    return value
                delivered_during_the_run = True
            return value

        def sink(result: Result) -> None:
            seen.append(result.case_id)
            first_arrived.set()

        run = await run_eval(_eval(task, "q1", "q2"), RunConfig(on_result=sink))

        assert delivered_during_the_run, (
            "q1's Result never reached the sink while q2 was still executing, "
            "so the table would stay empty until the whole eval finished"
        )
        assert sorted(seen) == ["q1", "q2"]
        assert len(run.results) == 2

    @pytest.mark.anyio
    async def test_every_result_is_announced_exactly_once(self) -> None:
        """A row painted twice reads as two cases, and the pass count in the
        footer would disagree with the table above it."""
        seen: list[str] = []

        async def task(value: str) -> str:
            return value

        await run_eval(
            _eval(task, "q1", "q2", "q3"),
            RunConfig(on_result=lambda r: seen.append(r.case_id)),
        )

        assert sorted(seen) == ["q1", "q2", "q3"]

    @pytest.mark.anyio
    async def test_repeats_are_announced_separately(self) -> None:
        """An execution is the thing with an outcome, so three repeats of one
        case are three rows — matching what the Run finally holds."""
        seen: list[tuple[str, int]] = []

        async def task(value: str) -> str:
            return value

        run = await run_eval(
            _eval(task, "q1", repeat=3),
            RunConfig(on_result=lambda r: seen.append((r.case_id, r.repeat_index))),
        )

        assert sorted(seen) == [("q1", 0), ("q1", 1), ("q1", 2)]
        assert len(run.results) == 3

    @pytest.mark.anyio
    async def test_a_deselected_case_is_never_announced(self) -> None:
        """`only` filters before execution, so a row for a case that never ran
        would be a row for work nobody paid for."""
        seen: list[str] = []

        async def task(value: str) -> str:
            return value

        await run_eval(
            _eval(task, "q1", "q2"),
            RunConfig(on_result=lambda r: seen.append(r.case_id)),
            only={("q1", 0)},
        )

        assert seen == ["q1"]


class TestTheAnnouncedResultIsTheStoredOne:
    @pytest.mark.anyio
    async def test_the_sink_sees_the_same_object_the_run_holds(self) -> None:
        """A live table showing one thing and the stored run another is the
        exact class of defect this project keeps finding: two reports that
        cannot both be true, with nothing saying which is."""
        seen: list[Result] = []

        async def task(value: str) -> str:
            return value

        run = await run_eval(_eval(task, "q1"), RunConfig(on_result=lambda r: seen.append(r)))

        assert seen[0] is run.results[0]

    @pytest.mark.anyio
    async def test_a_failing_case_is_announced_with_its_error(self) -> None:
        """A case that crashed still needs its row: an eval whose table simply
        stops at case 12 looks like a hang, not a failure."""
        seen: list[Result] = []

        async def task(value: str) -> str:
            raise RuntimeError("boom")

        await run_eval(_eval(task, "q1"), RunConfig(on_result=lambda r: seen.append(r)))

        assert seen[0].error == "RuntimeError: boom"
        assert seen[0].scores == []

    @pytest.mark.anyio
    async def test_the_announced_result_already_carries_its_scores(self) -> None:
        """Announced after scoring, not before. A row that appeared with an
        empty score column and filled in later would render every case as
        unmeasured for the instant it mattered."""
        seen: list[Result] = []

        async def task(value: str) -> str:
            return value

        await run_eval(_eval(task, "q1"), RunConfig(on_result=lambda r: seen.append(r)))

        assert [s.value for s in seen[0].scores] == [1.0]


class TestASinkCannotBreakTheRun:
    @pytest.mark.anyio
    async def test_a_raising_sink_does_not_cost_the_measurement(self) -> None:
        """By the time a Result exists the money is already spent. A renderer
        with a formatting bug must not destroy the results it was called to
        display."""

        async def task(value: str) -> str:
            return value

        def bad_sink(result: Result) -> None:
            raise ValueError("the renderer is broken")

        run = await run_eval(_eval(task, "q1", "q2"), RunConfig(on_result=bad_sink))

        assert len(run.results) == 2
        assert [r.error for r in run.results] == [None, None]

    @pytest.mark.anyio
    async def test_a_raising_sink_does_not_stop_later_cases_being_announced(
        self,
    ) -> None:
        """One bad row must not silence the rest of the table."""
        seen: list[str] = []

        async def task(value: str) -> str:
            return value

        def sink(result: Result) -> None:
            seen.append(result.case_id)
            if result.case_id == "q1":
                raise ValueError("boom")

        await run_eval(_eval(task, "q1", "q2", "q3"), RunConfig(on_result=sink))

        assert sorted(seen) == ["q1", "q2", "q3"]

    @pytest.mark.anyio
    async def test_no_sink_is_the_ordinary_case(self) -> None:
        """The runner is used without a UI by the pytest plugin, and must not
        require one."""

        async def task(value: str) -> str:
            return value

        run = await run_eval(_eval(task, "q1"))

        assert len(run.results) == 1


class TestOrderingIsUnaffected:
    @pytest.mark.anyio
    async def test_the_run_stays_in_declaration_order_however_cases_finish(self) -> None:
        """Announcement order is completion order — that is what makes it live —
        but the stored Run must still be diffable between runs.

        The later case is made to finish first, so text order and completion
        order genuinely disagree.
        """
        seen: list[str] = []

        async def task(value: str) -> str:
            if value == "q1":
                await asyncio.sleep(0.05)
            return value

        run = await run_eval(
            _eval(task, "q1", "q2"),
            RunConfig(on_result=lambda r: seen.append(r.case_id)),
        )

        assert seen == ["q2", "q1"], "the sink should fire in completion order"
        assert [r.case_id for r in run.results] == ["q1", "q2"]

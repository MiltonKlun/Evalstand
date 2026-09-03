"""The runner (task 3.1).

Concurrent execution with a semaphore, ordered result collection, per-case
timeouts, and errors captured on the Result rather than aborting the run.

The governing rule: one bad case must never take down the others. An eval is a
measurement, and losing 29 good measurements because the 30th timed out would
waste both the time and the money already spent on them.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from evalstand.api import Eval
from evalstand.models import Case, Score
from evalstand.runner import RunConfig, run_eval


def _scorer(output: object, expected: object, case: Case) -> Score:
    return Score(
        scorer_name="exact", value=1.0 if output == expected else 0.0, passed=output == expected
    )


def _eval(task: object, cases: list[Case] | None = None, **kwargs: object) -> Eval:
    return Eval(
        name="test-eval",
        cases=cases or [Case(id="q1", input="x", expected="x")],
        task=task,  # type: ignore[arg-type]
        scorers=[_scorer],
        filepath="test_eval.py",
        **kwargs,  # type: ignore[arg-type]
    )


class TestBasicExecution:
    @pytest.mark.anyio
    async def test_runs_every_case(self) -> None:
        cases = [Case(id=f"q{i}", input=str(i), expected=str(i)) for i in range(5)]
        run = await run_eval(_eval(lambda value: value, cases))

        assert len(run.results) == 5
        assert {r.case_id for r in run.results} == {f"q{i}" for i in range(5)}

    @pytest.mark.anyio
    async def test_results_come_back_in_case_order(self) -> None:
        """Deterministic ordering despite concurrent execution: a report whose
        rows shuffle between runs is unreadable and undiffable."""

        async def varying(value: str) -> str:
            # Later cases finish first, so ordering cannot be completion order.
            await asyncio.sleep(0.01 * (5 - int(value)))
            return value

        cases = [Case(id=f"q{i}", input=str(i), expected=str(i)) for i in range(5)]
        run = await run_eval(_eval(varying, cases))

        assert [r.case_id for r in run.results] == [f"q{i}" for i in range(5)]

    @pytest.mark.anyio
    async def test_scores_are_attached(self) -> None:
        run = await run_eval(_eval(lambda value: value))
        assert run.results[0].scores[0].value == 1.0

    @pytest.mark.anyio
    async def test_a_sync_task_works(self) -> None:
        run = await run_eval(_eval(lambda value: value.upper()))
        assert run.results[0].output == "X"

    @pytest.mark.anyio
    async def test_an_async_task_works(self) -> None:
        async def task(value: str) -> str:
            return value.upper()

        run = await run_eval(_eval(task))
        assert run.results[0].output == "X"

    @pytest.mark.anyio
    async def test_the_run_carries_its_identity(self) -> None:
        run = await run_eval(_eval(lambda value: value))
        assert run.name == "test-eval"
        assert run.filepath == "test_eval.py"
        assert run.started_at is not None
        assert run.finished_at is not None


class TestErrorsDoNotAbortTheRun:
    """Task 3.1's acceptance: one case raising still completes the run."""

    @pytest.mark.anyio
    async def test_a_raising_case_is_recorded_and_the_rest_complete(self) -> None:
        def task(value: str) -> str:
            if value == "bad":
                raise ValueError("task exploded")
            return value

        cases = [
            Case(id="ok1", input="a", expected="a"),
            Case(id="boom", input="bad", expected="bad"),
            Case(id="ok2", input="c", expected="c"),
        ]
        run = await run_eval(_eval(task, cases))

        assert len(run.results) == 3
        failed = next(r for r in run.results if r.case_id == "boom")
        assert failed.error is not None
        assert "task exploded" in failed.error
        assert all(r.error is None for r in run.results if r.case_id != "boom")

    @pytest.mark.anyio
    async def test_a_failed_case_is_not_scored(self) -> None:
        """There is no output to score, and inventing a zero would claim the
        task performed badly when it never ran to completion."""

        def task(value: str) -> str:
            raise ValueError("down")

        run = await run_eval(_eval(task))
        assert run.results[0].scores == []
        assert run.mean_score is None

    @pytest.mark.anyio
    async def test_a_raising_scorer_does_not_fail_the_case(self) -> None:
        def bad_scorer(output: object, expected: object, case: Case) -> Score:
            raise RuntimeError("judge down")

        declared = Eval(
            name="e",
            cases=[Case(id="q1", input="x", expected="x")],
            task=lambda value: value,
            scorers=[bad_scorer],
            filepath="f",
        )
        run = await run_eval(declared)

        assert run.results[0].error is None, "the task succeeded"
        assert run.results[0].scores[0].error is not None
        assert run.mean_score is None, "an errored score is excluded, not counted zero"

    @pytest.mark.anyio
    async def test_every_case_failing_still_produces_a_run(self) -> None:
        def task(value: str) -> str:
            raise ValueError("everything is down")

        cases = [Case(id=f"q{i}", input=str(i)) for i in range(3)]
        run = await run_eval(_eval(task, cases))

        assert len(run.results) == 3
        assert all(r.error for r in run.results)


class TestTimeouts:
    @pytest.mark.anyio
    async def test_a_slow_case_times_out_without_stopping_the_others(self) -> None:
        async def task(value: str) -> str:
            if value == "slow":
                await asyncio.sleep(10)
            return value

        cases = [
            Case(id="fast", input="a", expected="a"),
            Case(id="slow", input="slow", expected="slow"),
        ]
        run = await run_eval(_eval(task, cases), RunConfig(timeout_seconds=0.05))

        assert len(run.results) == 2
        slow = next(r for r in run.results if r.case_id == "slow")
        assert slow.error is not None
        assert "timed out" in slow.error.lower()
        assert next(r for r in run.results if r.case_id == "fast").error is None

    @pytest.mark.anyio
    async def test_the_timeout_error_names_the_limit(self) -> None:
        """A bare "timed out" leaves the user guessing what the limit was."""

        async def task(value: str) -> str:
            await asyncio.sleep(10)
            return value

        run = await run_eval(_eval(task), RunConfig(timeout_seconds=0.05))
        assert "0.05" in (run.results[0].error or "")

    @pytest.mark.anyio
    async def test_no_timeout_by_default(self) -> None:
        """A slow model is normal; a default limit would fail honest work."""

        async def task(value: str) -> str:
            await asyncio.sleep(0.05)
            return value

        run = await run_eval(_eval(task))
        assert run.results[0].error is None


class TestConcurrency:
    @pytest.mark.anyio
    async def test_cases_run_concurrently(self) -> None:
        """Task 3.1's acceptance: changing concurrency changes wall time."""

        async def task(value: str) -> str:
            await asyncio.sleep(0.05)
            return value

        cases = [Case(id=f"q{i}", input=str(i)) for i in range(8)]

        started = time.perf_counter()
        await run_eval(_eval(task, cases), RunConfig(concurrency=8))
        concurrent_seconds = time.perf_counter() - started

        started = time.perf_counter()
        await run_eval(_eval(task, cases), RunConfig(concurrency=1))
        serial_seconds = time.perf_counter() - started

        assert concurrent_seconds < serial_seconds / 2, (
            f"concurrency had no effect: {concurrent_seconds:.3f}s vs {serial_seconds:.3f}s"
        )

    @pytest.mark.anyio
    async def test_the_semaphore_bounds_simultaneous_cases(self) -> None:
        """Unbounded concurrency would hit provider rate limits immediately."""
        live = 0
        peak = 0

        async def task(value: str) -> str:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return value

        cases = [Case(id=f"q{i}", input=str(i)) for i in range(20)]
        await run_eval(_eval(task, cases), RunConfig(concurrency=3))

        assert peak <= 3, f"ran {peak} cases at once with concurrency=3"

    @pytest.mark.anyio
    async def test_concurrency_of_one_is_serial(self) -> None:
        order: list[str] = []

        async def task(value: str) -> str:
            order.append(f"start-{value}")
            await asyncio.sleep(0.001)
            order.append(f"end-{value}")
            return value

        cases = [Case(id=f"q{i}", input=str(i)) for i in range(3)]
        await run_eval(_eval(task, cases), RunConfig(concurrency=1))

        # Serial execution never interleaves a start between a start and its end.
        for index in range(0, len(order), 2):
            assert order[index].replace("start-", "") == order[index + 1].replace("end-", "")


class TestTracingIntegration:
    @pytest.mark.anyio
    async def test_each_case_gets_its_own_trace_tree(self) -> None:
        """Concurrent cases must not pool their traces into one another."""
        from evalstand.tracing import trace

        async def task(value: str) -> str:
            with trace(f"work-{value}"):
                await asyncio.sleep(0.001)
            return value

        cases = [Case(id=f"q{i}", input=str(i)) for i in range(5)]
        run = await run_eval(_eval(task, cases), RunConfig(concurrency=5))

        for result in run.results:
            assert len(result.traces) == 1, "a case collected another case's traces"
            assert result.traces[0].name == f"work-{result.case_id[1:]}"

    @pytest.mark.anyio
    async def test_traces_survive_a_failing_case(self) -> None:
        """The calls made before a task blew up still cost money."""
        from evalstand.tracing import trace

        def task(value: str) -> str:
            with trace("before-the-failure"):
                pass
            raise ValueError("down")

        run = await run_eval(_eval(task))
        assert run.results[0].error is not None
        assert [t.name for t in run.results[0].traces] == ["before-the-failure"]


class TestRepeats:
    @pytest.mark.anyio
    async def test_each_case_runs_n_times(self) -> None:
        run = await run_eval(_eval(lambda value: value, repeat=3))

        assert len(run.results) == 3
        assert sorted(r.repeat_index for r in run.results) == [0, 1, 2]

    @pytest.mark.anyio
    async def test_repeat_indices_are_recorded_per_case(self) -> None:
        cases = [Case(id="a", input="x"), Case(id="b", input="y")]
        run = await run_eval(_eval(lambda value: value, cases, repeat=2))

        assert len(run.results) == 4
        for case_id in ("a", "b"):
            indices = sorted(r.repeat_index for r in run.results if r.case_id == case_id)
            assert indices == [0, 1]

    @pytest.mark.anyio
    async def test_repeated_results_stay_ordered_by_case_then_index(self) -> None:
        cases = [Case(id="a", input="x"), Case(id="b", input="y")]
        run = await run_eval(_eval(lambda value: value, cases, repeat=2))

        assert [(r.case_id, r.repeat_index) for r in run.results] == [
            ("a", 0),
            ("a", 1),
            ("b", 0),
            ("b", 1),
        ]

    @pytest.mark.anyio
    async def test_a_nondeterministic_task_produces_distinct_outputs(self) -> None:
        """Task 3.3's acceptance, in the runner: repeats must actually re-execute
        rather than reuse one answer."""
        counter = 0

        def task(value: str) -> str:
            nonlocal counter
            counter += 1
            return f"answer-{counter}"

        run = await run_eval(_eval(task, repeat=5))
        assert len({r.output for r in run.results}) == 5


class TestRunTotals:
    """Task 3.5: per-run aggregates."""

    @pytest.mark.anyio
    async def test_reports_wall_time(self) -> None:
        run = await run_eval(_eval(lambda value: value))
        assert run.finished_at is not None
        assert run.started_at is not None
        assert run.finished_at >= run.started_at

    @pytest.mark.anyio
    async def test_reports_pass_count(self) -> None:
        cases = [
            Case(id="good", input="x", expected="x"),
            Case(id="bad", input="y", expected="different"),
        ]
        run = await run_eval(_eval(lambda value: value, cases))

        passed = [r for r in run.results if all(s.passed for s in r.scores)]
        assert len(passed) == 1

    @pytest.mark.anyio
    async def test_reports_mean_score(self) -> None:
        cases = [
            Case(id="good", input="x", expected="x"),
            Case(id="bad", input="y", expected="different"),
        ]
        run = await run_eval(_eval(lambda value: value, cases))
        assert run.mean_score == pytest.approx(0.5)

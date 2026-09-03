"""The plugin executes through the runner (Phase 3 wiring).

Phase 3 built concurrency, timeouts, tracing and a shared cache, and Phase 2's
plugin executed cases itself in `runtest` — so none of it reached the path users
actually run. These tests pin down the seam between the two: the plugin collects
and reports, the runner executes, and nothing executes twice.

**Why `runpytest_subprocess` and not the default in-process run.** pytester's
inline runner leaves `importlib.metadata` unable to enumerate distributions once
litellm has been imported inside the session — `entry_points(group="pytest11")`
returns nothing afterwards, so the *next* inline run cannot register pytest-cov
and dies with "unrecognized arguments: --no-cov". Since the runner imports
litellm transitively, every test here would poison the ones after it. A
subprocess is also the more honest check: it is what a user's shell does.
"""

from __future__ import annotations

import pytest

from evalstand.api import Eval
from evalstand.models import Case

pytest_plugins = ["pytester"]

# A task that records every input it saw, so a test can assert on what did NOT
# run. Counting provider calls is the only way to catch a selection bug: the
# outcome of a deselected case looks identical whether or not it was executed.
SPY_EVAL = """
import pathlib
from evalstand import Case, evaluate
from evalstand.scorers import exact

LOG = pathlib.Path(__file__).parent / "executed.log"


def task(value):
    with LOG.open("a") as fh:
        fh.write(value + chr(10))
    return value


evaluate(
    name="spy",
    cases=[Case(id="alpha", input="alpha", expected="alpha"),
           Case(id="bravo", input="bravo", expected="bravo"),
           Case(id="delta", input="delta", expected="delta")],
    task=task,
    scorers=[exact],
)
"""

# An eval whose task calls the model, with litellm patched at import time. This
# is the shape that matters: the task calls the provider itself, so the cache
# and the trace collector can only reach it ambiently.
MODEL_EVAL = """
from unittest.mock import MagicMock, patch
from evalstand import Case, evaluate
from evalstand.llm import call
from evalstand.scorers import exact


def _response(**_):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = "Paris"
    r.usage.prompt_tokens = 10
    r.usage.completion_tokens = 5
    r.usage.total_tokens = 15
    r.model = "gpt-4o-mini"
    return r


patch("evalstand.llm.litellm.completion", side_effect=_response).start()
patch("evalstand.llm.litellm.completion_cost", return_value=0.0002).start()


def task(country):
    return call("gpt-4o-mini", [{"role": "user", "content": country}]).text


evaluate(
    name="model",
    cases=[Case(id="q1", input="France", expected="Paris"),
           Case(id="q2", input="Spain", expected="Paris")],
    task=task,
    scorers=[exact],
)
"""


def _executed(pytester: pytest.Pytester) -> list[str]:
    """Which case inputs the task actually ran for."""
    log = pytester.path / "executed.log"
    return log.read_text().split() if log.exists() else []


class TestSelectionIsHonoured:
    """A deselected case must cost nothing.

    This is the one bug in this file with a price attached: every other mistake
    here produces a wrong report, but executing a deselected case sends real
    requests to a provider the user explicitly asked to skip.
    """

    def test_k_does_not_execute_the_cases_it_deselected(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(spy_eval=SPY_EVAL)
        result = pytester.runpytest_subprocess("-k", "alpha", "--no-cov")

        result.assert_outcomes(passed=1, deselected=2)
        assert _executed(pytester) == ["alpha"], "a deselected case reached the provider"

    def test_selecting_one_node_executes_only_that_node(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(spy_eval=SPY_EVAL)
        pytester.runpytest_subprocess("spy_eval.py::bravo", "--no-cov")

        assert _executed(pytester) == ["bravo"]

    def test_collect_only_executes_nothing(self, pytester: pytest.Pytester) -> None:
        """Listing what would run must not run it. `--collect-only` is what a
        user reaches for precisely when they do *not* want to spend money."""
        pytester.makepyfile(spy_eval=SPY_EVAL)
        pytester.runpytest_subprocess("--collect-only", "--no-cov")

        assert _executed(pytester) == []

    def test_no_selection_runs_everything(self, pytester: pytest.Pytester) -> None:
        """The filter must not be so eager it drops cases nobody deselected."""
        pytester.makepyfile(spy_eval=SPY_EVAL)
        result = pytester.runpytest_subprocess("--no-cov")

        result.assert_outcomes(passed=3)
        assert sorted(_executed(pytester)) == ["alpha", "bravo", "delta"]


class TestResultsReachTheItems:
    def test_every_case_reports_its_own_outcome(self, pytester: pytest.Pytester) -> None:
        """One Result per item, matched by (case, repeat). A mismatch here would
        report one case's verdict under another's name."""
        pytester.makepyfile(
            mixed_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact

evaluate(
    name="mixed",
    cases=[Case(id="good", input="x", expected="x"),
           Case(id="bad", input="x", expected="y")],
    task=lambda v: v,
    scorers=[exact],
)
"""
        )
        result = pytester.runpytest_subprocess("--no-cov")

        result.assert_outcomes(passed=1, failed=1)
        assert "mixed::bad" in "\n".join(result.outlines)

    def test_a_case_never_executed_is_not_a_pass(self, pytester: pytest.Pytester) -> None:
        """If nothing populated the Result, the item must not report a pass.

        A green tick for a case that never ran is the single most misleading
        thing an eval tool can do, so the guard is asserted directly rather than
        through a simulated plugin conflict — a `tryfirst` hook wins the
        `firstresult` race against a conftest, so that simulation would silently
        test the happy path instead.
        """
        from evalstand.plugin import EvalCaseUnmeasuredError, EvalItem

        item = EvalItem.__new__(EvalItem)
        item.case = Case(id="never-ran", input="x")
        item.declared = Eval(
            name="e", cases=[item.case], task=lambda v: v, scorers=[], filepath="f"
        )
        item.repeat_index = 0
        item.result = None

        with pytest.raises(EvalCaseUnmeasuredError, match="never executed"):
            item.runtest()


class TestPhase3CapabilitiesReachTheUser:
    """The point of the wiring: what the runner measures must be reported."""

    def test_traces_produce_token_and_cost_totals(self, pytester: pytest.Pytester) -> None:
        """Phase 2's plugin built Runs from item state, which held no traces, so
        the footer printed "-" for a run that really did cost money."""
        pytester.makepyfile(model_eval=MODEL_EVAL)
        joined = "\n".join(pytester.runpytest_subprocess("--no-cov").outlines)

        assert "20 in / 10 out tokens" in joined, "token totals never reached the summary"
        assert "$0.0004" in joined, "cost never reached the summary"

    def test_a_timeout_fails_one_case_and_spares_the_rest(self, pytester: pytest.Pytester) -> None:
        """--timeout is unreachable without the runner: pytest has no notion of
        abandoning one item. And one slow case must not cost the others."""
        pytester.makepyfile(
            slow_eval="""
import asyncio
from evalstand import Case, evaluate
from evalstand.scorers import exact


async def task(value):
    await asyncio.sleep(0 if value == "fast" else 30)
    return "ok"


evaluate(
    name="slow",
    cases=[Case(id="fast", input="fast", expected="ok"),
           Case(id="slow", input="slow", expected="ok")],
    task=task,
    scorers=[exact],
)
"""
        )
        result = pytester.runpytest_subprocess("--timeout", "1", "--no-cov")

        result.assert_outcomes(passed=1, failed=1)
        assert "timed out after 1.0s" in "\n".join(result.outlines)

    def test_cases_run_concurrently(self, pytester: pytest.Pytester) -> None:
        """Concurrency is the reason the runner exists. Asserted by overlap
        rather than wall-clock, which would be flaky on a loaded machine."""
        pytester.makepyfile(
            conc_eval="""
import asyncio, pathlib
from evalstand import Case, evaluate
from evalstand.scorers import exact

PEAK = pathlib.Path(__file__).parent / "peak.log"
_live = 0


async def task(value):
    global _live
    _live += 1
    PEAK.write_text(str(max(_live, int(PEAK.read_text() or 0) if PEAK.exists() else 0)))
    await asyncio.sleep(0.2)
    _live -= 1
    return "ok"


evaluate(
    name="conc",
    cases=[Case(id=f"c{i}", input=str(i), expected="ok") for i in range(4)],
    task=task,
    scorers=[exact],
)
"""
        )
        pytester.runpytest_subprocess("--no-cov")

        peak = int((pytester.path / "peak.log").read_text())
        assert peak > 1, f"cases ran one at a time (peak in flight: {peak})"

    def test_concurrency_one_serialises(self, pytester: pytest.Pytester) -> None:
        """The flag has to actually restrict, or it is decoration. This is the
        other half of the previous test: without it, a peak above 1 could just
        mean the flag was ignored."""
        pytester.makepyfile(
            conc_eval="""
import asyncio, pathlib
from evalstand import Case, evaluate
from evalstand.scorers import exact

PEAK = pathlib.Path(__file__).parent / "peak.log"
_live = 0


async def task(value):
    global _live
    _live += 1
    PEAK.write_text(str(max(_live, int(PEAK.read_text() or 0) if PEAK.exists() else 0)))
    await asyncio.sleep(0.05)
    _live -= 1
    return "ok"


evaluate(
    name="conc",
    cases=[Case(id=f"c{i}", input=str(i), expected="ok") for i in range(4)],
    task=task,
    scorers=[exact],
)
"""
        )
        pytester.runpytest_subprocess("--concurrency", "1", "--no-cov")

        assert int((pytester.path / "peak.log").read_text()) == 1

    def test_no_cache_is_reported_as_a_bypass(self, pytester: pytest.Pytester) -> None:
        """Bypassing spends real money, so the summary says so rather than
        showing a 0% hit rate that reads like a cold cache."""
        pytester.makepyfile(model_eval=MODEL_EVAL)
        joined = "\n".join(pytester.runpytest_subprocess("--no-cache", "--no-cov").outlines)

        assert "bypassed" in joined

    def test_a_normal_run_reports_a_hit_rate_not_a_bypass(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(model_eval=MODEL_EVAL)
        joined = "\n".join(pytester.runpytest_subprocess("--no-cov").outlines)

        assert "cache" in joined
        assert "bypassed" not in joined


class TestFlagValidation:
    def test_a_zero_concurrency_is_a_usage_error(self, pytester: pytest.Pytester) -> None:
        """Rejected with a one-line message and a non-zero exit, not a traceback
        from inside the runner — and never as a silent success."""
        pytester.makepyfile(spy_eval=SPY_EVAL)
        result = pytester.runpytest_subprocess("--concurrency", "0", "--no-cov")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        assert _executed(pytester) == [], "a rejected run must not execute anything"

    def test_a_negative_timeout_is_a_usage_error(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(spy_eval=SPY_EVAL)
        result = pytester.runpytest_subprocess("--timeout", "-1", "--no-cov")

        assert result.ret == pytest.ExitCode.USAGE_ERROR


class TestPlainSessionsAreUntouched:
    def test_a_session_with_no_evals_is_unaffected(self, pytester: pytest.Pytester) -> None:
        """evalstand installs a pytest11 plugin, so it loads in every pytest run
        on the machine. A test suite that never heard of evals must behave as if
        it were not installed — including printing no summary."""
        pytester.makepyfile(test_ordinary="def test_one():\n    assert True\n")
        result = pytester.runpytest_subprocess("--no-cov")

        result.assert_outcomes(passed=1)
        joined = "\n".join(result.outlines)
        assert "scorer" not in joined, "an eval summary leaked into a plain test run"

    def test_evals_and_tests_coexist(self, pytester: pytest.Pytester) -> None:
        """Both in one session: the tests run normally and the evals go through
        the runner. Taking the loop over entirely would have broken this."""
        pytester.makepyfile(spy_eval=SPY_EVAL)
        pytester.makepyfile(test_ordinary="def test_one():\n    assert True\n")
        result = pytester.runpytest_subprocess("--no-cov")

        result.assert_outcomes(passed=4)  # 3 eval cases + 1 test
        assert sorted(_executed(pytester)) == ["alpha", "bravo", "delta"]


class TestOneExecutionPerCase:
    def test_a_case_is_executed_exactly_once(self, pytester: pytest.Pytester) -> None:
        """The failure mode of moving execution: leaving the old path in place
        so every case runs twice and every bill doubles."""
        pytester.makepyfile(spy_eval=SPY_EVAL)
        pytester.runpytest_subprocess("--no-cov")

        executed = _executed(pytester)
        assert len(executed) == 3, f"cases executed more than once: {executed}"

    def test_repeats_execute_once_each(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            rep_eval="""
import pathlib
from evalstand import Case, evaluate
from evalstand.scorers import exact

LOG = pathlib.Path(__file__).parent / "executed.log"


def task(value):
    with LOG.open("a") as fh:
        fh.write(value + chr(10))
    return value


evaluate(
    name="rep",
    cases=[Case(id="a", input="a", expected="a")],
    task=task,
    scorers=[exact],
    repeat=3,
)
"""
        )
        result = pytester.runpytest_subprocess("--no-cov")

        result.assert_outcomes(passed=3)
        assert _executed(pytester) == ["a", "a", "a"]


class TestTaskErrors:
    def test_a_raising_task_fails_only_its_own_case(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            boom_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact


def task(value):
    if value == "boom":
        raise RuntimeError("exploded")
    return value


evaluate(
    name="boom",
    cases=[Case(id="ok", input="x", expected="x"),
           Case(id="boom", input="boom", expected="boom")],
    task=task,
    scorers=[exact],
)
"""
        )
        result = pytester.runpytest_subprocess("--no-cov")

        result.assert_outcomes(passed=1, failed=1)
        assert "RuntimeError: exploded" in "\n".join(result.outlines)

    def test_the_failure_names_the_users_file_and_function(self, pytester: pytest.Pytester) -> None:
        """The runner catches the exception to keep other cases alive, which
        destroys the traceback. Without frames captured at the raise, a report
        can say what broke but never where."""
        pytester.makepyfile(
            deep_eval="""
from evalstand import Case, evaluate
from evalstand.scorers import exact


def lookup(key):
    return {"x": "y"}[key]


def task(key):
    return lookup(key)


evaluate(name="deep", cases=[Case(id="q1", input="missing", expected="y")],
         task=task, scorers=[exact])
"""
        )
        lines = pytester.runpytest_subprocess("--no-cov").outlines
        # Scoped to the FAILURES block: a subprocess run prints its own header
        # naming pluggy and every installed plugin, so searching the whole
        # output would pass or fail on the banner rather than the traceback.
        start = next(i for i, line in enumerate(lines) if "FAILURES" in line)
        end = next(
            (i for i, line in enumerate(lines[start:], start) if "short test summary" in line),
            len(lines),
        )
        block = "\n".join(lines[start:end])

        assert "deep_eval.py" in block, "the user's file should be named"
        assert "in lookup" in block, "the frame that actually failed"

        # The machinery the user cannot act on. `to_thread` puts a worker frame
        # on top of every sync task's traceback; showing it would repeat exactly
        # the noise this plugin exists to strip out of pytest's own output.
        assert "concurrent" not in block
        assert "in _run_one" not in block
        assert "pluggy" not in block


class TestDrivingTheAsyncRunnerFromASyncHook:
    """`pytest_runtestloop` is synchronous; the runner is not.

    `asyncio.run` refuses to nest, and this hook is not guaranteed to run on a
    thread without a loop — another plugin may own one. The fallback turns
    "your whole session died" into "it ran".
    """

    def test_it_works_when_a_loop_already_owns_the_thread(self) -> None:
        import asyncio

        from evalstand.api import Eval
        from evalstand.plugin import _execute_sync
        from evalstand.runner import RunConfig

        declared = Eval(
            name="nested",
            cases=[Case(id="q1", input="x")],
            task=lambda value: value,
            scorers=[],
            filepath="f",
        )
        wanted = {"nested": (declared, {("q1", 0)})}

        async def under_a_running_loop() -> list:
            return _execute_sync(wanted, RunConfig())

        runs = asyncio.run(under_a_running_loop())

        assert len(runs) == 1
        assert [r.output for r in runs[0].results] == ["x"]

    def test_a_failure_inside_the_thread_is_not_swallowed(self) -> None:
        """A crash must surface, not vanish into a thread nobody is watching:
        silently reporting zero runs would look like an empty test session."""
        import asyncio

        from evalstand.plugin import _execute_sync
        from evalstand.runner import RunConfig

        class Exploding:
            name = "boom"

            def __getattr__(self, item: str) -> object:
                raise RuntimeError("cannot load this eval")

        async def under_a_running_loop() -> list:
            return _execute_sync({"boom": (Exploding(), {("q1", 0)})}, RunConfig())  # type: ignore[dict-item]

        with pytest.raises(RuntimeError, match="cannot load this eval"):
            asyncio.run(under_a_running_loop())

    def test_a_genuine_runtime_error_is_not_mistaken_for_a_nested_loop(self) -> None:
        """The loop check must not swallow a real RuntimeError.

        Detecting the nested case by catching RuntimeError around `asyncio.run`
        would treat any RuntimeError from a user's own code as "there is a loop
        running" and silently re-run the whole eval on a thread.
        """
        from evalstand.api import Eval
        from evalstand.plugin import _execute_sync
        from evalstand.runner import RunConfig

        attempts = 0

        def task(value: str) -> str:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("no running event loop")

        declared = Eval(
            name="tricky",
            cases=[Case(id="q1", input="x")],
            task=task,
            scorers=[],
            filepath="f",
        )
        runs = _execute_sync({"tricky": (declared, {("q1", 0)})}, RunConfig())

        # The runner records a task's raise on its Result rather than letting it
        # escape, so the run completes — and must have executed exactly once.
        assert attempts == 1, "the eval was executed twice"
        assert runs[0].results[0].error is not None


# Two eval files that append their own (eval name, start, end) to one shared
# file. A per-module in-flight counter cannot see this: each eval file is a
# separate module with its own globals, so neither can observe the other's
# cases. Overlapping time intervals can be checked from outside both.
SPAN_EVAL = """
import asyncio, pathlib, time

from evalstand import Case, evaluate
from evalstand.scorers import exact

SPANS = pathlib.Path(__file__).parent / "spans.log"


async def task(value):
    started = time.perf_counter()
    await asyncio.sleep(0.2)
    with SPANS.open("a") as fh:
        fh.write("NAME %.6f %.6f" % (started, time.perf_counter()) + chr(10))
    return "ok"


evaluate(
    name="NAME",
    cases=[Case(id="NAME-c%d" % i, input=str(i), expected="ok") for i in range(3)],
    task=task,
    scorers=[exact],
)
"""


def _spans(pytester: pytest.Pytester) -> dict[str, list[tuple[float, float]]]:
    spans: dict[str, list[tuple[float, float]]] = {}
    for line in (pytester.path / "spans.log").read_text().splitlines():
        name, started, ended = line.split()
        spans.setdefault(name, []).append((float(started), float(ended)))
    return spans


class TestEvalsDoNotRunConcurrentlyWithEachOther:
    """`--concurrency` is a promise about how many calls are in flight.

    Running eval files at once would quietly multiply it by the number of
    files and trip the rate limit the flag exists to avoid.
    """

    def test_two_evals_do_not_overlap_in_time(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(first_eval=SPAN_EVAL.replace("NAME", "first"))
        pytester.makepyfile(second_eval=SPAN_EVAL.replace("NAME", "second"))

        result = pytester.runpytest_subprocess("--concurrency", "3", "--no-cov")
        result.assert_outcomes(passed=6)

        spans = _spans(pytester)
        assert set(spans) == {"first", "second"}

        # One eval must be wholly finished before the other starts. Compared as
        # whole blocks rather than case by case, because cases *within* an eval
        # are expected to overlap — that is the concurrency working.
        first_end = max(end for _, end in spans["first"])
        second_start = min(start for start, _ in spans["second"])
        first_start = min(start for start, _ in spans["first"])
        second_end = max(end for _, end in spans["second"])

        sequential = first_end <= second_start or second_end <= first_start
        assert sequential, "the two evals overlapped, so --concurrency 3 allowed 6 calls in flight"

    def test_cases_within_one_eval_do_overlap(self, pytester: pytest.Pytester) -> None:
        """The other half. Without this, the assertion above would also hold if
        every case ran one at a time — which would mean concurrency was broken
        rather than correctly bounded."""
        pytester.makepyfile(first_eval=SPAN_EVAL.replace("NAME", "first"))

        pytester.runpytest_subprocess("--concurrency", "3", "--no-cov")

        cases = _spans(pytester)["first"]
        latest_start = max(start for start, _ in cases)
        earliest_end = min(end for _, end in cases)
        assert latest_start < earliest_end, "cases within one eval ran serially"


class TestAPlainSessionCostsNothing:
    def test_an_eval_flag_does_not_break_a_suite_with_no_evals(
        self, pytester: pytest.Pytester
    ) -> None:
        """evalstand ships a pytest11 plugin, so it loads in every pytest run on
        the machine. A suite that never heard of evals must not be affected by
        an evalstand flag — not even a rejected one, since validating a config
        nothing will use turns someone else's green suite red.
        """
        pytester.makepyfile(test_plain="def test_one():\n    assert True\n")

        result = pytester.runpytest_subprocess("--concurrency", "0", "--no-cov")

        result.assert_outcomes(passed=1)

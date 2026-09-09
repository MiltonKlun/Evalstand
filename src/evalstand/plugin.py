"""The pytest plugin: collect `*_eval.py` and run one item per execution.

Registered through the `pytest11` entry point, so bare `pytest` collects evals
with no extra configuration (parity item 19).

The unit of collection is one `(case, repeat_index)` pair, not one case. An item
is one execution with one outcome; collapsing several stochastic executions into
a single pass/fail would require inventing an aggregation rule — any/all/majority
— and that is a judgement the user did not make.

**Execution belongs to the runner, not to pytest.** pytest runs items strictly
one after another, so a per-item `runtest` can never be concurrent, and every
capability Phase 3 built — concurrency, timeouts, tracing, the shared cache —
would be unreachable through the path users actually run. So the plugin takes
over `pytest_runtestloop`: it hands each eval to `run_eval()`, then replays the
finished Results through the items so pytest still reports one outcome per case
and `-k`, `-x`, `--collect-only` and the rest keep working.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

from evalstand.api import Eval, _current_eval_file, registry
from evalstand.models import Case, Result, Run, Score
from evalstand.recording import BatchRecorder, open_recorder
from evalstand.runner import RunConfig, run_eval

logger = logging.getLogger("evalstand.plugin")

EVAL_FILE_SUFFIX = "_eval.py"

_SESSION_MARKER = "_evalstand_registry_reset"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Phase 3's execution controls, exposed on the path users actually run.

    These configure *how* a run executes, never what it measures, so they are
    flags rather than anything declared in an eval file.
    """
    group = parser.getgroup("evalstand", "LLM evaluation")
    group.addoption(
        "--concurrency",
        type=int,
        default=None,
        metavar="N",
        help="Cases to execute at once (default: 8).",
    )
    group.addoption(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Abandon a case after this long. One case timing out never stops the rest.",
    )
    group.addoption(
        "--no-cache",
        action="store_true",
        default=False,
        help="Call the provider even when a cached response exists.",
    )
    group.addoption(
        "--no-store",
        action="store_true",
        default=False,
        help="Run without recording anything to the local history database.",
    )
    group.addoption(
        "--allow-dirty",
        action="store_true",
        default=False,
        help="Persist results even though the working tree has uncommitted changes.",
    )
    group.addoption(
        "--threshold",
        type=float,
        default=None,
        metavar="MEAN",
        help="Fail the run when an eval's mean score falls below this value.",
    )
    group.addoption(
        "--output",
        choices=("terminal", "markdown"),
        default="terminal",
        help="How to print the summary. `markdown` suits a pull-request comment.",
    )
    group.addoption(
        "--html",
        default=None,
        metavar="PATH",
        help=(
            "Also write a self-contained HTML report here, for a CI artifact. "
            "Independent of --output: the terminal summary still prints."
        ),
    )
    group.addoption(
        "--fail-on-error",
        action="store_true",
        default=False,
        help=(
            "Exit 2 when any case or scorer errored, even if the means are fine. "
            "A mean over the cases that survived is not a measurement of the eval."
        ),
    )


def _run_config(config: pytest.Config) -> RunConfig:
    """Read the flags into a RunConfig, rejecting bad values as usage errors.

    `RunConfig` validates in `__post_init__`; surfacing that as a pytest usage
    error means `--concurrency 0` prints a one-line message instead of a
    traceback from inside the runner.
    """
    kwargs: dict[str, Any] = {"bypass_cache": bool(config.getoption("--no-cache", default=False))}

    concurrency = config.getoption("--concurrency", default=None)
    if concurrency is not None:
        kwargs["concurrency"] = concurrency

    timeout = config.getoption("--timeout", default=None)
    if timeout is not None:
        kwargs["timeout_seconds"] = timeout

    try:
        return RunConfig(**kwargs)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc


def _reset_registry_once_per_session(session: pytest.Session) -> None:
    """Empty the registry at the start of each collection.

    The registry is module state that outlives one collection. Without this, a
    second collection in the same process — watch mode re-running, or a test
    harness driving pytest twice — sees the previous collection's evals and
    reports every file as a duplicate of itself.
    """
    if not getattr(session, _SESSION_MARKER, False):
        registry.clear()
        setattr(session, _SESSION_MARKER, True)


def pytest_collect_file(parent: pytest.Collector, file_path: Path) -> pytest.Collector | None:
    """Collect any `*_eval.py` file."""
    if file_path.name.endswith(EVAL_FILE_SUFFIX):
        return EvalFile.from_parent(parent, path=file_path)
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> Any:
    """Claim `*_eval.py` before pytest's own module collector does.

    Without this, pytest imports the file through its assertion-rewriting
    importer for its own collection *and* we import it again, so every eval is
    declared twice and reported as a duplicate of itself.
    """
    if module_path.name.endswith(EVAL_FILE_SUFFIX):
        return EvalFile.from_parent(parent, path=module_path)
    return None


class EvalFile(pytest.File):
    """One eval module. Imports it, then reads what it declared."""

    def collect(self) -> Any:
        # The registry is module-level and outlives a single collection: watch
        # mode re-collects the same files in one process, and pytester runs
        # several collections back to back. Clearing per session — not per file
        # — keeps duplicate-name detection working across files while stopping a
        # re-collection from reporting a file as a duplicate of itself.
        _reset_registry_once_per_session(self.session)

        before = {declared.name for declared in registry.evals()}

        # The filepath must be set before *any* import of this file, because
        # pytest's assertion-rewriting importer may run first. It is what tells
        # a re-import of one file apart from two files colliding on a name.
        token = _current_eval_file.set(str(self.path))
        try:
            self._import_module()
        finally:
            _current_eval_file.reset(token)

        declared_here = [declared for declared in registry.evals() if declared.name not in before]
        for declared in declared_here:
            yield from self._items_for(declared)

    def _import_module(self) -> Any:
        """Import the eval file through pytest's own importer.

        Deliberately not a hand-rolled `importlib` import. pytest imports files
        matching its collection patterns through its assertion-rewriting
        importer regardless; doing our own import as well executed every eval
        file twice, which registered each eval twice and reported it as a
        duplicate of itself.

        Going through `import_path` means one import, with pytest's own
        `sys.modules` handling and rootdir semantics.

        The cached module is dropped first. `import_path` caches by a
        path-derived name, and `evaluate()` runs at import time — so a *second*
        pytest session in the same process re-imports nothing, registers
        nothing, and collects zero evals from a file that plainly declares one.

        That is not only a test artifact: `evalstand run` twice in one process
        hits it, which is what a script driving several runs does. Watch mode
        escaped it because `loading.py` pops its own modules; this is the same
        fix on the path pytest owns.
        """
        import sys

        from _pytest.pathlib import ImportMode, import_path, module_name_from_path

        cached = module_name_from_path(self.path, self.config.rootpath)
        sys.modules.pop(cached, None)

        return import_path(
            self.path,
            mode=ImportMode.importlib,
            root=self.config.rootpath,
            consider_namespace_packages=False,
        )

    def _items_for(self, declared: Eval) -> Any:
        cases = declared.load_cases()
        for case in cases:
            for repeat_index in range(declared.repeat):
                # The suffix appears only when repeats are on, so the common
                # case stays clean and `-k q1` still selects by prefix.
                name = case.id if declared.repeat == 1 else f"{case.id}[repeat={repeat_index + 1}]"
                yield EvalItem.from_parent(
                    self,
                    name=name,
                    declared=declared,
                    case=case,
                    repeat_index=repeat_index,
                )


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session: pytest.Session) -> bool | None:
    """Execute every selected eval through the runner, before any item runs.

    Returning None hands control back to pytest, which then walks the items as
    usual — each one now merely *reporting* a Result the runner already
    produced. Taking the loop over entirely would mean reimplementing `-x`,
    `--maxfail`, fixtures and reporting, all of which pytest already does well.

    Only selected items are executed. `-k q1` must not spend money on the cases
    it deselected, so the runner is told exactly which `(case, repeat)` pairs
    survived selection.
    """
    if session.config.option.collectonly:
        return None

    wanted: dict[str, tuple[Eval, set[tuple[str, int]]]] = {}
    for item in session.items:
        if isinstance(item, EvalItem):
            _, units = wanted.setdefault(item.declared.name, (item.declared, set()))
            units.add((item.case.id, item.repeat_index))

    if not wanted:
        # A plain test session must be untouched by a plugin it never asked for.
        return None

    config = _run_config(session.config)

    # Checked before anything executes. Discovering that results cannot be
    # recorded *after* paying for them would be the worst possible ordering.
    recorder = _open_recorder(session.config)

    runs = _execute_sync(wanted, config, recorder)

    by_eval: dict[str, dict[tuple[str, int], Result]] = {
        run.name: {(r.case_id, r.repeat_index): r for r in run.results} for run in runs
    }
    for item in session.items:
        if isinstance(item, EvalItem):
            item.result = by_eval.get(item.declared.name, {}).get((item.case.id, item.repeat_index))

    session.config._evalstand_runs = runs  # type: ignore[attr-defined]
    if recorder is not None:
        recorder.finish()
    return None


def _execute_sync(
    wanted: dict[str, tuple[Eval, set[tuple[str, int]]]],
    config: RunConfig,
    recorder: BatchRecorder | None = None,
) -> list[Run]:
    """Drive the async runner from pytest's synchronous hook.

    `asyncio.run` refuses to nest, and this hook is not guaranteed to run on a
    thread without a loop — another plugin may own one. Falling back to a
    dedicated thread costs nothing in the common case and turns "your whole
    session died" into "it ran".

    The check is made *before* building the coroutine rather than by catching
    the resulting RuntimeError: a `try` around `asyncio.run` would also swallow
    a genuine RuntimeError raised by a user's task, and it leaves an un-awaited
    coroutine behind when it does.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_execute(wanted, config, recorder))  # the ordinary path

    results: list[Run] = []
    error: BaseException | None = None

    def target() -> None:
        nonlocal error
        try:
            results.extend(asyncio.run(_execute(wanted, config, recorder)))
        except BaseException as exc:  # re-raised on the calling thread below
            error = exc

    thread = threading.Thread(target=target, name="evalstand-runner")
    thread.start()
    thread.join()

    if error is not None:
        # Raised here so the failure reaches pytest normally. A crash swallowed
        # in a thread nobody watches would look like an empty test session.
        raise error
    return results


def _open_recorder(config: pytest.Config) -> BatchRecorder | None:
    """Open the history recorder, or None when this run should not persist.

    Called before any case executes: a dirty tree is refused up front, because
    discovering that results cannot be recorded after paying for them would be
    the worst possible ordering.
    """
    from evalstand.provenance import DirtyTreeError

    try:
        return open_recorder(
            enabled=not config.getoption("--no-store", default=False),
            allow_dirty=bool(config.getoption("--allow-dirty", default=False)),
            root=config.rootpath,
        )
    except DirtyTreeError as exc:
        raise pytest.UsageError(str(exc)) from exc


async def _execute(
    wanted: dict[str, tuple[Eval, set[tuple[str, int]]]],
    config: RunConfig,
    recorder: BatchRecorder | None = None,
) -> list[Run]:
    """Run each eval in turn, concurrent *within* an eval but not across them.

    Evals are kept sequential on purpose: `--concurrency` is a promise about how
    many calls are in flight, and running four evals at once would quietly
    multiply it by four and trip the rate limit the flag exists to avoid.

    Each Run is recorded as it finishes rather than at the end, so a crash
    part-way keeps everything already measured.
    """
    runs: list[Run] = []
    for declared, units in wanted.values():
        batch_id = recorder.batch_id if recorder else "local"
        run = await run_eval(declared, config, only=units, batch_id=batch_id)
        runs.append(run)
        if recorder is not None:
            recorder.record(run, cases=await declared.aload_cases(), task=declared.task)
    return runs


def _is_internal(filename: str) -> bool:
    """Frames belonging to pytest, pluggy, or this plugin are not the user's."""
    lowered = filename.replace("\\", "/").lower()
    return (
        "/_pytest/" in lowered or "/pluggy/" in lowered or lowered.endswith("evalstand/plugin.py")
    )


class EvalCaseFailedError(AssertionError):
    """One case did not pass. Carries enough context to act on without re-running."""


class EvalTaskError(Exception):
    """The task itself raised or timed out, so there is no output to score.

    The runner catches the original exception and keeps its text on the Result,
    because one failing case must never abort the others. By the time pytest
    reports it the traceback is gone, so the message carries what is known.
    """


class EvalCaseUnmeasuredError(Exception):
    """Every scorer for a case errored, so the case has no verdict.

    Distinct from a failure: the task may well have been fine. What is absent is
    a measurement, and a run that measured nothing must not report success.
    """


class EvalItem(pytest.Item):
    """One execution of one case."""

    def __init__(
        self,
        *,
        declared: Eval,
        case: Case,
        repeat_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.declared = declared
        self.case = case
        self.repeat_index = repeat_index
        self.result: Result | None = None
        """Filled in by `pytest_runtestloop` before any item runs."""

    @property
    def output(self) -> Any:
        return self.result.output if self.result else None

    @property
    def scores(self) -> list[Score]:
        return self.result.scores if self.result else []

    def runtest(self) -> None:
        """Report what the runner already found.

        No execution happens here. By the time pytest walks the items the run is
        over, so this turns one Result into one pytest outcome.
        """
        if self.result is None:
            # Only reachable if something bypassed the run loop. Better a loud
            # error than a green tick for a case that never executed.
            raise EvalCaseUnmeasuredError(f"case {self.case.id!r} was never executed by the runner")

        if self.result.error is not None:
            raise EvalTaskError(self.result.error)

        failed = [s for s in self.scores if s.passed is False]
        if failed:
            raise EvalCaseFailedError(
                f"case {self.case.id!r} did not pass {', '.join(s.scorer_name for s in failed)}"
            )

        # A case whose scorers all errored was not measured. Reporting it as a
        # pass would be a false claim — a rate-limited judge is not evidence the
        # task did well — and it is not a task failure either, since the task
        # itself ran fine. What is missing is a verdict, so the run must not
        # exit zero as though one had been reached.
        if self.scores and not any(s.counts_towards_mean for s in self.scores):
            reasons = "; ".join(f"{s.scorer_name}: {s.error}" for s in self.scores)
            raise EvalCaseUnmeasuredError(
                f"case {self.case.id!r} produced no usable score ({reasons})"
            )

    def repr_failure(self, excinfo: Any, style: Any = None) -> str:
        """Show what happened, so a failure is actionable without a re-run."""
        if isinstance(excinfo.value, EvalCaseUnmeasuredError):
            lines = [
                f"eval:     {self.declared.name}",
                f"case:     {self.case.id}",
                f"input:    {self.case.input!r}",
                f"output:   {self.output!r}",
                "no usable score: every scorer errored, so this case has no verdict",
            ]
            lines += [f"  {s.scorer_name}: {s.error}" for s in self.scores if s.error]
            return "\n".join(lines)

        if isinstance(excinfo.value, EvalCaseFailedError):
            lines = [
                f"eval:     {self.declared.name}",
                f"case:     {self.case.id}",
                f"input:    {self.case.input!r}",
                f"output:   {self.output!r}",
                f"expected: {self.case.expected!r}",
                "scores:",
            ]
            lines += [
                f"  {s.scorer_name}: "
                + (f"error {s.error}" if s.error else f"{s.value:.3f}")
                + ("" if s.passed is None else f" ({'pass' if s.passed else 'fail'})")
                for s in self.scores
            ]
            return "\n".join(lines)

        if isinstance(excinfo.value, EvalTaskError):
            lines = [
                f"eval:     {self.declared.name}",
                f"case:     {self.case.id}",
                f"input:    {self.case.input!r}",
                f"the task failed: {excinfo.value}",
            ]
            # The runner caught the exception to keep the other cases running,
            # so the traceback is long gone by now. These frames were captured
            # at the raise for exactly this moment.
            if self.result is not None:
                lines += self.result.error_frames
            return "\n".join(lines)

        # Anything else escaping `runtest` is a bug in evalstand, not in the
        # user's task — the runner catches a task's own exceptions and reports
        # them as EvalTaskError above. Kept rather than left to pytest's default
        # repr, which would bury the one useful line under pluggy internals.
        error = excinfo.value
        theirs = [
            frame for frame in traceback.extract_tb(excinfo.tb) if not _is_internal(frame.filename)
        ]

        lines = [
            f"eval:     {self.declared.name}",
            f"case:     {self.case.id}",
            f"input:    {self.case.input!r}",
            f"unexpected {type(error).__name__}: {error}",
        ]
        for frame in theirs:
            lines.append(f"  {Path(frame.filename).name}:{frame.lineno} in {frame.name}")
            if frame.line:
                lines.append(f"    {frame.line}")
        return "\n".join(lines)

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, 0, f"{self.declared.name}::{self.name}"


# `_run_task`, `_score` and `_await` lived here until the runner took over
# execution. They are gone rather than kept "just in case": two execution paths
# that can drift apart is the exact risk this wiring was meant to remove, and a
# dead copy of the scoring rules is the most likely thing to be edited by
# mistake. The live versions are `runner._call_task` and `runner._score`.


EXIT_BELOW_THRESHOLD = 1
"""An eval's mean fell below `--threshold`. The measurement succeeded and the
answer was "worse than the bar"."""

EXIT_EXECUTION_ERROR = 2
"""Something did not run: a task raised, a scorer broke, or a file failed to
import. Distinct from 1 because it is a different message to whoever reads the
build — one says the model got worse, the other says we do not know."""


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Decide the session's exit code from what the evals actually did.

    Three outcomes, documented in `docs/ci.md` and kept apart on purpose:

    - **0** every eval met its bar.
    - **1** an eval's mean fell below `--threshold`. Without this, a continuous
      scorer can never fail a run: `levenshtein` and friends deliberately leave
      `passed` unset — they report where an answer sits on a scale and do not
      know where the line is — so a model answering every case with garbage
      exited zero and CI went green.
    - **2** with `--fail-on-error`, something did not run.

    The threshold is the user supplying the judgement the scorer declined to
    make, which is why it is opt-in: `evalstand` will not invent a pass mark. It
    judges the aggregate only and never sets any Score's pass flag, per
    CONTEXT.md.

    **2 outranks 1.** A run whose cases mostly errored has a mean over the few
    that survived, and that mean is not a measurement of the eval — it is a
    measurement of the subset that happened to work. Reporting "below
    threshold" there would name a cause the evidence does not support, and send
    whoever reads the build to look at the model instead of the outage.
    """
    runs = _collected_runs(session.config)
    if not runs:
        return

    config = session.config

    if config.getoption("--fail-on-error", default=False):
        failures = [
            (run.name, sum(1 for r in run.results if r.error), run.errored_score_count)
            for run in runs
            if any(r.error for r in run.results) or run.errored_score_count
        ]
        if failures:
            config._evalstand_error_failures = failures  # type: ignore[attr-defined]
            session.exitstatus = EXIT_EXECUTION_ERROR
            return

    threshold = config.getoption("--threshold", default=None)
    if threshold is None:
        return

    breaches = [
        (run.name, run.mean_score)
        for run in runs
        if run.mean_score is not None and run.mean_score < threshold
    ]

    # A run that measured nothing is not below the threshold — it is unmeasured,
    # and reporting it as a breach would put a number where there is none. It
    # already fails through its unmeasured cases.
    if breaches:
        config._evalstand_breaches = breaches  # type: ignore[attr-defined]
        session.exitstatus = EXIT_BELOW_THRESHOLD


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Print the eval summary after pytest's own report.

    Only when evals actually ran: a plain test session must look untouched.
    """
    runs = _collected_runs(config)
    if not runs:
        return

    _write_html(config, runs)

    if config.getoption("--output", default="terminal") == "markdown":
        _write_markdown(terminalreporter, config, runs)
        return

    from rich.console import Console

    from evalstand.reporting.console import render_failures, render_summary

    console = Console(file=terminalreporter._tw._file, highlight=False)
    console.print()
    console.print(render_summary(runs, wall_seconds=_session_seconds(terminalreporter)))

    failures = render_failures(runs)
    if failures is not None:
        console.print(failures)

    # Printed here rather than left to the exit code alone: a run that fails
    # for a reason the output never states is indistinguishable from a bug in
    # the tool, and the user would go looking for the wrong thing.
    breaches = getattr(config, "_evalstand_breaches", [])
    if breaches:
        threshold = config.getoption("--threshold", default=None)
        for name, mean in breaches:
            console.print(
                f"[red]FAILED[/red] {name}: mean {mean:.2f} is below the "
                f"threshold of {threshold:.2f}"
            )

    # Said separately from a breach, and never alongside one: the run exited 2
    # precisely because its mean does not describe the eval, so printing a
    # verdict about that mean here would contradict the exit code.
    for name, errored_cases, errored_scores in getattr(config, "_evalstand_error_failures", []):
        parts = []
        if errored_cases:
            parts.append(f"{errored_cases} case{'' if errored_cases == 1 else 's'} errored")
        if errored_scores:
            parts.append(f"{errored_scores} score{'' if errored_scores == 1 else 's'} errored")
        console.print(
            f"[red]ERROR[/red] {name}: {', '.join(parts)} "
            f"(--fail-on-error), so the mean does not measure the whole eval"
        )


def _write_html(config: pytest.Config, runs: list[Run]) -> None:
    """Write the HTML artifact, if one was asked for.

    A path rather than stdout, because the point of this format is a file CI can
    upload and somebody can open days later. Writing it to a stream would leave
    the caller to redirect it, and a half-written report from a failed
    redirection looks exactly like a complete one.

    A write failure is reported and swallowed. The run has already happened and
    been recorded; losing it to an unwritable directory would be the tail
    wagging the dog.
    """
    destination = config.getoption("--html", default=None)
    if not destination:
        return

    from evalstand.reporting.html import render_html

    path = Path(destination)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_html(
                runs,
                threshold=config.getoption("--threshold", default=None),
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("could not write the HTML report to %s: %s", path, exc)


def _write_markdown(terminalreporter: Any, config: pytest.Config, runs: list[Run]) -> None:
    """Print the summary as markdown instead of tables.

    Written to stdout rather than a file so the caller decides where it goes:
    `evalstand run --output markdown >> $GITHUB_STEP_SUMMARY` is the whole
    recipe, and a flag naming a path would be one more thing to get wrong in a
    workflow nobody can debug locally.

    Rich is bypassed entirely — it would wrap the lines to the terminal width,
    and a wrapped markdown table stops being a table.
    """
    from evalstand.reporting.markdown import render_markdown

    body = render_markdown(
        runs,
        threshold=config.getoption("--threshold", default=None),
        wall_seconds=_session_seconds(terminalreporter),
    )
    terminalreporter._tw._file.write("\n" + body)


def _session_seconds(terminalreporter: Any) -> float | None:
    start = getattr(terminalreporter, "_sessionstarttime", None)
    return time.time() - start if start else None


def _collected_runs(config: pytest.Config) -> list[Run]:
    """The Runs the runner produced.

    Assembly belongs to the runner, which is the only place that saw the traces,
    token counts and costs. Rebuilding Runs here from item state — as Phase 2
    did — would silently drop all three, which is exactly how a summary comes to
    print "-" for a run that really did cost money.
    """
    return list(getattr(config, "_evalstand_runs", []))

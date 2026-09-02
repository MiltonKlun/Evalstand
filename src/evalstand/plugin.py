"""The pytest plugin: collect `*_eval.py` and run one item per execution.

Registered through the `pytest11` entry point, so bare `pytest` collects evals
with no extra configuration (parity item 19).

The unit of collection is one `(case, repeat_index)` pair, not one case. An item
is one execution with one outcome; collapsing several stochastic executions into
a single pass/fail would require inventing an aggregation rule — any/all/majority
— and that is a judgement the user did not make.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

from evalstand.api import Eval, _current_eval_file, registry
from evalstand.models import Case, Result, Run, RunStatus, Score

EVAL_FILE_SUFFIX = "_eval.py"

_SESSION_MARKER = "_evalstand_registry_reset"


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
        """
        from _pytest.pathlib import ImportMode, import_path

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


def _is_internal(filename: str) -> bool:
    """Frames belonging to pytest, pluggy, or this plugin are not the user's."""
    lowered = filename.replace("\\", "/").lower()
    return (
        "/_pytest/" in lowered or "/pluggy/" in lowered or lowered.endswith("evalstand/plugin.py")
    )


class EvalCaseFailedError(AssertionError):
    """One case did not pass. Carries enough context to act on without re-running."""


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
        self.output: Any = None
        self.scores: list[Score] = []

    def runtest(self) -> None:
        try:
            self.output = _run_task(self.declared.task, self.case.input)
        except Exception as exc:
            # Recorded before re-raising: a case that blew up is a result, and
            # the summary must not silently omit it.
            self._record(error=f"{type(exc).__name__}: {exc}")
            raise
        self.scores = [
            _score(fn, self.output, self.case.expected, self.case) for fn in self.declared.scorers
        ]

        self._record()

        failed = [s for s in self.scores if s.passed is False]
        if failed:
            raise EvalCaseFailedError(
                f"case {self.case.id!r} did not pass {', '.join(s.scorer_name for s in failed)}"
            )

    def _record(self, error: str | None = None) -> None:
        """Keep this execution so the terminal summary can report on it."""
        store: dict[str, tuple[Eval, list[Result]]] = getattr(self.config, "_evalstand_results", {})
        _, results = store.setdefault(self.declared.name, (self.declared, []))
        results.append(
            Result(
                id=f"{self.declared.name}-{self.case.id}-{self.repeat_index}",
                case_id=self.case.id,
                repeat_index=self.repeat_index,
                output=self.output,
                error=error,
                scores=self.scores,
            )
        )
        self.config._evalstand_results = store  # type: ignore[attr-defined]

    def repr_failure(self, excinfo: Any, style: Any = None) -> str:
        """Show what happened, so a failure is actionable without a re-run."""
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

        # Show the user's own frames. pytest's default repr buries the actual
        # error under eight frames of runner and pluggy internals.
        error = excinfo.value
        theirs = [
            frame for frame in traceback.extract_tb(excinfo.tb) if not _is_internal(frame.filename)
        ]

        lines = [
            f"eval:     {self.declared.name}",
            f"case:     {self.case.id}",
            f"input:    {self.case.input!r}",
            f"the task raised {type(error).__name__}: {error}",
        ]
        for frame in theirs:
            lines.append(f"  {Path(frame.filename).name}:{frame.lineno} in {frame.name}")
            if frame.line:
                lines.append(f"    {frame.line}")
        return "\n".join(lines)

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, 0, f"{self.declared.name}::{self.name}"


def _run_task(task: Any, case_input: Any) -> Any:
    """Call the task, sync or async. The user never says which it is."""
    result = task(case_input)
    if inspect.isawaitable(result):
        return asyncio.run(_await(result))
    return result


def _score(fn: Any, output: Any, expected: Any, case: Case) -> Score:
    """Run one scorer, capturing a raise rather than letting it fail the case.

    A scorer that broke is not evidence the task did badly, so the error is
    recorded on the Score and excluded from means.
    """
    name = getattr(fn, "__name__", "scorer")
    try:
        result = fn(output, expected, case)
        if inspect.isawaitable(result):
            result = asyncio.run(_await(result))
    except Exception as exc:  # a scorer failure is data, not a crash
        return Score.from_error(name, f"{type(exc).__name__}: {exc}")

    if isinstance(result, Score):
        return result
    return Score(scorer_name=name, value=float(result))


async def _await(awaitable: Any) -> Any:
    return await awaitable


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Print the eval summary after pytest's own report.

    Only when evals actually ran: a plain test session must look untouched.
    """
    runs = _collected_runs(config)
    if not runs:
        return

    from rich.console import Console

    from evalstand.reporting.console import render_failures, render_summary

    console = Console(file=terminalreporter._tw._file, highlight=False)
    console.print()
    console.print(render_summary(runs, wall_seconds=_session_seconds(terminalreporter)))

    failures = render_failures(runs)
    if failures is not None:
        console.print(failures)


def _session_seconds(terminalreporter: Any) -> float | None:
    start = getattr(terminalreporter, "_sessionstarttime", None)
    return time.time() - start if start else None


def _collected_runs(config: pytest.Config) -> list[Run]:
    """Assemble Runs from the items that executed.

    Phase 2 builds these here so reporting has something real to render. Phase 3
    moves execution into the runner, which will own Run assembly instead.
    """
    results_by_eval: dict[str, tuple[Eval, list[Result]]] = getattr(
        config, "_evalstand_results", {}
    )
    return [
        Run(
            id=f"run-{name}",
            batch_id="batch-local",
            name=name,
            filepath=declared.filepath,
            status=RunStatus.COMPLETED,
            repeat_n=declared.repeat,
            results=results,
        )
        for name, (declared, results) in results_by_eval.items()
    ]

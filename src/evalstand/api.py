"""The public authoring API.

`evaluate()` is what a user writes at module scope. It **registers and returns** —
no I/O, no event loop, no model calls (ADR 0004). Something else executes later.

That deferral is what makes `pytest -k q1` able to select a single case, what
lets watch mode ask which Evals a changed file declares without running them,
and what keeps `import qa_eval` from spending money.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from evalstand.models import Case

__all__ = ["DuplicateEvalNameError", "Eval", "Registry", "evaluate", "registry"]

CaseSource: TypeAlias = (
    Sequence[Case | dict[str, Any]]
    | Callable[[], Sequence[Case | dict[str, Any]]]
    | Callable[[], Awaitable[Sequence[Case | dict[str, Any]]]]
)
"""Cases as a list, a callable returning one, or an async callable."""

Task: TypeAlias = Callable[..., Any]
ScorerFn: TypeAlias = Callable[..., Any]
Column: TypeAlias = Callable[[Any], Any]


class DuplicateEvalNameError(ValueError):
    """Two Evals share a name.

    A name is an Eval's identity: `history` and `compare` address Evals by it.
    Merging two silently would join unrelated histories into one meaningless
    series, so this is raised at collection time instead.
    """


@dataclass(frozen=True)
class Eval:
    """One declared `evaluate()` call. Inert until a runner executes it."""

    name: str
    cases: CaseSource
    task: Task
    scorers: Sequence[ScorerFn]
    filepath: str = ""
    repeat: int = 1
    columns: dict[str, Column] = field(default_factory=dict)

    def load_cases(self) -> list[Case]:
        """Resolve the case source from synchronous code.

        Deliberately not done at registration: loading may read files or call a
        network, and a loader that raises should fail its own Eval rather than
        break collection of every other one.

        An async loader is driven to completion here. Inside a running event
        loop use `aload_cases` instead — `asyncio.run` cannot nest.
        """
        source: Any = self.cases
        if callable(source):
            source = source()
            if inspect.isawaitable(source):
                source = asyncio.run(_await(source))
        return self._finalise(source)

    async def aload_cases(self) -> list[Case]:
        """Resolve the case source from async code.

        The runner is async, so this is the path it uses. Sync loaders work here
        too, which keeps callers from having to know which kind they were given.
        """
        source: Any = self.cases
        if callable(source):
            source = source()
            if inspect.isawaitable(source):
                source = await source
        return self._finalise(source)

    def _finalise(self, source: Any) -> list[Case]:
        """Normalise, number, and check a loaded case list.

        Shared by both loaders so the two cannot disagree about what a valid
        dataset is.
        """
        cases = [item if isinstance(item, Case) else Case(**item) for item in _numbered(source)]

        if not cases:
            raise ValueError(f"eval {self.name!r} has no cases")

        ids = [case.id for case in cases]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"eval {self.name!r} has duplicate case ids: {', '.join(duplicates)}")

        return cases

    @property
    def task_is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.task)


async def _await(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _numbered(source: Iterable[Any]) -> list[Any]:
    """Give ids to cases that arrived without them.

    A generated dataset rarely has natural ids. Numbering by position keeps
    history matchable — but only while the order holds, which `docs/writing-evals.md`
    says plainly. Content-derived ids are deliberately not used: they would make
    an edited case look like a different case, which is exactly what Amended
    Case detection has to catch.
    """
    numbered = []
    for index, item in enumerate(source, start=1):
        if isinstance(item, dict) and not item.get("id"):
            item = {**item, "id": f"case_{index:04d}"}
        numbered.append(item)
    return numbered


class Registry:
    """The Evals a module declared when it was imported."""

    def __init__(self) -> None:
        self._evals: dict[str, Eval] = {}

    def add(self, declared: Eval) -> None:
        """Register an Eval, refusing a genuine name collision.

        Re-registering the *same* name from the *same* file is idempotent and
        replaces the declaration: pytest imports a file twice when it is named
        twice on the command line, and watch mode re-imports on every edit.
        Neither is a collision, and the newer declaration is the current one.

        Two evals sharing a name from different files is a real collision, since
        a name is what `history` and `compare` address an Eval by.
        """
        existing = self._evals.get(declared.name)
        if existing is not None:
            same_known_file = bool(declared.filepath) and existing.filepath == declared.filepath
            if not same_known_file:
                raise DuplicateEvalNameError(
                    f"two evals are named {declared.name!r}: "
                    f"{existing.filepath or '<unknown>'} and "
                    f"{declared.filepath or '<unknown>'}. "
                    f"Names identify evals across runs, so they must be unique."
                )
        self._evals[declared.name] = declared

    def evals(self) -> list[Eval]:
        return list(self._evals.values())

    def get(self, name: str) -> Eval | None:
        return self._evals.get(name)

    def clear(self) -> None:
        self._evals.clear()

    def __len__(self) -> int:
        return len(self._evals)


registry = Registry()
"""Module-level registry. A runner reads it after importing an eval file."""

_current_eval_file: contextvars.ContextVar[str] = contextvars.ContextVar(
    "evalstand_current_eval_file", default=""
)
"""The file currently being imported for collection.

Set by the plugin so `evaluate()` knows which file declared it without the user
passing a path. That path is what tells a re-import apart from a real collision.
"""


def evaluate(
    *,
    name: str,
    cases: CaseSource,
    task: Task,
    scorers: Sequence[ScorerFn],
    filepath: str = "",
    repeat: int = 1,
    columns: dict[str, Column] | None = None,
) -> Eval:
    """Declare an eval.

    Registers and returns immediately. Nothing runs until a runner reads the
    registry — see ADR 0004.

    ```python
    evaluate(
        name="basic-qa",
        cases=load_cases,
        task=answer,
        scorers=[exact],
    )
    ```
    """
    if not name.strip():
        raise ValueError("an eval needs a non-blank name")
    if not scorers:
        raise ValueError(
            f"eval {name!r} declares no scorers; an eval with nothing to measure "
            f"would report an empty mean"
        )
    if repeat < 1:
        raise ValueError(f"eval {name!r} has repeat={repeat}; it must be at least 1")

    declared = Eval(
        name=name,
        cases=cases,
        task=task,
        scorers=scorers,
        filepath=filepath or _current_eval_file.get(),
        repeat=repeat,
        columns=columns or {},
    )
    registry.add(declared)
    return declared


def scorer(fn: ScorerFn) -> ScorerFn:
    """Mark a plain function as a scorer.

    Re-exported from `scorers.base`, where the protocol lives. Applying it is
    optional — a plain function passed to `evaluate(scorers=[...])` is already
    treated as a scorer — but it validates the signature at import time, so a
    scorer that could never be called fails before a run spends money.
    """
    from evalstand.scorers.base import scorer as _scorer

    return _scorer(fn)


def trace(name: str) -> Any:
    """Wrap an operation so it appears in a Result's trace tree.

    A placeholder in Phase 2; implemented in Phase 3, where the contextvar
    machinery lands.
    """
    raise NotImplementedError("trace() arrives in Phase 3")

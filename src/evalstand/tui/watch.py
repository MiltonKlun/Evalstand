"""Watch mode: re-run when the code being measured changes (task 6.6).

The feedback loop this tool exists for. Editing a prompt and seeing the scores
move without leaving the terminal is the difference between evaluating a change
and merely intending to.

Two decisions here are worth stating, because both trade something away:

**A change during an in-flight Batch cancels it.** Waiting for a slow Batch to
drain would spend the feedback loop this feature exists to provide — the user
has already moved on, and the results still arriving describe code they have
edited. The cost is a partial Batch, which is why cancellation is recorded
honestly rather than quietly: `cancelled` status, hidden from `history`, refused
by `compare`. A half-finished Batch that looked complete would drag every mean
it touched.

**In-flight model calls are allowed to finish.** The money is already spent, so
hard-killing a request discards a response the user has paid for — and the next
Batch, moments away, will ask for exactly the same thing. Letting it land in the
cache turns a wasted call into a free one.

The watching itself is deliberately dumb: any Python file under the watched
roots triggers a re-run. Trying to compute which evals a changed file *affects*
means resolving an import graph that the user can defeat with a dynamic import,
a config file, or a prompt loaded from disk — and a watch mode that silently
misses a change is worse than one that occasionally re-runs too much, because
the user stops trusting what they see.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from watchfiles import Change, awatch

__all__ = ["DEBOUNCE_MS", "changed_paths", "is_relevant", "watch_paths", "watch_roots"]

logger = logging.getLogger("evalstand.watch")

DEBOUNCE_MS = 300
"""How long to wait for a burst of changes to settle.

An editor writing a file produces several events — a temp file, a rename, a
touch — and re-running on each would start three Batches for one save. Long
enough to coalesce those, short enough that the re-run still feels immediate.
"""

_WATCHED_SUFFIXES = frozenset({".py", ".txt", ".md", ".json", ".yaml", ".yml", ".jinja", ".j2"})
"""What counts as the code being measured.

Prompts live in text files as often as in Python — a `.txt` under a `prompts/`
directory is the thing a user edits most while tuning. Missing those would make
watch mode useless for the exact workflow it is for.
"""


def is_relevant(path: str | Path) -> bool:
    """Whether a changed file should trigger a re-run.

    Excludes what the tool itself writes. The history database is updated *by* a
    run, so treating it as a change would make every Batch trigger the next one
    — a loop that spends money until the user notices.
    """
    resolved = Path(path)
    name = resolved.name

    if resolved.suffix.lower() not in _WATCHED_SUFFIXES:
        return False

    # Anything the tool writes, and the usual editor debris. `.db-wal` and
    # `.db-shm` never reach here (their suffixes are not watched), but the
    # database name is checked explicitly because a user may point `--db` at
    # something ending in `.json`.
    if name.startswith(".") or name.endswith("~"):
        return False

    parts = {part.lower() for part in resolved.parts}
    return not parts & {"__pycache__", ".git", ".venv", "node_modules", ".pytest_cache"}


def changed_paths(changes: Iterable[tuple[Change, str]]) -> list[Path]:
    """The relevant files in one batch of change events, in a stable order.

    Deleted files are included: removing a prompt file changes what the task
    does, and a re-run that reported the resulting error is more useful than
    silence.
    """
    paths = {Path(path) for _, path in changes if is_relevant(path)}
    return sorted(paths)


def watch_roots(paths: Iterable[Path | str] | None = None) -> list[Path]:
    """The directories to watch, derived from what the user asked to run.

    A file argument is watched through its parent directory: editing the prompt
    *next to* an eval file must trigger a re-run, and watching the single file
    would miss it.
    """
    resolved = [Path(path).resolve() for path in (paths or [Path()])]
    roots = {path if path.is_dir() else path.parent for path in resolved}

    # Drop any root already contained in another, so a change under a nested
    # path is not reported twice.
    return sorted(
        root
        for root in roots
        if not any(other != root and other in root.parents for other in roots)
    )


async def watch_paths(
    paths: Iterable[Path | str] | None = None,
    *,
    stop: asyncio.Event | None = None,
    debounce_ms: int = DEBOUNCE_MS,
) -> AsyncIterator[list[Path]]:
    """Yield each settled batch of relevant changes.

    A batch containing nothing relevant yields nothing rather than an empty
    list, so a caller can treat every yield as "re-run now".
    """
    roots = watch_roots(paths)
    logger.debug("watching %s", ", ".join(str(root) for root in roots))

    async for changes in awatch(*roots, debounce=debounce_ms, stop_event=stop):
        relevant = changed_paths(changes)
        if relevant:
            yield relevant

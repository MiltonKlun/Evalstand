"""What produced a run: the commit, the tree state, and the task's own source.

A stored result without provenance is a number nobody can act on. Six months
later "the score dropped" is only useful if you can say *between which commits*,
so this is recorded on every Batch and refused rather than guessed at.

The distinction that runs through this module: **absent is not the same as
false.** Not being in a git repository, or being in one with no commits yet, is
a normal way to use the tool — those runs are stored with a null SHA and a null
dirty flag, which says "unknown". Recording `dirty = False` there would assert
a fact nobody established, and a comparison built on it would be confidently
wrong.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["DirtyTreeError", "GitState", "git_state", "task_source_hash"]

logger = logging.getLogger("evalstand.provenance")

_TIMEOUT_SECONDS = 10
"""Long enough for a slow filesystem, short enough that a hung git — a stale
lock, a network filesystem — cannot wedge a run before it starts."""


class DirtyTreeError(RuntimeError):
    """The working tree has uncommitted changes and `--allow-dirty` was not given.

    Refused because a run recorded against a SHA whose tree it did not actually
    reflect is worse than an unrecorded one: it looks reproducible and is not.
    """


@dataclass(frozen=True)
class GitState:
    """The repository state a run was produced from.

    `sha` and `dirty` are both None when there is nothing to report — no
    repository, no commits, or no git. `dirty` is deliberately not defaulted to
    False: "not checked" and "checked and clean" are different claims, and only
    one of them can be made honestly.
    """

    sha: str | None = None
    dirty: bool | None = None

    @property
    def is_recorded(self) -> bool:
        """Whether this run can be tied to a commit at all."""
        return self.sha is not None

    def describe(self) -> str:
        """A short, honest phrase for a report."""
        if self.sha is None:
            return "no git"
        short = self.sha[:8]
        if self.dirty:
            return f"{short}-dirty"
        return short


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str] | None:
    """Run a git command, or None when git cannot be used at all.

    A missing git is not an error worth failing a run over — the user asked to
    measure a model, not to have their tooling audited.
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("git is unavailable; provenance will be unrecorded", exc_info=True)
        return None


def git_state(path: Path | str = ".") -> GitState:
    """The commit and tree state at `path`, as far as they can be determined.

    Returns an empty `GitState` rather than raising when there is no repository,
    no commits, or no git. Each of those is a legitimate way to run the tool,
    and refusing on first contact would be a worse failure than an unrecorded
    provenance the report states plainly.
    """
    directory = Path(path).resolve()
    if directory.is_file():
        directory = directory.parent

    revision = _git("rev-parse", "HEAD", cwd=directory)
    if revision is None or revision.returncode != 0:
        # A repository with no commits yet fails here *and prints "HEAD" to
        # stdout*, so trusting the output without checking the return code
        # would store the literal string "HEAD" as a commit SHA.
        return GitState()

    sha = revision.stdout.strip()
    if not _looks_like_a_sha(sha):
        logger.debug("git returned something that is not a sha: %r", sha)
        return GitState()

    # `--untracked-files=no` on purpose. A scratch file, a .env, or an
    # unignored output directory is not a change to the code being measured:
    # the commit recorded still describes exactly what ran.
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=directory)
    if status is None or status.returncode != 0:
        # The commit is known but the tree state is not. Reporting it as clean
        # would be the one lie this module exists to avoid.
        return GitState(sha=sha, dirty=None)

    return GitState(sha=sha, dirty=bool(status.stdout.strip()))


def _looks_like_a_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def require_clean(state: GitState, *, allow_dirty: bool) -> None:
    """Raise unless the run may be persisted.

    A dirty tree is refused without `--allow-dirty`; an *unknown* tree state is
    not. Being outside a repository is a normal way to try the tool, and the
    null SHA already tells any reader that this run cannot be tied to a commit.
    """
    if state.dirty and not allow_dirty:
        raise DirtyTreeError(
            f"the working tree has uncommitted changes, so results cannot be "
            f"tied to commit {state.sha[:8] if state.sha else '?'}. "
            f"Commit them, or pass --allow-dirty to persist anyway."
        )


def task_source_hash(task: Any) -> str | None:
    """A hash of the task's own source, or None when it cannot be read.

    Answers one question: did the code being measured change between two runs?
    Two runs with different hashes did not measure the same task, whatever else
    stayed the same — which is what stops a model change and a code change
    looking identical in a comparison.

    Covers the task function alone, not the whole eval file. Hashing the file
    would change when an unrelated comment or a second eval in it was edited,
    reporting a difference that is not one — and a field that cries wolf is a
    field readers learn to ignore. The cost is that a helper the task calls is
    not covered; `docs/ci.md` says so.
    """
    try:
        source = inspect.getsource(task)
    except (OSError, TypeError):
        # A lambda typed at a REPL, a C callable, a partial. Unreadable source
        # is not a reason to fail a run.
        logger.debug("could not read the task's source", exc_info=True)
        return None

    # Leading indentation varies with where the function was defined — a task
    # nested inside a fixture is indented, the same code at module level is not.
    # Normalising means moving a function does not read as changing it.
    normalised = inspect.cleandoc(source)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

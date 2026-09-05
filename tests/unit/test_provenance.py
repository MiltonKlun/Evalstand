"""Provenance (task 5.2): what produced a run.

A stored result without provenance is a number nobody can act on. The rule that
runs through this file: **absent is not the same as false.** Not being in a
repository, or being in one with no commits, is a normal way to use the tool —
those runs record a null SHA and a null dirty flag, which says "unknown".
Recording `dirty = False` there would assert a fact nobody established.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from evalstand.provenance import (
    DirtyTreeError,
    GitState,
    git_state,
    require_clean,
    task_source_hash,
)


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)


def _source(lines: list[str]) -> str:
    """Source text built from lines, so indentation is visible and deliberate."""
    return "\n".join(lines) + "\n"


def _hash_of(source: str) -> str | None:
    """Hash a fixed piece of source, whatever function is passed.

    A real pair of functions cannot differ *only* in indentation — the nested
    one needs a decorator or an enclosing `def`, and that line changes the hash
    on its own. Substituting the source is the only way to isolate the thing
    under test.
    """
    with patch("evalstand.provenance.inspect.getsource", return_value=source):
        return task_source_hash(_source)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real repository with one commit. Real git, not a mock: the behaviours
    that matter here are git's own, and a mock would only assert what I already
    believed."""
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "code.py").write_text("x = 1\n")
    _git("add", ".", cwd=tmp_path)
    _git("commit", "-qm", "first", cwd=tmp_path)
    return tmp_path


class TestReadingTheRepository:
    def test_a_clean_repo_reports_its_sha(self, repo: Path) -> None:
        state = git_state(repo)

        assert state.sha is not None
        assert len(state.sha) == 40
        assert state.dirty is False
        assert state.is_recorded

    def test_a_modified_tracked_file_is_dirty(self, repo: Path) -> None:
        (repo / "code.py").write_text("x = 2\n")

        assert git_state(repo).dirty is True

    def test_an_untracked_file_is_not_dirty(self, repo: Path) -> None:
        """A scratch file, a .env, or an unignored output directory is not a
        change to the code being measured — the commit recorded still describes
        exactly what ran. Counting it would have users passing --allow-dirty
        reflexively, which defeats the check."""
        (repo / "scratch.tmp").write_text("junk")

        assert git_state(repo).dirty is False

    def test_a_subdirectory_reports_the_same_repository(self, repo: Path) -> None:
        nested = repo / "evals" / "deep"
        nested.mkdir(parents=True)

        assert git_state(nested).sha == git_state(repo).sha

    def test_a_file_path_is_read_as_its_directory(self, repo: Path) -> None:
        """The plugin passes a rootpath that may be a file."""
        assert git_state(repo / "code.py").sha == git_state(repo).sha


class TestWhenThereIsNothingToRecord:
    """Each of these is a legitimate way to run the tool, and each has a wrong
    answer that would attach false provenance to a run."""

    def test_a_directory_that_is_not_a_repository(self, tmp_path: Path) -> None:
        state = git_state(tmp_path)

        assert state.sha is None
        assert state.dirty is None, "unknown, not clean"
        assert not state.is_recorded

    def test_a_repository_with_no_commits_yet(self, tmp_path: Path) -> None:
        """The trap. `git rev-parse HEAD` fails here *and prints "HEAD" to
        stdout*, so code that reads the output without checking the return code
        stores the literal string "HEAD" as a commit SHA."""
        _git("init", "-q", cwd=tmp_path)

        state = git_state(tmp_path)

        assert state.sha is None
        assert state.sha != "HEAD"

    def test_output_that_is_not_a_sha_is_refused(self, tmp_path: Path) -> None:
        """Belt and braces on the same trap: whatever git prints, only a real
        40-character hex digest is stored."""
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="HEAD\n", stderr="")

        with patch("evalstand.provenance._git", return_value=fake):
            assert git_state(tmp_path).sha is None

    def test_a_missing_git_does_not_fail_the_run(self, tmp_path: Path) -> None:
        """The user asked to measure a model, not to have their tooling
        audited."""
        with patch("evalstand.provenance.subprocess.run", side_effect=FileNotFoundError("git")):
            state = git_state(tmp_path)

        assert state == GitState()

    def test_a_known_commit_with_an_unknown_tree_is_not_reported_clean(self, repo: Path) -> None:
        """If `git status` fails, the commit is known but the tree state is
        not. Reporting it clean would be the one lie this module exists to
        avoid."""
        real = subprocess.run

        def status_fails(args: list[str], **kwargs: object) -> object:
            if "status" in args:
                return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="")
            return real(args, **kwargs)  # type: ignore[arg-type]

        with patch("evalstand.provenance.subprocess.run", side_effect=status_fails):
            state = git_state(repo)

        assert state.sha is not None
        assert state.dirty is None


class TestDescribingTheState:
    def test_a_clean_commit_is_named_by_its_short_sha(self) -> None:
        assert GitState(sha="a" * 40, dirty=False).describe() == "aaaaaaaa"

    def test_a_dirty_tree_says_so(self) -> None:
        assert GitState(sha="a" * 40, dirty=True).describe() == "aaaaaaaa-dirty"

    def test_an_unrecorded_run_says_so_plainly(self) -> None:
        """A reader must be able to see at a glance that a row cannot be tied
        to a commit."""
        assert GitState().describe() == "no git"


class TestTheDirtyTreeGate:
    def test_a_dirty_tree_is_refused(self) -> None:
        """A run recorded against a SHA whose tree it did not reflect is worse
        than an unrecorded one: it looks reproducible and is not."""
        with pytest.raises(DirtyTreeError, match="uncommitted changes"):
            require_clean(GitState(sha="a" * 40, dirty=True), allow_dirty=False)

    def test_allow_dirty_permits_it(self) -> None:
        require_clean(GitState(sha="a" * 40, dirty=True), allow_dirty=True)

    def test_a_clean_tree_passes(self) -> None:
        require_clean(GitState(sha="a" * 40, dirty=False), allow_dirty=False)

    def test_an_unknown_tree_state_is_not_refused(self) -> None:
        """Being outside a repository is a normal way to try the tool. Refusing
        there would fail a user on first contact to enforce a rule about a
        repository they do not have."""
        require_clean(GitState(), allow_dirty=False)

    def test_the_error_names_the_commit_and_the_way_out(self) -> None:
        with pytest.raises(DirtyTreeError) as caught:
            require_clean(GitState(sha="abcdef01" + "0" * 32, dirty=True), allow_dirty=False)

        message = str(caught.value)
        assert "abcdef01" in message
        assert "--allow-dirty" in message


class TestTheTaskSourceHash:
    """Answers one question: did the code being measured change between two
    runs? Two runs with different hashes did not measure the same task."""

    def test_different_tasks_hash_differently(self) -> None:
        def upper(value: str) -> str:
            return value.upper()

        def lower(value: str) -> str:
            return value.lower()

        assert task_source_hash(upper) != task_source_hash(lower)

    def test_the_same_task_hashes_the_same(self) -> None:
        def stable(value: str) -> str:
            return value

        assert task_source_hash(stable) == task_source_hash(stable)

    def test_it_is_a_sha256_digest(self) -> None:
        def task(value: str) -> str:
            return value

        digest = task_source_hash(task)
        assert digest is not None
        assert len(digest) == 64

    def test_indentation_is_normalised(self) -> None:
        """Moving a task into a nested scope must not read as changing it.

        Asserted on the source text directly, because a real pair of functions
        cannot differ *only* in indentation — the nested one needs a decorator
        or an enclosing def, and that line changes the hash on its own. Without
        `cleandoc`, `compare` would report a code change that never happened.
        """
        flat = _source(["def answer(q):", "    return q.upper()"])
        indented = _source(["    def answer(q):", "        return q.upper()"])

        assert _hash_of(flat) == _hash_of(indented), "moving a task changed its hash"

    def test_a_real_change_still_changes_the_hash(self) -> None:
        """The other half: normalisation must not be so eager it stops
        distinguishing edits."""
        before = _source(["def answer(q):", "    return q.upper()"])
        after = _source(["def answer(q):", "    return q.lower()"])

        assert _hash_of(before) != _hash_of(after)

    def test_an_unreadable_source_is_not_an_error(self) -> None:
        """A builtin, a C callable, a lambda typed at a REPL. Unreadable source
        is not a reason to fail a run."""
        assert task_source_hash(len) is None

    def test_a_non_callable_does_not_raise(self) -> None:
        assert task_source_hash(42) is None

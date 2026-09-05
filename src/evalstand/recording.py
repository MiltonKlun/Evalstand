"""Recording a batch of runs to the local history database.

The seam between running and storing. It exists so the two can fail
independently: **a storage failure must never cost a measurement.** By the time
a run is written the money is already spent, so a full disk or a locked database
degrades to a warning and a lost history entry, not a lost result.

The one exception is provenance, which is checked *before* anything executes.
Discovering that a run cannot be recorded after paying for it would be the worst
possible ordering, so a dirty tree without `--allow-dirty` refuses up front.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalstand.models import Batch, BatchKind, BatchStatus, Case, Run
from evalstand.provenance import GitState, git_state, require_clean, task_source_hash
from evalstand.storage import DatabaseTooNewError, RunStore

__all__ = ["BatchRecorder", "open_recorder"]

logger = logging.getLogger("evalstand.recording")


class BatchRecorder:
    """Writes one invocation's Runs as each finishes.

    Per run rather than per batch: a crash mid-batch then keeps everything
    already measured and loses only the eval in flight. Buffering the whole
    batch would discard results the user had already paid for.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        git: GitState,
        kind: BatchKind = BatchKind.FULL,
        owns_store: bool = False,
    ) -> None:
        self.store = store
        self.git = git
        # Only close what we opened. A caller who passed their own store
        # still needs it afterwards, and silently closing it turns a read
        # into 'Cannot operate on a closed database' far from the cause.
        self._owns_store = owns_store
        self.batch = Batch(
            id=f"batch-{uuid.uuid4().hex[:12]}",
            kind=kind,
            status=BatchStatus.RUNNING,
            started_at=datetime.now(UTC),
            git_sha=git.sha,
            git_dirty=git.dirty,
        )
        self._failed = False
        self._write(lambda: self.store.save_batch(self.batch))

    @property
    def batch_id(self) -> str:
        return self.batch.id

    def record(self, run: Run, *, cases: list[Case] | None = None, task: Any = None) -> None:
        """Store one finished Run, with the provenance of the code that made it."""
        run = run.model_copy(update={"task_source_hash": task_source_hash(task) if task else None})
        self._write(lambda: self.store.save_run(run, cases=cases))

    def finish(self, *, cancelled: bool = False) -> None:
        """Close the batch.

        A batch interrupted part-way is marked `cancelled` rather than
        `completed`, so `history` and `compare` can exclude it: its aggregate
        describes a subset of the cases and comparing it against a full run
        would be comparing different questions.
        """
        self.batch = self.batch.model_copy(
            update={
                "status": BatchStatus.CANCELLED if cancelled else BatchStatus.COMPLETED,
                "finished_at": datetime.now(UTC),
            }
        )
        self._write(lambda: self.store.save_batch(self.batch))
        if self._owns_store:
            self.store.close()

    def _write(self, action: Any) -> None:
        """Perform a write, downgrading any failure to a warning.

        By the time anything reaches here the run has already cost money. A
        locked database or a full disk must not turn a completed measurement
        into a crash — the result is still on screen, and only the history
        entry is lost.

        Warned once per batch: a message repeated per run would bury the
        summary the user actually came for.
        """
        try:
            action()
        except Exception:
            if not self._failed:
                self._failed = True
                logger.warning(
                    "could not write to the history database; this run will not be "
                    "recorded. The results above are unaffected.",
                    exc_info=True,
                )


def open_recorder(
    *,
    path: Path | str | None = None,
    enabled: bool = True,
    allow_dirty: bool = False,
    root: Path | str = ".",
) -> BatchRecorder | None:
    """Open a recorder, or None when this invocation should not persist.

    Raises `DirtyTreeError` when the tree has uncommitted changes and
    `allow_dirty` is not set — before any case runs, so nothing is paid for
    twice.

    Not being in a git repository is *not* refused. It is a normal way to try
    the tool, and the null SHA already tells any reader that this run cannot be
    tied to a commit. Refusing there would fail a user on first contact to
    enforce a rule about a repository they do not have.
    """
    if not enabled:
        return None

    state = git_state(root)
    require_clean(state, allow_dirty=allow_dirty)

    try:
        store = RunStore(path) if path is not None else RunStore()
    except DatabaseTooNewError:
        # Deliberately not downgraded. The plan requires this to be a hard
        # refusal: continuing would run the eval and then silently fail to
        # record it, leaving the user with a database they cannot read and no
        # sign that anything is wrong with their tooling.
        raise
    except Exception:
        # Any other failure to open — an unreadable file, a permission
        # problem — must not stop the run. The user asked to measure a model,
        # not to maintain a database.
        logger.warning(
            "could not open the history database; this run will not be recorded.", exc_info=True
        )
        return None

    return BatchRecorder(store, git=state, owns_store=True)

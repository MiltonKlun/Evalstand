"""The run store: where results become history.

Unlike the response cache, this is **load-bearing**. Deleting the cache costs
money; deleting or corrupting this loses the comparison data the tool exists to
provide. That difference drives every decision here.

**Forward-only migrations, applied atomically.** An older database is migrated
on open, each step inside its own transaction. A database whose version exceeds
what the installed code understands is *refused*, naming both versions — opening
it best-effort would let a newer column go unread and a comparison silently
answer the wrong question.

**One transaction per Run.** A run costs real money and takes real time, so each
Eval is committed the moment it finishes. A crash mid-batch keeps everything
already measured and loses only the eval in flight; buffering the whole batch
would discard results the user had already paid for.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from evalstand.migrations import LATEST_VERSION, MIGRATIONS
from evalstand.models import Batch, Case, Result, Run, Score, Trace

__all__ = ["DatabaseTooNewError", "RunStore"]

logger = logging.getLogger("evalstand.storage")

DEFAULT_PATH = Path(".evalstand") / "evalstand.db"
"""Project-local, per ADR 0005: history belongs to the repository it measures."""


class DatabaseTooNewError(RuntimeError):
    """The database was written by a newer evalstand than the one installed.

    Refused rather than opened. A newer schema may hold columns this code does
    not read, and a comparison built from a partial read would look right while
    answering a different question — the one failure this tool must never have.
    """


def _text(value: Any) -> str | None:
    """A readable form of a value, never raising.

    Same reasoning as `_json`: an object whose `__str__` raises must not take
    down the save of every other result in the run.
    """
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        logger.debug("could not stringify a value for storage", exc_info=True)
        return f"<unrepresentable {type(value).__name__}>"


def _json(value: Any) -> str | None:
    """Serialise a value, never raising.

    A task may return anything, and a whole run's results must not be lost
    because one output could not be encoded. `default=str` already handles most
    exotic objects; the fallbacks cover the two cases it does not:

    - a **circular reference**, which `json` rejects with a ValueError however
      the leaves are encoded;
    - an object whose `__repr__` itself raises, which escapes as whatever that
      code chose to throw. Catching only TypeError and ValueError there would
      let a hostile repr take down the save of every other result in the run.

    The last resort names the type, because knowing an output was a
    `ChatCompletion` is worth more than losing the row.
    """
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        pass

    try:
        return json.dumps(repr(value))
    except Exception:
        logger.debug("could not represent a value for storage", exc_info=True)
        return json.dumps(f"<unrepresentable {type(value).__name__}>")


class RunStore:
    """A SQLite-backed store of Batches, Runs, Results, Scores and Traces.

    Safe to share across threads. SQLite's own thread check is disabled so one
    store can serve a concurrent run, and a lock is what makes that safe — the
    two go together, and neither alone is enough.
    """

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        if self.path.parent != Path():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # isolation_level=None hands transaction control to us. Python's default
        # opens one implicitly before DML but not before DDL, which would leave
        # a migration's CREATE TABLE outside the transaction meant to protect
        # it.
        self.connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()

        with self._lock:
            # Both pragmas are per-connection and must be set outside any
            # transaction. Foreign keys are OFF by default in SQLite, so
            # without this the ON DELETE CASCADE rules are decoration and an
            # orphan row is accepted silently.
            self.connection.execute("PRAGMA foreign_keys = ON")
            # WAL lets a reader (a `history` command, the TUI) work while a run
            # is writing, instead of blocking on it.
            self.connection.execute("PRAGMA journal_mode = WAL")
            self._migrate()

    # ------------------------------------------------------------------ schema

    @property
    def schema_version(self) -> int:
        row = self.connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def _migrate(self) -> None:
        """Bring the database up to `LATEST_VERSION`, or refuse to open it.

        `user_version` rather than a table: it is a header field, so reading it
        needs no schema to already exist, and it cannot itself be missing from a
        database too old to have the table.
        """
        current = self.schema_version

        if current > LATEST_VERSION:
            raise DatabaseTooNewError(
                f"{self.path} was written by a newer evalstand: its schema is "
                f"version {current}, but this installation understands up to "
                f"{LATEST_VERSION}. Upgrade evalstand to read it."
            )

        for version in sorted(MIGRATIONS):
            if version <= current:
                continue
            self._apply(version, MIGRATIONS[version])

    def _apply(self, version: int, statements: list[str]) -> None:
        """Apply one migration atomically.

        Statements run one at a time. `executescript` would issue an implicit
        COMMIT and silently end this transaction, so a later failure could not
        be rolled back and the database would be left half-migrated — believing
        it holds columns it does not.
        """
        self.connection.execute("BEGIN")
        try:
            for statement in statements:
                self.connection.execute(statement)
            # In the same transaction as the statements it describes: a version
            # bumped separately could outlive a rolled-back migration.
            self.connection.execute(f"PRAGMA user_version = {version}")
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            logger.exception("migration to version %s failed; the database is unchanged", version)
            raise
        logger.debug("migrated %s to schema version %s", self.path, version)

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ writes

    def save_batch(self, batch: Batch) -> None:
        """Insert or update a Batch. Called at the start and end of an invocation."""
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO batches (id, kind, status, started_at, finished_at,
                                     git_sha, git_dirty)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    finished_at = excluded.finished_at
                """,
                (
                    batch.id,
                    batch.kind.value,
                    batch.status.value,
                    batch.started_at.isoformat() if batch.started_at else None,
                    batch.finished_at.isoformat() if batch.finished_at else None,
                    batch.git_sha,
                    # None means "not checked", which is not the same claim as
                    # "clean". Coercing it to 0 would assert a provenance fact
                    # nobody established.
                    None if batch.git_dirty is None else int(batch.git_dirty),
                ),
            )

    def save_run(self, run: Run, *, cases: list[Case] | None = None) -> None:
        """Write one Run and everything hanging off it, in one transaction.

        Committed as soon as the Eval finishes rather than at the end of the
        batch: a crash then loses only the eval in flight, not the ones already
        paid for.
        """
        with self._lock:
            self.connection.execute("BEGIN")
            try:
                self._insert_run(run)
                for result in run.results:
                    self._insert_result(run.id, result)
                for case in cases or []:
                    self._insert_case_snapshot(run.id, case)
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                logger.exception("failed to save run %s; nothing was written", run.id)
                raise

    def _insert_run(self, run: Run) -> None:
        self.connection.execute(
            """
            INSERT INTO runs (id, batch_id, name, filepath, started_at, finished_at,
                              model_config_json, task_source_hash, repeat_n,
                              total_cases, total_cost_usd, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.batch_id,
                run.name,
                run.filepath,
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                _json(run.model_config_used) if run.model_config_used else None,
                run.task_source_hash,
                run.repeat_n,
                len({r.case_id for r in run.results}),
                run.total_cost_usd,
                run.status.value,
            ),
        )

    def _insert_result(self, run_id: str, result: Result) -> None:
        self.connection.execute(
            """
            INSERT INTO results (id, run_id, case_id, repeat_index, output_text,
                                 output_json, latency_ms, input_tokens,
                                 output_tokens, cost_usd, error, error_frames)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.id,
                run_id,
                result.case_id,
                result.repeat_index,
                _text(result.output),
                _json(result.output),
                result.latency_ms,
                result.input_tokens,
                result.output_tokens,
                result.cost_usd,
                result.error,
                _json(result.error_frames) if result.error_frames else None,
            ),
        )
        for score in result.scores:
            self._insert_score(result.id, score)
        for trace in result.traces:
            self._insert_trace(result.id, trace)

    def _insert_score(self, result_id: str, score: Score) -> None:
        self.connection.execute(
            """
            INSERT INTO scores (result_id, scorer_name, value_float, passed,
                                error, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                score.scorer_name,
                score.value,
                # None stays None. Coercing an absent verdict to 0 would make a
                # scorer that declined to judge indistinguishable from one that
                # failed the case.
                None if score.passed is None else int(score.passed),
                score.error,
                _json(score.metadata) if score.metadata else None,
            ),
        )

    def _insert_trace(self, result_id: str, trace: Trace) -> None:
        self.connection.execute(
            """
            INSERT INTO traces (id, result_id, parent_id, name, started_at,
                                duration_ms, input_json, output_json, model,
                                tokens_json, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.id,
                result_id,
                trace.parent_id,
                trace.name,
                trace.started_at.isoformat() if trace.started_at else None,
                trace.duration_ms,
                _json(trace.input),
                _json(trace.output),
                trace.model,
                _json({"input": trace.input_tokens, "output": trace.output_tokens}),
                trace.cost_usd,
            ),
        )

    def _insert_case_snapshot(self, run_id: str, case: Case) -> None:
        self.connection.execute(
            """
            INSERT INTO case_snapshots (run_id, case_id, content_hash, input_json,
                                        expected_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                case.id,
                case.content_hash(),
                _json(case.input),
                _json(case.expected),
                _json(case.metadata) if case.metadata else None,
            ),
        )

    # ------------------------------------------------------------------- reads

    def run_count(self) -> int:
        with self._lock:
            return int(self.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def batch_ids(self) -> list[str]:
        with self._lock:
            return [
                row[0]
                for row in self.connection.execute("SELECT id FROM batches ORDER BY started_at")
            ]

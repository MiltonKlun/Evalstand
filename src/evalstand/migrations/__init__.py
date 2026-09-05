"""Sequential migrations for the evalstand store.

Each migration is a numbered list of SQL statements. They are applied in order,
each inside its own transaction, and the schema version is bumped in that same
transaction — so a migration either lands completely or not at all. A database
left half-migrated would be worse than one left old: the code would believe a
column exists that does not.

**Statements are executed one at a time, never through `executescript`.**
`sqlite3.Connection.executescript` issues an implicit COMMIT before it runs,
which silently ends the transaction wrapping the migration. A failure after that
point cannot be rolled back — verified directly: a script run inside `BEGIN`
left its tables behind and the subsequent ROLLBACK raised "cannot rollback - no
transaction is active".

**Forward only.** There is no downgrade path. A database whose version exceeds
what the installed code understands is refused rather than opened best-effort,
because a silently degraded read would corrupt exactly the comparison data the
tool exists to provide.
"""

from __future__ import annotations

__all__ = ["LATEST_VERSION", "MIGRATIONS"]

_V1 = [
    # A Batch is one invocation; a Run is one Eval within it. Separating them is
    # what lets a watch-mode re-run of three Evals be grouped as one event
    # rather than appearing as three unrelated executions.
    """
    CREATE TABLE batches (
        id          TEXT PRIMARY KEY,
        kind        TEXT NOT NULL,
        status      TEXT NOT NULL,
        started_at  TEXT,
        finished_at TEXT,
        git_sha     TEXT,
        -- Nullable on purpose: NULL means the tree was never checked, which is
        -- a different claim from "checked and clean". A NOT NULL DEFAULT 0
        -- would assert a provenance fact nobody established.
        git_dirty   INTEGER
    )
    """,
    """
    CREATE TABLE runs (
        id                TEXT PRIMARY KEY,
        batch_id          TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
        name              TEXT NOT NULL,
        filepath          TEXT NOT NULL,
        started_at        TEXT,
        finished_at       TEXT,
        model_config_json TEXT,
        task_source_hash  TEXT,
        repeat_n          INTEGER NOT NULL DEFAULT 1,
        total_cases       INTEGER NOT NULL DEFAULT 0,
        total_cost_usd    REAL,
        status            TEXT NOT NULL
    )
    """,
    # `output_text` and `output_json` are separate because a task may return a
    # string or a structure, and flattening a dict into text would lose the
    # ability to compare it field by field later.
    """
    CREATE TABLE results (
        -- Scoped by run. A Result's own id is `{eval}-{case}-{repeat}`, which
        -- is stable *within* a run and therefore identical across runs of the
        -- same eval -- so a bare PRIMARY KEY on it would silently reject every
        -- run after the first and leave history holding one entry forever.
        id            TEXT NOT NULL,
        run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        case_id       TEXT NOT NULL,
        repeat_index  INTEGER NOT NULL DEFAULT 0,
        output_text   TEXT,
        output_json   TEXT,
        latency_ms    INTEGER,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        cost_usd      REAL,
        error         TEXT,
        error_frames  TEXT,
        PRIMARY KEY (run_id, id),
        UNIQUE (run_id, case_id, repeat_index)
    )
    """,
    # `passed` is nullable and set only by scorers that genuinely know pass from
    # fail. Nothing derives it from a threshold. `value_float` is nullable too:
    # an errored Score has no value, and storing 0.0 would make an
    # infrastructure failure indistinguishable from a task that did badly.
    """
    CREATE TABLE scores (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        -- Both columns, because a Result is identified by (run_id, id): its
        -- own id repeats across runs of the same eval.
        run_id        TEXT NOT NULL,
        result_id     TEXT NOT NULL,
        scorer_name   TEXT NOT NULL,
        value_float   REAL,
        passed        INTEGER,
        error         TEXT,
        metadata_json TEXT,
        FOREIGN KEY (run_id, result_id) REFERENCES results(run_id, id) ON DELETE CASCADE
    )
    """,
    # `parent_id` is what makes a Result carry a tree rather than a list. It is
    # deliberately NOT a foreign key to traces(id): rows are inserted in
    # collection order, and a child may precede its parent.
    """
    CREATE TABLE traces (
        id          TEXT NOT NULL,
        run_id      TEXT NOT NULL,
        result_id   TEXT NOT NULL,
        parent_id   TEXT,
        name        TEXT NOT NULL,
        started_at  TEXT,
        duration_ms INTEGER,
        input_json  TEXT,
        output_json TEXT,
        model       TEXT,
        tokens_json TEXT,
        cost_usd    REAL,
        PRIMARY KEY (run_id, id),
        FOREIGN KEY (run_id, result_id) REFERENCES results(run_id, id) ON DELETE CASCADE
    )
    """,
    # Cases are snapshotted per run so history stays valid when the dataset
    # changes. Without this, editing a Case's expected value would silently
    # rewrite the meaning of every past run that used it.
    #
    # `content_hash` covers input as well as expected: an edited input under the
    # same case_id is no longer the same test either. Storing the hash makes
    # Amended Case detection an integer comparison rather than a JSON parse per
    # case.
    """
    CREATE TABLE case_snapshots (
        run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        case_id       TEXT NOT NULL,
        content_hash  TEXT NOT NULL,
        input_json    TEXT,
        expected_json TEXT,
        metadata_json TEXT,
        PRIMARY KEY (run_id, case_id)
    )
    """,
    # History and comparison both filter by eval name and order by time, so the
    # index matches the only query shape that matters.
    "CREATE INDEX idx_runs_name_started ON runs(name, started_at DESC)",
    "CREATE INDEX idx_runs_batch ON runs(batch_id)",
    "CREATE INDEX idx_results_run ON results(run_id)",
    "CREATE INDEX idx_scores_result ON scores(run_id, result_id)",
    "CREATE INDEX idx_traces_result ON traces(run_id, result_id)",
]

_V2 = [
    # Declaration order, recorded because it cannot be recovered afterwards.
    #
    # `runner.py` promises results come back in the order their cases were
    # declared: "a report whose rows shuffle between runs cannot be read or
    # diffed". Reading them back `ORDER BY case_id` broke that promise as soon
    # as a suite had ten cases, because text ordering puts q10 before q2.
    #
    # Nullable so the migration needs no backfill: rows written before this
    # column existed sort last, in their old order, rather than being
    # rewritten to a position nobody recorded.
    "ALTER TABLE results ADD COLUMN ordinal INTEGER",
]

MIGRATIONS: dict[int, list[str]] = {1: _V1, 2: _V2}
"""Version number to the statements that take the schema *to* that version.

Applied in ascending order. A migration is never edited once released — a
database already carrying that version would never re-run it, so the change
would apply only to new databases and the two would diverge silently. Add a new
numbered migration instead.
"""

LATEST_VERSION = max(MIGRATIONS)

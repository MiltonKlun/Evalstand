"""The response cache.

A store of previous model responses, keyed on the exact call, that exists purely
to avoid paying twice for the same request. It is **always safe to delete** and
never load-bearing for correctness: cache damage degrades to a miss, never to an
error, and a missing cache costs money rather than breaking a run.

This is not the mechanism that makes tests deterministic — cassettes are. The
two are kept apart deliberately: deleting a cassette breaks the suite, deleting
the cache only costs money.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from evalstand import __version__
from evalstand.llm import LLMResponse, cache_key

__all__ = ["ResponseCache", "cache_key"]  # cache_key re-exported from llm

logger = logging.getLogger("evalstand.cache")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key               TEXT PRIMARY KEY,
    model             TEXT NOT NULL,
    evalstand_version TEXT NOT NULL,
    response_json     TEXT NOT NULL,
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    hit_count         INTEGER NOT NULL DEFAULT 0
);
"""


class ResponseCache:
    """A SQLite-backed store of model responses."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(_SCHEMA)
        self.connection.commit()

    def get(self, key: str, *, bypass: bool = False) -> LLMResponse | None:
        """Return a cached response, or None.

        A miss is the normal, safe outcome: an entry written by a different
        version is ignored rather than trusted, and a damaged entry is treated
        as absent rather than raising into the caller's run.
        """
        if bypass:
            return None

        row = self.connection.execute(
            "SELECT response_json, evalstand_version FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None

        if row["evalstand_version"] != __version__:
            # Key canonicalisation may have changed between versions; a
            # mismatch is not a hit, it is a stale entry.
            logger.debug("ignoring cache entry from version %s", row["evalstand_version"])
            return None

        try:
            response = LLMResponse.model_validate_json(row["response_json"])
        except Exception as exc:  # a damaged entry is a miss, never a crash
            logger.debug("discarding unreadable cache entry: %s", exc)
            return None

        self.connection.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?", (key,))
        self.connection.commit()
        return response

    def set(self, key: str, model: str, response: LLMResponse, *, bypass: bool = False) -> None:
        """Store a response.

        `bypass` runs in both directions. A response produced under conditions
        that made it non-reusable — a repeat, say — must not later be served as
        *the* answer for its key.
        """
        if bypass:
            return

        self.connection.execute(
            """
            INSERT INTO cache (key, model, evalstand_version, response_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                model = excluded.model,
                evalstand_version = excluded.evalstand_version,
                response_json = excluded.response_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (key, model, __version__, response.model_dump_json()),
        )
        self.connection.commit()

    def hit_count(self, key: str) -> int:
        row = self.connection.execute(
            "SELECT hit_count FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return int(row["hit_count"]) if row else 0

    def entry_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS n FROM cache").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        self.connection.execute("DELETE FROM cache")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

# 0005 — The database is project-local

- **Status:** accepted
- **Date:** 2026-09-01

Run history lives in a project-local SQLite database at `.evalstand/`, which is
gitignored, rather than in a user-global cache directory.

History is only meaningful relative to a codebase: comparing Runs across two
unrelated projects is nonsense, and a global database makes `history` ambiguous
for anyone working on more than one. Project-local storage also keeps a Run's
provenance — the git SHA recorded in Phase 5.2 — aligned with the repository the
database sits in.

## Consequences

History is per-machine. It does not survive a fresh clone, and it does not
transfer between developers. This means CI has no prior Runs to compare against
unless it is explicitly given a database, so `--threshold` (an absolute bar) is
the CI gate, not `compare` (a relative one). `docs/ci.md` must say this plainly.

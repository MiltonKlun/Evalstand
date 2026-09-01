# 0004 — `evaluate()` registers, it does not execute

- **Status:** accepted
- **Date:** 2026-09-01

`evaluate()` is called at module top level, which makes it look like it should
run the eval right there. It does not: it records an Eval into a module-level
registry and returns immediately, performing no I/O, starting no event loop, and
making no model calls. Execution happens later, when a runner reads the registry.

Deferring is what makes the rest of the design possible. `pytest -k q1` must
select a single Case, which requires Cases to exist as collected items before
anything executes. Watch mode must know which Evals a changed file declares in
order to decide what to re-run — it cannot answer that by running them. And a
bare `import qa_eval` must not fire billable model calls.

## Consequences

Importing an eval file is side-effect-free and cheap, so any component may
enumerate Evals without executing them. The cost is one level of indirection:
the thing a user writes at module scope is a declaration, not a call, and the
docs must be explicit about that or the deferral will surprise people.

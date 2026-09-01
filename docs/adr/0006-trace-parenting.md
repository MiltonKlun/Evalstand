# 0006 — Trace parenting via a reset ContextVar

- **Status:** accepted
- **Date:** 2026-09-01

`evalstand` records Traces as a tree, which the implementation that inspired it
does not — its traces are a flat list. The parent of a node is whatever scope is
open when the call starts, tracked by a `contextvars.ContextVar` that is set on
entry and reset via its token in a `finally`.

The alternatives both fail under concurrency. Inferring parentage from the async
call stack is fragile and implementation-specific. A mutable stack of open nodes
breaks when a call raises: the missing pop leaves the next sibling parented to a
node that already closed. A token reset in `finally` survives both, and because
`contextvars` copies context per task, calls fired under `asyncio.gather`
naturally become siblings with no special handling.

## Consequences

When a parent cannot be determined confidently, the node attaches to the Result
root instead of guessing. An orphan at the top level is visibly incomplete; a
wrongly-parented node is a plausible-looking lie, and a trace tree that quietly
misrepresents which call invoked which is worse than no tree. If nesting proves
unreliable under load, the fallback is a flat list — not a best-effort tree.

This is the behaviour to test first, before the showcase example exists:
`contextvars` under `asyncio.gather` is where this design either holds or fails.

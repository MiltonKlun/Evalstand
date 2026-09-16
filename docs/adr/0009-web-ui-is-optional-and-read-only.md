# 0009 — The web UI is optional, and read-only over history

- **Status:** accepted
- **Date:** 2026-09-14

Phase 8 adds a FastAPI backend, an HTMX front end, and `evalstand serve`. Three
decisions shape all of it.

## It is an optional extra

`fastapi`, `uvicorn` and `sse-starlette` live in the `web` extra, not in
`dependencies`. `evalstand` is a local-first CLI and TUI: the primary way to
watch a run is `evalstand watch`, which Phase 6 already ships. A user who never
runs `serve` should not install a web server, an ASGI framework and their
transitive tree to get a test runner.

The cost is that `evalstand serve` fails on a bare install. It fails *loudly*,
naming the extra — an ImportError with installation instructions, not a
traceback — because the alternative is making every user pay for a feature the
plan itself calls an optional stretch.

## It reads history; it does not own it

The API serves what `RunStore` already holds. It does not compute a pass rate, a
mean, or a total of its own: those come from `reporting.console.pass_counts`
and the runner's own totals, the same functions the terminal and the markdown
reporter call.

This is the rule that matters most here. A second implementation of "which
Results count towards a pass rate" would eventually disagree with the first, and
a web page contradicting the terminal about whether a build passed is worse than
either number alone. The same reasoning made `pass_counts` public in Phase 7.

## Missing data stays missing

`None` is serialised as `null`, never as `0`. A JSON consumer that reads
`"cost_usd": 0` will sum it into a total that understates real spend while
looking authoritative. The terminal renders these as `-`; JSON has a real null
and should use it.

## Consequences

**The front end holds no state.** HTMX swaps server-rendered fragments and SSE
pushes new rows. There is no client-side model to drift from the server's, and
no build step — which keeps the stack entirely Python, as the plan asks.

**A browser disconnecting must not affect a run.** The SSE publisher is a
`ResultSink` like the TUI's, and `_announce` already guarantees an observer that
raises cannot cost the measurement it was observing. A closed browser tab is the
ordinary case, not an error.

**`serve` does not execute by default.** It opens the history database and
serves it. Triggering a run from a web page is a separate, explicit action —
running evals costs money, and a page that spends it on load would be a trap.

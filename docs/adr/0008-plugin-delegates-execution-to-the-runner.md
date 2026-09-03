# 0008 — The plugin collects and reports; the runner executes

- **Status:** accepted
- **Date:** 2026-09-03

The pytest plugin takes over `pytest_runtestloop`. It hands every selected Eval
to `run_eval()` before any item runs, then each item's `runtest` merely reports
the Result the runner already produced. The plugin no longer executes anything.

Phase 2's plugin executed each case inside its own `runtest`. That is the
obvious shape, and it is wrong for this tool: **pytest runs items strictly one
after another.** A per-item `runtest` therefore cannot be concurrent, cannot
abandon one case on a timeout while others continue, and cannot share a cache or
a trace collector across a run. Every capability Phase 3 built was unreachable
from the path users actually run — `pytest` and `evalstand run` both. The
summary printed `-` for cost on runs that really did spend money, because Runs
were rebuilt from item state that never held a trace.

The risk register anticipated this as *"the plugin fights the runner's async
model"* and prescribed falling back to a standalone runner with pytest
compatibility as a stretch. That fallback is not needed. `pytest_runtestloop`
resolves the conflict outright: only one component executes, so there is nothing
left to fight over, and parity item 19 (bare `pytest` collects and runs evals)
is preserved rather than waived.

## Why not take the loop over entirely

The hook returns `None` rather than `True`, so pytest still walks the items
itself. Claiming the loop would mean reimplementing `-x`, `--maxfail`, fixtures,
reporting and every other plugin's expectations — all of which pytest already
does well, and none of which is what this project is for. Ordinary tests and
evals therefore coexist in one session, each behaving normally.

## Consequences

**Selection has to reach the runner.** `run_eval(..., only=...)` filters
`(case_id, repeat_index)` pairs *before* anything executes. Filtering afterwards
would be a bug with a price tag: `-k q1` would send thirty requests and discard
twenty-nine. `--collect-only` executes nothing for the same reason.

**A missing Result must never read as a pass.** An item whose `result` is `None`
raises rather than reporting success — a green tick for a case that never ran is
the most misleading thing an eval tool can do.

**Tracebacks must be captured at the raise.** The runner catches a task's
exception so one bad case cannot end the run, which destroys the traceback long
before pytest reports it. `Result.error_frames` holds the user's own frames,
formatted at capture: storing the traceback object itself would pin every
frame's locals in memory and could carry an API key into the database. Frames
from evalstand, asyncio and `concurrent.futures` are stripped, since `to_thread`
puts a worker frame on top of every sync task's stack.

**Evals run sequentially with respect to each other.** `--concurrency` is a
promise about how many calls are in flight; running four evals at once would
quietly multiply it by four and trip the rate limit the flag exists to avoid.

## A testing note worth recording

These tests use `runpytest_subprocess`, not the default in-process run. Once
litellm has been imported inside a pytester inline session,
`importlib.metadata.entry_points()` returns nothing for the rest of the process
— so the next inline run cannot register pytest-cov and dies with
`unrecognized arguments: --no-cov`. Since the runner imports litellm
transitively, every inline test here would poison the ones after it. The
subprocess is also the more honest check: it is what a user's shell does.

This is a pytester artifact, not a defect in the design. Real runs — `pytest`
and `evalstand run` — are unaffected, which was verified directly before the
integration was accepted.

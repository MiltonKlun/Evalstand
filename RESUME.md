# Where this project stands

Written 2026-09-20, updated 2026-09-21. The Jev detour is finished and its
conclusion lives in `JEV.md` (short version: don't build).

## The one-line answer

Phases 0–8 are functionally complete. Every remaining checkbox in `PLAN.md` is
blocked on steps outside the code. Nothing is half-finished and nothing is
uncommitted.

## Verified state at `a115d18`

| | |
| --- | --- |
| Commit | `a115d18`, local and `origin/main` identical, working tree clean |
| Tests | 1588 passing, 8 deselected (the `live` marker) |
| Mutants | 279 in `scripts/mutate.py`, all killed, **zero stale anchors** |
| Gate | `ruff check` · `ruff format --check` · `mypy` · `mkdocs build --strict` all clean over the whole tree |
| CI | green on Python 3.11, 3.12, 3.13 |
| Docs | live at <https://miltonklun.github.io/Evalstand/> |
| Coverage | 95% overall; `web/` modules 95–100% |

## The three gates

These are the *only* open items. Each needs something I cannot supply.

1. **5.7 — the showcase baseline.** Needs an `OPENAI_API_KEY` and roughly 60
   model calls. `examples/pdf_extraction/baseline.py` produces the numbers from
   a stored run; `BASELINE.md` currently says the baseline is unrecorded rather
   than carrying invented figures.

2. **7.6 — publish to PyPI.** Needs a trusted publisher configured at pypi.org
   (project `evalstand`, workflow `release.yml`, environment `pypi`) and the
   version bumped off `0.0.0.dev0` in **both** `pyproject.toml` and
   `src/evalstand/__init__.py` — a test now fails if only one moves. Then tag
   `v1.0.0` and `release.yml` does the rest.

3. **6.8 / 7.7 — the demo GIF and the 3-minute video.** Need `vhs`
   (`winget install charmbracelet.vhs`) or `asciinema` + `agg` installed.
   `examples/demo/` already holds an offline eval and a VHS tape ready to
   record.

## The one unsolved problem — and a disproved theory

**The full suite once failed intermittently with `collected 0 items`** from a
nested `evalstand run`, naming a different test each time under fixed order.

### The theory I had, and why it is wrong

I repeatedly blamed ADR 0008's `entry_points()` poisoning: the suite spawns 55
in-process `pytest.main()` sessions, and importing litellm inside one can leave
`importlib.metadata.entry_points()` empty for the rest of the process.

**Tested directly on 2026-09-21 and disproved.** Five consecutive nested
`pytest.main()` runs in one process: `entry_points(pytest11)` stayed at 3
throughout, and all five collected and passed. The poisoning is real for
*pytester* sessions — which is what ADR 0008 actually documents, and why those
tests use `runpytest_subprocess` — but it does not happen for `pytest.main`
from a plain process. Do not inherit this theory.

### What the evidence actually supports

Every observed failure was on a working tree **before `c0b0e14`**, the commit
that removed the stale `importorskip("faker")` guard and un-skipped 33 PDF
tests. Since that commit, six consecutive full-suite runs have been clean
(1588 passed, zero collection errors), plus three clean isolated runs of
`test_cli.py`.

That is correlation across six runs, not proof. It is possible the defect was
tied to the tree state at the time, and it is possible it is merely dormant.

### If it reappears

Capture the **untruncated collection error** first — every earlier attempt
truncated it, which is why the real cause was never seen. `pytest --tb=long -rA`
and read the `errors` section, rather than the list of which tests failed. The
failing test names shift between runs and are not the signal.

## What Phase 8 shipped, for someone picking this up cold

- **8.1** `src/evalstand/web/api.py` — read-only JSON over `RunStore`. Computes
  no score, mean or pass count of its own; everything comes from
  `reporting.console.pass_counts` and the `Run` model's properties, so the API
  cannot drift from what the terminal prints.
- **8.2** `fragments.py`, `live.py`, `page.py` — server-rendered HTMX fragments
  and an SSE broker. htmx is vendored at `web/static/htmx.min.js` with its
  SHA-256 asserted by the suite.
- **8.4** `evalstand serve` in `cli.py` — reads history, runs nothing, binds
  localhost.
- **ADR 0009** records why the web UI is optional, read-only, and why `None`
  stays `null` in JSON.

## Conventions worth knowing before changing anything

- **Never run two `uv` commands against this repo at once**, and never run tests
  or edit `src/` while `scripts/mutate.py` is running — it rewrites source files
  and restores them in a `finally`. Concurrent runs produce phantom failures.
- **Verify CI checks over the whole tree**, not a subset. `ruff format --check .`
  covers everything; checking `src tests scripts` has twice let CI go red.
- **Prove a fix with a mutant that reintroduces the defect** — and confirm the
  mutant actually *applied* before believing the result. A mutation that fails
  to match its anchor is indistinguishable from a test that works; this has
  produced two false "the test doesn't catch it" readings.
- Plugin tests must use `runpytest_subprocess`, never the inline runner.
- No time estimates or deadline framing anywhere in this project.

## Resuming

Nothing to un-wind. `git status` should be clean at `a115d18`. Pick up by
clearing one of the three gates above. The intermittent collection failure is
not currently reproducible and its only hypothesis has been disproved, so it is
a watch item rather than a task.

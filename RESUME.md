# Where this project stands

Written 2026-09-20, before a detour to evaluate TypeSafe's Jev model. Delete
this file when the detour ends and the work below resumes.

## The one-line answer

Phases 0–8 are functionally complete. Every remaining checkbox in `PLAN.md` is
blocked on steps outside the code. Nothing is half-finished and nothing is
uncommitted.

## Verified state at `a626b07`

| | |
| --- | --- |
| Commit | `a626b07`, local and `origin/main` identical, working tree clean |
| Tests | 1587 passing, 8 deselected (the `live` marker) |
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
   `src/evalstand/__init__.py`. Then tag `v1.0.0` and `release.yml` does the
   rest. I can drive the tag and verify the workflow once the publisher exists.

3. **6.8 / 7.7 — the demo GIF and the 3-minute video.** Need `vhs`
   (`winget install charmbracelet.vhs`) or `asciinema` + `agg` installed.
   `examples/demo/` already holds an offline eval and a VHS tape ready to
   record.

## The one unsolved problem

**The full suite fails intermittently with `collected 0 items`** from a nested
`evalstand run`, naming a different test each time under fixed order. It did
not appear in any of the last four full runs, and `tests/unit` alone is
green 1477/1477.

I never found the cause. The likely mechanism: the suite spawns **46 in-process
`pytest.main()` sessions**, and ADR 0008 records how importing litellm inside
one can leave `importlib.metadata.entry_points()` returning nothing for the rest
of the process — which would make a later nested session register no plugin and
collect nothing. That predicts both the intermittency and the shifting failure
set.

This is not a Phase 8 regression: it predates the web UI. If it appears during a
release run, look there first rather than treating it as new. Fixing it properly
means reconsidering the nested-`pytest.main` design, which is its own piece of
work and probably deserves an ADR.

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

Nothing to un-wind. `git status` should be clean at `a626b07`. Pick up either by
clearing one of the three gates above, or by investigating the intermittent
collection failure.

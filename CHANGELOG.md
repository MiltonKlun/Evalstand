# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] — 2026-09-23

**Upgrade if you use `evalstand watch`.** In 1.0.0, watch mode re-ran the code
it started with rather than your edit: the status line said "changed", the rows
re-landed, and the old code ran. Every result after an edit was stale. This
release fixes that and the other defects found alongside it — edits to a task
file were never seen, latency was never recorded, and sub-cent costs printed
as `$0.0000` — all listed under *Fixed* below.

### Added
- Task 5.7: the showcase baseline, recorded. `BASELINE.md` now describes one
  real run — span selection with TypeSafe's `jev-1.13.0` over the 30-invoice
  corpus: 180/180 scalar fields, $0.0015. It covers the scalar fields only and
  says so beside the table; the generative eval's baseline still awaits a key.
- `examples/pdf_extraction/jev_extraction_eval.py`, the span-selection method as
  a recorded eval. Jev calls are traced with their exact cost via `record_call`.
- `baseline.py` reads the model from the run's own traces and names the eval
  file it describes. Both used to be hard-coded for the generative eval.

- Task 6.8: the demo GIF, `docs/demo.gif`, at the top of the README. Recorded
  from `examples/demo/`, offline; the live view's start-up is cut and the README
  says so.

### Fixed
- **Watch mode re-ran the code it started with, not the edit.** A file change
  triggered a re-run of the eval imported when the session began, so the status
  said "changed", the rows re-landed, and the old code ran. `r` had the same
  flaw. Both now reload the eval from disk; a file that no longer imports is
  reported and nothing is run.
- An edit to a **task file** the eval imports was never seen: the import was
  answered from Python's module cache. Modules under the watched folders are now
  forgotten before a reload, and their cached bytecode deleted.
- The loader could run a same-length edit's **stale bytecode**: Python accepts a
  `.pyc` whose source matches in size and whole-second mtime. Eval files are now
  compiled from source on every load.
- **Per-case latency was never recorded**, so every latency column showed `-`.
  The runner now times the task alone — not queueing, not scoring.
- The "changed: <file>" indicator vanished as soon as a fast re-run finished; it
  now stays on the finished run's status line.
- A priced cost under half a hundredth of a cent printed as `$0.0000` — the
  string reserved for "nobody knows", attached to a known, non-zero amount.
  Short calls to cheap models, including ordinary judge calls, were shown as
  free in the trace tree, the HTML report, the TUI and the web UI. Nine
  renderers each rounded money themselves; all now go through `format_usd`.

## [1.0.0] — 2026-09-21

First published release. Every phase of the plan is complete: the authoring
API and pytest collection, the concurrent runner with nested trace trees, ten
scorers, SQLite history with `history`/`show`/`compare`, the Textual TUI and
watch mode, CI integration with documented exit codes, and an optional web UI.

1588 tests, 279 mutants, green on Python 3.11, 3.12 and 3.13.

Two things this release deliberately does **not** claim: score deltas carry no
significance testing, and the LLM judge scorers are uncalibrated against human
labels. Both are stated in the README's Limitations section rather than left
for a user to discover.

### Added
- Phase 8 complete: the optional web UI. `evalstand serve` opens a browser view
  of run history, behind a `web` extra so a default install does not pull in a
  web server. A read-only JSON API, an HTMX front end with no build step, and a
  live table fed by server-sent events. The page computes no score, mean or
  pass count of its own — it renders through the same helpers the terminal uses,
  so it cannot drift from `evalstand show` (ADR 0009). 1587 tests.
- Task 8.3: `--html PATH` writes a self-contained HTML report for a CI artifact.
  One file, no external references, no JavaScript — so it renders in the
  sandboxed iframe CI systems serve artifacts from and survives being emailed.
  Unlike the terminal summary and the PR comment it shows everything: full
  outputs and whole trace trees, with failing cases expanded and passes
  collapsed.
- Phase 7 complete: CI integration, docs, and the release path. `--fail-on-error`
  and a documented exit-code contract (`0` met the bar, `1` below `--threshold`,
  `2` something did not run, with `2` outranking `1`); `--output markdown` for a
  pull-request comment; three copyable GitHub Actions recipes; an mkdocs-material
  site built strictly and deployed to Pages; a rewritten README with a
  Limitations section; and a tag-driven `release.yml` using trusted publishing.
  1404 tests, 243 mutants.
- Phase 6 complete: the TUI and watch mode. A live run view whose rows land as
  each case finishes, a case detail pane with the trace tree, a history screen
  that compares any two runs, custom columns via `evaluate(columns=)`, and watch
  mode that re-runs on a file change and records the cancelled batch honestly.
- Phase 5 complete: storage, history, and the showcase example. SQLite with
  migrations, provenance on every batch, `history`, `show` and `compare`, and an
  invoice-extraction example whose 30-PDF corpus regenerates byte-identically
  from a fixed seed. 1089 tests.
- Phase 4 complete: the scorer library. Ten scorers — `exact`,
  `normalised_exact`, `contains`, `regex_match`, `levenshtein`, `ratio`,
  `close_to`, `json_fields`, `judge`, `factuality` — plus the `@scorer`
  decorator and signature adaptation. 829 tests.
- Phase 3 complete: the runner, tracing, repeats, and streaming. Concurrent
  execution behind a semaphore, nested trace trees with per-node cost, repeats
  that bypass the cache in both directions, and streamed chunks reaching a live
  view as they arrive. 367 tests.
- Phase 2 complete: the authoring API and pytest collection. `evaluate()`,
  the pytest plugin, console reporting, a minimal `evalstand run`, and the toy
  example. 263 tests, 98% coverage.
- Phase 1 complete: the model-call layer. `models.py` (Case, Score, Trace,
  Result, Run, Batch), `llm.py` (calls, streaming, retry, cost), `cache.py`
  (SQLite response cache), `cassettes.py` (record/replay). 176 tests, 99%
  coverage, no API keys required.
- ADR 0006 (trace parenting), 0007 (LiteLLM over provider SDKs).
- Phase 0 complete: repository scaffold, tooling configuration, CI workflow,
  ADR process.
- `CONTEXT.md`: the project's domain model, 29 terms.
- ADR 0002 (design scope), 0003 (no variants in v1), 0004 (`evaluate()`
  registers rather than executes), 0005 (project-local database), 0006 (trace
  parenting).

### Changed
- `compare` exits `2` rather than `1` when it has nothing to compare — an absent
  run, or one whose batch never finished. A CI job seeing `1` could not tell
  "these runs differ badly" from "there was no measurement here".
- `evalstand.trace` is the working implementation. It had been the Phase 2
  placeholder that raised `NotImplementedError` since Phase 3: every internal
  caller reached past the export to `evalstand.tracing.trace`, so nothing
  noticed for four phases.
- `reporting.console` exports `UNKNOWN`, `pass_counts`, `format_score` and
  `format_cost`, so the markdown reporter and the TUI share one set of honesty
  rules rather than three copies that drift.
- Results are stored and replayed in declaration order. An `ordinal` column
  replaces the previous `ORDER BY case_id`, which returned `q10` before `q2`.
- Task 3.3 corrected: repeats now bypass the cache unconditionally. Its previous
  rule — bypass only when temperature > 0 — could not be implemented, because a
  Task makes its own model calls and the runner cannot inspect their parameters.
- Schema gains `batches`, `case_snapshots.content_hash`, `scores.error`,
  `traces.parent_id`, and `cache.evalstand_version`.
- The capability table records three rows as going beyond the reference
  implementation rather than matching it: nested traces, response caching, and
  stable case identity.

### Fixed
- A stale `pytest.importorskip("faker")` skipped the entire PDF example suite —
  33 tests — on every platform and every Python version, including CI. The
  dependency had been removed four commits earlier and the guard outlived it;
  `importorskip` on a package nothing installs is an unconditional skip. A test
  now fails any guard naming a package no dependency, extra or group declares.
- The missing-`web`-extra message told the user to `pip install 'evalstand'`:
  Rich read `[web]` as a style tag and deleted it. The one string that had to
  survive verbatim was the one being rewritten.
- `compare` printed "nothing differs between these runs" about a batch that had
  been cancelled part-way. Its membership note fires on case coverage, so a batch
  interrupted after every case had scored slipped past it entirely.
- `watch --store` wrote a batch row with no runs beneath it: the recorder was
  opened, cancelled and read from, but `record()` was never called.
- A task returning `None` scored 1.0 against a Case expecting the word "None",
  because `str(None)` is `"None"`.
- Punctuation folding made `-5` equal `5`, `$100` equal `100%`, and `C++` equal
  `C`.
- The mutation harness reported a stale anchor as a survivor, having tested
  nothing for four tasks. Stale anchors now fail the run and are reported apart
  from genuine gaps.

## [0.0.0.dev0] — 2026-09-01

Name-reservation release. Contains no working code.

### Added
- Name chosen and reserved on PyPI: `evalstand` (ADR 0001).
- Package skeleton, MIT licence, build configuration.

# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

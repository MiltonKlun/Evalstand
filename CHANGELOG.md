# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Task 3.3 corrected: repeats now bypass the cache unconditionally. Its previous
  rule — bypass only when temperature > 0 — could not be implemented, because a
  Task makes its own model calls and the runner cannot inspect their parameters.
- Schema gains `batches`, `case_snapshots.content_hash`, `scores.error`,
  `traces.parent_id`, and `cache.evalstand_version`.
- The capability table records three rows as going beyond the reference
  implementation rather than matching it: nested traces, response caching, and
  stable case identity.

## [0.0.0.dev0] — 2026-09-01

Name-reservation release. Contains no working code.

### Added
- Name chosen and reserved on PyPI: `evalstand` (ADR 0001).
- Package skeleton, MIT licence, build configuration.

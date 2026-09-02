# 0002 — Design scope

- **Status:** accepted
- **Date:** 2026-09-01

`evalstand` is an independent Python implementation of an idea proven in
TypeScript by `evalite` (MIT): that local LLM evaluation should feel like
running a test suite. This ADR records what that means concretely — which
behaviours we adopt, which we do differently, and which we add — after reading
the reference implementation. No code was copied; it was read to understand
behaviour and interface decisions.

## Adopted

The shape of the authoring experience, because it is the thing worth having:

- **A single entry function** taking cases, a task, and scorers, declared at
  module scope in a file the test runner collects automatically.
- **Collection by the ecosystem's test runner** rather than a bespoke one. The
  reference builds on Vitest (`describe`/`it` in `evalite.ts`); we build on
  pytest, which is the equivalent and gives bare-`pytest` compatibility for
  free.
- **Scores as floats in `[0, 1]`**, with scorers returning either a bare number
  or a structure carrying metadata (`create-scorer.ts`).
- **Repeats** for non-deterministic evaluation — its `trialCount`, our
  `--repeat N`.
- **Traces** captured automatically from calls made inside a task, associated
  with the running case through per-task context storage. It uses
  `AsyncLocalStorage` (`traces.ts`); `contextvars` is the direct Python
  analogue, which validates the approach.
- **SQLite persistence** of runs, results, scores, and traces.
- **A batch/eval split in storage.** Its `runs` table is the invocation, tagged
  `full` or `partial`, with an `evals` table hanging off it. We keep the split —
  watch mode needs it to group a partial re-run — but name the levels `Batch`
  and `Run`.

## Done differently

- **A Textual TUI instead of a React web UI.** The TUI reaches functional parity
  faster, keeps the whole stack in Python, and demos well in a terminal
  recording. A web UI is a stretch phase.
- **`evaluate()` registers rather than executes.** The reference calls Vitest's
  `describe`/`it` at import time, so importing an eval file runs it. Ours
  records into a registry and returns; a runner executes later. See ADR 0004 —
  this is what makes `-k` selection, watch-mode planning, and side-effect-free
  imports possible.
- **One score scale, not two.** Its scores are `[0, 1]` while `scoreThreshold`
  is `0-100` (`command.ts`). We use `[0, 1]` everywhere; a unit mismatch between
  the value and the gate that judges it is an easy mistake to make and a
  confusing one to debug.
- **A missing score is excluded from the mean, not counted as zero.** The
  reference averages with `score.score ?? 0` (`reporter.ts:229`). Treating an
  absent or errored score as a zero silently conflates "the scorer broke" with
  "the task did badly", which corrupts both the mean and any gate reading it.
- **No variants.** See ADR 0003.

## Added

Three capabilities the reference does not have. They are marked **beyond** in
the plan's capability table and stated in the README, because the honest framing
is stronger than implying parity:

- **Stable case identity.** The reference identifies results positionally, by
  `col_order` — there is no case ID in its schema. Inserting a case at the top
  of a dataset therefore re-pairs every later case with the wrong history. Our
  `Case.id` is durable, which is what makes `history` and `compare`
  trustworthy at all.
- **Nested traces.** Its `traces` table has no `parent_id` and its `Trace` type
  has no children — traces are a flat list ordered for display. Ours form a
  tree, so a judge scorer's call appears beneath the task call it judges. See
  ADR 0006 for the parenting rule and its fallback.
- **Model response caching.** The reference has none; "cache" there is only the
  directory its SQLite file lives in. Ours is a real response cache, kept
  strictly separate from the cassettes that make tests deterministic — the cache
  is disposable, the cassettes are not.

## Consequences

Two of the three additions — nested traces and durable identity — are also the
two the risk register names as deciding whether this is a real tool or a
wrapper. They are not incidental extras; they are the reason the port is worth
doing rather than reaching for the TypeScript original.

Being genuinely independent also means the reference cannot be consulted as an
oracle when behaviour is ambiguous. Where it is silent or wrong for Python, the
decision is ours and belongs in an ADR.

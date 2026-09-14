# Architecture

How `evalstand` is put together, and why the pieces are divided where they are.

This page is for someone changing the code, or deciding whether the design will
hold up under a change they have in mind. If you only want to *use* the tool,
[Quickstart](quickstart.md) and [Writing evals](writing-evals.md) are the pages
you want.

## The shape of a run

Five stages, in order. Each one hands off a value and stops caring what happens
next.

```
authoring        registration        execution         scoring        reporting
──────────       ────────────        ─────────         ───────        ─────────
evaluate()  ──►  Registry      ──►   run_eval()   ──►  Scorer    ──►  console
in *_eval.py     (in api.py)         (runner.py)       protocol       markdown
                                          │                           html
                                          ├── llm.py ──► cache.py     TUI
                                          └── tracing.py              storage
```

The two entry points — `pytest` and `evalstand run` — join this pipeline at the
same place. Neither has an execution path of its own, which is the point of
[ADR 0008](adr/0008-plugin-delegates-execution-to-the-runner.md).

## The vocabulary

Four words with four distinct meanings, defined once in `models.py` and not
treated as synonyms anywhere:

| Term | What it is |
| --- | --- |
| **Case** | One input, and what a correct answer looks like. Written by the user. |
| **Result** | What happened when one Case met the Task **once**. |
| **Run** | Every Result for one Eval, plus its totals and provenance. |
| **Batch** | Every Run from one invocation. |

The distinction that matters most in practice: with `--repeat 3`, one Case
produces three Results. Any design that collapses those into a single
pass/fail has to invent an aggregation rule — any, all, majority — and that is
a judgement the user never made.

## Registration is not execution

`evaluate()` is called at module top level, which makes it look like it should
run the eval right there. It does not. It records an `Eval` into a module-level
registry and returns, performing no I/O, starting no event loop, and making no
model calls.

That deferral buys three things that are otherwise unreachable:

- `pytest -k q1` can select a single case, because Cases exist as collected
  items before anything runs.
- Watch mode can ask which Evals a changed file declares without running them.
- `import qa_eval` does not spend money.

The full reasoning is [ADR 0004](adr/0004-evaluate-registers-defers.md).

## The runner owns concurrency

`runner.py` executes an Eval's cases against an `asyncio.Semaphore`, collects
Results, and computes totals. Two rules govern it.

**One bad case never takes down the others.** A raising task, a timeout, a
broken scorer — each is recorded on its own Result and the run continues.
Discarding 29 good measurements because the 30th timed out would throw away
time and money already spent.

**Ordering is by case, not by completion.** Cases finish in whatever order the
provider answers. Results are collected back into declaration order, because a
report whose rows shuffle between runs cannot be read or diffed.

The reason execution lives here rather than in the pytest plugin is that
**pytest runs items strictly one after another.** A per-item `runtest` can never
be concurrent, so every capability in this module would be unreachable through
the path most users actually run.

## Two caches, and why they are not one

This is the distinction most likely to be collapsed by a well-meaning change,
so it is worth stating plainly.

| | `cache.py` | `cassettes.py` |
| --- | --- | --- |
| Purpose | Avoid paying twice for the same call | Pin real provider behaviour for tests |
| Keyed on | Call identity | A named fixture |
| Safe to delete? | **Always** — degrades to a miss | **No** — breaks the suite |
| Cost of losing it | Money | Correctness |

The cache is an optimisation. Cassettes are a correctness fixture: they let
tests run offline against a payload shape that genuinely came back from a
provider, rather than one a test author *believed* a provider returns. That is
not academic — a streamed-cost bug early in the project existed precisely
because a hand-written mock agreed with the wrong assumption.

`--repeat N` with `N > 1` bypasses the cache unconditionally. Repeating a call
to measure variance, and then being served the same cached answer three times,
would report zero variance with total confidence.

## Storage is load-bearing; the cache is not

The same asymmetry again, one layer up. `storage.py` holds the history the tool
exists to provide, so:

- **Migrations are forward-only and atomic**, each step in its own transaction.
- **A database newer than the code is refused**, naming both versions. Opening
  it best-effort would let a newer column go unread and a comparison silently
  answer the wrong question.
- **One transaction per Run.** A crash mid-batch keeps every run that finished.

`recording.py` is the seam between running and storing, and it exists so the two
can fail independently: **a storage failure must never cost a measurement.** By
the time a run is written, the money is already spent, so a full disk degrades
to a warning and a lost history entry — not a lost result.

Provenance is the one thing checked *before* execution. Discovering a run cannot
be recorded after paying for it would be the worst possible ordering, so a dirty
tree without `--allow-dirty` refuses up front.

The database is project-local and gitignored
([ADR 0005](adr/0005-project-local-database.md)): history is only meaningful
relative to a codebase, and CI therefore starts every build with none.

## Traces are a tree

Every model call is captured automatically, and `trace(name)` gives any
operation a node of its own. A `ContextVar` holds the currently open node;
entering a span sets it and keeps the token, leaving resets that token in a
`finally`.

That single rule is what survives both `asyncio.gather` — where contextvars are
copied per task, so siblings share a parent for free — and an exception
mid-call, where the `finally` stops a raising span corrupting the parentage of
the one after it. The alternatives (inferring parentage from the async call
stack, or a mutable stack of open nodes) both break under concurrency.
[ADR 0006](adr/0006-trace-parenting.md) has the comparison.

This is also a deliberate difference from the implementation that inspired the
project, whose traces are a flat list. A tree puts a judge scorer's call
*beneath* the task call it judges rather than beside it.

## Missing data is `None`, never zero

`llm.py` reports `None` when a token count or a price is unavailable.

A zero cost is a *claim that the call was free*. A run total built from such
claims understates real spend while looking authoritative, and the person
reading it has no way to tell. The reporting layer carries this through: an
unmeasured value renders as unknown rather than as `0.00`.

## One provider interface, borrowed

Every model call goes through LiteLLM rather than provider SDKs or a
hand-rolled adapter per vendor. An eval tool is provider-agnostic by nature,
and a user comparing two models across vendors is the normal case, not the
exotic one. [ADR 0007](adr/0007-litellm-over-provider-sdks.md) records what that
costs as well as what it buys.

## Reporting is four renderers over one model

`reporting/` holds `console.py`, `markdown.py` and `html.py`; the TUI in `tui/`
is the fourth. All four read the same `Run`, and none of them computes a score,
a total, or a pass count of its own — those come from the runner. A renderer
that recomputed a total could disagree with the one the gate checked.

`html.py` emits a self-contained page with no external references and no
JavaScript, because its job is to be openable from a CI artifact page where
nothing else will load.

## The public surface is seven names

```python
from evalstand import Case, Result, Score, Trace, evaluate, scorer, trace
```

`Run` and `Batch` are deliberately not exported: users read those in reports,
they never construct them. Widening this list needs an ADR — the cap exists to
force the question rather than to answer it.

## Where to look for what

| I want to change… | Start in |
| --- | --- |
| What a user writes in an eval file | `api.py` |
| How cases are executed, timed, or limited | `runner.py` |
| What `pytest` collects, or an exit code | `plugin.py` |
| A scorer, or the `Scorer` protocol | `scorers/` |
| The history schema | `storage.py` + `migrations/` |
| What a report looks like | `reporting/` |
| The live table, or watch mode | `tui/` |
| Model calls, cost, or token accounting | `llm.py` |

## The decisions behind all of this

Every non-obvious choice above has an ADR recording the alternatives that were
rejected and why. The [decision index](adr/index.md) lists them.

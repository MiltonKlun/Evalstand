# Decision index

Architecture decision records: the choices that are not obvious from the code,
and the alternatives that were rejected.

Each one states what was decided, what else was considered, and what the decision
costs. They are written to be read by someone who is about to change the thing
they describe — including the author, months later, who no longer remembers why
it is like that.

An ADR is **immutable once accepted.** A decision that turns out to be wrong gets
a new record that supersedes the old one; the original stays, because the
reasoning that led to a mistake is part of the history.
[ADR 0000](0000-adr-process.md) describes the process itself.

## The records

| # | Decision | Status | Date |
| --- | --- | --- | --- |
| [0000](0000-adr-process.md) | ADR process | accepted | 2026-09-01 |
| [0001](0001-name.md) | Project name: `evalstand` | accepted | 2026-09-01 |
| [0002](0002-design-scope.md) | Design scope | accepted | 2026-09-01 |
| [0003](0003-no-variants-in-v1.md) | No variants in v1 | accepted | 2026-09-01 |
| [0004](0004-evaluate-registers-defers.md) | `evaluate()` registers, it does not execute | accepted | 2026-09-01 |
| [0005](0005-project-local-database.md) | The database is project-local | accepted | 2026-09-01 |
| [0006](0006-trace-parenting.md) | Trace parenting via a reset `ContextVar` | accepted | 2026-09-01 |
| [0007](0007-litellm-over-provider-sdks.md) | LiteLLM rather than provider SDKs | accepted | 2026-09-01 |
| [0008](0008-plugin-delegates-execution-to-the-runner.md) | The plugin collects and reports; the runner executes | accepted | 2026-09-03 |
| [0009](0009-web-ui-is-optional-and-read-only.md) | The web UI is optional, and read-only over history | accepted | 2026-09-14 |

## By what they answer

**"Why does this project exist, and what is it not?"**
[0001](0001-name.md) on the name,
[0002](0002-design-scope.md) on what is adopted from the TypeScript
implementation that proved the idea, what is done differently, and what is
added.
[0003](0003-no-variants-in-v1.md) records the one significant capability
deliberately waived rather than missing.

**"Why is the code structured this way?"**
[0004](0004-evaluate-registers-defers.md) on why `evaluate()` performs no I/O,
and [0008](0008-plugin-delegates-execution-to-the-runner.md) on why the pytest
plugin executes nothing. Together these two explain the central seam:
collection and execution are separate, because pytest runs items one at a time
and concurrency has to live somewhere else.

**"Why does my data live there?"**
[0005](0005-project-local-database.md) on the project-local SQLite database —
and therefore on why CI starts every build with no history.

**"Why does it do that at runtime?"**
[0006](0006-trace-parenting.md) on the single `ContextVar` rule that makes trace
trees survive `asyncio.gather` and mid-call exceptions.
[0007](0007-litellm-over-provider-sdks.md) on reaching every provider through
one library, including what that costs.

## Writing a new one

Copy the shape of an existing record: a title line, a status, a date, then
context, the decision, and the consequences — including the ones you would
rather not write down. Number it sequentially, add it to
[the nav](https://github.com/MiltonKlun/Evalstand/blob/main/mkdocs.yml) and to
the table above.

A test enforces that every ADR on disk appears in the nav, so a decision written
and never linked will fail the suite rather than quietly become unfindable.

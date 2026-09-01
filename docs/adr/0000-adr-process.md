# 0000 — ADR process

- **Status:** accepted
- **Date:** 2026-09-01

## Context

This project makes design decisions that are not obvious from the code, and it
is built over several weeks. Without a record, the reasoning behind a decision
is lost and gets relitigated.

## Decision

Every non-obvious decision gets an ADR in `docs/adr/NNNN-title.md`, numbered
sequentially. Each records **context**, **decision**, and **consequences**.
Keep them short — a screen or less. An ADR is never edited to change its
decision; it is superseded by a later ADR that references it.

An ADR is required for:

- Adding a dependency outside the approved list in `PLAN.md` Section 4.
- Waiving or changing a row in the Section 2 capability table.
- Any architectural choice a future contributor would reasonably question.

## Consequences

Slight overhead per decision. In exchange, the reasoning survives, and the
"why is it like this?" question has an answer that is not a guess.

## Template

```markdown
# NNNN — Title

- **Status:** proposed | accepted | superseded by [NNNN](NNNN-title.md)
- **Date:** YYYY-MM-DD

## Context
What forced the decision.

## Decision
What we chose.

## Consequences
What this makes easy, and what it costs.
```

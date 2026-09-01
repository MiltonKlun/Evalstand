# 0003 — No variants in v1

- **Status:** accepted
- **Date:** 2026-09-01

The reference implementation supports *variants* (`variantName`, `variantGroup`,
typed through the task signature) for running the same cases against several
prompt or model configurations and comparing them side by side. `evalstand` does
not implement this in v1, and the capability table records it as waived rather
than missing.

Two reasons. Variants multiply the result space — cases × repeats × variants —
which complicates every aggregate, the results table, and the TUI before any of
those exist. More importantly, an A/B comparison invites the conclusion "B is
better," and `evalstand` has no significance testing in v1; the same reasoning
that stops `compare` from using verdict language stops us from shipping a
feature whose entire purpose is to invite one.

## Consequences

The schema must not preclude variants later. `Result` is keyed on
`(run_id, case_id, repeat_index)`, which leaves room for a fourth key column
without restructuring. The term stays reserved in `CONTEXT.md` so it is not
reused for something else.

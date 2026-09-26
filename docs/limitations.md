# Limitations

What `evalstand` does not do, or does only partly. Stated plainly and kept
accurate as the project changes, because a number quoted without its caveat
stops being honest.

## A delta is not a verdict

`compare` reports the arithmetic difference between two runs' means. There is
**no significance testing**, so a difference between two runs of a stochastic
system may be noise. Nothing here will call a change a regression.

## LLM judge scorers are unvalidated

`judge` and `factuality` work, but their verdicts have not been calibrated
against human labels on your data. Treat them as a signal, not a measurement.
Every judge Score carries `unvalidated: True` in its metadata so the caveat
travels with the number. See [Scorers](scorers.md).

## The showcase baseline covers the scalar fields only, and one method

`examples/pdf_extraction/BASELINE.md` records one run of span selection with
TypeSafe's `jev-1.13.0`: a regex finds candidate values and the model picks
among them. It scored 30/30 invoices on six scalar fields for $0.0015. It does
not score line items — span selection cannot produce a list of records — and
the generative eval, which does, has not yet been run against a real model. One
run of one method on a synthetic corpus is a record, not a verdict on either.

## The live view takes a few seconds to start

`evalstand watch` imports LiteLLM at start-up — over three seconds on its own —
even for an eval that never calls a model. The demo recording cuts that wait and
says so.

## Cost figures are lower bounds when a model is not in LiteLLM's pricing table

Unpriced calls are counted and declared, never silently treated as free. A
total that includes unpriced calls is marked as a lower bound wherever it
appears.

## The web UI has no authentication

`evalstand serve` binds localhost by default for that reason. The database holds
every prompt and completion your evals sent and received, so `--host 0.0.0.0`
publishes all of it to anyone who can reach the port. There is no login, and
adding one is not planned — put it behind something that does auth if it needs
to leave the machine. See [Web UI](web.md).

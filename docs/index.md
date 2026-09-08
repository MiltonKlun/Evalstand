# evalstand

Evaluating an LLM application should feel like running a test suite.

`evalstand` is a local-first evaluation tool for Python. You write an eval file,
run a watch command, and results stream into a terminal UI with scores, nested
call traces, token counts, latency and cost. Everything runs on your machine and
persists to a local SQLite database, so you can compare a run against the one
before it.

```python
from evalstand import Case, evaluate, llm
from evalstand.scorers import exact


async def answer(question: str) -> str:
    reply = await llm.acall("gpt-4o-mini", [{"role": "user", "content": question}])
    return reply.text


evaluate(
    name="capitals",
    cases=[Case(id="france", input="Capital of France?", expected="Paris")],
    task=answer,
    scorers=[exact],
)
```

```bash
evalstand watch
```

## Start here

- **[Quickstart](quickstart.md)** — install, write an eval, run it
- [Writing evals](writing-evals.md) — cases, tasks, repeats, custom columns
- [Scorers](scorers.md) — the built-in library, and writing your own
- [Traces](traces.md) — what your task did, and what it cost
- [Watching](watching.md) — the live view and watch mode
- [CI](ci.md) — thresholds, exit codes, pull-request comments
- [Architecture decisions](adr/0000-adr-process.md) — why the design is the way
  it is

## What makes it different

Three things go beyond the tool this one is modelled on:

**Nested trace trees.** A call made inside another call is its child, so a
Result carries a tree rather than a flat list. When a case scores badly, the
tree is what tells you which step went wrong.

**A response cache.** Re-running an eval after editing one prompt should not
re-buy every answer that did not change.

**Stable case identity.** Cases are matched across runs by `id`, not by
position. Inserting a case at the top does not silently re-pair every later case
with the wrong history — which is what makes `history` and `compare` worth
trusting.

## What it will not tell you

A short list, because a tool that overstates what it knows is worse than one
that knows less.

**A delta is not a verdict.** `compare` reports the arithmetic difference
between two runs' means. There is no significance testing, so a difference
between two runs of a stochastic system may be noise. Nothing here will call a
change a regression.

**An unknown is never a zero.** A `-` in a report means the value could not be
determined — a call the pricing table did not cover, a scorer that declined to
judge. It is never rendered as `0`, because a reader would act on a zero.

**A scorer that broke is not a bad answer.** Errored scores are excluded from
means and counted separately. A mean over 27 of 30 cases is always reported as
covering 27.

**LLM judge scorers are unvalidated.** `judge` and `factuality` work, but
nobody has measured how well their verdicts agree with a human's on your data.
Treat them as a signal, not a measurement.

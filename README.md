# evalstand

> **Status: in development.** Phases 0-3 of 7 complete: evals run through
> pytest or the CLI, concurrently, with response caching, nested traces, and
> per-run token and cost totals. The scorer library, persistence, and the live
> TUI are not built yet, so the published release remains a placeholder — build
> from source to try it.

Evaluating an LLM application should feel like running a test suite.

`evalstand` is a local-first LLM evaluation tool for Python. You write an eval
file, run a watch command, and results stream into a live terminal UI — scores,
nested call traces, token counts, latency, and cost. Everything runs on your
machine and persists to a local SQLite database, so you can compare a run
against the one before it.

```python
from evalstand import Case, evaluate, llm
from evalstand.scorers import exact, levenshtein


def load_cases() -> list[Case]:
    return [
        Case(id="q1", input="What is the capital of France?", expected="Paris"),
        Case(id="q2", input="What is 2 + 2?", expected="4"),
    ]


async def answer(question: str) -> str:
    resp = await llm.acall("gpt-4o-mini", [{"role": "user", "content": question}])
    return resp.text


evaluate(
    name="basic-qa",
    cases=load_cases,
    task=answer,
    scorers=[exact, levenshtein],
)
```

`levenshtein` arrives with the scorer library in Phase 4; `exact` works today.

Save that as `qa_eval.py` and run it either way:

```bash
evalstand run qa_eval.py     # live TUI, watch mode, traces
pytest qa_eval.py            # plain test runner, CI-friendly
```

Both go through the same runner, so both accept the same execution controls:

```bash
pytest qa_eval.py --concurrency 4   # cases in flight at once (default 8)
pytest qa_eval.py --timeout 30      # abandon a case after 30s; the rest continue
pytest qa_eval.py --no-cache        # call the provider even when a response is cached
```

These control *how* a run executes, never what it measures. One case failing —
raising, or timing out — never ends the run: it is recorded as that case's
result and the others carry on, because discarding twenty-nine good
measurements to punish the thirtieth wastes the money already spent on them.

## Why

Existing Python options are either heavyweight platforms that push you toward a
hosted service, or bare metric libraries with no runner, no persistence, and no
live feedback loop. `evalstand` is the middle: a real runner with a real UI that
stays on your machine.

## Planned capabilities

See [PLAN.md](PLAN.md) for the full build plan and the capability checklist that
defines v1. Three of those capabilities go beyond what inspired them:

- **Stable case identity.** Cases are matched across runs by a durable `id`, not
  by position, so inserting a case does not silently re-pair every later case
  with the wrong history.
- **Nested traces.** A call made inside another call is recorded as its child,
  so a judge scorer's request appears under the task request it is judging
  rather than beside it.
- **Response caching.** Identical model calls are served from a local cache, so
  iterating on an eval does not pay for the same request twice.

## Limitations

Stated up front, and kept accurate as the project grows:

- Score differences between runs are reported as plain deltas. There is **no
  statistical significance testing** in v1, so a delta is not evidence of a real
  regression or improvement.
- LLM-as-judge scorers are **unvalidated** — they have not been calibrated
  against human labels.

## Licence

MIT. See [LICENSE](LICENSE).

---

<sub>Inspired by [evalite](https://github.com/mattpocock/evalite) (MIT), which
showed that local LLM evals could feel like running tests. `evalstand` is an
independent Python implementation.</sub>

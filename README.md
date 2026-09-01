# evalstand

> **Status: in development.** Phase 0 of 7. Not yet usable, not yet on PyPI.

Evaluating an LLM application should feel like running a test suite.

`evalstand` is a local-first LLM evaluation tool for Python. You write an eval
file, run a watch command, and results stream into a live terminal UI — scores,
nested call traces, token counts, latency, and cost. Everything runs on your
machine and persists to a local SQLite database, so you can compare a run
against the one before it.

```python
from evalstand import Case, evaluate
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

Save that as `qa_eval.py` and run it either way:

```bash
evalstand run qa_eval.py     # live TUI, watch mode, traces
pytest qa_eval.py            # plain test runner, CI-friendly
```

## Why

Existing Python options are either heavyweight platforms that push you toward a
hosted service, or bare metric libraries with no runner, no persistence, and no
live feedback loop. `evalstand` is the middle: a real runner with a real UI that
stays on your machine.

## Planned capabilities

See [PLAN.md](PLAN.md) for the full build plan and the capability checklist that
defines v1.

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

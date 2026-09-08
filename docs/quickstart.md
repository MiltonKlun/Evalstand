# Quickstart

## Install

```bash
uv add evalstand      # or: pip install evalstand
```

## Write an eval

An eval file is a Python file whose name ends in `_eval.py`. It declares cases,
a task, and the scorers that judge the output:

```python
# capitals_eval.py
from evalstand import Case, evaluate, llm
from evalstand.scorers import exact


def load_cases() -> list[Case]:
    return [
        Case(id="france", input="What is the capital of France?", expected="Paris"),
        Case(id="japan", input="What is the capital of Japan?", expected="Tokyo"),
        Case(id="peru", input="What is the capital of Peru?", expected="Lima"),
    ]


async def answer(question: str) -> str:
    reply = await llm.acall(
        "gpt-4o-mini",
        [{"role": "user", "content": f"{question} Answer with the city name only."}],
    )
    return reply.text.strip()


evaluate(
    name="capitals",
    cases=load_cases,
    task=answer,
    scorers=[exact],
)
```

Three things, and only three: **Cases** are the inputs, the **Task** is your
function under test, and **Scorers** judge what it returned.

`evaluate()` registers and returns immediately. Nothing runs at import time, so
`import capitals_eval` never spends money.

## Run it

```bash
export OPENAI_API_KEY=sk-...
evalstand run
```

```
+------------------------------------------+
| eval     | scorer | mean | passed | cost |
|----------+--------+------+--------+------|
| capitals | exact  | 0.67 |    2/3 |  $0.0002 |
+------------------------------------------+
              needs attention
+-----------------------------------------+
| eval     | case | reason       | output |
|----------+------+--------------+--------|
| capitals | peru | failed exact | Cusco  |
+-----------------------------------------+
```

Evals also run under bare `pytest`, because `evalstand` registers a pytest
plugin. Each `(case, repeat)` pair is one test item, so `-k`, `-x` and `--lf`
all work:

```bash
pytest                 # your tests and your evals
pytest -k france       # one case
```

## Watch it

The live view is the point of the tool. Rows appear as each case finishes, and
editing your prompt re-runs the eval:

```bash
evalstand watch
```

`enter` opens a case with its full output and trace tree, `f` filters to what
needs attention, `/` searches, and `q` quits. See [Watching](watching.md).

## Compare two runs

Every run is recorded in a project-local SQLite database:

```bash
evalstand history              # what has run
evalstand show <run-id>        # one run in detail
evalstand compare <old> <new>  # what changed
```

`compare` reports differences, never verdicts. There is no significance testing
here, so a delta between two runs of a stochastic system may be noise — and
saying otherwise would be a claim the tool cannot support.

## Gate a pull request

```bash
evalstand run --threshold 0.85 --fail-on-error
```

Exit `0` met the bar, `1` fell below it, `2` something did not run. See
[CI](ci.md) for the workflow and the full exit-code table.

## Where next

- [Writing evals](writing-evals.md) — cases, tasks, repeats, custom columns
- [Scorers](scorers.md) — the built-in library and how to write your own
- [Traces](traces.md) — what your task actually did, and what it cost
- [Watching](watching.md) — the live view and watch mode
- [CI](ci.md) — thresholds, exit codes, pull-request comments

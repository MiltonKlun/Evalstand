# Writing evals

An eval is a file named `*_eval.py` that declares one or more `evaluate()` calls.

## A complete example

```python
from evalstand import Case, evaluate
from evalstand.scorers import exact


def load_cases() -> list[Case]:
    return [
        Case(id="france", input="What is the capital of France?", expected="Paris"),
        Case(id="japan", input="What is the capital of Japan?", expected="Tokyo"),
    ]


def answer(question: str) -> str:
    return "Paris" if "France" in question else "Tokyo"


evaluate(
    name="capitals",
    cases=load_cases,
    task=answer,
    scorers=[exact],
)
```

Run it either way:

```bash
evalstand run capitals_eval.py    # live UI, watch mode, traces
pytest capitals_eval.py           # plain test runner
```

## The three things you write

**Cases** are your test inputs. Each needs an `id`, an `input`, and optionally an
`expected` value. `cases=` accepts a list, a function returning one, or an async
function — loading is deferred, so a loader that reads files or calls an API does
not run at import time.

**The task** is your function under test. It receives a Case's input and returns
an output. It may be sync or async; you never have to say which.

**Scorers** judge one output. `exact` is built in; `evalstand.scorers` holds the
rest.

## Case ids

An id is what makes a Case the same Case across runs, which is what `history` and
`compare` match on. Supply one wherever you can.

Cases given as plain dicts without an `id` are numbered by position
(`case_0001`, `case_0002`, …). That works, but **reordering a generated dataset
breaks its history**, because case `case_0002` then refers to different content
than it did before. Ids are never derived from content: a content-derived id
would make an edited Case look like a brand-new one, hiding the fact that the
dataset changed.

## `evaluate()` does not run anything

It registers the eval and returns. Nothing executes until a runner asks for it.

That is why importing an eval file is free and side-effect-free, why `pytest -k`
can select a single case before anything runs, and why watch mode can work out
which evals a changed file declares without executing them.

## Repeats

`repeat=N` runs each case N times, for tasks whose output varies:

```python
evaluate(name="creative", cases=cases, task=write, scorers=[judge], repeat=5)
```

Each execution is its own test item with its own outcome. They are not collapsed
into a single pass or fail, because any rule for doing so — any, all, majority —
is a judgement you did not make.

## Names must be unique

An eval's name is its identity across runs. Two evals sharing a name is an error
raised at collection time, naming both files, rather than a silent merge of two
unrelated histories.

# Running evals in CI

## Bare pytest

`evalstand` registers a pytest plugin, so `*_eval.py` files are collected with no
configuration:

```bash
pytest                    # runs your tests and your evals
pytest examples/toy       # just the evals in one directory
pytest -k france          # one case
```

Each `(case, repeat)` pair is one test item with one outcome, so evals participate
in `-k`, `-x`, `--lf`, and everything else pytest offers. A repository with no
eval files behaves exactly as it did before.

After an eval run, a summary table is printed:

```
+----------------------------------------------+
| eval         | scorer | mean | passed | cost |
|--------------+--------+------+--------+------|
| toy-capitals | exact  | 1.00 |    3/3 |    - |
+----------------------------------------------+
```

A `-` means the value is genuinely unknown — no priced calls, or no scorer that
judged pass/fail. It never means zero.

## A GitHub Actions workflow

```yaml
name: evals

on: [pull_request]

jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run pytest --no-header -q
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## What CI can and cannot check

**A threshold is an absolute bar.** `--threshold` (Phase 7) fails a run whose mean
score falls below a number you choose. That number is a human decision taken from
a committed baseline — never computed from the most recent run, which would let
the bar drift downward every time quality dropped.

**Comparison is not available in CI.** Run history lives in a project-local,
gitignored database (ADR 0005), so a fresh clone starts with no prior runs. CI has
nothing to compare against. `--threshold` works on an empty database because it is
absolute; `compare` correctly reports that it has nothing to compare rather than
falsely passing.

**A delta is not a verdict.** When you do compare two runs locally, `evalstand`
reports the plain difference between their means. It does not call a change a
regression or an improvement, because it has no significance testing: a difference
between two runs of a stochastic system may be noise, and asserting otherwise
would be a claim the tool cannot support.

**Errored scores are excluded, and counted.** A scorer that raised — a rate-limited
judge, an unparseable output — is not evidence the task did badly. Those scores are
left out of the mean, and the number excluded is always reported alongside it. A
mean over 27 of 30 cases presented as though it covered all 30 would be quietly
wrong.

**A case with no usable score does not pass.** If every scorer for a case errored,
the case has no verdict: the task may have been fine, but nothing measured it.
Such a case fails rather than passing, so a run that measured nothing cannot exit
zero and be read as success.

A case where *some* scorers worked is still judged on those. Turning any errored
score into a failure is what `--fail-on-error` will do (Phase 7); until then, only
the total absence of a measurement is treated as a failure.

## Keeping CI free

The test suite runs with no API keys and no spend. Tests that would call a provider
use committed cassettes; tests that must hit a real model are marked
`@pytest.mark.live` and **deselected by default**:

```bash
pytest                    # live tests deselected; no key needed, nothing spent
pytest -m live            # opt in, with a key set
```

Deselected rather than skipped, deliberately. Skipping depends on no key being
present, so a developer with `OPENAI_API_KEY` exported would spend money on a
plain `pytest` without meaning to. Running them is always an explicit act.

They exist because everything else is mocked or replayed: nothing else in the
suite would notice if LiteLLM changed the shape of what it returns. A live run
costs a fraction of a cent — the cheapest model, prompts of a few tokens.

## Mutation testing

Coverage shows which lines ran, not which behaviours are asserted on. A line can
be fully covered by a test that would pass no matter what the line did.

```bash
python scripts/mutate.py          # break each behaviour, check the suite notices
python scripts/mutate.py --list   # see what it would run
```

A **survivor** is a mutation the suite did not catch: the behaviour is covered
but unverified. Three survivors from the post-Phase-2 audit each turned out to be
a real gap — an unmeasured eval exiting zero, a numeric unknown-placeholder, and
a supplied case id that could be silently overwritten.

The harness purges `__pycache__` around every mutant. Restoring the source file
is not enough on its own: during that audit a mutated `.pyc` outlived its
restored source, and `UNKNOWN` read as `"0"` while the file on disk said `"-"`.
Any harness that edits source in place has to clear bytecode, or its results
cannot be trusted.

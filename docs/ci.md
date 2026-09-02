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

## Keeping CI free

The test suite runs with no API keys and no spend. Tests that would call a provider
use committed cassettes; tests that must hit a real model are marked
`@pytest.mark.live` and excluded by default:

```bash
pytest -m "not live"      # the default in CI
pytest -m live            # opt in locally, with keys set
```

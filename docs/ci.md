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

The simplest thing that works — run the evals, fail the build if quality drops:

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
      - run: uv run evalstand run --threshold 0.85 --fail-on-error
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Posting the summary as a pull-request comment

`--output markdown` prints a body suitable for a comment. Two ways to use it,
and the first is worth trying before the second:

**The job summary** needs no permissions, no token, and no third-party action.
It appears on the run's own page:

```yaml
      - run: uv run evalstand run --threshold 0.85 --output markdown >> "$GITHUB_STEP_SUMMARY"
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

**A comment on the pull request** puts the numbers where the review happens:

```yaml
permissions:
  contents: read
  pull-requests: write

jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync

      # `tee` so the body is kept even when the gate fails, and
      # `continue-on-error` so the comment is still posted — a build that fails
      # silently teaches nobody why.
      - id: evals
        continue-on-error: true
        run: |
          uv run evalstand run --threshold 0.85 --fail-on-error             --output markdown | tee summary.md
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - uses: actions/github-script@v7
        with:
          script: |
            const body = require('fs').readFileSync('summary.md', 'utf8');
            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body,
            });

      # The gate, restored after the comment is posted.
      - if: steps.evals.outcome == 'failure'
        run: exit 1
```

`continue-on-error` with an explicit re-fail is deliberate. Without it the job
stops at the eval step and the comment never posts, so the one artefact
explaining the failure is the thing the failure suppresses.

**Forked pull requests cannot post comments.** `GITHUB_TOKEN` is read-only for
them, by design — otherwise anyone could open a PR that writes to your
repository. Use the job summary for those, or trigger on `pull_request_target`
and understand what you are accepting before you do.

### An HTML report as a build artifact

`--html PATH` writes a self-contained report alongside whatever the terminal
prints. Unlike the summary and the comment, it shows **everything**: the full
output of every case and the complete trace tree, which is the one thing a build
log cannot give you.

```yaml
      - run: uv run evalstand run --threshold 0.85 --html report/index.html
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - uses: actions/upload-artifact@v4
        if: always()          # the failing run is the one worth reading
        with:
          name: eval-report
          path: report/
```

`if: always()` matters. Without it the upload is skipped exactly when the
threshold gate fired, and the report explaining the failure is the thing the
failure discards.

The file has no external references — no stylesheet, no script, no sibling
assets — so it renders correctly in the sandboxed iframe CI systems serve
artifacts from, and it still works when emailed to somebody. Failing cases start
expanded; passing ones are collapsed, because a thirty-case report with every
prompt open is a page nobody scrolls.

`--html` is independent of `--output`: the terminal summary still prints, since a
log made less useful in exchange for a file nobody has opened yet is a poor
trade. A report that cannot be written logs a warning and leaves the run alone —
the measurement has already been paid for.

## What CI can and cannot check

**A threshold is an absolute bar.** `--threshold` fails a run whose mean score
falls below a number you choose. That number is a human decision taken from
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

```
$ evalstand compare run-old run-new

 scorer | before | after | changed by
 exact  |   0.75 |  0.50 |      -0.25

 cases whose pass state changed
 q1     | fail -> pass
 q3     | pass -> fail

evalstand has no significance testing: a delta is an arithmetic difference,
not evidence of a real change.
```

That last line is printed on every comparison, not left to this document. A
reader who takes a delta as proof of a change has been misled by the tool, and
the tool is the only thing present at the moment they might do so.

**An edited case is not evidence about the task.** If a case's input or expected
value changed between two runs, its pass state may have moved for a reason that
has nothing to do with the model. Those cases are listed separately, under
"cases edited between these runs", and are never counted as flips. `evalstand`
knows which is which because it snapshots every case's content hash with the run
that used it.

**Two runs may not be comparable at all.** `compare` says so when the runs are of
different evals, when the task's source changed between them, or when they did
not cover the same set of cases — each of which makes a mean-to-mean difference
mean something other than it appears to.

**A case that stopped being measured is a finding, not a silence.** If the task
began crashing, or every scorer for a case errored, that case has no verdict and
no value — so it cannot flip and cannot move. Those appear under *cases whose
measurement changed*, with the error that caused it:

```
 cases whose measurement changed
 q1 | was judged, now not run | RuntimeError: the API key expired
```

Reported apart from flips deliberately. "Was passing, now crashes" is not the
task getting worse — the task did not run, and saying otherwise would attribute
an infrastructure failure to the model.

**Errored scores are excluded, and counted.** A scorer that raised — a rate-limited
judge, an unparseable output — is not evidence the task did badly. Those scores are
left out of the mean, and the number excluded is always reported alongside it. A
mean over 27 of 30 cases presented as though it covered all 30 would be quietly
wrong.

**A case with no usable score does not pass.** If every scorer for a case errored,
the case has no verdict: the task may have been fine, but nothing measured it.
Such a case fails rather than passing, so a run that measured nothing cannot exit
zero and be read as success.

A case where *some* scorers worked is still judged on those. `--fail-on-error`
is what turns any errored score into a failed build; without it, only the total
absence of a measurement is treated as a failure.

## Exit codes

A CI job can gate on these, and they mean different things on purpose:

| code | meaning | when |
| --- | --- | --- |
| `0` | every eval met its bar | |
| `1` | an eval fell below `--threshold` | the measurement worked; the answer was worse than your bar |
| `2` | something did not run | only with `--fail-on-error`; also what `compare` returns when it has nothing to compare |

```bash
evalstand run --threshold 0.85 --fail-on-error
```

**`2` outranks `1`.** A run whose cases mostly errored still has a mean — over
the few that survived — and that mean is not a measurement of the eval. Reporting
it as "below threshold" would name a cause the evidence does not support, and
send whoever reads the build to look at the model when the real problem is an
expired key or a rate limit. One code says *the model got worse*; the other says
*we do not know*.

That is also why `--fail-on-error` is opt-in rather than the default: with a
flaky provider, some teams would rather have the partial number than no build at
all. The flag is how you say which you want.

**`compare` uses `2` for the same reason.** Asked about a run id that is not
recorded, or one whose Batch never finished, it exits `2` — not `1`. A job that
saw `1` could not tell "these runs differ badly" from "there was no measurement
here", which is exactly the confusion the table exists to prevent.

Every code in this table is reproduced in a test: see
`tests/unit/test_exit_codes.py`, including the empty-database case below.

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

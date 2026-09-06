# The live view and watch mode

`evalstand run` prints a summary when everything has finished. `evalstand watch`
opens a terminal UI instead: rows appear as each case completes, and the eval
re-runs when you edit a file.

```
evalstand watch                  # the eval in the current directory
evalstand watch evals/           # a directory
evalstand watch qa_eval.py       # one file
evalstand watch --eval basic-qa  # when a path declares several
```

The view runs **one** eval. A path declaring several is refused rather than
guessed at, because every number on screen would otherwise be about something
you did not ask for.

`--once` opens the live view without watching for changes — useful when you want
the table and the trace trees but are not editing.

`--concurrency` and `--timeout` behave exactly as they do for `evalstand run`.

## What the table shows

| column | meaning |
| --- | --- |
| case | the Case id, with `#n` for repeats |
| status | `pass`, `fail`, `error`, `unmeasured`, or `scored` |
| score | the mean of that case's scores, or `-` |
| latency | wall time for the Task |
| cost | what the case's calls cost, or `-` when they were never priced |

Five statuses rather than pass/fail, because collapsing them hides the two
failures that are not the model's fault:

- **`error`** — the Task raised. It never produced an answer, so it did not
  answer wrongly.
- **`unmeasured`** — the Task answered and every Scorer broke. The model's
  performance is *unknown*, not bad. Reporting a scorer outage as a regression
  sends you to debug a prompt that is working.
- **`scored`** — a continuous Scorer gave a value and no verdict. It has not
  failed anything, and the table does not decide otherwise on its behalf.

The cost column shows `-` rather than `$0.0000` when nothing could be priced: a
call nobody could price has not been shown to be free. The footer marks a total
as a lower bound the same way — `$0.0140 (+3 unpriced)`.

## Keys

| key | does |
| --- | --- |
| `enter` | open the selected case: full output, per-scorer breakdown, trace tree |
| `f` | show only cases that failed, errored, or went unmeasured |
| `/` | filter by case id (composes with `f`; `escape` clears it) |
| `r` | re-run now |
| `c` | compare this run with the previous run of the same eval |
| `h` | browse past runs, and compare any two |
| `y` | copy the selected case id |
| `q` | quit |

`h` and `c` need recorded runs, so they are available when you pass `--store`.

## What a change re-runs

Any watched file under the paths you named: `.py`, `.txt`, `.md`, `.json`,
`.yaml`, `.yml`, `.jinja`, `.j2`. Prompts live in text files at least as often
as in Python, and missing those would make watch mode useless for the workflow
it exists for.

Generated files are ignored — `__pycache__`, `.git`, `.venv`, `node_modules`,
editor backups. The history database is ignored too: a run writes to it, so
reacting to it would make every Batch trigger the next one.

Changes are debounced by 300ms, because one save produces a burst of filesystem
events and re-running on each would start three Batches for one edit.

Which evals a change *affects* is deliberately not computed. That means
resolving an import graph you can defeat with a dynamic import or a prompt read
from disk, and a watch mode that silently misses a change is worse than one that
occasionally re-runs too much — you stop trusting what you see.

## Editing during a run

**A change cancels the run in flight.** Waiting for a slow Batch to drain would
spend the feedback loop the feature exists to provide: you have already moved
on, and the results still arriving describe code you have edited.

In-flight model calls are left to finish rather than being killed. The money is
already spent, so discarding the response wastes it — and the next Batch, moments
away, asks for the same thing. Letting it land in the Cache turns a wasted call
into a free one.

The cancellation is recorded honestly. The Batch gets a terminal `cancelled`
status, `evalstand history` does not list it, and `evalstand compare` refuses it:

```
$ evalstand compare run-abc123 run-def456
run-def456 did not run to completion, so comparing it would compare different
questions.
```

A half-finished Batch that looked complete would drag every mean it touched, and
you would attribute the movement to your edit.

## Recording

Watch mode does **not** write to the history database unless you pass `--store`.

That is the opposite of `evalstand run`, and deliberate. Watch mode is for use
*while editing*, so the tree is dirty by construction and every run would be
tied to a commit whose code it did not reflect. Filling history with dozens of
unreproducible runs would bury the deliberate ones you actually measure against.

With `--store`, runs are recorded and the batch notes that the tree was dirty —
so no later reader mistakes one for something reproducible.

## Custom columns

`evaluate(columns=...)` adds derived columns to the table:

```python
evaluate(
    name="extraction",
    cases=load_cases,
    task=extract,
    scorers=[json_fields],
    columns={
        "fields correct": lambda result: sum(
            1 for s in result.scores if s.passed
        ),
    },
)
```

A plain `dict[str, Callable[[Result], Any]]`. A column that raises shows `!` in
that cell rather than taking down the row: a bug in a display helper must not
cost you the measurement it was called to display.

# Baseline

**Not yet recorded.** This file is a placeholder until the eval has been run
against a real model.

It is deliberately empty of numbers rather than filled with plausible ones. A
baseline is the figure people quote — and one produced from fixtures, or written
by hand to look reasonable, would be the most misleading document in this
repository. There is no honest way to state per-field accuracy without having
measured it.

## Producing it

```bash
uv sync --extra examples
python generate.py --seed 42        # build the corpus (free)

export OPENAI_API_KEY=...
pytest extraction_eval.py           # ~30 calls plus 30 judge calls

evalstand history                   # find the run id
python baseline.py <run_id>         # overwrite this file
```

`baseline.py` reads a **stored run** and writes the whole document: scorer
means, per-field accuracy, cost per document, and every case that failed with
the field names that failed on it. Nothing is typed by hand, so the numbers
cannot drift from the run they claim to describe.

It also refuses to be quietly wrong in two ways. A run whose calls all cost
exactly the same is labelled as coming from a mocked provider, because real
completions vary in length and therefore in price. And a run with no priced
calls reports `-` rather than `$0.0000`, because nothing has been shown to be
free.

## What it will and will not tell you

It will say what one model did on one day: which fields it got right, what each
document cost, and which documents failed on which fields.

It will not say whether that is good. `evalstand` has no significance testing,
so a later run scoring differently has not necessarily got better or worse — and
the generated document uses no verdict language for the same reason `compare`
does not. Use a baseline to notice a change worth investigating, never to
conclude one.

The `line_items` scorer is an **LLM judge and is unvalidated**: a second model's
opinion, not a measurement, uncalibrated against human labels. Read its number
with that in mind. See [docs/scorers.md](../../docs/scorers.md).

## What the corpus is built to catch

Whatever the numbers turn out to be, these are the documents worth reading the
failures on — each exists to provoke a specific mistake:

| Documents | The mistake they provoke |
|---|---|
| 03, 11, 24 | Line items run onto **page two** and the total is only on the last page, so a model that stops reading early reports items that do not sum to it. |
| 05, 12, 19, 26 | **A second currency**, moving the symbol and changing the separator. |
| 07, 15, 22 | **No due date exists.** The honest answer is `null`, and a plausible invention is wrong in a way that looks right. |
| 09, 17, 28 | **An ambiguous date** printed `DD/MM/YYYY`, whose intended reading the ground truth records. |

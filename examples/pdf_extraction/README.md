# Invoice extraction

Thirty synthetic invoices, extracted to JSON and scored three ways.

```bash
uv sync --extra examples
python generate.py --seed 42     # build the corpus (once)
pytest extraction_eval.py        # run the eval
```

Needs one API key (`OPENAI_API_KEY`). The corpus is generated locally and costs
nothing to build.

## Why the ground truth is trustworthy

**It is true by construction.** `generate.py` chooses each invoice's numbers and
*then* renders them, so the expected values are not a model's opinion of what
another model should have said. An eval whose ground truth came from an LLM
measures agreement between two models and calls it accuracy.

**The corpus regenerates identically for a given seed** — the same invoices with
the same ground truth, on any machine. That is what the eval is scored against,
and what `--check` verifies:

```bash
python generate.py --seed 42 --check    # regenerate and compare, writing nothing
```

A PDF normally embeds a creation timestamp, so two runs of the same code would
otherwise produce different files and a "reproducible" corpus would drift every
time anyone regenerated it. `invariant=1` turns that off, and `checksums.txt`
records the result.

**The names, dates and descriptions come from committed word lists sampled with
`random.Random`.** Python specifies that stream, so it reproduces on every
platform and version. A synthetic-data library does not make that promise and did
not keep one: the corpus was generated with `faker` until CI found all thirty
invoices differing on Linux — every drifting field faker-derived, every stable
field drawn from `random`. Pinning the version did not help, because the version
was already identical.

**Those checksums are a local change-detector, not a cross-platform promise.**
reportlab's byte output depends on its build, so a corpus generated on Linux does
not match checksums recorded on Windows — while the data inside is identical.
`--check` therefore reports a byte difference and fails only on a difference in
the *truth*. Assuming otherwise is what failed this example's first CI run.

`ground_truth.json` and `checksums.txt` are committed; the PDFs are gitignored
and rebuilt on demand. So the truth is reviewable in a diff, and a dependency
upgrade that changed what the generator produces fails the check rather than
shifting the corpus underneath a committed baseline.

## What the corpus is designed to break

| Documents | What they test |
|---|---|
| 03, 11, 24 | Line items spill onto **page two**, and the total is only on the last page. A model that stops reading early reports items that do not sum to it. |
| 05, 12, 19, 26 | **A second currency.** The symbol moves and the separator changes. |
| 07, 15, 22 | **No due date at all.** The honest answer is `null`, and a model that invents a plausible date is wrong in a way that looks right. |
| 09, 17, 28 | **An ambiguous date**: `04/03/2026` is printed, and the ground truth records which reading was meant. |

## Why three scorers

`json_fields` scores the object as a whole and, crucially, reports *which* field
was wrong — a single pass/fail would collapse "got the vendor wrong" and "got
everything wrong" into the same number.

`close_to` scores the total separately with a cent of tolerance, because a
rounding difference is not the same kind of error as reading the wrong number
off the page.

A **judge** scores line-item completeness. No string comparison can tell "missed
two items" from "worded the descriptions differently", and that is exactly the
distinction worth measuring. Like every judge, it is unvalidated — a second
model's opinion, not a measurement. See [docs/scorers.md](../../docs/scorers.md).

## A trap worth knowing about

`pypdfium2` wraps a C library that is **not thread-safe**. `evalstand` runs cases
concurrently and offloads a sync task to a thread pool, so the first version of
this example killed the interpreter outright — an access violation, not a Python
exception the runner could record.

`read_pdf` holds a lock for exactly this reason. If your task calls into a C
extension, do the same, or run with `--concurrency 1`.

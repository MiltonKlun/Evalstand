"""Extracting structured data from invoices.

The showcase example: thirty synthetic invoices, each with ground truth that is
**true by construction** — the numbers were chosen first and then rendered, so
nothing here is a model's opinion of what another model should have said.

    python generate.py --seed 42     # build the corpus (once)
    pytest extraction_eval.py        # run the eval

Scored three ways, because "did it extract the invoice correctly" is really
three different questions:

- `json_fields` for the object as a whole, which reports *which* field was wrong
  rather than collapsing five fields into one pass or fail;
- `close_to` for the total, where a rounding difference of a cent is not the
  same kind of error as reading the wrong number;
- a judge for line-item completeness, where no string comparison can tell
  "missed two items" from "worded them differently".

The corpus varies deliberately. Multi-page documents put the total on page two,
so a model that stops reading early reports items that do not sum to it. Three
invoices have **no due date at all**: the honest answer is null, and a model
that invents a plausible one is wrong in a way that looks right. Three more
print an ambiguous `DD/MM/YYYY` date whose true reading is recorded in the
ground truth.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from evalstand import Case, evaluate, llm
from evalstand.models import Score
from evalstand.scorers import close_to, json_fields, judge

HERE = Path(__file__).parent
INVOICES = HERE / "invoices"
GROUND_TRUTH = HERE / "ground_truth.json"

MODEL = "gpt-4o-mini"

PROMPT = """\
Extract the following fields from this invoice and reply with JSON only.

invoice_number  the invoice's own reference
vendor_name     who issued it
invoice_date    ISO format, YYYY-MM-DD
due_date        ISO format, or null if the invoice does not state one
currency        USD or EUR
total           a number, no currency symbol or thousands separator
line_items      a list of {description, quantity, unit_price, amount}

Invoice text:
{text}"""


def load_cases() -> list[Case]:
    """One case per invoice, with the generator's own record as the expected value.

    Raises rather than silently running on nothing: an eval that reports 0 cases
    looks like a pass, which is the worst way for a missing corpus to announce
    itself.
    """
    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH.name} is missing. Run `python generate.py --seed 42` first."
        )

    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    missing = [entry["file"] for entry in truth if not (INVOICES / entry["file"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} invoice PDF(s) are missing, starting with {missing[0]}. "
            f"Run `python generate.py --seed 42` to rebuild them."
        )

    return [
        Case(
            id=entry["file"].removesuffix(".pdf"),
            input=entry["file"],
            # Only the scalar fields. Line items are judged separately, because
            # comparing them field by field would score a model that reworded a
            # description the same as one that missed an item entirely.
            expected={
                "invoice_number": entry["invoice_number"],
                "vendor_name": entry["vendor_name"],
                "invoice_date": entry["invoice_date"],
                "due_date": entry["due_date"],
                "currency": entry["currency"],
                "total": entry["total"],
            },
            metadata={
                "line_items": entry["line_items"],
                "total": entry["total"],
                "pdf": entry["file"],
            },
        )
        for entry in truth
    ]


_PDF_LOCK = threading.Lock()
"""pypdfium2 wraps a C library that is **not thread-safe**.

`evalstand` runs cases concurrently, offloading a sync task to a thread pool so
one slow call cannot stall the others. That is right for a task that spends its
time waiting on a network — but eight threads inside the same C library at once
corrupt its heap, and the process dies with an access violation rather than a
Python exception the runner could catch and record on the Result.

Found by running this example: the first attempt killed the interpreter
outright. If your task calls into a C extension, wrap it the same way, or run
with `--concurrency 1`.
"""


def read_pdf(name: str) -> str:
    """The invoice's text.

    Imported lazily so `load_cases` can raise a useful message about a missing
    corpus before anything demands an optional dependency.
    """
    import pypdfium2

    with _PDF_LOCK:
        document = pypdfium2.PdfDocument(INVOICES / name)
        return "\n".join(page.get_textpage().get_text_range() for page in document)


def extract(pdf_name: str) -> Any:
    """Ask the model for the invoice as JSON."""
    response = llm.call(
        MODEL,
        [{"role": "user", "content": PROMPT.replace("{text}", read_pdf(pdf_name))}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return response.text


fields = json_fields()


def total_within_a_cent(output: Any, expected: Any, case: Case) -> Score:
    """The invoice total, allowing a cent of rounding.

    Separate from `json_fields` because a total is the one field where being
    close genuinely counts: a cent of rounding is not the same kind of error as
    reading the wrong number off the page, and folding both into one exact
    comparison would make them indistinguishable.
    """
    from evalstand.scorers.json_field import parse_object

    parsed = parse_object(output)
    if parsed is None:
        return Score.from_error("total", f"the output is not an object: {output!r}")

    result = close_to(abs_tol=0.01)(parsed.get("total"), case.metadata["total"])
    return result.model_copy(update={"scorer_name": "total"})


LINE_ITEM_RUBRIC = """\
Compare the line items the model extracted with the ones the invoice actually
contains. Judge completeness and accuracy of the amounts, not the exact wording
of descriptions.

(A) Every item is present with the right quantity and amount.
(B) Every item is present, but at least one quantity or amount is wrong.
(C) At least one item is missing, or an item was invented.
(D) The extracted line items bear no useful resemblance to the real ones."""


def line_items_complete(output: Any, expected: Any, case: Case) -> Any:
    """A judge for the list of line items.

    No string comparison can tell "missed two items" from "worded them
    differently", and that is the distinction worth measuring. Unvalidated, like
    every judge: a second model's opinion, not a measurement.
    """
    scorer = judge(
        rubric=LINE_ITEM_RUBRIC,
        choices={"A": 1.0, "B": 0.5, "C": 0.25, "D": 0.0},
        model=MODEL,
        name="line_items",
        temperature=0.0,
    )
    reference = json.dumps(case.metadata["line_items"], indent=2)
    return scorer(output, reference, case)


evaluate(
    name="invoice-extraction",
    cases=load_cases,
    task=extract,
    scorers=[fields, total_within_a_cent, line_items_complete],
)

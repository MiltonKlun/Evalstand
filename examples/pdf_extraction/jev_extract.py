"""Extract invoice fields with Jev, following the pre-parsed value cookbook.

Regex over-finds candidates; one Jev request asks every field's question at once
against the same state; code copies the picked span verbatim. Jev never produces
a value, so it cannot transpose a digit — it can only choose a span the regex
already found, or `none`.

    python jev_extract.py            # all 30 invoices, writes jev_results.json

The same `extract` backs `jev_extraction_eval.py`, which runs it under evalstand
so the result is recorded, traced and costed like any other run.

The three traps in the corpus are deliberate and are what this measures:
  - three invoices have no due date, where the honest answer is `none`
  - three print an ambiguous DD/MM/YYYY date
  - multi-page invoices put the total on page two
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from typesafe_sdk import TypeSafeClient

EXAMPLE = Path(__file__).parent
MODEL = "jev-1.13.0"
"""Pinned, not `jev-latest`. Aliases move, and a baseline recorded against an
alias describes whichever model the alias pointed at that day."""

NONE = "none"

PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000
"""$0.042 per million input tokens; output tokens are free.

Exact rather than a lower bound: the response reports `usage.input_tokens`, so
there is no pricing table to miss the model.
"""

FIELDS = ("invoice_number", "vendor_name", "invoice_date", "due_date", "currency", "total")

# Tuned to over-find. A candidate the regex misses is one Jev cannot pick.
MONEY_RE = re.compile(r"\$?\d[\d,]*\.\d{2}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}")
INVOICE_NO_RE = re.compile(r"\bINV-[A-Z0-9-]+\b")

_PDF_LOCK = threading.Lock()
"""pypdfium2 wraps a C library that is not thread-safe.

The standalone script reads one PDF at a time, but under evalstand the runner
offloads this sync task to a thread pool, and eight threads inside the library
at once kill the interpreter with an access violation. `extraction_eval.py`
learned this the hard way; the lock is the same one.
"""


@dataclass(frozen=True)
class Extraction:
    """One invoice's answer, and what it cost to get."""

    fields: dict[str, Any]
    """Normalised values, ready to compare against the ground truth."""
    picks: dict[str, str]
    """The spans Jev actually chose, before normalisation."""
    confidences: dict[str, float | None]
    input_tokens: int
    output_tokens: int | None
    seconds: float

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_PER_INPUT_TOKEN


def find(pattern: re.Pattern[str], text: str) -> list[str]:
    """Deduped matches in document order."""
    seen: set[str] = set()
    out: list[str] = []
    for match in pattern.findall(text):
        span = match.strip()
        if span and span not in seen:
            seen.add(span)
            out.append(span)
    return out


def read_pdf(name: str) -> str:
    import pypdfium2

    with _PDF_LOCK:
        document = pypdfium2.PdfDocument(EXAMPLE / "invoices" / name)
        return "\n".join(page.get_textpage().get_text_range() for page in document)


def vendor_candidates(text: str) -> list[str]:
    """Lines that could be a vendor name.

    The vendor is not a pattern a regex can characterise, so every short line
    that is not obviously something else becomes a candidate and Jev chooses.
    """
    out: list[str] = []
    for line in text.splitlines()[:6]:
        line = line.strip()
        if not line or line.upper() == "INVOICE":
            continue
        if INVOICE_NO_RE.search(line) or DATE_RE.search(line):
            continue
        if line.lower().startswith(("description", "total", "date", "due")):
            continue
        if line not in out:
            out.append(line)
    return out


def options(candidates: list[str], describe: str) -> dict[str, str]:
    """Candidate spans as Choice options, always with a `none` escape hatch.

    Without `none`, a field the document does not contain — three invoices have
    no due date — would force a pick among the dates that *are* there, and the
    answer would be a real date in the wrong field: wrong in a way that looks
    right.
    """
    opts = dict.fromkeys(candidates, describe)
    opts[NONE] = "no candidate in the document fits this field"
    return opts


def questions_for(text: str) -> dict[str, Any]:
    """The six questions, each a Choice over spans found in `text`."""
    from typesafe_sdk import Choice

    money = find(MONEY_RE, text)
    dates = find(DATE_RE, text)
    numbers = find(INVOICE_NO_RE, text)
    vendors = vendor_candidates(text)

    return {
        "invoice_number": Choice(
            instructions=(
                "Which candidate is this invoice's own reference number, "
                "as printed on the document?"
            ),
            criteria=options(numbers, "a string appearing in the invoice"),
        ),
        "vendor_name": Choice(
            instructions=(
                "Which candidate is the name of the company that ISSUED this "
                "invoice? Not the customer, and not a product or service name."
            ),
            criteria=options(vendors, "a line of text from the top of the invoice"),
        ),
        "invoice_date": Choice(
            instructions=(
                "Which candidate is the date the invoice was ISSUED — the date "
                "labelled 'Date'? Not the due date."
            ),
            criteria=options(dates, "a date string appearing in the invoice"),
        ),
        "due_date": Choice(
            instructions=(
                "Which candidate is the date payment is DUE — the date labelled "
                "'Due'? If the invoice states no due date at all, answer 'none'."
            ),
            criteria=options(dates, "a date string appearing in the invoice"),
        ),
        "total": Choice(
            instructions=(
                "Which candidate is the invoice TOTAL — the final amount payable, "
                "labelled 'Total'? Not an individual line item's amount, and not a "
                "unit price."
            ),
            criteria=options(money, "a money amount appearing in the invoice"),
        ),
        "currency": Choice(
            instructions="Which currency is this invoice denominated in?",
            criteria={
                "USD": "US dollars, usually shown with $",
                "EUR": "euros, shown with € or the code EUR",
                NONE: "the currency cannot be determined",
            },
        ),
    }


def normalise(field: str, value: str) -> Any:
    """Copy the picked span, normalised only in ways code can do exactly.

    Date normalisation is the cookbook's step 3, and skipping it is what cost
    the first run three fields. Jev picked the right span every time — the
    document prints `02/10/2018` and the ground truth stores `2018-10-02` — so
    the model was correct and the code had simply not converted the format.

    DD/MM is the reading, not MM/DD. That is a fact about this corpus, recorded
    in its README, and it is exactly the kind of decision the jaggedness page
    says to keep in code: Jev reads dates as text and cannot be asked which
    component is the day.
    """
    if value == NONE:
        return None
    if field == "total":
        return value.lstrip("$").replace(",", "")
    if field in ("invoice_date", "due_date") and "/" in value:
        day, month, year = value.split("/")
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return value


def extract(name: str, client: TypeSafeClient) -> Extraction:
    """One invoice, one request, all six questions answered in parallel."""
    text = read_pdf(name)
    questions = questions_for(text)

    started = time.perf_counter()
    response = client.system_one(state={"invoice_text": text}, questions=questions, model=MODEL)
    elapsed = time.perf_counter() - started

    payload = response.model_dump()
    answers = payload["answers"]
    usage = payload["usage"]
    picks = {key: answers[key]["choice"] for key in questions}

    return Extraction(
        fields={key: normalise(key, value) for key, value in picks.items()},
        picks=picks,
        confidences={key: answers[key].get("confidence") for key in questions},
        input_tokens=usage["input_tokens"],
        output_tokens=usage.get("output_tokens"),
        seconds=elapsed,
    )


def main() -> int:
    if not os.environ.get("TYPESAFE_API_KEY"):
        # Named before anything runs. The SDK's own failure arrives as an
        # authentication error partway through the corpus, which reads like a
        # service problem rather than a missing variable.
        print("TYPESAFE_API_KEY is not set. Export it; .env files are not loaded.")
        return 1

    from typesafe_sdk import TypeSafeClient

    truth = json.loads((EXAMPLE / "ground_truth.json").read_text(encoding="utf-8"))
    client = TypeSafeClient()

    rows = []
    total_tokens = 0
    total_seconds = 0.0

    for entry in truth:
        name = entry["file"]
        try:
            got = extract(name, client)
        except Exception as exc:  # recorded, never hidden
            print(f"  {name}: ERROR {type(exc).__name__}: {exc}", flush=True)
            rows.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        total_tokens += got.input_tokens
        total_seconds += got.seconds

        expected = {field: entry[field] for field in FIELDS}
        correct = {field: got.fields[field] == expected[field] for field in FIELDS}

        rows.append(
            {
                "file": name,
                "got": got.fields,
                "picks": got.picks,
                "expected": expected,
                "correct": correct,
                "confidences": got.confidences,
                "input_tokens": got.input_tokens,
                "seconds": round(got.seconds, 3),
            }
        )
        hits = sum(correct.values())
        print(
            f"  {name}: {hits}/{len(FIELDS)} fields  {got.input_tokens} tok  {got.seconds:.2f}s",
            flush=True,
        )

    out = EXAMPLE / "jev_results.json"
    out.write_text(
        json.dumps(
            {
                "model": MODEL,
                "rows": rows,
                "total_input_tokens": total_tokens,
                "total_cost_usd": total_tokens * PRICE_PER_INPUT_TOKEN,
                "total_seconds": round(total_seconds, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

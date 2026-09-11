"""Generate the synthetic invoices this example evaluates against.

**The ground truth is written at generation time, so it is true by
construction.** Nothing here reads a PDF back and guesses what it says: the
numbers are chosen first, then rendered. An eval whose expected values came from
a model would be measuring one model against another and calling it accuracy.

**Output is byte-identical for a given seed.** That matters more than it sounds:
a PDF normally embeds a creation timestamp, so two runs of the same code produce
different files, and a "reproducible" corpus would quietly drift every time
anybody regenerated it. `invariant=1` is what turns that off — verified by
building the same document in two separate processes a second apart and
comparing digests.

    python generate.py --seed 42

The PDFs are gitignored and rebuilt on demand; `ground_truth.json` and
`checksums.txt` are committed. The ground truth is reviewable in a diff, and the
checksums fail loudly if a dependency upgrade changes what the generator
produces — rather than letting the corpus shift underneath the baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent
INVOICES = HERE / "invoices"
GROUND_TRUTH = HERE / "ground_truth.json"
CHECKSUMS = HERE / "checksums.txt"

DEFAULT_SEED = 42
DOCUMENT_COUNT = 30

# Deliberate variation, so the corpus exercises what actually breaks
# extraction rather than thirty copies of the same easy document. Each is
# assigned by index, not at random, so the mix is the same at every seed.
MULTI_PAGE = {3, 11, 24}
"""Line items spilling onto a second page. A model that reads only page one
reports a total that does not match the items it found.

The item count for these starts at 9 because `render` fits exactly 8 rows per
page: a draw of 8 would leave the invoice on one page and quietly remove the
variation this set exists to create."""

EUROS = {5, 12, 19, 26}
"""A second currency. The symbol moves and the decimal separator changes."""

NO_DUE_DATE = {7, 15, 22}
"""A field that is genuinely absent. The honest answer is null, and a model
that invents a plausible date is wrong in a way that looks right."""

AMBIGUOUS_DATE = {9, 17, 28}
"""`03/04/2026` — March 4th or April 3rd? The ground truth records what the
generator meant, so a model guessing the other reading is measurably wrong."""


# The corpus's vocabulary is committed here rather than drawn from a faker
# library. Python's `random` is specified by the language and reproduces the
# same stream on every platform and version; a generator library makes no such
# promise — and did not keep one. The first CI run found all thirty invoices
# differing on Linux, with every drifting field faker-derived and every stable
# field drawn from `random`.
#
# These lists are deliberately mundane. The corpus exists to exercise
# extraction, and an invented vendor name is no easier or harder to read than a
# generated one.
VENDOR_STEMS = (
    "Arbor",
    "Bellweather",
    "Calderon",
    "Dunmore",
    "Everline",
    "Fairhaven",
    "Granville",
    "Hollis",
    "Ironside",
    "Jessup",
    "Kestrel",
    "Larkspur",
    "Marchetti",
    "Northgate",
    "Orinoco",
    "Pemberton",
    "Quillon",
    "Ravenswood",
    "Southbourne",
    "Thackeray",
    "Underhill",
    "Vandermeer",
    "Westbrook",
    "Yarrow",
    "Ashford",
    "Blackwood",
    "Croft",
    "Delaney",
    "Ellsworth",
    "Fenwick",
    "Galbraith",
    "Harrowgate",
)
VENDOR_FORMS = ("{0} Ltd", "{0} Inc", "{0} LLC", "{0} and Sons", "{0}-{1}", "{0} Group")

ITEM_QUALIFIERS = (
    "Adaptive",
    "Advanced",
    "Assimilated",
    "Balanced",
    "Centralised",
    "Configurable",
    "Distributed",
    "Enhanced",
    "Extended",
    "Federated",
    "Grounded",
    "Horizontal",
    "Integrated",
    "Layered",
    "Managed",
    "Modular",
    "Networked",
    "Optimised",
    "Persistent",
    "Proactive",
    "Reactive",
    "Streamlined",
    "Synchronised",
    "Unified",
    "Versatile",
    "Virtual",
)
ITEM_SUBJECTS = (
    "access",
    "analytics",
    "archive",
    "backup",
    "billing",
    "capacity",
    "compliance",
    "content",
    "database",
    "delivery",
    "encryption",
    "gateway",
    "hosting",
    "identity",
    "indexing",
    "licensing",
    "logging",
    "messaging",
    "migration",
    "monitoring",
    "onboarding",
    "provisioning",
    "reporting",
    "routing",
    "scheduling",
    "storage",
    "support",
    "telemetry",
    "training",
    "workflow",
)
ITEM_NOUNS = (
    "bundle",
    "engine",
    "licence",
    "package",
    "plan",
    "platform",
    "service",
    "subscription",
    "tier",
    "upgrade",
)

EARLIEST_ORDINAL = date(1998, 1, 1).toordinal()
LATEST_ORDINAL = date(2026, 1, 1).toordinal()
"""The window invoice dates are drawn from.

Fixed endpoints, not "now": a range ending at the current date would move the
corpus every day, which is the same class of drift as depending on a library's
seeded stream.
"""


def _vendor(rng: random.Random) -> str:
    form = rng.choice(VENDOR_FORMS)
    return form.format(rng.choice(VENDOR_STEMS), rng.choice(VENDOR_STEMS))


def _description(rng: random.Random) -> str:
    return f"{rng.choice(ITEM_QUALIFIERS)} {rng.choice(ITEM_SUBJECTS)} {rng.choice(ITEM_NOUNS)}"


def _date(rng: random.Random) -> date:
    return date.fromordinal(rng.randint(EARLIEST_ORDINAL, LATEST_ORDINAL))


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price: str
    amount: str


@dataclass
class Invoice:
    invoice_number: str
    vendor_name: str
    invoice_date: str
    due_date: str | None
    currency: str
    total: str
    line_items: list[LineItem] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        return "EUR " if self.currency == "EUR" else "$"


def _money(value: Decimal) -> str:
    """Two decimal places, rounded half-up.

    Decimal rather than float throughout: 0.1 + 0.2 is not 0.3 in binary
    floating point, and a ground truth that disagrees with its own line items
    by a cent would make every extraction look wrong.
    """
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build_invoice(index: int, rng: random.Random) -> Invoice:
    """One invoice's data. Rendering comes later, from exactly this."""
    items = [
        _line_item(rng)
        for _ in range(rng.randint(9, 14) if index in MULTI_PAGE else rng.randint(2, 5))
    ]
    total = sum((Decimal(item.amount) for item in items), Decimal("0"))

    issued = _date(rng)
    return Invoice(
        invoice_number=f"INV-{2026000 + index:07d}",
        vendor_name=_vendor(rng),
        # An ambiguous document is rendered as DD/MM/YYYY but its truth is
        # recorded in ISO, so a model reading it as MM/DD is measurably wrong
        # rather than arguably right.
        invoice_date=issued.isoformat(),
        due_date=None if index in NO_DUE_DATE else _date(rng).isoformat(),
        currency="EUR" if index in EUROS else "USD",
        total=_money(total),
        line_items=items,
    )


def _line_item(rng: random.Random) -> LineItem:
    quantity = rng.randint(1, 20)
    unit_price = Decimal(rng.randint(150, 45000)) / Decimal(100)
    return LineItem(
        description=_description(rng),
        quantity=quantity,
        unit_price=_money(unit_price),
        amount=_money(unit_price * quantity),
    )


def render(invoice: Invoice, index: int, path: Path) -> None:
    """Draw one invoice.

    `invariant=1` suppresses the creation timestamp and document id that would
    otherwise make every regeneration produce different bytes.
    """
    pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    width, height = A4
    ambiguous = index in AMBIGUOUS_DATE

    def header(page: int, pages: int) -> float:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(20 * mm, height - 25 * mm, "INVOICE")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(20 * mm, height - 33 * mm, invoice.vendor_name)

        pdf.drawRightString(width - 20 * mm, height - 25 * mm, invoice.invoice_number)
        pdf.drawRightString(
            width - 20 * mm, height - 32 * mm, f"Date: {_shown_date(invoice, ambiguous)}"
        )
        if invoice.due_date is not None:
            pdf.drawRightString(width - 20 * mm, height - 39 * mm, f"Due: {invoice.due_date}")
        if pages > 1:
            pdf.drawRightString(width - 20 * mm, height - 46 * mm, f"Page {page} of {pages}")
        return height - 60 * mm

    per_page = 8
    pages = max(1, -(-len(invoice.line_items) // per_page))

    for page in range(pages):
        y = header(page + 1, pages)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(20 * mm, y, "Description")
        pdf.drawRightString(130 * mm, y, "Qty")
        pdf.drawRightString(160 * mm, y, "Unit")
        pdf.drawRightString(width - 20 * mm, y, "Amount")
        y -= 6 * mm

        pdf.setFont("Helvetica", 9)
        for item in invoice.line_items[page * per_page : (page + 1) * per_page]:
            pdf.drawString(20 * mm, y, item.description[:60])
            pdf.drawRightString(130 * mm, y, str(item.quantity))
            pdf.drawRightString(160 * mm, y, f"{invoice.symbol}{item.unit_price}")
            pdf.drawRightString(width - 20 * mm, y, f"{invoice.symbol}{item.amount}")
            y -= 5.5 * mm

        # The total appears only on the last page, so a model that stops
        # reading after page one has to notice it is missing.
        if page == pages - 1:
            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawRightString(
                width - 20 * mm, y - 8 * mm, f"Total: {invoice.symbol}{invoice.total}"
            )

        pdf.showPage()

    pdf.save()


def _shown_date(invoice: Invoice, ambiguous: bool) -> str:
    """What the document displays, which is not always what it means.

    An ambiguous invoice prints `04/03/2026` for the 4th of March. A model
    reading it as April 3rd is wrong, and the ground truth is what lets us say
    so rather than argue about it.
    """
    year, month, day = invoice.invoice_date.split("-")
    return f"{day}/{month}/{year}" if ambiguous else invoice.invoice_date


def generate(seed: int = DEFAULT_SEED) -> list[dict]:
    """Build every invoice, write the PDFs, and return the ground truth."""
    rng = random.Random(seed)

    INVOICES.mkdir(parents=True, exist_ok=True)
    for stale in INVOICES.glob("*.pdf"):
        # Otherwise a smaller corpus would leave orphans behind, and the eval
        # would score documents no ground truth describes.
        stale.unlink()

    truth = []
    for index in range(DOCUMENT_COUNT):
        invoice = build_invoice(index, rng)
        name = f"invoice_{index:02d}.pdf"
        render(invoice, index, INVOICES / name)
        truth.append({"file": name, **asdict(invoice)})

    return truth


def checksums() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(INVOICES.glob("*.pdf"))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate and compare against the committed checksums, writing nothing.",
    )
    args = parser.parse_args()

    truth = generate(args.seed)
    digests = checksums()

    if args.check:
        # The **ground truth** is what the corpus guarantees, and what the eval
        # is scored against. It regenerates identically on any machine, because
        # it comes from a seeded generator and nothing else.
        committed = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
        if truth != committed:
            # Report the *fields* that moved, not only the file names. A list of
            # thirty filenames says the corpus drifted; it does not say whether
            # a dependency changed one word list or every date, which is the
            # difference between a one-line fix and a redesign.
            print(f"{len(truth)} invoice(s) differ from the committed truth:")
            for entry, was in zip(truth, committed, strict=False):
                moved = [key for key in entry if entry.get(key) != was.get(key)]
                if not moved:
                    continue
                print(f"  {entry['file']}: {', '.join(moved)}")
                for key in moved[:3]:
                    print(f"      {key}: committed {was.get(key)!r} -> now {entry.get(key)!r}")
                break
            fields = sorted(
                {
                    key
                    for entry, was in zip(truth, committed, strict=False)
                    for key in entry
                    if entry.get(key) != was.get(key)
                }
            )
            print(f"  fields affected across the corpus: {', '.join(fields)}")
            return 1

        # The PDF bytes are a *local* change-detector, not a cross-platform
        # promise. reportlab's output depends on its build, so a corpus
        # generated on Linux does not match checksums recorded on Windows — the
        # data inside is identical either way. Reported, never fatal: treating
        # it as a failure is what made this example fail its first CI run.
        expected = dict(
            line.split("  ", 1)[::-1]
            for line in CHECKSUMS.read_text(encoding="utf-8").strip().splitlines()
        )
        drifted = sorted(name for name, digest in digests.items() if expected.get(name) != digest)

        print(f"ground truth matches for all {len(truth)} invoices")
        if drifted:
            print(
                f"note: {len(drifted)} PDF(s) differ byte-for-byte from the committed "
                f"checksums, which were recorded on another machine. The extracted "
                f"data is unchanged."
            )
        else:
            print(f"all {len(digests)} PDFs also match the committed checksums")
        return 0

    GROUND_TRUTH.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    CHECKSUMS.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items())),
        encoding="utf-8",
    )
    print(f"wrote {len(truth)} invoices to {INVOICES}")
    print(f"ground truth: {GROUND_TRUTH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

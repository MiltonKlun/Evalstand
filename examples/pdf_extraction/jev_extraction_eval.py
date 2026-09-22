"""The invoice corpus, extracted by span selection with Jev.

The same corpus and the same scalar ground truth as `extraction_eval.py`, by a
different method: a regex finds candidate spans and TypeSafe's `jev-1.13.0`
selects among them (see `jev_extract.py`). The model never writes a value, only
picks one the document already contains.

    export TYPESAFE_API_KEY=...
    evalstand run jev_extraction_eval.py

**What it does not measure: line items.** Jev selects; it does not generate, so
it cannot produce a list of `{description, quantity, unit_price, amount}`
records. `extraction_eval.py` scores line items with an LLM judge; this eval
does not score them at all, and a baseline drawn from it covers the six scalar
fields only. Comparing its means against the generative eval's would be
comparing two different questions.

Jev is not reached through LiteLLM, so its calls are not traced automatically.
Each one is recorded with `record_call`, carrying its exact cost — input tokens
at $0.042 per million, output free — so the run's total is a figure rather than
the lower bound an unpriced call would leave.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

from evalstand import Case, evaluate
from evalstand.models import Score
from evalstand.scorers import close_to, json_fields
from evalstand.tracing import record_call

HERE = Path(__file__).parent
GROUND_TRUTH = HERE / "ground_truth.json"


def _helpers() -> Any:
    """`jev_extract.py`, loaded by path.

    Not imported by name: an eval file is imported by evalstand from wherever it
    lives, and whether its directory is on `sys.path` at that moment depends on
    how it was collected. Loading by path does not.
    """
    name = "_evalstand_example_jev_extract"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "jev_extract.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


jev = _helpers()


def load_cases() -> list[Case]:
    """One case per invoice, expecting the six scalar fields.

    Raises on a missing corpus: an eval reporting 0 cases looks like a pass.
    """
    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH.name} is missing. Run `python generate.py --seed 42` first."
        )

    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    return [
        Case(
            id=entry["file"].removesuffix(".pdf"),
            input=entry["file"],
            expected={field: entry[field] for field in jev.FIELDS},
            metadata={"total": entry["total"]},
        )
        for entry in truth
    ]


@cache
def _client() -> Any:
    """One client for the run.

    Built on first use rather than at import, so collecting this file — or
    listing it with `--collect-only` — never needs a key.
    """
    from typesafe_sdk import TypeSafeClient

    return TypeSafeClient()


def extract(pdf_name: str) -> dict[str, Any]:
    """Select each field's span, and record the call in the case's trace."""
    got = jev.extract(pdf_name, _client())

    record_call(
        name="jev.system_one",
        model=jev.MODEL,
        duration_ms=round(got.seconds * 1000),
        input={"questions": list(got.picks)},
        output={"picks": got.picks, "confidences": got.confidences},
        input_tokens=got.input_tokens,
        output_tokens=got.output_tokens,
        cost_usd=got.cost_usd,
    )
    return got.fields


fields = json_fields()


def total_within_a_cent(output: Any, expected: Any, case: Case) -> Score:
    """The total, allowing a cent of rounding — the same scorer the generative
    eval uses, so the two runs agree on what "right" means for this field."""
    result = close_to(abs_tol=0.01)(output.get("total"), case.metadata["total"])
    return result.model_copy(update={"scorer_name": "total"})


evaluate(
    name="invoice-extraction-jev",
    cases=load_cases,
    task=extract,
    scorers=[fields, total_within_a_cent],
)

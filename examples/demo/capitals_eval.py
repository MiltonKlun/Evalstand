"""The eval the README's demo GIF records.

Deliberately **offline**: the task answers from a dict and sleeps briefly rather
than calling a provider. A recording that needed an API key would cost money to
re-record, drift as the model changed, and could not run in CI — and none of
that would make the demo any more honest, because what it demonstrates is the
*interface*, not a model's accuracy.

The sleeps are what make the recording show anything. Without them every case
completes in the same frame and the GIF shows a finished table, which is
precisely the impression this phase exists to correct.

The showcase example with real measurements is `examples/pdf_extraction/`.
"""

from __future__ import annotations

import asyncio

from evalstand import evaluate, trace
from evalstand.models import Score

CAPITALS = {
    "france": "Paris",
    "japan": "Tokyo",
    "peru": "Lima",
    "kenya": "Nairobi",
    "norway": "Oslo",
    "chile": "Santiago",
}

# One deliberate mistake, so the demo shows a failure and the `f` filter has
# something to find. A demo where everything passes teaches nothing about what
# the tool is for.
ANSWERS = {**CAPITALS, "peru": "Cusco"}

CASES = [
    {"id": country, "input": country, "expected": capital} for country, capital in CAPITALS.items()
]


async def answer(country: str) -> str:
    """Look up a capital, slowly enough to watch.

    Wrapped in a `trace` so the detail view has a tree to show — the same
    mechanism a real task's LLM calls flow through.
    """
    with trace("lookup"):
        await asyncio.sleep(0.4)
        return ANSWERS.get(country, "unknown")


def exact(output: str, expected: str) -> Score:
    correct = output == expected
    return Score(
        scorer_name="exact",
        value=1.0 if correct else 0.0,
        passed=correct,
        metadata={"expected": expected, "got": output},
    )


evaluate(
    name="capitals",
    cases=CASES,
    task=answer,
    scorers=[exact],
    columns={"length": lambda result: len(str(result.output))},
)

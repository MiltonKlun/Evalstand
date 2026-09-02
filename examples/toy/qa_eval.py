"""The toy example: three cases, one scorer, no external files.

Every subsequent phase develops against this. It uses a task that does no model
call, so it runs anywhere with no API key and no spend — the point here is the
authoring and collection experience, not the model.
"""

from evalstand import Case, evaluate
from evalstand.scorers import exact

CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Peru": "Lima",
}


def load_cases() -> list[Case]:
    return [
        Case(id="france", input="What is the capital of France?", expected="Paris"),
        Case(id="japan", input="What is the capital of Japan?", expected="Tokyo"),
        Case(id="peru", input="What is the capital of Peru?", expected="Lima"),
    ]


def answer(question: str) -> str:
    """Stand in for a model call, so the example runs offline."""
    for country, capital in CAPITALS.items():
        if country in question:
            return capital
    return "I don't know"


evaluate(
    name="toy-capitals",
    cases=load_cases,
    task=answer,
    scorers=[exact],
)

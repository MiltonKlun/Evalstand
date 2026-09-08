"""evalstand — a local-first LLM evaluation tool for Python.

Evaluating an LLM application should feel like running a test suite: write an
eval file, run a watch command, and watch results stream in with scores,
traces, token counts, and cost.

The public surface is deliberately seven names. `Run` and `Batch` are not among
them: users read those in reports, they never construct them. Widening this list
needs an ADR — the cap exists to force the question.
"""

from evalstand.api import evaluate, scorer
from evalstand.models import Case, Result, Score, Trace
from evalstand.tracing import trace

__version__ = "0.0.0.dev0"

__all__ = [
    "Case",
    "Result",
    "Score",
    "Trace",
    "evaluate",
    "scorer",
    "trace",
]

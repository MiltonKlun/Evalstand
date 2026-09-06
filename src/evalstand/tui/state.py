"""What the run view is showing, independent of any terminal.

The widgets are a projection of this, not the other way round. That split is
what makes the UI testable: a Textual app needs a terminal, an event loop and a
driver to assert anything about, whereas the question that actually matters —
*does this row say the right thing about this Result* — is a pure function and
should be tested like one.

The honesty rules are the same as `reporting/console.py`, and the formatting
helpers are imported from it rather than reimplemented. Two renderers that
disagree about whether an unpriced call costs `$0.0000` or `-` would put a
number on screen that the summary contradicts, and nothing would say which is
true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalstand.models import Result, Run
from evalstand.reporting.console import UNKNOWN

__all__ = ["CaseRow", "RunState", "Totals"]

_STATUS_UNMEASURED = "unmeasured"
"""Scored, but every scorer errored. Distinct from a failure: the task may have
answered perfectly and the scorer broke."""


@dataclass(frozen=True)
class CaseRow:
    """One row of the results table.

    Frozen, and built from a Result rather than holding one, so a row cannot
    drift from what was measured — and so the table can be re-sorted or filtered
    without carrying the whole trace tree around with every row.
    """

    case_id: str
    repeat_index: int
    status: str
    score: str
    latency: str
    cost: str
    extra: dict[str, str] = field(default_factory=dict)
    """User-defined columns from `evaluate(columns=...)`, already rendered."""

    @property
    def key(self) -> str:
        """A stable identity for the row, matching the Result's own.

        Includes the repeat index: three repeats of one case are three rows, and
        keying on `case_id` alone would make each overwrite the last — a table
        showing one row where three executions happened.
        """
        return f"{self.case_id}#{self.repeat_index}"


def _status_of(result: Result) -> str:
    """What happened to this case, in one word.

    The four states are kept apart deliberately. "failed" and "unmeasured" look
    alike in a table and mean opposite things: one is the task getting the
    answer wrong, the other is the scorer breaking and the task's performance
    being unknown. Folding them together would report a scorer outage as a model
    regression.
    """
    if result.error:
        return "error"
    if result.scores and all(score.error for score in result.scores):
        return _STATUS_UNMEASURED

    verdicts = [score.passed for score in result.scores if score.passed is not None]
    if not verdicts:
        # Continuous scorers give a value and no verdict. Inventing one from the
        # value is the inference CONTEXT.md forbids.
        return "scored"
    return "pass" if all(verdicts) else "fail"


def _score_of(result: Result) -> str:
    return UNKNOWN if result.mean_score is None else f"{result.mean_score:.2f}"


def _latency_of(result: Result) -> str:
    return UNKNOWN if result.latency_ms is None else f"{result.latency_ms}ms"


def _cost_of(result: Result) -> str:
    """This case's cost, or unknown when nothing it did was priced.

    A case whose calls were never priced has not been shown to be free, so it
    shows a placeholder. A case that genuinely made no calls costs nothing and
    says so.
    """
    priced = [trace for trace in result.traces if trace.cost_usd is not None]
    if result.traces and not priced:
        return UNKNOWN
    return f"${sum(t.cost_usd or 0.0 for t in priced):.4f}"


def _extra_of(result: Result, columns: dict[str, Any]) -> dict[str, str]:
    """Render the user's derived columns.

    A column that raises shows `!` rather than taking down the row. The user
    asked to measure a model, and a bug in a display helper must not cost them
    the measurement — the same rule the runner applies to sinks and scorers.
    """
    rendered: dict[str, str] = {}
    for name, fn in columns.items():
        try:
            value = fn(result)
        except Exception:
            rendered[name] = "!"
            continue
        rendered[name] = UNKNOWN if value is None else str(value)
    return rendered


def row_for(result: Result, columns: dict[str, Any] | None = None) -> CaseRow:
    """Project one Result into one row."""
    return CaseRow(
        case_id=result.case_id,
        repeat_index=result.repeat_index,
        status=_status_of(result),
        score=_score_of(result),
        latency=_latency_of(result),
        cost=_cost_of(result),
        extra=_extra_of(result, columns or {}),
    )


@dataclass(frozen=True)
class Totals:
    """The footer: what the run has done so far.

    Every field is over the Results that have *landed*, not the ones expected,
    so the footer is true at every instant rather than only at the end.
    """

    completed: int
    expected: int | None
    passed: int
    judged: int
    errors: int
    cost_usd: float
    unpriced_calls: int
    model_calls: int
    cache_hits: int

    @property
    def progress(self) -> float | None:
        """Fraction done, or None when the total is not yet known.

        None rather than 0.0: a run whose case count is still loading has no
        progress to report, and a bar sitting at 0% would be a claim that
        nothing has happened.
        """
        if not self.expected:
            return None
        return min(1.0, self.completed / self.expected)

    @property
    def cost(self) -> str:
        """Total spend, marked as a lower bound when calls went unpriced.

        `Run.total_cost_usd` sums what is known, so with unpriced calls the true
        figure is higher. Showing it bare would understate the user's bill —
        the same class of error as a false pass, quietly wrong in the direction
        that costs money.
        """
        if not self.model_calls and not self.cost_usd:
            return UNKNOWN
        bound = f" (+{self.unpriced_calls} unpriced)" if self.unpriced_calls else ""
        return f"${self.cost_usd:.4f}{bound}"

    @property
    def pass_rate(self) -> str:
        """Passes over *judged* cases.

        Unjudged cases are excluded from the denominator rather than counted as
        failures: a continuous scorer that declined to judge has not failed
        anything.
        """
        if not self.judged:
            return UNKNOWN
        return f"{self.passed}/{self.judged}"

    @property
    def cache_hit_rate(self) -> str:
        if not self.model_calls:
            return UNKNOWN
        return f"{self.cache_hits / self.model_calls:.0%}"


class RunState:
    """The live state of one eval executing.

    Accumulates Results as they land and answers what the widgets need to
    paint. Deliberately not a Textual object: this is where the logic that can
    be wrong lives, and it is tested without a terminal.
    """

    def __init__(
        self,
        name: str,
        *,
        expected: int | None = None,
        columns: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.expected = expected
        self.columns = columns or {}
        self.results: list[Result] = []
        self._seen: set[tuple[str, int]] = set()

    def add(self, result: Result) -> CaseRow | None:
        """Record a landed Result, returning the row to paint.

        A duplicate `(case_id, repeat_index)` is refused rather than appended.
        The same guarantee `Run` enforces: that pair names one execution, and
        two rows for it would make the footer's count disagree with the table
        above it. Returns None so the caller paints nothing.
        """
        key = (result.case_id, result.repeat_index)
        if key in self._seen:
            return None

        self._seen.add(key)
        self.results.append(result)
        return row_for(result, self.columns)

    def rows(self) -> list[CaseRow]:
        return [row_for(result, self.columns) for result in self.results]

    def failures(self) -> list[CaseRow]:
        """Rows worth a second look: a wrong answer, a crash, or no measurement.

        `scored` is excluded — a continuous score without a verdict is not a
        failure, and putting it in a failures filter would invent the verdict
        the scorer declined to give.
        """
        return [row for row in self.rows() if row.status in {"fail", "error", _STATUS_UNMEASURED}]

    def totals(self) -> Totals:
        """The footer, computed from what has landed."""
        judged = [
            result
            for result in self.results
            if any(score.passed is not None for score in result.scores)
        ]
        passed = [
            result for result in judged if all(score.passed is not False for score in result.scores)
        ]
        traces = [trace for result in self.results for trace in result.traces]

        return Totals(
            completed=len(self.results),
            expected=self.expected,
            passed=len(passed),
            judged=len(judged),
            errors=sum(1 for result in self.results if result.error),
            cost_usd=sum(trace.cost_usd or 0.0 for trace in traces),
            unpriced_calls=sum(1 for trace in traces if trace.cost_usd is None),
            model_calls=sum(1 for trace in traces if trace.model is not None),
            cache_hits=sum(1 for trace in traces if trace.name == "cached call"),
        )

    def mean_scores_by_scorer(self) -> dict[str, float]:
        """Per-scorer means over what has landed.

        Per scorer rather than one overall figure, for the reason
        `Run.mean_scores_by_scorer` gives: averaging across scorers that measure
        different things produces a number that means nothing.
        """
        buckets: dict[str, list[float]] = {}
        for result in self.results:
            for score in result.scores:
                if score.counts_towards_mean and score.value is not None:
                    buckets.setdefault(score.scorer_name, []).append(score.value)
        return {name: sum(values) / len(values) for name, values in buckets.items()}

    def result_for(self, key: str) -> Result | None:
        """The Result behind a row, for the detail view."""
        for result in self.results:
            if f"{result.case_id}#{result.repeat_index}" == key:
                return result
        return None

    def as_run(self, run: Run) -> Run:
        """Adopt the finished Run, so the view shows stored truth once it exists.

        The accumulated Results and the Run's should be identical; taking the
        Run's is what guarantees the view cannot drift from what was recorded.
        """
        self.results = list(run.results)
        self._seen = {(r.case_id, r.repeat_index) for r in run.results}
        return run

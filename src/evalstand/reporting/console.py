"""Console reporting.

The summary is where a user learns what a run did, so the governing rule is that
it never states a figure the data does not support. A mean over scores that
errored, a cost of $0.00 for calls that were never priced, or a pass rate that
counts an absent judgement as a failure would each be a confident-looking lie.

Where a number cannot honestly be given, this prints a placeholder rather
than a zero.
"""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from evalstand.models import Batch, BatchStatus, Run

__all__ = ["render_failures", "render_summary"]

UNKNOWN = "-"
"""Shown where a value is genuinely unknown. Never 0, which is a claim.

An ASCII hyphen rather than an em dash: consoles that cannot encode it render a
replacement character, and a summary that looks corrupted undermines trust in
the numbers beside it.
"""

_PARTIAL_STATUSES = frozenset({BatchStatus.CANCELLED, BatchStatus.FAILED})

_MAX_CELL = 60


def _truncate(value: Any, limit: int = _MAX_CELL) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _pass_counts(run: Run) -> tuple[int, int]:
    """Passes and judged results.

    Only Results carrying an explicit pass/fail are counted. A continuous scorer
    that declined to judge is not a failure, and a Result with no scores at all
    is not a pass; folding either into the denominator would misstate the rate.
    """
    judged = [r for r in run.results if any(s.passed is not None for s in r.scores)]
    passed = [r for r in judged if all(s.passed is not False for s in r.scores)]
    return len(passed), len(judged)


def _format_score(value: float | None) -> str:
    return UNKNOWN if value is None else f"{value:.2f}"


def _format_cost(runs: list[Run]) -> str:
    """Total cost, or unknown when nothing was priced.

    A run whose calls were never priced has not been shown to be free.
    """
    priced = [
        trace
        for run in runs
        for result in run.results
        for trace in result.traces
        if trace.cost_usd is not None
    ]
    if not priced:
        return UNKNOWN
    return f"${sum(t.cost_usd or 0.0 for t in priced):.4f}"


def render_summary(
    runs: list[Run],
    *,
    wall_seconds: float | None = None,
    batch: Batch | None = None,
) -> RenderableType:
    """A per-eval summary: scorer means, pass count, cost, and wall time."""
    parts: list[RenderableType] = []

    if batch is not None:
        label = Text(f"batch: {batch.kind.value}", style="dim")
        if batch.status in _PARTIAL_STATUSES:
            # A cancelled or failed batch's numbers are partial; saying so stops
            # them being read as a finished result. A running batch is normal
            # and needs no warning.
            label.append(f"  ({batch.status.value}: results are partial)", style="yellow")
        parts.append(label)

    table = Table(title=None, show_header=True, header_style="bold", expand=False)
    table.add_column("eval")
    table.add_column("scorer")
    table.add_column("mean", justify="right")
    table.add_column("passed", justify="right")
    table.add_column("cost", justify="right")

    for run in runs:
        means = run.mean_scores_by_scorer()
        passed, judged = _pass_counts(run)
        pass_cell = f"{passed}/{judged}" if judged else UNKNOWN
        cost_cell = _format_cost([run])

        if not means:
            table.add_row(run.name, UNKNOWN, UNKNOWN, pass_cell, cost_cell)
            continue

        for index, (scorer_name, mean) in enumerate(sorted(means.items())):
            table.add_row(
                run.name if index == 0 else "",
                scorer_name,
                _format_score(mean),
                pass_cell if index == 0 else "",
                cost_cell if index == 0 else "",
            )

    parts.append(table)

    footer_bits: list[str] = []
    errored = sum(run.errored_score_count for run in runs)
    if errored:
        # A mean computed over fewer scores than expected is a different claim
        # from one computed over all of them, so the gap is always stated.
        footer_bits.append(f"{errored} errored {'score' if errored == 1 else 'scores'} excluded")
    if wall_seconds is not None:
        footer_bits.append(f"{wall_seconds:.1f}s")
    total_cost = _format_cost(runs)
    if total_cost != UNKNOWN and len(runs) > 1:
        footer_bits.append(f"total {total_cost}")

    if footer_bits:
        parts.append(Text("  |  ".join(footer_bits), style="dim"))

    return Group(*parts)


def render_failures(runs: list[Run]) -> RenderableType | None:
    """Cases that failed, errored, or could not be scored. None when all passed.

    A task error and a low score are different events, so the reason column
    names which one happened rather than showing both as a failure.
    """
    rows: list[tuple[str, str, str, str]] = []

    for run in runs:
        for result in run.results:
            if result.error:
                rows.append((run.name, result.case_id, "task error", _truncate(result.error)))
                continue

            errored = [s for s in result.scores if s.error]
            if errored:
                rows.append(
                    (
                        run.name,
                        result.case_id,
                        "scorer error",
                        _truncate(f"{errored[0].scorer_name}: {errored[0].error}"),
                    )
                )
                continue

            failed = [s for s in result.scores if s.passed is False]
            if failed:
                rows.append(
                    (
                        run.name,
                        result.case_id,
                        f"failed {', '.join(s.scorer_name for s in failed)}",
                        _truncate(result.output),
                    )
                )

    if not rows:
        return None

    table = Table(title="failures", show_header=True, header_style="bold red", expand=False)
    table.add_column("eval")
    table.add_column("case")
    table.add_column("reason")
    table.add_column("output")
    for row in rows:
        table.add_row(*row)
    return table

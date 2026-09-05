"""Console reporting.

The summary is where a user learns what a run did, so the governing rule is that
it never states a figure the data does not support. A mean over scores that
errored, a cost of $0.00 for calls that were never priced, or a pass rate that
counts an absent judgement as a failure would each be a confident-looking lie.

Where a number cannot honestly be given, this prints a placeholder rather
than a zero.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from evalstand.comparison import (
    AmendedCase,
    Comparison,
    Flip,
    MeasurementChange,
    ScoreMove,
    ScorerDelta,
)
from evalstand.models import Batch, BatchStatus, Result, Run, Score, Trace

__all__ = [
    "render_comparison",
    "render_failures",
    "render_history",
    "render_run_detail",
    "render_summary",
]

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

    tokens_in = sum(r.input_tokens or 0 for run in runs for r in run.results)
    tokens_out = sum(r.output_tokens or 0 for run in runs for r in run.results)
    if tokens_in or tokens_out:
        footer_bits.append(f"{tokens_in} in / {tokens_out} out tokens")

    footer_bits.extend(_cache_note(runs))

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


def _cache_note(runs: list[Run]) -> list[str]:
    """What the cache did, or why it did nothing.

    A bypassed run paid for every call. Showing a 0% hit rate would read as a
    cache that missed, when in truth it was deliberately skipped — and the plan
    calls repeats the easiest way to run up a bill by accident, so the number of
    calls actually paid for is stated.
    """
    calls = sum(run.model_calls for run in runs)
    if not calls:
        return []

    if any(run.cache_bypassed for run in runs):
        return [f"cache bypassed: {calls} calls made"]

    hits = sum(run.cache_hits for run in runs)
    return [f"cache {hits}/{calls} ({hits / calls:.0%})"]


def render_failures(runs: list[Run]) -> RenderableType | None:
    """Cases worth looking at: failed, errored, unscorable, or scored zero.

    Each is a different event, and the reason column names which one happened
    rather than flattening them into "failed". A scorer that declined to give a
    verdict must not have one put in its mouth by the report.
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
                continue

            # A case that scored zero without a verdict. Continuous scorers do
            # not set `passed` — correctly, since they do not know where the
            # line sits — so listing only `passed is False` made the worst
            # possible result invisible: a model answering every case with
            # garbage produced an empty failures table.
            #
            # Reported as "scored 0.00" rather than "failed": the scorer
            # declined to give a verdict and this table must not invent one. It
            # says what happened and leaves the judgement to the reader.
            zeroed = [s for s in result.scores if s.passed is None and s.value == 0.0]
            if zeroed:
                rows.append(
                    (
                        run.name,
                        result.case_id,
                        f"scored 0.00 on {', '.join(s.scorer_name for s in zeroed)}",
                        _truncate(result.output),
                    )
                )

    if not rows:
        return None

    table = Table(title="needs attention", show_header=True, header_style="bold red", expand=False)
    table.add_column("eval")
    table.add_column("case")
    table.add_column("reason")
    table.add_column("output")
    for row in rows:
        table.add_row(*row)
    return table


def render_history(entries: list[tuple[Run, Batch | None]]) -> RenderableType:
    """Past runs, newest first: when, from which commit, and what they measured.

    Every figure comes from the Run object's own properties, which are the same
    ones the live summary uses. Recomputing them in SQL would create a second
    definition of "mean" that could drift from the first, and a history table
    quietly disagreeing with a fresh run is the exact failure this project keeps
    finding.
    """
    table = Table(title=None, show_header=True, header_style="bold", expand=False)
    table.add_column("run")
    table.add_column("eval")
    table.add_column("when")
    table.add_column("commit")
    table.add_column("mean", justify="right")
    table.add_column("passed", justify="right")
    table.add_column("cost", justify="right")

    for run, batch in entries:
        passed, judged = _pass_counts(run)
        table.add_row(
            run.id,
            run.name,
            _when(run.started_at),
            _commit(batch),
            _format_score(run.mean_score),
            f"{passed}/{judged}" if judged else UNKNOWN,
            _run_cost(run),
        )

    return table


def _when(moment: datetime | None) -> str:
    """A timestamp a human can scan. Local time, because history is read by the
    person who produced it and UTC would make them do arithmetic."""
    if moment is None:
        return UNKNOWN
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def _commit(batch: Batch | None) -> str:
    """The commit a run came from, or an honest gap.

    A dirty tree is marked, because a run recorded against a SHA whose tree it
    did not reflect looks reproducible and is not.
    """
    if batch is None or batch.git_sha is None:
        return UNKNOWN
    short = batch.git_sha[:8]
    return f"{short}*" if batch.git_dirty else short


def _run_cost(run: Run) -> str:
    """One run's cost, marked when it is only part of the story.

    `total_cost_usd` sums what was priced, so a run with unpriced calls reports
    a lower bound. Showing that as an exact figure would understate a bill --
    the same class of error as a false pass, and in the direction that costs
    money. The `+` says "at least this much".
    """
    priced = [
        trace for result in run.results for trace in result.traces if trace.cost_usd is not None
    ]
    if not priced:
        # Either no calls were made, or none could be priced. Neither has been
        # shown to be free, and `$0.0000` would claim exactly that — the same
        # rule `_format_cost` already applies to the live summary.
        return UNKNOWN

    formatted = f"${run.total_cost_usd:.4f}"
    return formatted if run.cost_is_complete else f"{formatted}+"


def render_run_detail(
    run: Run,
    batch: Batch | None = None,
    *,
    full: bool = False,
) -> RenderableType:
    """One run in full: what it was, what each case did, and what it called.

    `full` prints the prompts and completions themselves. They are withheld by
    default because one 4000-token prompt fills a screen and buries the tree it
    belongs to — and because a trace input can hold a customer record that
    nobody meant to display on a shared terminal.
    """
    parts: list[RenderableType] = [_run_header(run, batch)]

    for result in run.results:
        parts.append(Text())
        parts.append(_case_panel(result, full=full))

    return Group(*parts)


def _run_header(run: Run, batch: Batch | None) -> RenderableType:
    """The run's identity and headline numbers."""
    table = Table(title=None, show_header=False, box=None, expand=False)
    table.add_column(style="dim")
    table.add_column()

    table.add_row("run", run.id)
    table.add_row("eval", f"{run.name}  ({run.filepath})")
    table.add_row("when", _when(run.started_at))
    table.add_row("commit", _commit(batch))
    if run.task_source_hash:
        # Short, because its only use is comparing two runs at a glance.
        table.add_row("task", run.task_source_hash[:12])
    table.add_row("mean", _format_score(run.mean_score))

    passed, judged = _pass_counts(run)
    table.add_row("passed", f"{passed}/{judged}" if judged else UNKNOWN)
    table.add_row("cost", _run_cost(run))

    if run.errored_score_count:
        # A mean over fewer scores than expected is a different claim from one
        # over all of them, so the gap is always stated.
        table.add_row(
            "note",
            Text(
                f"{run.errored_score_count} errored "
                f"{'score' if run.errored_score_count == 1 else 'scores'} excluded from the mean",
                style="yellow",
            ),
        )

    return table


def _case_panel(result: Result, *, full: bool) -> RenderableType:
    """One case: its verdict, its scores, and the calls it made."""
    parts: list[RenderableType] = [Text(result.case_id, style="bold")]

    if result.error:
        parts.append(Text(f"  task error: {result.error}", style="red"))
        # The frames were captured at the raise precisely so a stored failure
        # stays debuggable months later.
        parts.extend(Text(f"  {frame}", style="dim") for frame in result.error_frames)
    else:
        parts.append(Text(f"  output: {_truncate(result.output, 200 if full else _MAX_CELL)}"))

    for score in result.scores:
        parts.append(Text(f"  {_score_line(score)}"))

    if result.traces:
        parts.append(_trace_tree(result.traces, full=full))

    return Group(*parts)


def _score_line(score: Score) -> str:
    """One score, saying which of the three things it is.

    A verdict, a bare value and an error are different claims, and collapsing
    any pair of them is how a false pass gets made.
    """
    if score.error:
        return f"{score.scorer_name}: errored ({score.error})"

    value = UNKNOWN if score.value is None else f"{score.value:.3f}"
    if score.passed is None:
        # No verdict was given, and the report must not invent one.
        return f"{score.scorer_name}: {value}"
    return f"{score.scorer_name}: {value} ({'pass' if score.passed else 'fail'})"


def _trace_tree(traces: list[Trace], *, full: bool) -> RenderableType:
    """The calls a case made, as the tree they actually form.

    Built **iteratively**. A recursive walk overflows the stack at about a
    thousand levels, and losing a whole run's display to a judge that called a
    judge would be a poor way to fail. The Result validator has already
    guaranteed a forest — no cycles, no dangling parents — so this can trust
    the shape and only has to survive its size.

    Children may appear before their parents in the list, so the index is built
    in one pass before anything is walked.
    """
    children: dict[str | None, list[Trace]] = {}
    for trace in traces:
        children.setdefault(trace.parent_id, []).append(trace)

    tree = Tree("traces", guide_style="dim")

    # (parent renderable, trace) pairs, deepest last — a stack, so siblings
    # come out in the order they were recorded.
    pending: list[tuple[Tree, Trace]] = [
        (tree, trace) for trace in reversed(children.get(None, []))
    ]
    while pending:
        parent, trace = pending.pop()
        node = parent.add(_trace_label(trace, full=full))
        pending.extend((node, child) for child in reversed(children.get(trace.id, [])))

    return tree


def _trace_label(trace: Trace, *, full: bool) -> RenderableType:
    """One call: what it was and what it cost."""
    line = Text(trace.name, style="bold")
    if trace.model:
        line.append(f"  {trace.model}", style="cyan")
    line.append(f"  {trace.duration_ms}ms", style="dim")

    if trace.input_tokens is not None or trace.output_tokens is not None:
        tokens = f"{trace.input_tokens or 0} in / {trace.output_tokens or 0} out"
        line.append(f"  {tokens}", style="dim")

    # An unpriced call shows a placeholder, never $0.0000: it has not been
    # shown to be free.
    line.append(f"  {UNKNOWN if trace.cost_usd is None else f'${trace.cost_usd:.4f}'}", style="dim")

    if not full:
        # A size rather than the content. Enough to know something was sent,
        # without spraying a prompt across the terminal.
        for label, value in (("input", trace.input), ("output", trace.output)):
            if value is not None:
                line.append(f"\n{label}: {_describe(value)}", style="dim")
        return line

    for label, value in (("input", trace.input), ("output", trace.output)):
        if value is not None:
            line.append(f"\n{label}: {value!r}", style="dim")
    return line


def _describe(value: Any) -> str:
    """How much there is, without showing it."""
    if isinstance(value, list):
        return f"{len(value)} message{'' if len(value) == 1 else 's'} ({len(str(value))} chars)"
    return f"{len(str(value))} chars"


def render_comparison(comparison: Comparison) -> RenderableType:
    """Two runs, side by side, with no claim about what the difference means.

    Every word here is chosen to withhold a verdict. A delta is "changed", not
    "improved" or "regressed"; a flip is "pass -> fail", not "a regression".
    `evalstand` has no significance testing, so it cannot tell a real change
    from noise — and a tool that said "regression" without being able to
    support it would be worse than one that stayed quiet, because the word
    would be believed.
    """
    parts: list[RenderableType] = [_comparison_header(comparison)]

    parts.append(Text())
    parts.append(_delta_table(comparison.scorer_deltas))

    # First, because a case that did not run cannot also have flipped — and
    # because a run whose cases all stopped being measured is the finding, not
    # a footnote to one.
    if comparison.measurement_changes:
        parts.append(Text())
        parts.append(_measurement_table(comparison.measurement_changes))

    if comparison.flips:
        parts.append(Text())
        parts.append(_flip_table(comparison.flips))

    if comparison.moves:
        parts.append(Text())
        parts.append(_move_table(comparison.moves))

    if comparison.amended:
        parts.append(Text())
        parts.append(_amended_table(comparison.amended))

    if comparison.only_before or comparison.only_after:
        parts.append(Text())
        parts.append(_membership_note(comparison))

    if comparison.is_empty:
        parts.append(Text())
        parts.append(Text("nothing differs between these runs.", style="dim"))

    # Stated every time, not buried in documentation. A reader who takes a
    # delta as proof of a change has been misled by the tool, and the tool is
    # the only thing present at the moment they might do so.
    parts.append(Text())
    parts.append(
        Text(
            "evalstand has no significance testing: a delta is an arithmetic "
            "difference, not evidence of a real change.",
            style="dim",
        )
    )

    return Group(*parts)


def _comparison_header(comparison: Comparison) -> RenderableType:
    table = Table(title=None, show_header=False, box=None, expand=False)
    table.add_column(style="dim")
    table.add_column()

    before, after = comparison.before, comparison.after
    table.add_row("before", f"{before.id}  ({_when(before.started_at)})")
    table.add_row("after", f"{after.id}  ({_when(after.started_at)})")

    if before.name != after.name:
        # Two different evals measure different things, so their means are not
        # comparable at all. Said plainly rather than left for the reader to
        # notice in the ids.
        table.add_row(
            "note",
            Text(
                f"these are different evals ({before.name} and {after.name}); "
                f"their scores measure different things",
                style="yellow",
            ),
        )

    if (
        before.task_source_hash
        and after.task_source_hash
        and before.task_source_hash != after.task_source_hash
    ):
        # Without this a reader would attribute a code change to the model.
        table.add_row("note", Text("the task's source changed between these runs", style="yellow"))

    return table


def _delta_table(deltas: list[ScorerDelta]) -> RenderableType:
    table = Table(title="scorer means", show_header=True, header_style="bold", expand=False)
    table.add_column("scorer")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("changed by", justify="right")

    for delta in deltas:
        table.add_row(
            delta.scorer_name,
            _format_score(delta.before),
            _format_score(delta.after),
            # A scorer measured in only one run has not moved; the comparison
            # cannot be made, and a number here would invent one.
            f"{delta.delta:+.2f}" if delta.delta is not None else UNKNOWN,
        )

    return table


def _flip_table(flips: list[Flip]) -> RenderableType:
    table = Table(
        title="cases whose pass state changed",
        show_header=True,
        header_style="bold",
        expand=False,
    )
    table.add_column("case")
    table.add_column("was")
    table.add_column("now")
    table.add_column("output before")
    table.add_column("output after")

    for flip in flips:
        table.add_row(
            flip.case_id,
            "pass" if flip.passed_before else "fail",
            "pass" if flip.passed_after else "fail",
            _truncate(flip.output_before, 40),
            _truncate(flip.output_after, 40),
        )

    return table


def _measurement_table(changes: list[MeasurementChange]) -> RenderableType:
    """Cases that gained or lost a measurement.

    Separate from the flip table on purpose. "Was passing, now crashes" is not
    the task getting worse — the task did not run, and saying otherwise would
    attribute an infrastructure failure to the model.
    """
    table = Table(
        title="cases whose measurement changed",
        show_header=True,
        header_style="bold yellow",
        expand=False,
    )
    table.add_column("case")
    table.add_column("change")
    table.add_column("detail")

    for change in changes:
        table.add_row(
            change.case_id,
            change.description,
            _truncate(change.detail or "", 60),
        )

    return table


def _move_table(moves: list[ScoreMove]) -> RenderableType:
    """Cases whose score moved without any pass state to flip.

    Titled by what happened rather than by what it might mean: continuous
    scorers declined to give a verdict, and this table must not supply one on
    their behalf.
    """
    table = Table(
        title="largest score changes", show_header=True, header_style="bold", expand=False
    )
    table.add_column("case")
    table.add_column("scorer")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("changed by", justify="right")

    for move in moves:
        table.add_row(
            move.case_id,
            move.scorer_name,
            f"{move.before:.2f}",
            f"{move.after:.2f}",
            f"{move.delta:+.2f}",
        )

    return table


def _amended_table(amended: list[AmendedCase]) -> RenderableType:
    """Cases whose test itself changed.

    Kept apart from flips because a pass state that moved because the
    expectation moved says nothing about the task. Folding these in would
    attribute a dataset edit to the model.
    """
    table = Table(
        title="cases edited between these runs",
        show_header=True,
        header_style="bold",
        expand=False,
    )
    table.add_column("case")
    table.add_column("note")

    for case in amended:
        note = (
            "its input or expected value changed, and so did its pass state"
            if case.verdict_moved
            else "its input or expected value changed"
        )
        table.add_row(case.case_id, note)

    return table


def _membership_note(comparison: Comparison) -> RenderableType:
    """Cases present in one run and not the other.

    A mean computed over a different set of cases is a different measurement,
    so the reader is told the sets differ rather than left to assume they match.
    """
    lines: list[str] = []
    if comparison.only_before:
        lines.append(f"only in the earlier run: {', '.join(comparison.only_before)}")
    if comparison.only_after:
        lines.append(f"only in the later run: {', '.join(comparison.only_after)}")
    lines.append("the two runs did not cover the same cases, so their means are not like-for-like")
    return Text("\n".join(lines), style="yellow")

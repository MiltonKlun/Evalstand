"""Server-rendered HTML fragments for the HTMX front end (task 8.2).

HTMX swaps fragments the server renders; there is no client-side model, so
there is nothing to drift out of step with the server's, and no build step. The
stack stays entirely Python, which is what the plan asks for.

Every value is escaped through `reporting.html.escape`, and status comes from
`reporting.html.status_of`. Both are imported rather than reimplemented: a
completion containing `<script>` is an inevitability, not an attack, and a page
with its own escaping rule would eventually disagree with the artifact
renderer's about which values are safe.
"""

from __future__ import annotations

from evalstand.models import Result, Run
from evalstand.reporting.console import UNKNOWN, format_score, pass_counts
from evalstand.reporting.html import escape, result_cost, status_of


def run_row(run: Run) -> str:
    """One row of the history table.

    `hx-get` on the row rather than a link inside it: the whole row is the
    target, which is what a user expects from a table they can click.
    """
    passed, judged = pass_counts(run)
    pass_cell = f"{passed}/{judged}" if judged else UNKNOWN

    return (
        f"<tr class='row' hx-get='/ui/runs/{escape(run.id)}' "
        f"hx-target='#detail' hx-swap='innerHTML'>"
        f"<td>{escape(run.name)}</td>"
        f"<td class='mono'>{escape(_short(run.id))}</td>"
        f"<td>{escape(_when(run))}</td>"
        f"<td class='num'>{escape(format_score(run.mean_score))}</td>"
        f"<td class='num'>{escape(pass_cell)}</td>"
        f"<td class='num'>{escape(_run_cost(run))}</td>"
        f"</tr>"
    )


def run_table(runs: list[Run]) -> str:
    """The history table, or a statement that there is none.

    An empty database renders a sentence rather than an empty table. "No runs
    yet" and "runs exist but none matched" send a user looking in completely
    different places, and an empty `<tbody>` says neither.
    """
    if not runs:
        return (
            "<p class='note'>No runs recorded yet. "
            "Run <code>evalstand run</code> to record one.</p>"
        )

    rows = "".join(run_row(run) for run in runs)
    return (
        "<div class='scroll'><table><thead><tr>"
        "<th>eval</th><th>run</th><th>when</th>"
        "<th class='num'>mean</th><th class='num'>passed</th><th class='num'>cost</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def case_row(result: Result) -> str:
    """One case in the live table.

    `id` is the `(case_id, repeat)` pair rather than the Result id, because SSE
    swaps this row by id when a later repeat of the same case lands, and the
    Result id is not what the page knows the row by.
    """
    status = status_of(result)
    return (
        f"<tr id='case-{escape(result.case_id)}-{result.repeat_index}' class='row'>"
        f"<td class='mono'>{escape(result.case_id)}</td>"
        f"<td><span class='badge {status}'>{status}</span></td>"
        f"<td class='num'>{escape(_score_of(result))}</td>"
        f"<td class='num'>{escape(_latency(result))}</td>"
        f"<td class='num'>{escape(result_cost(result))}</td>"
        f"</tr>"
    )


def run_detail(run: Run) -> str:
    """One run opened: its totals, then every case.

    Cost carries its completeness beside it. `total_cost_usd` sums only the
    calls that could be priced, so the bare figure is a lower bound wearing the
    name of a total — a reader shown it alone understates their bill.
    """
    passed, judged = pass_counts(run)
    rows = "".join(case_row(result) for result in run.results)

    incomplete = ""
    if not run.cost_is_complete:
        incomplete = (
            f" <span class='note'>(a lower bound: {run.unpriced_call_count} "
            f"call{'' if run.unpriced_call_count == 1 else 's'} could not be priced)</span>"
        )

    return (
        f"<h2>{escape(run.name)}</h2>"
        f"<p class='sub'>{escape(_short(run.id))} &middot; {escape(_when(run))}</p>"
        "<dl class='totals'>"
        f"<dt>mean</dt><dd>{escape(format_score(run.mean_score))}</dd>"
        f"<dt>passed</dt><dd>{passed}/{judged}</dd>"
        f"<dt>cost</dt><dd>{escape(_run_cost(run))}{incomplete}</dd>"
        f"<dt>cache</dt><dd>{escape(_cache(run))}</dd>"
        "</dl>"
        "<div class='scroll'><table><thead><tr>"
        "<th>case</th><th>status</th><th class='num'>score</th>"
        "<th class='num'>latency</th><th class='num'>cost</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _short(run_id: str) -> str:
    """Enough of an id to recognise, not so much it crowds the row."""
    return run_id[:12]


def _when(run: Run) -> str:
    return UNKNOWN if run.started_at is None else run.started_at.strftime("%Y-%m-%d %H:%M")


def _run_cost(run: Run) -> str:
    """A run's spend, or unknown when nothing was priced.

    Unknown rather than `$0.0000`: a run whose calls were never priced has not
    been shown to be free.
    """
    if run.unpriced_call_count and run.total_cost_usd == 0.0:
        return UNKNOWN
    return f"${run.total_cost_usd:.4f}"


def _cache(run: Run) -> str:
    """Hit rate, or why there isn't one.

    A bypassed run is named as such rather than shown as 0%: bypassing spends
    the full amount every time, and a 0% hit rate reads like a merely cold
    cache.
    """
    if run.cache_bypassed:
        return "bypassed"
    rate = run.cache_hit_rate
    return UNKNOWN if rate is None else f"{rate:.0%}"


def _score_of(result: Result) -> str:
    """The mean of this Result's measured scores.

    Scores that errored are excluded, matching `Run.mean_score`. A scorer that
    crashed did not score zero, and averaging its absence in would drag the
    number towards a failure that was never measured.
    """
    values = [s.value for s in result.scores if s.value is not None and not s.error]
    if not values:
        return UNKNOWN
    return format_score(sum(values) / len(values))


def _latency(result: Result) -> str:
    return UNKNOWN if result.latency_ms is None else f"{result.latency_ms} ms"

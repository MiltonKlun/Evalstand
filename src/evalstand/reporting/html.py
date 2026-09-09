"""A self-contained HTML report, for a CI artifact (task 8.3).

The plan calls this worth doing without the rest of the web UI, and the reason is
what CI does to a terminal report: it scrolls away. A build log is searched
once, by whoever is already debugging. An HTML file can be downloaded from the
build, opened by someone who does not have the project installed, and attached
to a review.

**One file, no assets, no network.** Everything — CSS included — is inline. A
report that fetched a stylesheet would render unstyled inside the sandboxed
iframes CI systems display artifacts in, and one that needed a sibling file
would break the moment it was emailed to somebody.

The honesty rules come from `reporting.console` (`UNKNOWN`, `pass_counts`,
`format_cost`) rather than being restated here, for the same reason the markdown
reporter borrows them: three copies of "what counts as a pass" would drift, and
then three reports would disagree about whether a build passed.

Where this differs from the other two reporters: it shows **everything**. The
terminal truncates to fit a screen and the PR comment truncates to fit GitHub's
limit, but a file has room for the full output and the whole trace tree, which is
the one thing a build log cannot give you.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

from evalstand.models import Result, Run, Score, Trace
from evalstand.reporting.console import UNKNOWN, format_cost, format_score, pass_counts

__all__ = ["render_html"]

_STYLE = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --pass: #047857; --fail: #b91c1c; --warn: #b45309; --scored: #0369a1;
  --panel: #f9fafb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e5e7eb; --muted: #9ca3af; --line: #272b33;
    --pass: #34d399; --fail: #f87171; --warn: #fbbf24; --scored: #60a5fa;
    --panel: #16191f;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 68rem; margin: 0 auto; }
h1 { font-size: 1.35rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 .6rem; }
h3 { font-size: .95rem; margin: 0; font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 1.5rem; font-size: .875rem; }
table { border-collapse: collapse; width: 100%; font-size: .875rem; }
/* Wide tables scroll inside their own box; the page never scrolls sideways. */
.scroll { overflow-x: auto; }
th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid var(--line); }
th { font-weight: 600; color: var(--muted); font-size: .8rem; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: var(--pass); } .fail { color: var(--fail); font-weight: 600; }
.error { color: var(--fail); font-weight: 600; }
.unmeasured { color: var(--warn); font-weight: 600; }
.scored { color: var(--scored); }
.note {
  margin: 1rem 0; padding: .7rem .9rem; background: var(--panel);
  border-left: 3px solid var(--warn); color: var(--muted); font-size: .875rem;
}
details.case {
  border: 1px solid var(--line); border-radius: 6px; margin: .5rem 0;
  background: var(--panel);
}
details.case > summary {
  cursor: pointer; padding: .6rem .9rem; display: flex; gap: .75rem;
  align-items: center; font-size: .875rem;
}
details.case > div { padding: 0 .9rem .9rem; }
.field { margin: .7rem 0 0; }
.field > .label {
  color: var(--muted); font-size: .75rem; text-transform: uppercase;
  letter-spacing: .04em; margin-bottom: .2rem;
}
pre {
  margin: 0; padding: .6rem .7rem; background: var(--bg);
  border: 1px solid var(--line); border-radius: 4px; overflow-x: auto;
  font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; word-break: break-word;
}
ul.tree { list-style: none; margin: .3rem 0 0; padding-left: 1rem; }
ul.tree ul { margin: 0; padding-left: 1.1rem; border-left: 1px solid var(--line); }
ul.tree li { padding: .2rem 0; font-size: .8rem; }
.tname { font-weight: 600; }
.tmeta { color: var(--muted); }
footer {
  margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line);
  color: var(--muted); font-size: .8rem;
}
"""


def _e(value: object) -> str:
    """Escape for HTML.

    Applied to every interpolated value without exception, including model
    output. A completion containing `<script>` is not an attack so much as an
    inevitability — models are asked about HTML all the time — and an artifact
    that executed it would be a stored-XSS hole in a file people open from a
    build page.
    """
    return html.escape("" if value is None else str(value), quote=True)


def _status(result: Result) -> str:
    """The same five states the live view uses, and for the same reason.

    "failed" and "unmeasured" look alike in a table and mean opposite things:
    one is the model answering wrongly, the other is the scorer breaking and the
    model's performance being unknown.
    """
    if result.error:
        return "error"
    if result.scores and all(score.error for score in result.scores):
        return "unmeasured"

    verdicts = [s.passed for s in result.scores if s.passed is not None]
    if not verdicts:
        return "scored"
    return "pass" if all(verdicts) else "fail"


def _result_cost(result: Result) -> str:
    """This case's spend, or unknown when nothing it did was priced."""
    priced = [t for t in result.traces if t.cost_usd is not None]
    if result.traces and not priced:
        return UNKNOWN
    return f"${sum(t.cost_usd or 0.0 for t in priced):.4f}"


def _summary_rows(runs: list[Run]) -> str:
    rows: list[str] = []
    for run in runs:
        passed, judged = pass_counts(run)
        verdict = UNKNOWN if not judged else f"{passed}/{judged}"
        cost = format_cost([run])
        means = run.mean_scores_by_scorer()

        if not means:
            # An eval whose every scorer errored still gets a row. Dropping it
            # would leave a reader believing it was never part of the run.
            rows.append(
                f"<tr><td>{_e(run.name)}</td><td>{UNKNOWN}</td>"
                f"<td class='num'>{UNKNOWN}</td><td class='num'>{_e(verdict)}</td>"
                f"<td class='num'>{_e(cost)}</td></tr>"
            )
            continue

        for scorer_name, mean in sorted(means.items()):
            rows.append(
                f"<tr><td>{_e(run.name)}</td><td>{_e(scorer_name)}</td>"
                f"<td class='num'>{_e(format_score(mean))}</td>"
                f"<td class='num'>{_e(verdict)}</td>"
                f"<td class='num'>{_e(cost)}</td></tr>"
            )
    return "\n".join(rows)


def _score_line(score: Score) -> str:
    """One score, saying which of the three things it is.

    A verdict, a bare value and an error are different claims; collapsing any
    pair of them is how a false pass gets made.
    """
    if score.error:
        return (
            f"<span class='error'>{_e(score.scorer_name)}: errored</span> "
            f"<span class='tmeta'>({_e(score.error)})</span>"
        )

    value = UNKNOWN if score.value is None else f"{score.value:.3f}"
    if score.passed is None:
        return f"<span class='scored'>{_e(score.scorer_name)}: {_e(value)}</span>"
    verdict = "pass" if score.passed else "fail"
    css = "pass" if score.passed else "fail"
    return f"<span class='{css}'>{_e(score.scorer_name)}: {_e(value)} ({verdict})</span>"


def _trace_list(traces: list[Trace]) -> str:
    """The calls a case made, as the tree they actually form.

    Built **iteratively**, like the console renderer: a recursive walk overflows
    the stack at about a thousand levels, and losing a whole report to a judge
    that called a judge would be a poor way to fail. The Result validator has
    already rejected cycles and dangling parents, so this only has to survive
    the shape's size.
    """
    if not traces:
        return ""

    children: dict[str | None, list[Trace]] = {}
    for trace in traces:
        children.setdefault(trace.parent_id, []).append(trace)

    parts = ["<ul class='tree'>"]
    # (trace, depth) with an explicit close-marker, so nesting is emitted
    # without recursion. Reversed because this is a stack and siblings must come
    # out in the order they were recorded.
    stack: list[Trace | None] = list(reversed(children.get(None, [])))

    while stack:
        node = stack.pop()
        if node is None:
            parts.append("</ul></li>")
            continue

        kids = children.get(node.id, [])
        parts.append(f"<li>{_trace_label(node)}")
        if kids:
            parts.append("<ul>")
            stack.append(None)
            stack.extend(reversed(kids))
        else:
            parts.append("</li>")

    parts.append("</ul>")
    return "".join(parts)


def _trace_label(trace: Trace) -> str:
    """One call: what it was, what it took, and what it cost."""
    bits = [f"<span class='tname'>{_e(trace.name)}</span>"]
    if trace.model:
        bits.append(f"<span class='tmeta'>{_e(trace.model)}</span>")
    bits.append(f"<span class='tmeta'>{trace.duration_ms}ms</span>")

    if trace.input_tokens is not None or trace.output_tokens is not None:
        tokens = f"{trace.input_tokens or 0} in / {trace.output_tokens or 0} out"
        bits.append(f"<span class='tmeta'>{_e(tokens)}</span>")

    # An unpriced call shows a placeholder, never $0.0000: it has not been shown
    # to be free.
    cost = UNKNOWN if trace.cost_usd is None else f"${trace.cost_usd:.4f}"
    bits.append(f"<span class='tmeta'>{_e(cost)}</span>")

    label = " ".join(bits)
    for name, value in (("input", trace.input), ("output", trace.output)):
        if value is not None:
            label += (
                f"<div class='field'><div class='label'>{name}</div><pre>{_e(value)}</pre></div>"
            )
    return label


def _case_block(run: Run, result: Result) -> str:
    """One case, collapsed by default.

    Collapsed because a thirty-case report with every prompt expanded is a page
    nobody scrolls; open because the detail is the reason this format exists.
    Failures start open, since they are what the reader came for.
    """
    status = _status(result)
    score = UNKNOWN if result.mean_score is None else f"{result.mean_score:.2f}"
    latency = UNKNOWN if result.latency_ms is None else f"{result.latency_ms}ms"
    label = (
        result.case_id if result.repeat_index == 0 else (f"{result.case_id} #{result.repeat_index}")
    )

    parts = [
        f"<details class='case'{' open' if status in {'fail', 'error', 'unmeasured'} else ''}>",
        "<summary>",
        f"<h3>{_e(label)}</h3>",
        f"<span class='{status}'>{status}</span>",
        f"<span class='tmeta'>{_e(score)}</span>",
        f"<span class='tmeta'>{_e(latency)}</span>",
        f"<span class='tmeta'>{_e(_result_cost(result))}</span>",
        "</summary><div>",
    ]

    if result.error:
        parts.append(
            f"<div class='field'><div class='label'>task error</div>"
            f"<pre class='error'>{_e(result.error)}</pre></div>"
        )
        if result.error_frames:
            frames = "\n".join(result.error_frames)
            parts.append(
                f"<div class='field'><div class='label'>where</div><pre>{_e(frames)}</pre></div>"
            )
    else:
        parts.append(
            f"<div class='field'><div class='label'>output</div>"
            f"<pre>{_e(result.output)}</pre></div>"
        )

    if result.scores:
        lines = "<br>".join(_score_line(score) for score in result.scores)
        parts.append(f"<div class='field'><div class='label'>scores</div>{lines}</div>")

    traces = _trace_list(result.traces)
    if traces:
        parts.append(f"<div class='field'><div class='label'>traces</div>{traces}</div>")

    parts.append("</div></details>")
    return "".join(parts)


def _notes(runs: list[Run]) -> str:
    """Every caveat the numbers above depend on, stated where they are read.

    A footnote on another page is a footnote nobody sees, and these are the
    claims that make the difference between a figure and a guess.
    """
    notes: list[str] = []

    unpriced = sum(run.unpriced_call_count for run in runs)
    if unpriced:
        notes.append(
            f"{unpriced} model call{'' if unpriced == 1 else 's'} could not be priced, "
            f"so every cost above is a lower bound."
        )

    errored = sum(run.errored_score_count for run in runs)
    if errored:
        notes.append(
            f"{errored} score{'' if errored == 1 else 's'} errored and "
            f"{'is' if errored == 1 else 'are'} excluded from the means, so those "
            f"means cover fewer cases than the runs contain."
        )

    return "".join(f"<p class='note'>{_e(note)}</p>" for note in notes)


def render_html(
    runs: list[Run],
    *,
    title: str = "evalstand report",
    threshold: float | None = None,
    generated_at: datetime | None = None,
) -> str:
    """A complete HTML document for one batch of runs.

    `generated_at` is injectable so a test can assert on a fixed document. It
    defaults to now, because an artifact with no timestamp is one nobody can
    place against a build.
    """
    when = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")

    if not runs:
        # Distinct from a passing run. "No evals ran" and "every eval passed" are
        # opposite findings, and a page showing an empty table for both would let
        # a broken collection read as success.
        body = "<p class='note'>No evals ran.</p>"
    else:
        cases = "".join(_case_block(run, result) for run in runs for result in run.results)
        breaches = [
            (run.name, run.mean_score)
            for run in runs
            if threshold is not None and run.mean_score is not None and run.mean_score < threshold
        ]
        gate = ""
        if breaches:
            named = ", ".join(
                f"{_e(name)} scored {_e(format_score(mean))}" for name, mean in breaches
            )
            gate = (
                f"<p class='note'><strong class='fail'>Below the threshold of "
                f"{threshold:.2f}:</strong> {named}</p>"
            )

        body = (
            "<h2>Summary</h2>"
            "<div class='scroll'><table><thead><tr><th>eval</th><th>scorer</th>"
            "<th class='num'>mean</th><th class='num'>passed</th>"
            "<th class='num'>cost</th></tr></thead>"
            f"<tbody>{_summary_rows(runs)}</tbody></table></div>"
            f"{gate}{_notes(runs)}"
            f"<h2>Cases</h2>{cases}"
        )

    total = sum(len(run.results) for run in runs)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<h1>{_e(title)}</h1>
<p class="sub">{len(runs)} eval{"" if len(runs) == 1 else "s"},
{total} case{"" if total == 1 else "s"} &middot; {_e(when)}</p>
{body}
<footer>
Generated by evalstand. A dash means a value could not be determined &mdash;
never that it is zero. Score differences carry no significance testing.
</footer>
</main>
</body>
</html>
"""

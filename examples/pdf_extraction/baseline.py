"""Turn a stored run into `BASELINE.md`.

    pytest extraction_eval.py          # produce a run (needs an API key)
    evalstand history                  # find its id
    python baseline.py <run_id>        # write BASELINE.md from it

The baseline is generated from a **recorded run**, never typed by hand. A
hand-written figure drifts from the code the moment either changes, and a
baseline that quietly disagrees with the tool is worse than none: it is the
number people quote.

Everything here is a count of what happened. There is no "good" or "poor", no
grade, and no comparison against an expectation — the plan is explicit that
`evalstand` has no significance testing, and a baseline that editorialised would
be asserting more than one run can support.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from evalstand.models import Run
from evalstand.storage import RunStore

HERE = Path(__file__).parent
BASELINE = HERE / "BASELINE.md"

FIELD_SCORER = "json_fields"
"""The scorer whose metadata carries the per-field breakdown."""


def field_accuracy(run: Run) -> dict[str, tuple[int, int]]:
    """How often each field was right, as (correct, attempted).

    Read from `json_fields`' own metadata rather than recomputed here: the
    scorer already decided what "matched" means, and a second implementation
    would be a second definition that can drift from the first.

    A field that was *missing* counts as attempted-and-wrong. The model was
    asked for it and did not supply it, which is a different failure from
    supplying the wrong value but not a better one.
    """
    correct: Counter[str] = Counter()
    attempted: Counter[str] = Counter()

    for result in run.results:
        for score in result.scores:
            if score.scorer_name != FIELD_SCORER or score.error:
                continue
            for field in score.metadata.get("matched", []):
                correct[field] += 1
                attempted[field] += 1
            for field in score.metadata.get("wrong", []) + score.metadata.get("missing", []):
                attempted[field] += 1

    return {field: (correct[field], attempted[field]) for field in sorted(attempted)}


def failure_modes(run: Run) -> list[tuple[str, list[str]]]:
    """Which documents failed, and on which fields.

    Grouped by document rather than by field, because a document that got three
    fields wrong is usually one problem rather than three — and reading them
    apart hides that.
    """
    modes: list[tuple[str, list[str]]] = []

    for result in run.results:
        problems: list[str] = []
        if result.error:
            problems.append(f"the task raised: {result.error}")

        for score in result.scores:
            if score.error:
                problems.append(f"{score.scorer_name} errored: {score.error}")
                continue
            if score.scorer_name == FIELD_SCORER:
                wrong = score.metadata.get("wrong", [])
                missing = score.metadata.get("missing", [])
                if wrong:
                    problems.append(f"wrong: {', '.join(wrong)}")
                if missing:
                    problems.append(f"missing: {', '.join(missing)}")
            elif score.value is not None and score.value < 1.0:
                problems.append(f"{score.scorer_name} scored {score.value:.2f}")

        if problems:
            modes.append((result.case_id, problems))

    return modes


def _cost_per_document(run: Run) -> str:
    """What one document cost, or an honest gap.

    A run with unpriced calls reports a lower bound, so the figure is marked.
    Presenting it as exact would understate a bill — the same class of error as
    a false pass, in the direction that costs money.
    """
    if not run.results:
        return "-"

    priced = any(trace.cost_usd is not None for result in run.results for trace in result.traces)
    if not priced:
        return "-"

    per_document = run.total_cost_usd / len(run.results)
    return f"${per_document:.4f}" if run.cost_is_complete else f"${per_document:.4f}+"


def _total_cost(run: Run) -> str:
    """The run's total, or a gap when nothing was priced.

    `$0.0000` would claim a run was free; `-` says nobody knows. The same
    distinction the summary and history tables already draw.
    """
    priced = any(trace.cost_usd is not None for result in run.results for trace in result.traces)
    if not priced:
        return "-"
    total = f"${run.total_cost_usd:.4f}"
    return total if run.cost_is_complete else f"{total}+"


def _scorer_line(name: str, mean: float | None, errored: int) -> str:
    value = "-" if mean is None else f"{mean:.3f}"
    note = f" ({errored} errored, excluded)" if errored else ""
    return f"| `{name}` | {value}{note} |"


def _errored_by_scorer(run: Run) -> Counter[str]:
    return Counter(
        score.scorer_name for result in run.results for score in result.scores if score.error
    )


def _looks_mocked(run: Run) -> bool:
    """Whether every call in this run cost exactly the same.

    A crude but reliable signal that a provider was mocked: real completions
    vary in length and therefore in price, so thirty identical costs mean a
    fixture returned a constant. Worth detecting, because a baseline generated
    from mock data and committed as though it were a measurement would be the
    most misleading document in the repository.
    """
    costs = {
        trace.cost_usd
        for result in run.results
        for trace in result.traces
        if trace.cost_usd is not None
    }
    return len(run.results) > 2 and len(costs) == 1


def render(run: Run, model: str) -> str:
    """The baseline document.

    Deliberately plain. Every number is a count of what one run did, and the
    caveats are stated where the numbers are rather than in a footnote nobody
    reads.
    """
    lines: list[str] = [
        "# Baseline",
        "",
        "One run of `extraction_eval.py` over the 30-invoice corpus at seed 42.",
        "",
        "**This is a record, not a target.** It describes what one model did on",
        "one day, and `evalstand` has no significance testing — a later run that",
        "scores differently has not necessarily got better or worse. Use it to",
        "notice a change worth investigating, not to conclude one.",
        "",
    ]

    if _looks_mocked(run):
        lines += [
            "> **This run was produced against a mocked provider, not a real model.**",
            "> Every call cost the same amount, which does not happen with real",
            "> completions. The numbers below describe the fixture, not any model's",
            "> ability. Replace this file with a real run before quoting it.",
            "",
        ]

    lines += [
        "| | |",
        "|---|---|",
        f"| model | `{model}` |",
        f"| run | `{run.id}` |",
        f"| when | {run.started_at.astimezone().strftime('%Y-%m-%d') if run.started_at else '-'} |",
        f"| documents | {len(run.results)} |",
        f"| cost per document | {_cost_per_document(run)} |",
        f"| total cost | {_total_cost(run)} |",
        "",
        "## Scorer means",
        "",
        "| scorer | mean |",
        "|---|---|",
    ]

    errored = _errored_by_scorer(run)
    means = run.mean_scores_by_scorer()
    for name in sorted(set(means) | set(errored)):
        lines.append(_scorer_line(name, means.get(name), errored[name]))

    lines += [
        "",
        "`line_items` is an **LLM judge and is unvalidated** — a second model's",
        "opinion, not a measurement. It has not been calibrated against human",
        "labels. See [docs/scorers.md](../../docs/scorers.md).",
        "",
        "## Per-field accuracy",
        "",
        "| field | correct | of | rate |",
        "|---|---|---|---|",
    ]

    for field, (correct, attempted) in field_accuracy(run).items():
        rate = f"{correct / attempted:.0%}" if attempted else "-"
        lines.append(f"| `{field}` | {correct} | {attempted} | {rate} |")

    lines += ["", "## Observed failure modes", ""]

    modes = failure_modes(run)
    if not modes:
        lines.append("No document failed any scorer in this run.")
    else:
        lines.append(f"{len(modes)} of {len(run.results)} documents had at least one problem.")
        lines += ["", "| document | what went wrong |", "|---|---|"]
        for case_id, problems in modes:
            lines.append(f"| `{case_id}` | {'; '.join(problems)} |")

    lines += [
        "",
        "---",
        "",
        f"Generated by `python baseline.py {run.id}` on "
        f"{datetime.now(UTC).strftime('%Y-%m-%d')}. Regenerate it rather than",
        "editing by hand: a figure typed in drifts from the run it claims to",
        "describe, and a baseline that disagrees with the tool is worse than",
        "none — it is the number people quote.",
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="the run to describe, from `evalstand history`")
    parser.add_argument("--model", default="gpt-4o-mini", help="which model produced it")
    parser.add_argument("--db", type=Path, default=None, help="a database other than the default")
    args = parser.parse_args()

    store = RunStore(args.db) if args.db else RunStore()
    with store:
        run = store.load_run(args.run_id)

    if run is None:
        print(f"no run with id {args.run_id}. Try `evalstand history`.")
        return 1

    BASELINE.write_text(render(run, args.model), encoding="utf-8")
    print(f"wrote {BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

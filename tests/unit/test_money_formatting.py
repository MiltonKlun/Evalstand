"""A priced cost is never rounded down to "free".

Found by the first Jev baseline: $0.0015 over thirty invoices is $0.00005 each,
and at four decimals that printed `cost per document | $0.0000`. The string this
project refuses to show for an *unknown* cost had appeared beside a *known,
non-zero* one.

It was not a Jev problem. A short `gpt-4o-mini` judge call costs well under a
hundredth of a cent, so every trace tree, HTML report, TUI row and web page was
already showing such calls as `$0.0000` — nine renderers, each with its own
`:.4f`. The fix is one formatter, and these tests check that every renderer
actually goes through it: a correct function that one renderer bypasses is the
same bug in one fewer place.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from evalstand.models import Result, Run, RunStatus, Score, Trace
from evalstand.reporting.console import format_usd

TINY = 0.0000075
"""Roughly one short gpt-4o-mini call. Four decimals would print zero."""


def _result(cost: float | None = TINY) -> Result:
    return Result(
        id="r1",
        case_id="q1",
        output="answer",
        scores=[Score(scorer_name="exact", value=1.0, passed=True)],
        traces=[Trace(id="t1", name="call", duration_ms=5, model="m", cost_usd=cost)],
    )


def _run(cost: float | None = TINY) -> Run:
    return Run(
        id="run-1",
        batch_id="b1",
        name="demo",
        filepath="demo_eval.py",
        status=RunStatus.COMPLETED,
        started_at=datetime(2026, 9, 22, tzinfo=UTC),
        results=[_result(cost)],
        model_calls=1,
    )


class TestFormatUsd:
    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (0.0015, "$0.0015"),
            (1.23456, "$1.2346"),
            (0.0000498, "$0.000050"),
            (0.0000075, "$0.0000075"),
        ],
    )
    def test_it_keeps_two_significant_figures(self, value: float, shown: str) -> None:
        assert format_usd(value) == shown

    def test_a_real_zero_is_still_zero(self) -> None:
        """The one case where `$0.0000` is true: a call priced at nothing."""
        assert format_usd(0.0) == "$0.0000"

    @pytest.mark.parametrize("value", [0.00004, 0.0000001, 0.000049999])
    def test_no_positive_amount_prints_as_zero(self, value: float) -> None:
        shown = format_usd(value)

        assert float(shown.lstrip("$")) > 0, f"{value} was shown as {shown}"


def _no_false_zero(text: str) -> None:
    assert "$0.0000" not in text.replace("$0.0000075", ""), "a priced cost rendered as free"
    assert "0.0000075" in text, "the tiny cost does not appear at all"


class TestEveryRendererUsesIt:
    """Each renderer, fed one call costing a fraction of a hundredth of a cent."""

    def test_the_console_summary(self) -> None:
        from evalstand.reporting.console import format_cost

        _no_false_zero(format_cost([_run()]))

    def test_the_console_trace_tree(self) -> None:
        from rich.console import Console

        from evalstand.reporting.console import render_case

        console = Console(record=True, width=200)
        console.print(render_case(_result(), full=True))

        _no_false_zero(console.export_text())

    def test_the_html_report(self) -> None:
        from evalstand.reporting.html import render_html

        _no_false_zero(render_html([_run()]))

    def test_the_tui_row(self) -> None:
        from evalstand.tui.state import row_for

        _no_false_zero(row_for(_result()).cost)

    def test_the_web_run_list(self) -> None:
        pytest.importorskip("fastapi", reason="the web extra is not installed")
        from evalstand.web.fragments import run_row

        _no_false_zero(run_row(_run()))

    def test_the_web_case_row(self) -> None:
        pytest.importorskip("fastapi", reason="the web extra is not installed")
        from evalstand.web.fragments import case_row

        _no_false_zero(case_row(_result()))


class TestNoRendererFormatsMoneyItself:
    def test_every_dollar_amount_goes_through_format_usd(self) -> None:
        """How the bug reached nine places: each renderer had its own `:.4f`.

        Asserted on the source because the renderers are many and the next one
        written will not be in the list above. A new `f"${...:.4f}"` anywhere in
        the package is the same defect waiting for a cheap enough model.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "src" / "evalstand"
        offenders = [
            f"{path.relative_to(root)}:{number}"
            for path in root.rglob("*.py")
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"\$\{[^}]*:\.\d+f\}", line)
        ]

        assert not offenders, f"money formatted outside format_usd: {offenders}"

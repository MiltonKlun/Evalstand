"""Every factual claim in `docs/watching.md` (task 6.6).

Same rule as `test_docs_scorers.py`: documentation that drifts from the code is
worse than none, because a user who follows a wrong instruction loses time and
then trust.

This file is the reason the doc can state exact keys, statuses and file
extensions rather than hedging. Each of those is a claim that can silently
become false when the code changes, so each is asserted here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evalstand.models import Result, Score, Trace
from evalstand.tui.app import EvalApp
from evalstand.tui.state import RunState, row_for
from evalstand.tui.watch import DEBOUNCE_MS, is_relevant
from tests.conftest import help_text

DOC = Path(__file__).resolve().parents[2] / "docs" / "watching.md"


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


def _result(scores: list[Score] | None = None, **kwargs: object) -> Result:
    return Result(
        id="e-q1-0",
        case_id="q1",
        scores=scores if scores is not None else [Score(scorer_name="s", value=1.0, passed=True)],
        **kwargs,  # type: ignore[arg-type]
    )


class TestTheCommandsItShows:
    def test_watch_is_a_real_command(self, text: str) -> None:
        from evalstand.cli import app

        names = {command.name or command.callback.__name__ for command in app.registered_commands}
        assert "watch" in names

    @pytest.mark.parametrize("flag", ["--eval", "--once", "--store"])
    def test_every_documented_flag_exists(self, text: str, flag: str) -> None:
        assert flag in text

        assert flag in help_text("watch")


class TestTheStatusesItDescribes:
    """The doc names five, and explains why they are not collapsed to two."""

    def test_a_crashed_task_says_error(self) -> None:
        assert row_for(_result(scores=[], error="boom")).status == "error"

    def test_every_scorer_erroring_says_unmeasured(self) -> None:
        assert row_for(_result(scores=[Score.from_error("s", "boom")])).status == "unmeasured"

    def test_a_verdictless_score_says_scored(self) -> None:
        assert row_for(_result(scores=[Score(scorer_name="r", value=0.4)])).status == "scored"

    def test_pass_and_fail_are_the_other_two(self) -> None:
        assert row_for(_result()).status == "pass"
        assert row_for(
            _result(scores=[Score(scorer_name="s", value=0.0, passed=False)])
        ).status == ("fail")

    def test_the_doc_lists_exactly_those_five(self, text: str) -> None:
        documented = set(re.findall(r"`(pass|fail|error|unmeasured|scored)`", text))

        assert documented == {"pass", "fail", "error", "unmeasured", "scored"}


class TestTheCostClaims:
    def test_an_unpriced_case_shows_a_dash_not_zero(self, text: str) -> None:
        assert "has not been shown to be free" in text

        traced = _result(traces=[Trace(id="t", name="call", duration_ms=1, cost_usd=None)])
        assert row_for(traced).cost == "-"

    def test_the_lower_bound_format_is_what_the_doc_prints(self, text: str) -> None:
        """The doc shows `$0.0140 (+3 unpriced)`, so the format is pinned."""
        assert "(+3 unpriced)" in text

        state = RunState("e")
        state.add(
            _result(
                traces=[
                    Trace(id="a", name="c", duration_ms=1, cost_usd=0.014),
                    *(Trace(id=f"u{i}", name="c", duration_ms=1, cost_usd=None) for i in range(3)),
                ]
            )
        )

        assert state.totals().cost == "$0.0140 (+3 unpriced)"


class TestTheKeysItLists:
    @pytest.mark.parametrize(
        ("key", "action"),
        [
            ("enter", "open_case"),
            ("f", "toggle_failures"),
            ("slash", "search"),
            ("r", "rerun"),
            ("c", "compare_previous"),
            ("h", "history"),
            ("y", "copy_case"),
            ("q", "quit"),
        ],
    )
    def test_each_documented_key_is_bound_to_a_real_action(self, key: str, action: str) -> None:
        bound = {
            binding.key: binding.action for binding in EvalApp.BINDINGS if hasattr(binding, "key")
        }

        assert bound.get(key) == action
        assert hasattr(EvalApp, f"action_{action}"), f"{action} is bound but not implemented"

    def test_escape_clears_the_search(self) -> None:
        bound = {
            binding.key: binding.action for binding in EvalApp.BINDINGS if hasattr(binding, "key")
        }

        assert bound.get("escape") == "clear_search"


class TestTheFileTypesItNames:
    @pytest.mark.parametrize(
        "suffix", [".py", ".txt", ".md", ".json", ".yaml", ".yml", ".jinja", ".j2"]
    )
    def test_every_documented_extension_triggers_a_rerun(self, text: str, suffix: str) -> None:
        assert f"`{suffix}`" in text
        assert is_relevant(f"prompts/file{suffix}")

    @pytest.mark.parametrize("ignored", ["__pycache__", ".git", ".venv", "node_modules"])
    def test_every_documented_exclusion_is_ignored(self, text: str, ignored: str) -> None:
        assert ignored in text
        assert not is_relevant(f"{ignored}/thing.py")

    def test_the_debounce_is_the_number_the_doc_states(self, text: str) -> None:
        assert "300ms" in text
        assert DEBOUNCE_MS == 300


class TestTheRecordingPolicy:
    def test_watch_does_not_persist_without_store(self, text: str) -> None:
        """The doc says this is the opposite of `run`, so the default is pinned:
        a silent change here would fill history with unreproducible runs."""
        assert "does **not** write to the history database unless you pass `--store`" in text

        import inspect

        from evalstand.cli import watch

        signature = inspect.signature(watch)
        assert signature.parameters["store"].default is False

    def test_the_refusal_message_is_what_compare_actually_prints(self, text: str) -> None:
        """The doc quotes the message verbatim in a code block."""
        from evalstand.comparison import NotComparableError, refuse_partial_runs
        from evalstand.models import Batch, BatchKind, BatchStatus, Run

        assert "did not run to completion, so comparing it would compare different" in text

        run = Run(id="run-def456", batch_id="b", name="qa", filepath="q_eval.py")
        cancelled = Batch(id="b", kind=BatchKind.FULL, status=BatchStatus.CANCELLED)

        with pytest.raises(NotComparableError) as caught:
            refuse_partial_runs([run], lambda _: cancelled)

        assert "run-def456 did not run to completion" in str(caught.value)
        assert "compare different questions" in str(caught.value)


class TestTheCustomColumnExample:
    def test_the_example_compiles(self, text: str) -> None:
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)

        assert blocks, "the doc should carry a python example"
        for block in blocks:
            compile(block, "<docs/watching.md>", "exec")

    def test_a_raising_column_shows_the_marker_the_doc_promises(self, text: str) -> None:
        assert "shows `!` in" in text

        def boom(result: Result) -> object:
            raise ValueError("bad")

        state = RunState("e", columns={"broken": boom})
        row = state.add(_result())

        assert row is not None and row.extra == {"broken": "!"}

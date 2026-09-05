"""Rendering one run in full (task 5.4).

Two themes. The first is the trace tree: the Result validator guarantees a
forest — no cycles, no dangling parents, no duplicate ids — so the renderer can
trust the *shape* and only has to survive its *size*. A recursive walk overflows
the stack at about a thousand levels, and losing a whole run's display to a
judge that called a judge would be a poor way to fail.

The second is what a figure claims. `show` displays the same numbers as the
live summary and as `history`, so the three must agree — a detail view quietly
disagreeing with the table it was opened from is the silent wrongness this
project keeps finding.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from evalstand.models import (
    Batch,
    BatchKind,
    BatchStatus,
    Result,
    Run,
    RunStatus,
    Score,
    Trace,
)
from evalstand.reporting.console import render_run_detail
from evalstand.storage import RunStore


def _batch(**kwargs: Any) -> Batch:
    defaults: dict[str, Any] = {
        "kind": BatchKind.FULL,
        "status": BatchStatus.COMPLETED,
        "started_at": datetime.now(UTC),
    }
    return Batch(id="b1", **(defaults | kwargs))


def _run(**kwargs: Any) -> Run:
    defaults: dict[str, Any] = {
        "name": "qa",
        "filepath": "qa_eval.py",
        "status": RunStatus.COMPLETED,
        "started_at": datetime.now(UTC),
        "results": [
            Result(
                id="run-1-q1-0",
                case_id="q1",
                output="Paris",
                scores=[Score(scorer_name="exact", value=1.0, passed=True)],
            )
        ],
    }
    return Run(id="run-1", batch_id="b1", **(defaults | kwargs))


def _trace(trace_id: str, parent: str | None = None, **kwargs: Any) -> Trace:
    defaults: dict[str, Any] = {"name": "model call", "duration_ms": 10}
    return Trace(id=trace_id, parent_id=parent, **(defaults | kwargs))


def _rendered(run: Run, batch: Batch | None = None, *, full: bool = False) -> str:
    console = Console(width=200, force_terminal=False, record=True)
    console.print(render_run_detail(run, batch, full=full))
    return console.export_text()


class TestTheTraceTree:
    def test_a_child_appears_under_its_parent(self) -> None:
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[_trace("root", name="task"), _trace("call", "root")],
                )
            ]
        )
        lines = [line for line in _rendered(run).splitlines() if "task" in line or "call" in line]

        assert len(lines) == 2
        task_indent = len(lines[0]) - len(lines[0].lstrip())
        call_indent = len(lines[1]) - len(lines[1].lstrip())
        assert call_indent > task_indent, "the child was not nested under its parent"

    def test_a_forest_shows_every_root(self) -> None:
        """The validator permits several roots, so the renderer must not assume
        one and silently drop the rest."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[_trace("a", name="first"), _trace("b", name="second")],
                )
            ]
        )
        rendered = _rendered(run)

        assert "first" in rendered
        assert "second" in rendered

    def test_a_child_listed_before_its_parent_still_nests(self) -> None:
        """Traces arrive in collection order, and a child can be recorded
        before the parent that opened it. The index is built in one pass for
        exactly this reason."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[_trace("call", "root"), _trace("root", name="task")],
                )
            ]
        )
        rendered = _rendered(run)

        assert "task" in rendered
        assert "call" in rendered

    def test_a_tree_deeper_than_the_recursion_limit_still_renders(self) -> None:
        """The reason the walk is iterative. A recursive renderer raises
        RecursionError at about a thousand levels and loses the entire run's
        display — every case, not just the deep one."""
        depth = sys.getrecursionlimit() * 3
        chain = [_trace(f"n{index}", f"n{index - 1}" if index else None) for index in range(depth)]
        run = _run(results=[Result(id="r1", case_id="q1", traces=chain)])

        rendered = _rendered(run)  # must not raise

        assert "model call" in rendered

    def test_siblings_keep_the_order_they_were_recorded(self) -> None:
        """Trace order is the order calls happened. Shuffling it would make a
        tree that cannot be read against a log."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[
                        _trace("root", name="task"),
                        _trace("a", "root", name="first-call"),
                        _trace("b", "root", name="second-call"),
                    ],
                )
            ]
        )
        rendered = _rendered(run)

        assert rendered.index("first-call") < rendered.index("second-call")

    def test_a_case_with_no_traces_renders_without_an_empty_tree(self) -> None:
        run = _run(results=[Result(id="r1", case_id="q1", output="x")])
        assert "traces" not in _rendered(run)


class TestWhatIsShownAboutEachCall:
    def test_a_priced_call_shows_its_cost(self) -> None:
        run = _run(results=[Result(id="r1", case_id="q1", traces=[_trace("t", cost_usd=0.0002)])])
        assert "$0.0002" in _rendered(run)

    def test_an_unpriced_call_is_not_shown_as_free(self) -> None:
        """`$0.0000` claims a call cost nothing. It has not been shown to."""
        run = _run(results=[Result(id="r1", case_id="q1", traces=[_trace("t")])])
        assert "$0.0000" not in _rendered(run)

    def test_tokens_and_model_are_shown(self) -> None:
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[_trace("t", model="gpt-4o-mini", input_tokens=10, output_tokens=3)],
                )
            ]
        )
        rendered = _rendered(run)

        assert "gpt-4o-mini" in rendered
        assert "10 in / 3 out" in rendered


class TestPromptsAreWithheldByDefault:
    """One 4000-token prompt fills a screen and buries the tree it belongs to —
    and a trace input can hold a customer record nobody meant to display on a
    shared terminal."""

    def test_the_prompt_itself_is_not_printed(self) -> None:
        secret = "the customer's account number is 4111111111111111"
        run = _run(
            results=[
                Result(id="r1", case_id="q1", traces=[_trace("t", input=[{"content": secret}])])
            ]
        )

        assert secret not in _rendered(run)

    def test_but_its_size_is(self) -> None:
        """Enough to know something was sent, without showing it."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[_trace("t", input=[{"content": "hello"}], output="Paris")],
                )
            ]
        )
        rendered = _rendered(run)

        assert "1 message" in rendered
        assert "chars" in rendered

    def test_full_prints_the_prompt(self) -> None:
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[_trace("t", input=[{"content": "capital of France?"}])],
                )
            ]
        )

        assert "capital of France?" in _rendered(run, full=True)

    def test_full_prints_the_completion(self) -> None:
        run = _run(
            results=[Result(id="r1", case_id="q1", traces=[_trace("t", output="Paris it is")])]
        )

        assert "Paris it is" in _rendered(run, full=True)


class TestScoresSayWhichOfTheThreeThingsTheyAre:
    """A verdict, a bare value and an error are different claims. Collapsing
    any pair of them is how a false pass gets made."""

    def test_a_verdict_is_shown_as_pass_or_fail(self) -> None:
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    scores=[Score(scorer_name="exact", value=1.0, passed=True)],
                )
            ]
        )
        assert "(pass)" in _rendered(run)

    def test_a_value_without_a_verdict_claims_neither(self) -> None:
        """A continuous scorer does not know where the line sits, and the
        report must not invent one for it."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    scores=[Score(scorer_name="levenshtein", value=0.62)],
                )
            ]
        )
        rendered = _rendered(run)

        assert "0.620" in rendered
        assert "(pass)" not in rendered
        assert "(fail)" not in rendered

    def test_an_errored_score_says_so_rather_than_showing_zero(self) -> None:
        """`0.000` would report that the task did badly. The judge failed."""
        run = _run(
            results=[
                Result(id="r1", case_id="q1", scores=[Score.from_error("judge", "rate limited")])
            ]
        )
        rendered = _rendered(run)

        assert "errored" in rendered
        assert "rate limited" in rendered
        assert "0.000" not in rendered

    def test_the_excluded_count_is_stated_beside_the_mean(self) -> None:
        """A mean over fewer scores than expected is a different claim from one
        over all of them."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    scores=[
                        Score(scorer_name="exact", value=1.0, passed=True),
                        Score.from_error("judge", "down"),
                    ],
                )
            ]
        )

        assert "excluded" in _rendered(run)


class TestAFailedCase:
    def test_the_task_error_is_shown(self) -> None:
        run = _run(results=[Result(id="r1", case_id="q1", error="RuntimeError: boom")])
        assert "RuntimeError: boom" in _rendered(run)

    def test_the_captured_frames_are_shown(self) -> None:
        """They were captured at the raise precisely so a stored failure stays
        debuggable months later."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    error="KeyError: 'missing'",
                    error_frames=["  qa_eval.py:12 in answer"],
                )
            ]
        )

        assert "qa_eval.py:12" in _rendered(run)


class TestTheHeaderAgreesWithTheOtherViews:
    def test_the_mean_matches_the_run(self) -> None:
        run = _run(
            results=[Result(id="r1", case_id="q1", scores=[Score(scorer_name="s", value=0.5)])]
        )
        assert run.mean_score == 0.5
        assert "0.50" in _rendered(run)

    def test_an_unmeasured_run_shows_no_mean(self) -> None:
        run = _run(
            results=[Result(id="r1", case_id="q1", scores=[Score.from_error("judge", "down")])]
        )
        assert run.mean_score is None
        assert "0.00" not in _rendered(run)

    def test_a_partly_priced_run_is_marked_as_a_lower_bound(self) -> None:
        """The same rule `history` applies, because the two views must not
        disagree about the same run."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[_trace("a", cost_usd=0.0002), _trace("b")],
                )
            ]
        )
        assert "$0.0002+" in _rendered(run)

    def test_the_commit_is_shown_and_a_dirty_tree_marked(self) -> None:
        rendered = _rendered(_run(), _batch(git_sha="abcdef12" + "0" * 32, git_dirty=True))
        assert "abcdef12*" in rendered

    def test_the_task_hash_is_shown_when_recorded(self) -> None:
        """What lets a reader tell a changed model from changed code."""
        assert "abcdef123456" in _rendered(_run(task_source_hash="abcdef123456" + "0" * 52))


class TestTheShowCommand:
    def test_it_renders_a_stored_run(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "h.db"
        with RunStore(database) as store:
            store.save_batch(_batch())
            store.save_run(_run())

        result = CliRunner().invoke(app, ["show", "run-1", "--db", str(database)])

        assert result.exit_code == 0
        assert "q1" in result.output

    def test_an_unknown_run_exits_non_zero_with_guidance(self, tmp_path: Path) -> None:
        """A typo and an empty database send a user looking in different
        places, so they are answered differently."""
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "h.db"
        with RunStore(database) as store:
            store.save_batch(_batch())
            store.save_run(_run())

        result = CliRunner().invoke(app, ["show", "run-nope", "--db", str(database)])

        assert result.exit_code != 0
        assert "evalstand history" in result.output

    def test_an_empty_database_says_so(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "empty.db"
        RunStore(database).close()

        result = CliRunner().invoke(app, ["show", "run-1", "--db", str(database)])

        assert result.exit_code != 0
        assert "no runs recorded" in result.output

    def test_full_is_opt_in(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "h.db"
        secret = "account 4111111111111111"
        with RunStore(database) as store:
            store.save_batch(_batch())
            store.save_run(
                _run(
                    results=[
                        Result(
                            id="r1",
                            case_id="q1",
                            traces=[_trace("t", input=[{"content": secret}])],
                        )
                    ]
                )
            )

        default = CliRunner().invoke(app, ["show", "run-1", "--db", str(database)])
        expanded = CliRunner().invoke(app, ["show", "run-1", "--full", "--db", str(database)])

        assert "4111111111111111" not in default.output
        assert "4111111111111111" in expanded.output

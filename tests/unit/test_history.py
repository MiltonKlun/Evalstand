"""Reading history back (task 5.3).

The danger in a read path is not that it crashes — it is that it returns
something that looks right and differs from what was stored. So most of this
file is a round trip: write a Run, read it back, and assert that every
distinction the model draws survived.

The one that matters most is **what a figure claims**. `$0.0000` says a run was
free; `-` says nobody knows. A cost summed over partly-unpriced calls is a lower
bound, not a total, and presenting it as exact understates a bill — the same
class of error as a false pass, in the direction that costs money.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
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
from evalstand.reporting.console import render_history
from evalstand.storage import RunStore


@pytest.fixture
def store(tmp_path: Path) -> Any:
    with RunStore(tmp_path / "history.db") as opened:
        yield opened


def _batch(batch_id: str = "b1", **kwargs: Any) -> Batch:
    defaults: dict[str, Any] = {
        "kind": BatchKind.FULL,
        "status": BatchStatus.COMPLETED,
        "started_at": datetime.now(UTC),
    }
    return Batch(id=batch_id, **(defaults | kwargs))


def _run(run_id: str = "run-1", batch_id: str = "b1", **kwargs: Any) -> Run:
    defaults: dict[str, Any] = {
        "name": "qa",
        "filepath": "qa_eval.py",
        "status": RunStatus.COMPLETED,
        "started_at": datetime.now(UTC),
        "finished_at": datetime.now(UTC),
        "results": [
            Result(
                id=f"{run_id}-q1-0",
                case_id="q1",
                output="Paris",
                scores=[Score(scorer_name="exact", value=1.0, passed=True)],
            )
        ],
    }
    return Run(id=run_id, batch_id=batch_id, **(defaults | kwargs))


def _trace(trace_id: str, cost: float | None) -> Trace:
    return Trace(
        id=trace_id,
        name="model call",
        model="gpt-4o-mini",
        duration_ms=10,
        input_tokens=10,
        output_tokens=3,
        cost_usd=cost,
    )


def _rendered(runs: list[Run], batches: list[Batch | None] | None = None) -> str:
    pairs = list(zip(runs, batches or [None] * len(runs), strict=True))
    console = Console(width=140, force_terminal=False, record=True)
    console.print(render_history(pairs))
    return console.export_text()


class TestARunSurvivesTheRoundTrip:
    """A read that loses a distinction is worse than one that fails: the
    resulting table looks correct."""

    def test_the_aggregates_match_what_was_stored(self, store: RunStore) -> None:
        """The reason whole Runs are loaded rather than aggregated in SQL. Two
        definitions of "mean" would drift, and neither would say so."""
        original = _run(
            results=[
                Result(
                    id="run-1-q1-0",
                    case_id="q1",
                    output="Paris",
                    scores=[
                        Score(scorer_name="exact", value=1.0, passed=True),
                        Score(scorer_name="levenshtein", value=0.6),
                        Score.from_error("judge", "rate limited"),
                    ],
                    traces=[_trace("t1", 0.0002)],
                )
            ]
        )
        store.save_batch(_batch())
        store.save_run(original)

        loaded = store.load_run("run-1")

        assert loaded is not None
        assert loaded.mean_score == original.mean_score
        assert loaded.total_cost_usd == original.total_cost_usd
        assert loaded.errored_score_count == original.errored_score_count

    def test_the_three_kinds_of_score_stay_distinguishable(self, store: RunStore) -> None:
        """A verdict, a bare value, and an error are three different things.
        Flattening any pair of them recreates a false pass."""
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(
                        id="run-1-q1-0",
                        case_id="q1",
                        output="x",
                        scores=[
                            Score(scorer_name="exact", value=1.0, passed=True),
                            Score(scorer_name="levenshtein", value=0.6),
                            Score.from_error("judge", "rate limited"),
                        ],
                    )
                ]
            )
        )

        loaded = store.load_run("run-1")
        assert loaded is not None
        by_name = {s.scorer_name: s for s in loaded.results[0].scores}

        assert by_name["exact"].passed is True
        assert by_name["levenshtein"].passed is None, "an absent verdict became a failure"
        assert by_name["judge"].value is None
        assert by_name["judge"].error is not None
        assert by_name["judge"].counts_towards_mean is False

    def test_the_trace_tree_survives(self, store: RunStore) -> None:
        """`parent_id` is what makes a Result carry a tree rather than a list."""
        parent = Trace(id="t-root", name="task", duration_ms=50)
        child = Trace(id="t-call", parent_id="t-root", name="model call", model="m", duration_ms=40)
        store.save_batch(_batch())
        store.save_run(
            _run(results=[Result(id="run-1-q1-0", case_id="q1", traces=[parent, child])])
        )

        loaded = store.load_run("run-1")
        assert loaded is not None
        parents = {t.id: t.parent_id for t in loaded.results[0].traces}

        assert parents == {"t-root": None, "t-call": "t-root"}

    def test_a_structured_output_keeps_its_type(self, store: RunStore) -> None:
        """`compare` will need to read these field by field."""
        store.save_batch(_batch())
        store.save_run(_run(results=[Result(id="run-1-q1-0", case_id="q1", output={"total": 42})]))

        loaded = store.load_run("run-1")
        assert loaded is not None
        assert loaded.results[0].output == {"total": 42}

    def test_an_errored_result_keeps_its_frames(self, store: RunStore) -> None:
        """Captured at the raise precisely so a failure stays debuggable."""
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(
                        id="run-1-q1-0",
                        case_id="q1",
                        error="RuntimeError: boom",
                        error_frames=["  qa_eval.py:12 in answer"],
                    )
                ]
            )
        )

        loaded = store.load_run("run-1")
        assert loaded is not None
        assert loaded.results[0].error_frames == ["  qa_eval.py:12 in answer"]

    def test_an_unknown_run_is_none_not_an_error(self, store: RunStore) -> None:
        assert store.load_run("no-such-run") is None


class TestListingRuns:
    def test_runs_come_back_newest_first(self, store: RunStore) -> None:
        store.save_batch(_batch())
        now = datetime.now(UTC)
        for index, offset in enumerate([2, 0, 1]):
            store.save_run(_run(f"run-{index}", started_at=now - timedelta(hours=offset)))

        assert [r.id for r in store.runs_for()] == ["run-1", "run-2", "run-0"]

    def test_a_run_with_no_timestamp_falls_to_the_bottom(self, store: RunStore) -> None:
        """NULLs sort last under DESC in SQLite. Worth pinning: the opposite
        would let an undated run displace real history at the top."""
        store.save_batch(_batch())
        store.save_run(_run("run-dated", started_at=datetime.now(UTC)))
        store.save_run(_run("run-undated", started_at=None))

        assert [r.id for r in store.runs_for()] == ["run-dated", "run-undated"]

    def test_filtering_by_name(self, store: RunStore) -> None:
        store.save_batch(_batch())
        store.save_run(_run("run-a", name="qa"))
        store.save_run(_run("run-b", name="summarise"))

        assert [r.id for r in store.runs_for("qa")] == ["run-a"]

    def test_the_limit_is_honoured(self, store: RunStore) -> None:
        store.save_batch(_batch())
        for index in range(5):
            store.save_run(_run(f"run-{index}"))

        assert len(store.runs_for(limit=2)) == 2

    def test_a_cancelled_batch_is_excluded(self, store: RunStore) -> None:
        """Its aggregate describes a subset of the cases, so listing it beside
        a full run invites a comparison between different questions."""
        store.save_batch(_batch("b-done", status=BatchStatus.COMPLETED))
        store.save_batch(_batch("b-stopped", status=BatchStatus.CANCELLED))
        store.save_run(_run("run-done", batch_id="b-done"))
        store.save_run(_run("run-stopped", batch_id="b-stopped"))

        assert [r.id for r in store.runs_for()] == ["run-done"]

    def test_eval_names_are_listed(self, store: RunStore) -> None:
        """So a user who mistypes an eval name is told what does exist."""
        store.save_batch(_batch())
        store.save_run(_run("run-a", name="qa"))
        store.save_run(_run("run-b", name="summarise"))

        assert store.eval_names() == ["qa", "summarise"]

    def test_the_batch_is_reachable_from_a_run(self, store: RunStore) -> None:
        """Provenance lives on the Batch, and the history table needs it."""
        store.save_batch(_batch(git_sha="a" * 40, git_dirty=True))
        store.save_run(_run())

        batch = store.batch_for("run-1")
        assert batch is not None
        assert batch.git_sha == "a" * 40
        assert batch.git_dirty is True


class TestTheCostColumnNeverOverstatesCertainty:
    """`$0.0000` and `-` are different claims, and so are `$0.0021` and
    `$0.0021+`. Each pair differs in what a reader is entitled to conclude."""

    def test_a_fully_priced_run_shows_an_exact_figure(self) -> None:
        run = _run(results=[Result(id="r1", case_id="q1", traces=[_trace("t1", 0.0002)])])
        assert "$0.0002" in _rendered([run])
        assert "$0.0002+" not in _rendered([run])

    def test_a_partly_priced_run_is_marked_as_a_lower_bound(self) -> None:
        """The sum is what was priced, not what was spent. Showing it as exact
        understates a bill."""
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    traces=[_trace("t1", 0.0002), _trace("t2", None)],
                )
            ]
        )
        assert "$0.0002+" in _rendered([run])

    def test_a_run_with_no_priced_calls_shows_nothing(self) -> None:
        """Nothing has been shown to be free."""
        run = _run(results=[Result(id="r1", case_id="q1", traces=[_trace("t1", None)])])
        assert "$0.0000" not in _rendered([run])

    def test_a_run_that_made_no_calls_shows_nothing(self) -> None:
        """The bug this caught: with no traces at all, `cost_is_complete` is
        True and the total is 0, so the naive path printed `$0.0000` — claiming
        a run was free when in truth there was nothing to price."""
        run = _run(results=[Result(id="r1", case_id="q1", output="x")])
        assert "$0.0000" not in _rendered([run])


class TestTheCommitColumnIsHonest:
    def test_a_clean_commit_shows_its_short_sha(self) -> None:
        rendered = _rendered([_run()], [_batch(git_sha="abcdef12" + "0" * 32, git_dirty=False)])
        assert "abcdef12" in rendered
        assert "abcdef12*" not in rendered

    def test_a_dirty_tree_is_marked(self) -> None:
        """A run recorded against a SHA whose tree it did not reflect looks
        reproducible and is not."""
        assert "abcdef12*" in _rendered(
            [_run()], [_batch(git_sha="abcdef12" + "0" * 32, git_dirty=True)]
        )

    def test_a_run_with_no_commit_shows_a_gap(self) -> None:
        rendered = _rendered([_run()], [_batch()])
        assert "-" in rendered

    def test_a_missing_batch_does_not_break_the_table(self) -> None:
        assert _rendered([_run()], [None])


class TestTheTableAgreesWithTheRun:
    """Every figure comes from the Run's own properties, so the table cannot
    disagree with a fresh run about the same data."""

    def test_the_mean_matches_the_model(self, store: RunStore) -> None:
        run = _run(
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    scores=[Score(scorer_name="s", value=0.5)],
                )
            ]
        )
        assert "0.50" in _rendered([run])
        assert run.mean_score == 0.5

    def test_an_unmeasured_run_shows_no_mean(self) -> None:
        """Every score errored. `0.00` would claim the task did badly."""
        run = _run(
            results=[Result(id="r1", case_id="q1", scores=[Score.from_error("judge", "down")])]
        )
        assert run.mean_score is None
        rendered = _rendered([run])
        assert "0.00" not in rendered

    def test_unjudged_cases_are_not_counted_as_passes(self) -> None:
        """A continuous scorer sets no verdict, so the pass column has no
        denominator to report."""
        run = _run(
            results=[
                Result(id="r1", case_id="q1", scores=[Score(scorer_name="levenshtein", value=0.9)])
            ]
        )
        rendered = _rendered([run])
        assert "1/1" not in rendered


class TestTheHistoryCommand:
    def test_it_lists_stored_runs(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "h.db"
        with RunStore(database) as store:
            store.save_batch(_batch(git_sha="a" * 40, git_dirty=False))
            store.save_run(_run())

        result = CliRunner().invoke(app, ["history", "--db", str(database)])

        assert result.exit_code == 0
        assert "run-1" in result.output
        assert "qa" in result.output

    def test_an_empty_database_says_so_rather_than_printing_nothing(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "empty.db"
        RunStore(database).close()

        result = CliRunner().invoke(app, ["history", "--db", str(database)])

        assert result.exit_code == 0
        assert "no runs recorded" in result.output

    def test_an_unknown_eval_name_lists_the_known_ones(self, tmp_path: Path) -> None:
        """ "No runs for X" and "no runs at all" send a user looking in
        completely different places."""
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "h.db"
        with RunStore(database) as store:
            store.save_batch(_batch())
            store.save_run(_run(name="qa"))

        result = CliRunner().invoke(app, ["history", "typo", "--db", str(database)])

        assert "typo" in result.output
        assert "qa" in result.output, "the user is not told what does exist"

    def test_a_dirty_run_is_explained_not_just_marked(self, tmp_path: Path) -> None:
        """An asterisk with no legend is a puzzle, not information."""
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "h.db"
        with RunStore(database) as store:
            store.save_batch(_batch(git_sha="a" * 40, git_dirty=True))
            store.save_run(_run())

        result = CliRunner().invoke(app, ["history", "--db", str(database)])

        assert "uncommitted changes" in result.output


class TestTheReadPathPreservesUncertainty:
    """The write path was already careful to store "unknown" as NULL. These
    assert the *read* path does not quietly turn it back into a claim — a
    distinction that survives storage and dies on retrieval is no distinction
    at all.
    """

    def test_an_unchecked_tree_is_read_back_as_unknown(self, store: RunStore) -> None:
        """`git_dirty` is None when the tree was never checked. Reading it as
        False would assert a clean checkout nobody verified, and the history
        table would show a commit with no warning beside it."""
        store.save_batch(_batch(git_sha="a" * 40, git_dirty=None))
        store.save_run(_run())

        batch = store.batch_for("run-1")
        assert batch is not None
        assert batch.git_dirty is None, "an unchecked tree was read back as clean"

    def test_a_clean_tree_is_still_read_back_as_clean(self, store: RunStore) -> None:
        """The other half: the fix must not make every tree unknown."""
        store.save_batch(_batch(git_sha="a" * 40, git_dirty=False))
        store.save_run(_run())

        batch = store.batch_for("run-1")
        assert batch is not None
        assert batch.git_dirty is False

    def test_an_unpriced_call_is_read_back_as_unpriced(self, store: RunStore) -> None:
        """`cost_usd` None means the call could not be priced. Reading it as
        0.0 would make `cost_is_complete` true and turn a lower bound into an
        exact figure — understating a bill with no sign that it had."""
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(
                        id="run-1-q1-0",
                        case_id="q1",
                        traces=[_trace("t1", 0.0002), _trace("t2", None)],
                    )
                ]
            )
        )

        loaded = store.load_run("run-1")
        assert loaded is not None
        assert loaded.unpriced_call_count == 1
        assert loaded.cost_is_complete is False

    def test_an_unreadable_timestamp_does_not_lose_the_run(self, store: RunStore) -> None:
        """A malformed timestamp must not make a whole run unreadable. The
        scores are the measurement; losing them to a formatting problem would
        be a worse failure than an unknown time."""
        store.save_batch(_batch())
        store.save_run(_run())
        store.connection.execute("UPDATE runs SET started_at = 'not a timestamp'")

        loaded = store.load_run("run-1")

        assert loaded is not None, "one bad timestamp discarded the whole run"
        assert loaded.started_at is None
        assert loaded.mean_score == 1.0, "the measurement survived"


class TestResultsComeBackInDeclarationOrder:
    """`runner.py` promises it: "a report whose rows shuffle between runs
    cannot be read or diffed, so results are collected back into declaration
    order."

    Storage broke that promise as soon as a suite had ten cases, because it
    read them back `ORDER BY case_id` and text ordering puts q10 before q2. A
    user reading a 30-case report could not match it against their eval file.
    """

    def test_ten_or_more_cases_keep_their_order(self, store: RunStore) -> None:
        """Ten is where it starts: with nine cases, alphabetical and
        declaration order happen to agree, and the bug is invisible."""
        declared = [f"q{index}" for index in range(1, 13)]
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[Result(id=f"run-1-{case}", case_id=case, output=case) for case in declared]
            )
        )

        loaded = store.load_run("run-1")
        assert loaded is not None
        assert [r.case_id for r in loaded.results] == declared

    def test_an_arbitrary_declared_order_is_preserved(self, store: RunStore) -> None:
        """Cases are not always named in a sortable way. Whatever order the
        eval file declared is the order the report must show."""
        declared = ["zebra", "apple", "monkey", "banana"]
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[Result(id=f"run-1-{case}", case_id=case, output=case) for case in declared]
            )
        )

        loaded = store.load_run("run-1")
        assert loaded is not None
        assert [r.case_id for r in loaded.results] == declared

    def test_repeats_of_one_case_stay_in_index_order(self, store: RunStore) -> None:
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(id=f"run-1-q1-{index}", case_id="q1", repeat_index=index)
                    for index in range(4)
                ]
            )
        )

        loaded = store.load_run("run-1")
        assert loaded is not None
        assert [r.repeat_index for r in loaded.results] == [0, 1, 2, 3]

    def test_rows_written_before_the_ordinal_existed_still_load(self, store: RunStore) -> None:
        """The column is nullable so the migration needs no backfill. A row
        with no ordinal sorts last rather than jumbling the ones that have
        one — a database that could not be read after upgrading would be a
        worse failure than a wrong order."""
        store.save_batch(_batch())
        store.save_run(
            _run(
                results=[
                    Result(id="run-1-q1", case_id="q1"),
                    Result(id="run-1-q2", case_id="q2"),
                ]
            )
        )
        store.connection.execute("UPDATE results SET ordinal = NULL WHERE case_id = 'q1'")

        loaded = store.load_run("run-1")
        assert loaded is not None
        assert {r.case_id for r in loaded.results} == {"q1", "q2"}
        assert loaded.results[-1].case_id == "q1", "an unordered row should sort last"

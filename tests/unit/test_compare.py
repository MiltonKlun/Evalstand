"""Comparing two runs (task 5.5).

The acceptance criterion is unusual: it is about what the output must *not*
say. `evalstand` has no significance testing, so it cannot tell a real change
from noise — and a tool that said "regression" without being able to support the
claim would be worse than one that stayed quiet, because the word would be
believed.

The second theme is evidence. A case whose pass state moved because somebody
edited its expected value says nothing about the task. `case_snapshots` exists
so those can be told apart, and without the check every dataset edit would be
reported as a finding about the model.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from evalstand.comparison import compare_runs
from evalstand.models import (
    Batch,
    BatchKind,
    BatchStatus,
    Case,
    Result,
    Run,
    RunStatus,
    Score,
)
from evalstand.reporting.console import render_comparison
from evalstand.storage import RunStore

# Every word CONTEXT.md tells us to avoid for a Delta or a Flip, plus the
# obvious synonyms. A comparison that uses any of them is asserting something
# this version cannot support.
VERDICT_WORDS = [
    "regress",
    "improve",
    "better",
    "worse",
    "degrad",
    "deteriorat",
    "gain",
    "drop",
    "decline",
    "progress",
    "success",
    "failure rate",
]


def _case(case_id: str, expected: str = "Paris") -> Case:
    return Case(id=case_id, input="a question", expected=expected)


def _run(
    run_id: str,
    verdicts: dict[str, bool | None],
    *,
    values: dict[str, float] | None = None,
    name: str = "qa",
    **kwargs: Any,
) -> Run:
    """A run whose cases carry the given pass states.

    `values` overrides the score value, so a case can have a value without a
    verdict — the shape a continuous scorer produces.
    """
    results = []
    for case_id, verdict in verdicts.items():
        value = (values or {}).get(case_id)
        if value is None:
            value = 1.0 if verdict else 0.0
        results.append(
            Result(
                id=f"{run_id}-{case_id}-0",
                case_id=case_id,
                output=f"output from {run_id}",
                scores=[Score(scorer_name="exact", value=value, passed=verdict)],
            )
        )

    defaults: dict[str, Any] = {
        "batch_id": "b1",
        "name": name,
        "filepath": "qa_eval.py",
        "status": RunStatus.COMPLETED,
        "started_at": datetime.now(UTC),
    }
    return Run(id=run_id, results=results, **(defaults | kwargs))


def _rendered(comparison: Any) -> str:
    console = Console(width=200, force_terminal=False, record=True)
    console.print(render_comparison(comparison))
    return console.export_text()


class TestTheOutputAssertsNoVerdict:
    """Task 5.5's acceptance, checked mechanically rather than by eye."""

    def test_no_verdict_word_appears_when_scores_fell(self) -> None:
        comparison = compare_runs(
            _run("a", {"q1": True, "q2": True}), _run("b", {"q1": False, "q2": False})
        )
        rendered = _rendered(comparison).lower()

        found = [word for word in VERDICT_WORDS if word in rendered]
        assert not found, f"the comparison asserted a verdict: {found}"

    def test_no_verdict_word_appears_when_scores_rose(self) -> None:
        """The same restraint in the flattering direction. A tool that stays
        quiet about bad news but celebrates good news is not neutral."""
        comparison = compare_runs(
            _run("a", {"q1": False, "q2": False}), _run("b", {"q1": True, "q2": True})
        )
        rendered = _rendered(comparison).lower()

        found = [word for word in VERDICT_WORDS if word in rendered]
        assert not found, f"the comparison asserted a verdict: {found}"

    def test_the_limitation_is_stated_every_time(self) -> None:
        """Not buried in documentation. A reader who takes a delta as proof has
        been misled by the tool, and the tool is the only thing present at the
        moment they might do so."""
        rendered = _rendered(compare_runs(_run("a", {"q1": True}), _run("b", {"q1": True})))

        assert "no significance testing" in rendered

    def test_the_delta_is_shown_as_an_arithmetic_difference(self) -> None:
        comparison = compare_runs(
            _run("a", {"q1": True, "q2": True}), _run("b", {"q1": True, "q2": False})
        )
        rendered = _rendered(comparison)

        assert "-0.50" in rendered
        assert "changed by" in rendered


class TestFlips:
    def test_a_case_that_changed_verdict_is_reported(self) -> None:
        comparison = compare_runs(_run("a", {"q1": False}), _run("b", {"q1": True}))

        assert [(f.case_id, f.passed_before, f.passed_after) for f in comparison.flips] == [
            ("q1", False, True)
        ]

    def test_a_case_that_did_not_change_is_not_reported(self) -> None:
        comparison = compare_runs(_run("a", {"q1": True}), _run("b", {"q1": True}))

        assert comparison.flips == []

    def test_both_outputs_are_shown_side_by_side(self) -> None:
        """The point of a flip table: seeing what the model said before and
        after, without opening two runs by hand."""
        rendered = _rendered(compare_runs(_run("a", {"q1": True}), _run("b", {"q1": False})))

        assert "output from a" in rendered
        assert "output from b" in rendered

    def test_a_case_with_no_verdict_on_either_side_is_not_a_flip(self) -> None:
        """A continuous scorer gives no pass state, so there is nothing to
        flip. Inventing one would be the inference CONTEXT.md forbids."""
        comparison = compare_runs(
            _run("a", {"q1": None}, values={"q1": 0.9}),
            _run("b", {"q1": None}, values={"q1": 0.2}),
        )

        assert comparison.flips == []

    def test_a_case_scored_by_repeats_is_not_compared(self) -> None:
        """A case run five times has five outcomes and no single pass state.
        Picking one would be a choice with no basis."""
        repeated = Run(
            id="a",
            batch_id="b1",
            name="qa",
            filepath="f",
            status=RunStatus.COMPLETED,
            results=[
                Result(
                    id=f"a-q1-{index}",
                    case_id="q1",
                    repeat_index=index,
                    scores=[Score(scorer_name="exact", value=1.0, passed=True)],
                )
                for index in range(3)
            ],
        )

        comparison = compare_runs(repeated, _run("b", {"q1": False}))

        assert comparison.flips == []


class TestAmendedCasesAreNotEvidence:
    """A pass state that moved because the expectation moved says nothing about
    the task. Conflating the two attributes a dataset edit to the model."""

    def test_an_edited_case_is_reported_separately(self) -> None:
        before = {"q1": _case("q1", "Paris").content_hash()}
        after = {"q1": _case("q1", "Paris, France").content_hash()}

        comparison = compare_runs(
            _run("a", {"q1": True}),
            _run("b", {"q1": False}),
            hashes_before=before,
            hashes_after=after,
        )

        assert comparison.flips == [], "an edited case was reported as a flip"
        assert [a.case_id for a in comparison.amended] == ["q1"]

    def test_an_edited_input_counts_too(self) -> None:
        """An edited input under the same id is no longer the same test."""
        before = {"q1": Case(id="q1", input="France", expected="Paris").content_hash()}
        after = {"q1": Case(id="q1", input="FRANCE", expected="Paris").content_hash()}

        comparison = compare_runs(
            _run("a", {"q1": True}),
            _run("b", {"q1": False}),
            hashes_before=before,
            hashes_after=after,
        )

        assert comparison.flips == []
        assert len(comparison.amended) == 1

    def test_an_unedited_case_is_still_a_flip(self) -> None:
        """The check must not be so eager it suppresses genuine evidence."""
        digest = _case("q1").content_hash()

        comparison = compare_runs(
            _run("a", {"q1": True}),
            _run("b", {"q1": False}),
            hashes_before={"q1": digest},
            hashes_after={"q1": digest},
        )

        assert [f.case_id for f in comparison.flips] == ["q1"]
        assert comparison.amended == []

    def test_the_amended_table_explains_why_it_is_separate(self) -> None:
        before = {"q1": _case("q1", "Paris").content_hash()}
        after = {"q1": _case("q1", "Lyon").content_hash()}
        rendered = _rendered(
            compare_runs(
                _run("a", {"q1": True}),
                _run("b", {"q1": False}),
                hashes_before=before,
                hashes_after=after,
            )
        )

        assert "edited between these runs" in rendered

    def test_missing_snapshots_do_not_mark_everything_amended(self) -> None:
        """An old run recorded before snapshots existed would otherwise have
        every case reported as edited, which is noise rather than information.
        """
        comparison = compare_runs(
            _run("a", {"q1": True}), _run("b", {"q1": False}), hashes_before={}, hashes_after={}
        )

        assert comparison.amended == []
        assert len(comparison.flips) == 1


class TestScoreMovesKeepContinuousScorersVisible:
    """A suite scored only by levenshtein or a judge has no pass states, so the
    flip list is empty even when every number moved."""

    def test_a_large_move_is_reported(self) -> None:
        comparison = compare_runs(
            _run("a", {"q1": None}, values={"q1": 0.9}),
            _run("b", {"q1": None}, values={"q1": 0.2}),
        )

        assert [(m.case_id, m.before, m.after) for m in comparison.moves] == [("q1", 0.9, 0.2)]

    def test_a_tiny_move_is_not(self) -> None:
        """A noise floor, not a significance test: a table of two hundred cases
        must not bury the ones that moved a long way."""
        comparison = compare_runs(
            _run("a", {"q1": None}, values={"q1": 0.90}),
            _run("b", {"q1": None}, values={"q1": 0.91}),
        )

        assert comparison.moves == []

    def test_moves_are_ordered_by_size(self) -> None:
        comparison = compare_runs(
            _run("a", {"q1": None, "q2": None}, values={"q1": 0.9, "q2": 0.5}),
            _run("b", {"q1": None, "q2": None}, values={"q1": 0.8, "q2": 0.1}),
        )

        assert [m.case_id for m in comparison.moves] == ["q2", "q1"]

    def test_the_table_is_titled_by_what_happened_not_what_it_means(self) -> None:
        rendered = _rendered(
            compare_runs(
                _run("a", {"q1": None}, values={"q1": 0.9}),
                _run("b", {"q1": None}, values={"q1": 0.2}),
            )
        )

        assert "largest score changes" in rendered
        assert "regress" not in rendered.lower()


class TestWhatCannotBeCompared:
    def test_an_unmeasured_run_has_no_delta(self) -> None:
        """A run whose scores all errored has no mean. Subtracting from None
        would invent a number, and "unmeasured" is not "unchanged"."""
        errored = Run(
            id="b",
            batch_id="b1",
            name="qa",
            filepath="f",
            status=RunStatus.COMPLETED,
            results=[Result(id="b-q1-0", case_id="q1", scores=[Score.from_error("exact", "down")])],
        )

        comparison = compare_runs(_run("a", {"q1": True}), errored)
        delta = comparison.scorer_deltas[0]

        assert delta.delta is None
        assert delta.is_comparable is False

    def test_a_scorer_in_only_one_run_shows_no_delta(self) -> None:
        rendered = _rendered(
            compare_runs(_run("a", {"q1": True}), _run("b", {"q1": True}, name="qa"))
        )
        assert "changed by" in rendered

    def test_cases_present_in_only_one_run_are_flagged(self) -> None:
        """A mean over a different set of cases is a different measurement."""
        comparison = compare_runs(
            _run("a", {"q1": True, "q2": True}), _run("b", {"q1": True, "q3": True})
        )

        assert comparison.only_before == ["q2"]
        assert comparison.only_after == ["q3"]
        assert "not like-for-like" in _rendered(comparison)

    def test_comparing_two_different_evals_says_so(self) -> None:
        """Two evals measure different things, so their means are not
        comparable at all."""
        rendered = _rendered(
            compare_runs(_run("a", {"q1": True}), _run("b", {"q1": True}, name="summarise"))
        )

        assert "different evals" in rendered

    def test_a_changed_task_source_is_flagged(self) -> None:
        """Without this a reader would attribute a code change to the model."""
        rendered = _rendered(
            compare_runs(
                _run("a", {"q1": True}, task_source_hash="a" * 64),
                _run("b", {"q1": False}, task_source_hash="b" * 64),
            )
        )

        assert "task's source changed" in rendered

    def test_identical_runs_say_nothing_differs(self) -> None:
        comparison = compare_runs(_run("a", {"q1": True}), _run("b", {"q1": True}))

        assert comparison.is_empty
        assert "nothing differs" in _rendered(comparison)


class TestTheCompareCommand:
    @pytest.fixture
    def seeded(self, tmp_path: Path) -> Path:
        database = tmp_path / "h.db"
        with RunStore(database) as store:
            store.save_batch(
                Batch(
                    id="b1",
                    kind=BatchKind.FULL,
                    status=BatchStatus.COMPLETED,
                    started_at=datetime.now(UTC),
                )
            )
            now = datetime.now(UTC)
            store.save_run(
                _run("run-old", {"q1": False, "q2": True}, started_at=now - timedelta(hours=1)),
                cases=[_case("q1"), _case("q2", "Tokyo")],
            )
            store.save_run(
                _run("run-new", {"q1": True, "q2": True}, started_at=now),
                cases=[_case("q1"), _case("q2", "Tokyo")],
            )
        return database

    def test_it_compares_two_stored_runs(self, seeded: Path) -> None:
        from typer.testing import CliRunner

        from evalstand.cli import app

        result = CliRunner().invoke(app, ["compare", "run-old", "run-new", "--db", str(seeded)])

        assert result.exit_code == 0
        assert "q1" in result.output
        assert "no significance testing" in result.output

    def test_it_uses_the_stored_snapshots(self, tmp_path: Path) -> None:
        """The command must pass the hashes. Without them every edited case is
        reported as a finding about the task."""
        from typer.testing import CliRunner

        from evalstand.cli import app

        database = tmp_path / "h.db"
        with RunStore(database) as store:
            store.save_batch(
                Batch(
                    id="b1",
                    kind=BatchKind.FULL,
                    status=BatchStatus.COMPLETED,
                    started_at=datetime.now(UTC),
                )
            )
            store.save_run(_run("run-old", {"q1": True}), cases=[_case("q1", "Paris")])
            store.save_run(_run("run-new", {"q1": False}), cases=[_case("q1", "Lyon")])

        result = CliRunner().invoke(app, ["compare", "run-old", "run-new", "--db", str(database)])

        assert "edited between these runs" in result.output
        assert "pass state changed" not in result.output

    def test_an_unknown_run_exits_non_zero(self, seeded: Path) -> None:
        from typer.testing import CliRunner

        from evalstand.cli import app

        result = CliRunner().invoke(app, ["compare", "run-old", "run-nope", "--db", str(seeded)])

        assert result.exit_code != 0
        assert "run-nope" in result.output

    def test_the_output_carries_no_verdict(self, seeded: Path) -> None:
        """The acceptance criterion, asserted on the real command."""
        from typer.testing import CliRunner

        from evalstand.cli import app

        result = CliRunner().invoke(app, ["compare", "run-old", "run-new", "--db", str(seeded)])
        lowered = result.output.lower()

        found = [word for word in VERDICT_WORDS if word in lowered]
        assert not found, f"the command asserted a verdict: {found}"


class TestVerdictsAcrossSeveralScorers:
    """A case is judged by every scorer that gave a verdict, not by whichever
    one happened to be first or most generous."""

    def test_a_case_with_one_failing_scorer_has_not_passed(self) -> None:
        """`all`, not `any`. A case that failed one check has failed, and
        reporting it as a pass because another scorer liked it would be a false
        pass built out of disagreement."""
        mixed = Run(
            id="a",
            batch_id="b1",
            name="qa",
            filepath="f",
            status=RunStatus.COMPLETED,
            results=[
                Result(
                    id="a-q1-0",
                    case_id="q1",
                    scores=[
                        Score(scorer_name="exact", value=1.0, passed=True),
                        Score(scorer_name="contains", value=0.0, passed=False),
                    ],
                )
            ],
        )
        all_passing = Run(
            id="b",
            batch_id="b1",
            name="qa",
            filepath="f",
            status=RunStatus.COMPLETED,
            results=[
                Result(
                    id="b-q1-0",
                    case_id="q1",
                    scores=[
                        Score(scorer_name="exact", value=1.0, passed=True),
                        Score(scorer_name="contains", value=1.0, passed=True),
                    ],
                )
            ],
        )

        comparison = compare_runs(mixed, all_passing)

        assert [(f.passed_before, f.passed_after) for f in comparison.flips] == [(False, True)]

    def test_a_scorer_with_no_verdict_does_not_veto_the_others(self) -> None:
        """A continuous scorer alongside a binary one must not erase the
        binary one's judgement."""
        run = Run(
            id="a",
            batch_id="b1",
            name="qa",
            filepath="f",
            status=RunStatus.COMPLETED,
            results=[
                Result(
                    id="a-q1-0",
                    case_id="q1",
                    scores=[
                        Score(scorer_name="exact", value=1.0, passed=True),
                        Score(scorer_name="levenshtein", value=0.7),
                    ],
                )
            ],
        )

        comparison = compare_runs(run, _run("b", {"q1": False}))

        assert [f.passed_before for f in comparison.flips] == [True]


class TestSnapshotsOnOnlyOneSide:
    """Runs recorded before snapshots existed, compared against newer ones."""

    def test_a_run_with_no_snapshots_does_not_mark_everything_amended(self) -> None:
        """The older run has no case_snapshots rows. Treating that absence as
        evidence of an edit would hide every genuine flip behind a wall of
        "this case was edited" — noise that buries the finding."""
        digest = _case("q1").content_hash()

        comparison = compare_runs(
            _run("a", {"q1": True}),
            _run("b", {"q1": False}),
            hashes_before={"q1": digest},
            hashes_after={},
        )

        assert comparison.amended == [], "an absent snapshot was read as an edit"
        assert [f.case_id for f in comparison.flips] == ["q1"]

    def test_the_reverse_direction_too(self) -> None:
        comparison = compare_runs(
            _run("a", {"q1": True}),
            _run("b", {"q1": False}),
            hashes_before={},
            hashes_after={"q1": _case("q1").content_hash()},
        )

        assert comparison.amended == []
        assert [f.case_id for f in comparison.flips] == ["q1"]


class TestTheDocumentedLimitation:
    """`docs/ci.md` is required by the plan to state the limitation explicitly.
    A doc that drifts from the code is worse than none: a reader who follows it
    loses time and then trust."""

    def test_ci_md_says_a_delta_is_not_a_verdict(self) -> None:
        doc = (Path(__file__).resolve().parents[2] / "docs" / "ci.md").read_text(encoding="utf-8")

        assert "no significance testing" in doc
        assert "regression or an improvement" in doc

    def test_ci_md_explains_amended_cases(self) -> None:
        """The distinction is subtle enough that a user will not infer it from
        a table heading alone."""
        doc = (Path(__file__).resolve().parents[2] / "docs" / "ci.md").read_text(encoding="utf-8")

        assert "edited between these runs" in doc

    def test_the_documented_caveat_is_the_one_the_tool_prints(self) -> None:
        """The doc quotes the line the command emits. If the wording in code
        changes, this fails rather than the two drifting apart."""
        doc = (Path(__file__).resolve().parents[2] / "docs" / "ci.md").read_text(encoding="utf-8")
        rendered = _rendered(compare_runs(_run("a", {"q1": True}), _run("b", {"q1": True})))

        quoted = "a delta is an arithmetic difference"
        assert quoted in doc
        assert quoted in rendered

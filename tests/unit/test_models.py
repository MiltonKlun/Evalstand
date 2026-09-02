"""Round-trip serialisation and invariants for the core models (task 1.1)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from evalstand.models import (
    Batch,
    BatchKind,
    BatchStatus,
    Case,
    Result,
    Run,
    RunStatus,
    Score,
    Trace,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _roundtrip(model: ModelT) -> ModelT:
    """Serialise to JSON text and back, proving the model survives storage."""
    return type(model).model_validate_json(model.model_dump_json())


class TestCase:
    def test_roundtrips(self) -> None:
        case = Case(id="q1", input="capital of France?", expected="Paris")
        assert _roundtrip(case) == case

    def test_expected_is_optional(self) -> None:
        """Not every Case has a reference output; scorers may judge output alone."""
        case = Case(id="q1", input="write a poem")
        assert case.expected is None
        assert _roundtrip(case) == case

    def test_id_is_required(self) -> None:
        """Identity is supplied, never derived from content."""
        with pytest.raises(ValidationError):
            Case(input="no id")  # type: ignore[call-arg]

    def test_id_must_not_be_blank(self) -> None:
        with pytest.raises(ValidationError):
            Case(id="   ", input="x")

    def test_input_may_be_structured(self) -> None:
        case = Case(id="q1", input={"doc": "invoice.pdf", "page": 2}, expected=["a", "b"])
        assert _roundtrip(case) == case

    def test_content_hash_is_stable_across_instances(self) -> None:
        a = Case(id="q1", input="x", expected="y")
        b = Case(id="q1", input="x", expected="y")
        assert a.content_hash() == b.content_hash()

    def test_content_hash_ignores_key_order(self) -> None:
        a = Case(id="q1", input={"a": 1, "b": 2})
        b = Case(id="q1", input={"b": 2, "a": 1})
        assert a.content_hash() == b.content_hash()

    def test_content_hash_covers_input_and_expected(self) -> None:
        """An edited input under the same id is no longer the same test."""
        base = Case(id="q1", input="x", expected="y")
        assert base.content_hash() != Case(id="q1", input="CHANGED", expected="y").content_hash()
        assert base.content_hash() != Case(id="q1", input="x", expected="CHANGED").content_hash()

    def test_content_hash_excludes_id_and_metadata(self) -> None:
        """The hash answers 'is this the same test?', not 'is this the same record?'."""
        a = Case(id="q1", input="x", expected="y", metadata={"note": "one"})
        b = Case(id="q2", input="x", expected="y", metadata={"note": "two"})
        assert a.content_hash() == b.content_hash()


class TestScore:
    def test_roundtrips(self) -> None:
        score = Score(scorer_name="exact", value=1.0, passed=True)
        assert _roundtrip(score) == score

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_accepts_the_unit_interval(self, value: float) -> None:
        assert Score(scorer_name="s", value=value).value == value

    @pytest.mark.parametrize("value", [-0.01, 1.01, 100.0])
    def test_rejects_values_outside_the_unit_interval(self, value: float) -> None:
        """One scale everywhere: a 0-100 threshold judging a 0-1 value is a trap."""
        with pytest.raises(ValidationError):
            Score(scorer_name="s", value=value)

    def test_passed_defaults_to_none(self) -> None:
        """A continuous scorer leaves it unset rather than inventing a cutoff."""
        assert Score(scorer_name="levenshtein", value=0.87).passed is None

    def test_errored_score_carries_no_value(self) -> None:
        score = Score.from_error("judge", "rate limited after 3 attempts")
        assert score.error == "rate limited after 3 attempts"
        assert score.value is None
        assert score.passed is None
        assert _roundtrip(score) == score

    def test_a_score_cannot_be_both_valued_and_errored(self) -> None:
        with pytest.raises(ValidationError):
            Score(scorer_name="s", value=0.5, error="boom")

    def test_a_score_must_be_one_or_the_other(self) -> None:
        with pytest.raises(ValidationError):
            Score(scorer_name="s")

    def test_counts_towards_mean_only_when_it_succeeded(self) -> None:
        assert Score(scorer_name="s", value=0.0).counts_towards_mean is True
        assert Score.from_error("s", "boom").counts_towards_mean is False


class TestTrace:
    def test_roundtrips(self) -> None:
        trace = Trace(
            id="t1",
            name="chat",
            started_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
            duration_ms=1234,
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.0001,
        )
        assert _roundtrip(trace) == trace

    def test_root_trace_has_no_parent(self) -> None:
        assert Trace(id="t1", name="chat", duration_ms=1).parent_id is None

    def test_child_points_at_its_parent(self) -> None:
        """Nesting is what distinguishes a tree from a flat list."""
        child = Trace(id="t2", parent_id="t1", name="judge", duration_ms=1)
        assert child.parent_id == "t1"

    def test_a_trace_cannot_parent_itself(self) -> None:
        with pytest.raises(ValidationError):
            Trace(id="t1", parent_id="t1", name="chat", duration_ms=1)


class TestResult:
    def test_roundtrips(self) -> None:
        result = Result(
            id="r1",
            case_id="q1",
            repeat_index=0,
            output="Paris",
            latency_ms=120,
            scores=[Score(scorer_name="exact", value=1.0, passed=True)],
        )
        assert _roundtrip(result) == result

    def test_mean_score_excludes_errored_scores(self) -> None:
        """An infrastructure failure is not evidence the task did badly."""
        result = Result(
            id="r1",
            case_id="q1",
            output="x",
            scores=[
                Score(scorer_name="a", value=1.0),
                Score(scorer_name="b", value=0.0),
                Score.from_error("judge", "429"),
            ],
        )
        assert result.mean_score == pytest.approx(0.5)
        assert result.errored_score_count == 1

    def test_mean_score_is_none_when_every_score_errored(self) -> None:
        """Zero would be a fabricated judgement."""
        result = Result(
            id="r1", case_id="q1", output="x", scores=[Score.from_error("judge", "429")]
        )
        assert result.mean_score is None
        assert result.errored_score_count == 1

    def test_mean_score_is_none_without_scores(self) -> None:
        assert Result(id="r1", case_id="q1", output="x").mean_score is None

    def test_a_failed_task_records_the_error(self) -> None:
        """One case raising must not abort the run."""
        result = Result(id="r1", case_id="q1", output=None, error="task raised ValueError")
        assert result.error == "task raised ValueError"
        assert _roundtrip(result) == result

    def test_traces_nest_into_a_tree(self) -> None:
        result = Result(
            id="r1",
            case_id="q1",
            output="x",
            traces=[
                Trace(id="t1", name="task", duration_ms=10),
                Trace(id="t2", parent_id="t1", name="judge", duration_ms=5),
            ],
        )
        assert _roundtrip(result) == result
        roots = [t for t in result.traces if t.parent_id is None]
        assert len(roots) == 1

    def test_total_cost_sums_its_traces(self) -> None:
        result = Result(
            id="r1",
            case_id="q1",
            output="x",
            traces=[
                Trace(id="t1", name="a", duration_ms=1, cost_usd=0.01),
                Trace(id="t2", parent_id="t1", name="b", duration_ms=1, cost_usd=0.02),
            ],
        )
        assert result.total_cost_usd == pytest.approx(0.03)


class TestRun:
    def test_roundtrips(self) -> None:
        run = Run(
            id="run1",
            batch_id="b1",
            name="basic-qa",
            filepath="examples/toy/qa_eval.py",
            status=RunStatus.COMPLETED,
        )
        assert _roundtrip(run) == run

    def test_mean_score_excludes_errored_scores_across_results(self) -> None:
        run = Run(
            id="run1",
            batch_id="b1",
            name="qa",
            filepath="x.py",
            results=[
                Result(
                    id="r1", case_id="q1", output="x", scores=[Score(scorer_name="a", value=1.0)]
                ),
                Result(
                    id="r2", case_id="q2", output="x", scores=[Score(scorer_name="a", value=0.0)]
                ),
                Result(id="r3", case_id="q3", output="x", scores=[Score.from_error("a", "429")]),
            ],
        )
        assert run.mean_score == pytest.approx(0.5)
        assert run.errored_score_count == 1

    def test_mean_score_is_reported_per_scorer(self) -> None:
        run = Run(
            id="run1",
            batch_id="b1",
            name="qa",
            filepath="x.py",
            results=[
                Result(
                    id="r1",
                    case_id="q1",
                    output="x",
                    scores=[
                        Score(scorer_name="exact", value=1.0),
                        Score(scorer_name="levenshtein", value=0.5),
                    ],
                ),
                Result(
                    id="r2",
                    case_id="q2",
                    output="x",
                    scores=[
                        Score(scorer_name="exact", value=0.0),
                        Score(scorer_name="levenshtein", value=0.9),
                    ],
                ),
            ],
        )
        assert run.mean_scores_by_scorer() == {
            "exact": pytest.approx(0.5),
            "levenshtein": pytest.approx(0.7),
        }


class TestBatch:
    def test_roundtrips(self) -> None:
        batch = Batch(id="b1", kind=BatchKind.FULL, status=BatchStatus.COMPLETED)
        assert _roundtrip(batch) == batch

    def test_a_cancelled_batch_is_excluded_from_history(self) -> None:
        """A half-finished batch that looked complete would drag every mean."""
        cancelled = Batch(id="b1", kind=BatchKind.PARTIAL, status=BatchStatus.CANCELLED)
        assert cancelled.is_comparable is False
        assert Batch(id="b2", kind=BatchKind.FULL, status=BatchStatus.COMPLETED).is_comparable

    def test_a_running_batch_is_not_yet_comparable(self) -> None:
        assert (
            Batch(id="b1", kind=BatchKind.FULL, status=BatchStatus.RUNNING).is_comparable is False
        )


class TestJsonSerialisability:
    """Every model must survive the database round trip (task 1.1 acceptance)."""

    def test_all_models_dump_to_plain_json(self) -> None:
        models = [
            Case(id="q1", input="x", expected="y"),
            Score(scorer_name="s", value=1.0),
            Trace(id="t1", name="chat", duration_ms=1),
            Result(id="r1", case_id="q1", output="x"),
            Run(id="run1", batch_id="b1", name="qa", filepath="x.py"),
            Batch(id="b1", kind=BatchKind.FULL),
        ]
        for model in models:
            assert json.loads(model.model_dump_json())

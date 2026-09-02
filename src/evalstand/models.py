"""The core domain models.

The vocabulary here is defined in `CONTEXT.md` and should not drift from it. In
particular: a Batch contains Runs, a Run contains Results, and a Result is what
happened when one Case met the Task once. Those three names are distinct
granularities, not synonyms for "execution".
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Batch",
    "BatchKind",
    "BatchStatus",
    "Case",
    "Result",
    "Run",
    "RunStatus",
    "Score",
    "Trace",
]


def _canonical_json(value: Any) -> str:
    """Serialise deterministically, so equal content always hashes equally."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Case(_Model):
    """One test input, with an optional reference output.

    `id` is supplied rather than derived: a content-derived id would make an
    edited Case look like a different Case, which is exactly what Amended Case
    detection needs to catch.
    """

    id: Annotated[str, Field(min_length=1)]
    input: Any
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _id_is_not_blank(self) -> Self:
        if not self.id.strip():
            raise ValueError("Case.id must not be blank")
        return self

    def content_hash(self) -> str:
        """Identify *what this Case tests*, ignoring its id and metadata.

        Covers `expected` and `input` both: an edited input under an unchanged
        id is no longer the same test, and reporting that as a genuine flip
        would be a claim about the Task that the evidence does not support.
        """
        payload = _canonical_json({"input": self.input, "expected": self.expected})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Score(_Model):
    """One Scorer's judgement of one Result.

    A Score either succeeded (it has a `value`) or errored (it has an `error`),
    never both and never neither. An errored Score is excluded from means rather
    than counted as zero: a scorer that broke is not evidence the Task did badly.
    """

    scorer_name: Annotated[str, Field(min_length=1)]
    value: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    passed: bool | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _value_xor_error(self) -> Self:
        if self.value is not None and self.error is not None:
            raise ValueError("a Score carries either a value or an error, not both")
        if self.value is None and self.error is None:
            raise ValueError("a Score must carry either a value or an error")
        return self

    @classmethod
    def from_error(cls, scorer_name: str, error: str) -> Score:
        return cls(scorer_name=scorer_name, error=error)

    @property
    def counts_towards_mean(self) -> bool:
        return self.error is None


class Trace(_Model):
    """A record of one model call made inside a Task.

    Traces nest: `parent_id` is what makes a Result carry a tree rather than a
    flat list, so a judge scorer's call appears beneath the call it judges.
    """

    id: Annotated[str, Field(min_length=1)]
    parent_id: str | None = None
    name: str
    started_at: datetime | None = None
    duration_ms: Annotated[int, Field(ge=0)]
    model: str | None = None
    input: Any = None
    output: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @model_validator(mode="after")
    def _no_self_parenting(self) -> Self:
        if self.parent_id is not None and self.parent_id == self.id:
            raise ValueError("a Trace cannot be its own parent")
        return self


def _mean(values: list[float]) -> float | None:
    """Mean of the values that exist. None, never zero, when there are none."""
    return sum(values) / len(values) if values else None


class Result(_Model):
    """What happened when one Case met the Task once.

    With repeats, one Case produces several Results in the same Run,
    distinguished by `repeat_index`.
    """

    id: Annotated[str, Field(min_length=1)]
    case_id: Annotated[str, Field(min_length=1)]
    repeat_index: Annotated[int, Field(ge=0)] = 0
    output: Any = None
    error: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    scores: list[Score] = Field(default_factory=list)
    traces: list[Trace] = Field(default_factory=list)

    @property
    def mean_score(self) -> float | None:
        return _mean(
            [s.value for s in self.scores if s.counts_towards_mean and s.value is not None]
        )

    @property
    def errored_score_count(self) -> int:
        return sum(1 for s in self.scores if not s.counts_towards_mean)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd or 0.0 for t in self.traces)


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Run(_Model):
    """One execution of one Eval, covering all its Cases."""

    id: Annotated[str, Field(min_length=1)]
    batch_id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    filepath: str
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    repeat_n: Annotated[int, Field(ge=1)] = 1
    results: list[Result] = Field(default_factory=list)

    def _successful_scores(self) -> list[Score]:
        return [s for r in self.results for s in r.scores if s.counts_towards_mean]

    @property
    def mean_score(self) -> float | None:
        return _mean([s.value for s in self._successful_scores() if s.value is not None])

    @property
    def errored_score_count(self) -> int:
        return sum(r.errored_score_count for r in self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.total_cost_usd for r in self.results)

    def mean_scores_by_scorer(self) -> dict[str, float]:
        """Per-scorer means. Reported separately because averaging across
        scorers that measure different things produces a meaningless number."""
        buckets: dict[str, list[float]] = {}
        for score in self._successful_scores():
            if score.value is not None:
                buckets.setdefault(score.scorer_name, []).append(score.value)
        return {name: sum(vs) / len(vs) for name, vs in buckets.items()}


class BatchKind(StrEnum):
    FULL = "full"
    PARTIAL = "partial"


class BatchStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Batch(_Model):
    """One invocation of the tool, containing every Run it produced.

    Exists so a watch-mode re-run of three Evals is grouped rather than
    appearing as three unrelated executions.
    """

    id: Annotated[str, Field(min_length=1)]
    kind: BatchKind
    status: BatchStatus = BatchStatus.RUNNING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    runs: list[Run] = Field(default_factory=list)

    @property
    def is_comparable(self) -> bool:
        """Only a Batch that ran to completion may inform history or comparison."""
        return self.status is BatchStatus.COMPLETED

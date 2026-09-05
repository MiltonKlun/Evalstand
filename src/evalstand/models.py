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

        # A verdict that contradicts its own value is the most dangerous thing a
        # Score can hold: `passed=True, value=0.0` is counted in the pass column
        # and omitted from the failures table, so a wrong answer is reported as
        # correct *and* made invisible. Rejected here rather than reconciled,
        # because there is no way to know which half the scorer meant.
        #
        # Only the endpoints are constrained. A scorer is free to pass at 0.8 or
        # fail at 0.3 — where the line sits is its judgement to make — but
        # nothing can both score zero and have passed.
        if self.passed is True and self.value == 0.0:
            raise ValueError("a Score cannot pass with a value of 0.0")
        if self.passed is False and self.value == 1.0:
            raise ValueError("a Score cannot fail with a value of 1.0")
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
    error_frames: list[str] = Field(default_factory=list)
    """The user's own stack frames, when the task raised.

    Captured at the raise, because whoever catches an exception to keep the run
    alive is also the last place the traceback still exists. Without this a
    report can say *what* broke but never *where*, which is the difference
    between a message and a fix.
    """
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    scores: list[Score] = Field(default_factory=list)
    traces: list[Trace] = Field(default_factory=list)

    @model_validator(mode="after")
    def _traces_form_a_forest(self) -> Self:
        """Reject trace shapes that are not a forest.

        ADR 0006 promises a tree. A cycle has no root, so the tree is
        unreachable and a naive walk descends forever; a dangling parent orphans
        a node silently. Both are caught here, where the data is built, rather
        than in the renderer that would hang on them.
        """
        if not self.traces:
            return self

        ids = [trace.id for trace in self.traces]
        if len(ids) != len(set(ids)):
            duplicates = sorted({node for node in ids if ids.count(node) > 1})
            raise ValueError(f"duplicate trace ids: {', '.join(duplicates)}")

        parents = {trace.id: trace.parent_id for trace in self.traces}
        known = set(parents)

        for trace in self.traces:
            if trace.parent_id is not None and trace.parent_id not in known:
                raise ValueError(f"trace {trace.id!r} names an unknown parent {trace.parent_id!r}")

        # Walk upward from each node; a node that revisits itself is in a
        # cycle. Nodes already proven acyclic are not walked again, which keeps
        # this linear rather than quadratic on deep chains.
        settled: set[str] = set()
        for start in known:
            if start in settled:
                continue
            path: list[str] = []
            seen: set[str] = set()
            current: str | None = start
            while current is not None and current not in settled:
                if current in seen:
                    raise ValueError(f"traces form a cycle through {current!r}")
                seen.add(current)
                path.append(current)
                current = parents[current]
            settled.update(path)

        return self

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

    model_config_used: dict[str, Any] = Field(default_factory=dict)
    """What the run was configured with, for provenance (task 5.2).

    Named `model_config_used` because pydantic reserves `model_config` on every
    BaseModel; using that name would silently shadow the class's own settings.
    """

    task_source_hash: str | None = None
    """A hash of the task's source, so a comparison can tell a changed model
    from a changed task. Two runs with different hashes measured different
    code, whatever else stayed the same."""

    model_calls: Annotated[int, Field(ge=0)] = 0
    cache_hits: Annotated[int, Field(ge=0)] = 0
    cache_bypassed: bool = False
    """True when this run skipped the cache — repeats always do. Recorded
    because a bypassed run spends the full amount every time, and the plan
    calls that the easiest way to run up a bill by accident."""

    @model_validator(mode="after")
    def _results_are_distinct_executions(self) -> Self:
        """Reject two Results claiming to be the same execution.

        `(case_id, repeat_index)` names one execution, so a duplicate means the
        Run holds two answers to a question that has one. The mean would average
        both and the pass count would read 1/2 for a single case — a report that
        cannot be reconciled with what actually ran.

        The runner cannot produce this, but the model is what guarantees it, and
        Runs are also assembled from stored rows and from user code.
        """
        seen: set[tuple[str, int]] = set()
        duplicates: set[tuple[str, int]] = set()
        for result in self.results:
            key = (result.case_id, result.repeat_index)
            if key in seen:
                duplicates.add(key)
            seen.add(key)

        if duplicates:
            named = ", ".join(f"{case}#{index}" for case, index in sorted(duplicates))
            raise ValueError(f"duplicate executions in one run: {named}")
        return self

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

    @property
    def unpriced_call_count(self) -> int:
        """Model calls whose cost could not be determined.

        `total_cost_usd` sums what is known, so a run with unpriced calls
        reports a **lower bound** rather than a total. A reader shown that
        figure as exact would understate their bill, which is the same class of
        error as a false pass: quietly wrong in the direction that matters.

        Callers that display a cost must consult this and say so.
        """
        return sum(1 for r in self.results for t in r.traces if t.cost_usd is None)

    @property
    def cost_is_complete(self) -> bool:
        """Whether `total_cost_usd` is the whole cost or only part of it."""
        return self.unpriced_call_count == 0

    @property
    def cache_hit_rate(self) -> float | None:
        """Hits over model calls, or None when nothing was called.

        None rather than 0.0: a run that made no calls did not have a 0% hit
        rate, it had no rate at all.
        """
        if not self.model_calls:
            return None
        return self.cache_hits / self.model_calls

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

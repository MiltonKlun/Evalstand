"""What the run view claims, tested without a terminal.

The TUI's widgets are a projection of `RunState`, so this is where the mistakes
that matter can be caught: a row that calls a broken scorer a failing model, a
footer that reports an unpriced run as free, a pass rate that counts an absent
judgement against the task.

Every rule here already exists in `reporting/console.py`. The risk being tested
is a *second* renderer quietly disagreeing with the first — two views of one run
that cannot both be true, with nothing on screen saying which is.
"""

from __future__ import annotations

from evalstand.models import Result, Run, Score, Trace
from evalstand.tui.state import RunState, Totals, row_for


def _result(
    case_id: str = "q1",
    *,
    repeat_index: int = 0,
    scores: list[Score] | None = None,
    traces: list[Trace] | None = None,
    error: str | None = None,
    output: object = "answer",
    latency_ms: int | None = None,
) -> Result:
    return Result(
        id=f"e-{case_id}-{repeat_index}",
        case_id=case_id,
        repeat_index=repeat_index,
        output=output,
        error=error,
        latency_ms=latency_ms,
        scores=scores if scores is not None else [Score(scorer_name="s", value=1.0, passed=True)],
        traces=traces or [],
    )


def _trace(
    name: str = "call", *, cost: float | None = 0.001, model: str | None = "gpt-4o"
) -> Trace:
    return Trace(
        id=f"t-{name}-{cost}-{model}",
        name=name,
        duration_ms=10,
        model=model,
        cost_usd=cost,
    )


class TestStatusTellsTheFourOutcomesApart:
    """ "failed" and "unmeasured" look alike in a table and mean opposite
    things. One is the model getting it wrong; the other is the scorer breaking
    and the model's performance being unknown."""

    def test_a_passing_case_says_pass(self) -> None:
        assert row_for(_result()).status == "pass"

    def test_a_failing_case_says_fail(self) -> None:
        row = row_for(_result(scores=[Score(scorer_name="s", value=0.0, passed=False)]))

        assert row.status == "fail"

    def test_a_crashed_task_says_error_not_fail(self) -> None:
        """The task never produced an answer, so it did not answer wrongly."""
        row = row_for(_result(error="RuntimeError: boom", scores=[]))

        assert row.status == "error"

    def test_a_case_whose_scorers_all_errored_says_unmeasured(self) -> None:
        """The most dangerous confusion in the table: reporting a scorer outage
        as a model regression would send someone to debug a prompt that is
        working fine."""
        row = row_for(_result(scores=[Score.from_error("s", "no api key")]))

        assert row.status == "unmeasured"

    def test_a_continuous_score_with_no_verdict_says_scored(self) -> None:
        """A scorer that gave a value and declined to judge has not failed the
        case, and the table must not decide otherwise on its behalf."""
        row = row_for(_result(scores=[Score(scorer_name="ratio", value=0.42)]))

        assert row.status == "scored"

    def test_one_failing_scorer_fails_the_case(self) -> None:
        """A case passes only if nothing failed it."""
        row = row_for(
            _result(
                scores=[
                    Score(scorer_name="a", value=1.0, passed=True),
                    Score(scorer_name="b", value=0.0, passed=False),
                ]
            )
        )

        assert row.status == "fail"

    def test_a_verdict_survives_a_broken_sibling_scorer(self) -> None:
        """One scorer erroring does not erase the judgement another gave."""
        row = row_for(
            _result(
                scores=[
                    Score(scorer_name="a", value=1.0, passed=True),
                    Score.from_error("b", "boom"),
                ]
            )
        )

        assert row.status == "pass"


class TestCostIsNeverOverstatedAsZero:
    def test_an_unpriced_call_shows_a_placeholder_not_free(self) -> None:
        """A call nobody could price has not been shown to cost nothing.
        `$0.0000` is a claim; `-` is the truth."""
        row = row_for(_result(traces=[_trace(cost=None)]))

        assert row.cost == "-"

    def test_a_case_that_made_no_calls_costs_nothing(self) -> None:
        """Genuinely free is different from unknown, and both must be sayable."""
        assert row_for(_result(traces=[])).cost == "$0.0000"

    def test_a_partly_priced_case_shows_what_is_known(self) -> None:
        row = row_for(_result(traces=[_trace("a", cost=0.002), _trace("b", cost=None)]))

        assert row.cost == "$0.0020"

    def test_absent_latency_shows_a_placeholder(self) -> None:
        assert row_for(_result(latency_ms=None)).latency == "-"


class TestRowsAreOnePerExecution:
    def test_repeats_get_distinct_keys(self) -> None:
        """Keying on case_id alone would make three repeats overwrite each other
        into one row, so the table would show one execution where three
        happened."""
        keys = {row_for(_result("q1", repeat_index=i)).key for i in range(3)}

        assert keys == {"q1#0", "q1#1", "q1#2"}

    def test_a_duplicate_execution_is_refused(self) -> None:
        """The same guarantee `Run` enforces. Two rows for one execution would
        make the footer's count disagree with the table above it."""
        state = RunState("e")

        assert state.add(_result("q1")) is not None
        assert state.add(_result("q1")) is None
        assert len(state.results) == 1


class TestTotalsAreTrueAtEveryInstant:
    def test_the_pass_rate_excludes_unjudged_cases(self) -> None:
        """A continuous scorer that declined to judge has failed nothing, and
        counting it in the denominator would understate the pass rate."""
        state = RunState("e")
        state.add(_result("q1"))
        state.add(_result("q2", scores=[Score(scorer_name="ratio", value=0.5)]))

        assert state.totals().pass_rate == "1/1"

    def test_an_unmeasured_run_has_no_pass_rate(self) -> None:
        """Zero passes out of zero judged is not 0%; it is unknown."""
        state = RunState("e")
        state.add(_result("q1", scores=[Score.from_error("s", "boom")]))

        assert state.totals().pass_rate == "-"

    def test_cost_is_marked_as_a_lower_bound_when_calls_went_unpriced(self) -> None:
        """Showing the known sum bare would understate the user's bill — wrong
        in the direction that costs money."""
        state = RunState("e")
        state.add(_result("q1", traces=[_trace("a", cost=0.005), _trace("b", cost=None)]))

        assert state.totals().cost == "$0.0050 (+1 unpriced)"

    def test_a_fully_priced_run_states_its_cost_plainly(self) -> None:
        state = RunState("e")
        state.add(_result("q1", traces=[_trace(cost=0.005)]))

        assert state.totals().cost == "$0.0050"

    def test_progress_is_unknown_before_the_case_count_is(self) -> None:
        """A bar at 0% claims nothing has happened; None says we do not know
        how much there is to do."""
        assert RunState("e").totals().progress is None

    def test_progress_reflects_what_has_landed(self) -> None:
        state = RunState("e", expected=4)
        state.add(_result("q1"))

        assert state.totals().progress == 0.25

    def test_progress_never_exceeds_one(self) -> None:
        """A miscounted expectation must not render a bar past its end."""
        state = RunState("e", expected=1)
        state.add(_result("q1"))
        state.add(_result("q2"))

        assert state.totals().progress == 1.0

    def test_the_cache_hit_rate_is_unknown_when_nothing_was_called(self) -> None:
        """A run that made no calls did not have a 0% hit rate; it had no rate."""
        assert RunState("e").totals().cache_hit_rate == "-"

    def test_the_cache_hit_rate_counts_hits_against_calls(self) -> None:
        state = RunState("e")
        state.add(
            _result("q1", traces=[_trace("cached call", cost=0.0), _trace("call", cost=0.001)])
        )

        assert state.totals().cache_hit_rate == "50%"

    def test_errors_are_counted(self) -> None:
        state = RunState("e")
        state.add(_result("q1", error="boom", scores=[]))
        state.add(_result("q2"))

        assert state.totals().errors == 1


class TestFailuresFilter:
    def test_it_shows_wrong_answers_crashes_and_unmeasured_cases(self) -> None:
        state = RunState("e")
        state.add(_result("ok"))
        state.add(_result("wrong", scores=[Score(scorer_name="s", value=0.0, passed=False)]))
        state.add(_result("crashed", error="boom", scores=[]))
        state.add(_result("broken", scores=[Score.from_error("s", "boom")]))

        assert [row.case_id for row in state.failures()] == ["wrong", "crashed", "broken"]

    def test_a_continuous_score_is_not_a_failure(self) -> None:
        """Filtering it into the failures list would invent the verdict the
        scorer deliberately declined to give."""
        state = RunState("e")
        state.add(_result("q1", scores=[Score(scorer_name="ratio", value=0.1)]))

        assert state.failures() == []


class TestPerScorerMeans:
    def test_means_are_reported_per_scorer(self) -> None:
        """Averaging across scorers that measure different things produces a
        figure that means nothing."""
        state = RunState("e")
        state.add(
            _result(
                "q1",
                scores=[
                    Score(scorer_name="exact", value=1.0, passed=True),
                    Score(scorer_name="ratio", value=0.5),
                ],
            )
        )
        state.add(
            _result(
                "q2",
                scores=[
                    Score(scorer_name="exact", value=0.0, passed=False),
                    Score(scorer_name="ratio", value=0.7),
                ],
            )
        )

        assert state.mean_scores_by_scorer() == {"exact": 0.5, "ratio": 0.6}

    def test_an_errored_score_is_excluded_rather_than_counted_as_zero(self) -> None:
        """A scorer that broke is not evidence the task did badly."""
        state = RunState("e")
        state.add(_result("q1", scores=[Score(scorer_name="s", value=1.0, passed=True)]))
        state.add(_result("q2", scores=[Score.from_error("s", "boom")]))

        assert state.mean_scores_by_scorer() == {"s": 1.0}

    def test_exclusion_is_by_errored_state_not_by_a_missing_value(self) -> None:
        """The filter asks `counts_towards_mean`, not merely whether a value is
        present.

        Today those agree, because a `Score` carries a value or an error and
        never both. But that invariant lives in `models.py`, and a mean here
        that silently depends on it would start counting broken scorers as
        zeroes the moment the model was relaxed — reporting a scorer outage as
        the task performing badly, with nothing on screen to say otherwise.

        Asserted with a hand-built Score that has both, which the model forbids
        and this module must therefore never meet by accident.
        """
        state = RunState("e")
        state.add(_result("q1", scores=[Score(scorer_name="s", value=1.0, passed=True)]))

        broken = Score.model_construct(scorer_name="s", value=0.0, error="boom", metadata={})
        state.results.append(_result("q2", scores=[]).model_copy(update={"scores": [broken]}))

        assert state.mean_scores_by_scorer() == {"s": 1.0}


class TestCustomColumns:
    """Task 6.5. A plain dict of callables, so this spends none of the
    remaining public-API slots."""

    def test_a_derived_column_is_rendered(self) -> None:
        state = RunState("e", columns={"length": lambda r: len(str(r.output))})
        row = state.add(_result("q1", output="hello"))

        assert row is not None and row.extra == {"length": "5"}

    def test_a_raising_column_does_not_cost_the_row(self) -> None:
        """A bug in a display helper must not destroy the measurement it was
        called to display — the rule the runner applies to sinks and scorers."""

        def boom(result: Result) -> object:
            raise ValueError("bad column")

        state = RunState("e", columns={"broken": boom})
        row = state.add(_result("q1"))

        assert row is not None
        assert row.extra == {"broken": "!"}
        assert row.status == "pass"

    def test_a_column_returning_none_shows_a_placeholder_not_the_word_none(self) -> None:
        """`str(None)` is the string "None", which has already produced one
        false pass in this project."""
        state = RunState("e", columns={"maybe": lambda r: None})
        row = state.add(_result("q1"))

        assert row is not None and row.extra == {"maybe": "-"}


class TestAdoptingTheFinishedRun:
    def test_the_view_takes_the_stored_run_as_truth(self) -> None:
        """Once the Run exists, the view shows what was recorded rather than
        what it accumulated — so the screen cannot drift from the database."""
        state = RunState("e")
        state.add(_result("q1"))

        run = Run(
            id="run-1",
            batch_id="b",
            name="e",
            filepath="x_eval.py",
            results=[_result("q1"), _result("q2")],
        )
        state.as_run(run)

        assert [row.case_id for row in state.rows()] == ["q1", "q2"]

    def test_a_result_can_be_found_for_the_detail_view(self) -> None:
        state = RunState("e")
        state.add(_result("q1", repeat_index=1))

        found = state.result_for("q1#1")

        assert found is not None and found.case_id == "q1"

    def test_an_unknown_key_finds_nothing(self) -> None:
        assert RunState("e").result_for("nope#0") is None


class TestTotalsIsHonestByConstruction:
    def test_a_run_with_no_calls_and_no_cost_reports_unknown(self) -> None:
        """Distinct from `$0.0000`: nothing was measured about cost at all."""
        totals = Totals(
            completed=0,
            expected=None,
            passed=0,
            judged=0,
            errors=0,
            cost_usd=0.0,
            unpriced_calls=0,
            model_calls=0,
            cache_hits=0,
        )

        assert totals.cost == "-"

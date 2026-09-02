"""The public API: `evaluate()` and the registry (task 2.1).

`evaluate()` registers and returns. It performs no I/O, starts no event loop, and
makes no model calls — see ADR 0004. Every test here asserts that boundary in
some form, because the whole design rests on it.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalstand import Case, evaluate
from evalstand.api import DuplicateEvalNameError, Eval, registry


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """Each test gets an empty registry; module-level state leaks otherwise."""
    registry.clear()
    yield
    registry.clear()


CASES = [
    Case(id="q1", input="capital of France?", expected="Paris"),
    Case(id="q2", input="2 + 2?", expected="4"),
]


def _task(question: str) -> str:
    return "Paris"


def _scorer(output: Any, expected: Any, case: Case) -> float:
    return 1.0 if output == expected else 0.0


class TestRegistration:
    def test_registers_an_eval(self) -> None:
        evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer])
        assert [e.name for e in registry.evals()] == ["qa"]

    def test_returns_the_registered_eval(self) -> None:
        """Returning it lets a caller inspect what was declared."""
        declared = evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer])
        assert isinstance(declared, Eval)
        assert declared.name == "qa"

    def test_several_evals_coexist(self) -> None:
        evaluate(name="first", cases=CASES, task=_task, scorers=[_scorer])
        evaluate(name="second", cases=CASES, task=_task, scorers=[_scorer])
        assert sorted(e.name for e in registry.evals()) == ["first", "second"]

    def test_a_duplicate_name_is_refused(self) -> None:
        """Names are identity; a silent merge would join two evals' histories."""
        evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer])
        with pytest.raises(DuplicateEvalNameError, match="qa"):
            evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer])

    def test_the_duplicate_error_names_both_files(self) -> None:
        """A collision is unhelpful without saying where the other one is."""
        evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer], filepath="a/first_eval.py")
        with pytest.raises(DuplicateEvalNameError) as caught:
            evaluate(
                name="qa", cases=CASES, task=_task, scorers=[_scorer], filepath="b/second_eval.py"
            )
        assert "a/first_eval.py" in str(caught.value)
        assert "b/second_eval.py" in str(caught.value)

    def test_a_blank_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name"):
            evaluate(name="  ", cases=CASES, task=_task, scorers=[_scorer])

    def test_at_least_one_scorer_is_required(self) -> None:
        """An eval with no scorers measures nothing and would report a
        meaningless empty mean."""
        with pytest.raises(ValueError, match="scorer"):
            evaluate(name="qa", cases=CASES, task=_task, scorers=[])


class TestDefersEverything:
    """ADR 0004: registration must not execute anything."""

    def test_the_task_is_not_called(self) -> None:
        calls = 0

        def counting_task(question: str) -> str:
            nonlocal calls
            calls += 1
            return "x"

        evaluate(name="qa", cases=CASES, task=counting_task, scorers=[_scorer])
        assert calls == 0

    def test_a_callable_case_loader_is_not_called(self) -> None:
        """Loading cases may read files or hit a network; registration must not."""
        calls = 0

        def load() -> list[Case]:
            nonlocal calls
            calls += 1
            return CASES

        evaluate(name="qa", cases=load, task=_task, scorers=[_scorer])
        assert calls == 0

    def test_no_scorer_is_called(self) -> None:
        calls = 0

        def counting_scorer(output: Any, expected: Any, case: Case) -> float:
            nonlocal calls
            calls += 1
            return 1.0

        evaluate(name="qa", cases=CASES, task=_task, scorers=[counting_scorer])
        assert calls == 0


class TestCaseLoading:
    def test_accepts_a_plain_list(self) -> None:
        declared = evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer])
        assert [c.id for c in declared.load_cases()] == ["q1", "q2"]

    def test_accepts_a_callable(self) -> None:
        declared = evaluate(name="qa", cases=lambda: CASES, task=_task, scorers=[_scorer])
        assert [c.id for c in declared.load_cases()] == ["q1", "q2"]

    def test_accepts_an_async_callable(self) -> None:
        async def load() -> list[Case]:
            return CASES

        declared = evaluate(name="qa", cases=load, task=_task, scorers=[_scorer])
        assert [c.id for c in declared.load_cases()] == ["q1", "q2"]

    def test_loading_happens_on_demand_not_at_registration(self) -> None:
        """A loader that raises must not break collection of other evals."""

        def exploding() -> list[Case]:
            raise RuntimeError("dataset unavailable")

        declared = evaluate(name="qa", cases=exploding, task=_task, scorers=[_scorer])
        with pytest.raises(RuntimeError, match="dataset unavailable"):
            declared.load_cases()

    def test_auto_numbers_cases_that_arrive_without_ids(self) -> None:
        """A generated dataset rarely has natural ids; positional ones are
        assigned so history still has something stable to match on."""
        declared = evaluate(
            name="qa",
            cases=[{"input": "a", "expected": "b"}, {"input": "c", "expected": "d"}],
            task=_task,
            scorers=[_scorer],
        )
        assert [c.id for c in declared.load_cases()] == ["case_0001", "case_0002"]

    def test_a_supplied_dict_id_is_never_overwritten(self) -> None:
        """Q13: ids are supplied, never derived. Auto-numbering fills a gap; it
        must not replace an id the user chose, or history silently re-points at
        different content."""
        declared = evaluate(
            name="qa",
            cases=[{"id": "my-own-id", "input": "a"}, {"id": "second", "input": "b"}],
            task=_task,
            scorers=[_scorer],
        )
        assert [c.id for c in declared.load_cases()] == ["my-own-id", "second"]

    def test_numbering_fills_only_the_gaps(self) -> None:
        """A dataset may mix supplied and absent ids; each keeps what it had."""
        declared = evaluate(
            name="qa",
            cases=[{"input": "a"}, {"id": "kept", "input": "b"}, {"input": "c"}],
            task=_task,
            scorers=[_scorer],
        )
        assert [c.id for c in declared.load_cases()] == ["case_0001", "kept", "case_0003"]

    def test_numbering_follows_position_not_sequence(self) -> None:
        """case_0003 is the third case, whether or not the second was numbered.
        Position is what a regenerated dataset preserves; a running counter
        would shift every later id when one case gains an explicit id."""
        declared = evaluate(
            name="qa",
            cases=[{"id": "a", "input": "x"}, {"input": "y"}],
            task=_task,
            scorers=[_scorer],
        )
        assert [c.id for c in declared.load_cases()] == ["a", "case_0002"]

    def test_a_blank_dict_id_is_treated_as_absent(self) -> None:
        declared = evaluate(
            name="qa",
            cases=[{"id": "", "input": "a"}],
            task=_task,
            scorers=[_scorer],
        )
        assert [c.id for c in declared.load_cases()] == ["case_0001"]

    def test_a_case_object_id_is_never_touched(self) -> None:
        """Case objects carry their own id by construction; numbering must not
        reach them at all."""
        declared = evaluate(
            name="qa",
            cases=[Case(id="explicit", input="a")],
            task=_task,
            scorers=[_scorer],
        )
        assert [c.id for c in declared.load_cases()] == ["explicit"]

    def test_ids_are_not_derived_from_content(self) -> None:
        """Two cases with identical content but different ids stay distinct: a
        content-derived id would make an edited case look like a new one, which
        is what Amended Case detection has to catch."""
        declared = evaluate(
            name="qa",
            cases=[{"id": "first", "input": "same"}, {"id": "second", "input": "same"}],
            task=_task,
            scorers=[_scorer],
        )
        assert [c.id for c in declared.load_cases()] == ["first", "second"]

    def test_duplicate_case_ids_are_refused(self) -> None:
        """Two cases sharing an id make per-case history ambiguous."""
        declared = evaluate(
            name="qa",
            cases=[Case(id="same", input="a"), Case(id="same", input="b")],
            task=_task,
            scorers=[_scorer],
        )
        with pytest.raises(ValueError, match="duplicate case id"):
            declared.load_cases()

    @pytest.mark.anyio
    async def test_an_async_loader_works_inside_a_running_loop(self) -> None:
        """Phase 3's runner is async, so it will call this from inside a loop.
        `asyncio.run` cannot nest, so the sync path must not be the only one."""

        async def load() -> list[Case]:
            return CASES

        declared = evaluate(name="qa", cases=load, task=_task, scorers=[_scorer])
        assert [c.id for c in await declared.aload_cases()] == ["q1", "q2"]

    @pytest.mark.anyio
    async def test_a_sync_loader_also_works_through_the_async_path(self) -> None:
        declared = evaluate(name="qa", cases=lambda: CASES, task=_task, scorers=[_scorer])
        assert [c.id for c in await declared.aload_cases()] == ["q1", "q2"]

    @pytest.mark.anyio
    async def test_a_plain_list_works_through_the_async_path(self) -> None:
        declared = evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer])
        assert [c.id for c in await declared.aload_cases()] == ["q1", "q2"]

    def test_an_empty_dataset_is_refused(self) -> None:
        declared = evaluate(name="qa", cases=[], task=_task, scorers=[_scorer])
        with pytest.raises(ValueError, match="no cases"):
            declared.load_cases()


class TestPublicSurface:
    def test_exports_exactly_the_seven_agreed_names(self) -> None:
        """The cap exists to force the question; widening it needs an ADR."""
        import evalstand

        assert set(evalstand.__all__) == {
            "Case",
            "Result",
            "Score",
            "Trace",
            "evaluate",
            "scorer",
            "trace",
        }

    def test_run_and_batch_are_not_exported(self) -> None:
        """Users read them in reports; they never construct them."""
        import evalstand

        assert "Run" not in evalstand.__all__
        assert "Batch" not in evalstand.__all__

    def test_every_exported_name_resolves(self) -> None:
        import evalstand

        for name in evalstand.__all__:
            assert getattr(evalstand, name, None) is not None, name


class TestReRegistration:
    """Re-importing the same eval file must be idempotent.

    pytest imports a file twice when it is named twice on the command line, and
    watch mode re-imports on every change. Neither is a name collision, but a
    naive check cannot tell them apart from two different files clashing.
    """

    def test_the_same_eval_from_the_same_file_is_idempotent(self) -> None:
        evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer], filepath="a/qa_eval.py")
        evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer], filepath="a/qa_eval.py")
        assert len(registry) == 1

    def test_the_same_name_from_a_different_file_still_collides(self) -> None:
        evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer], filepath="a/one_eval.py")
        with pytest.raises(DuplicateEvalNameError):
            evaluate(
                name="qa", cases=CASES, task=_task, scorers=[_scorer], filepath="b/two_eval.py"
            )

    def test_re_registration_replaces_the_declaration(self) -> None:
        """Watch mode re-imports an edited file; the new definition must win."""
        evaluate(
            name="qa",
            cases=[Case(id="old", input="x")],
            task=_task,
            scorers=[_scorer],
            filepath="a/qa_eval.py",
        )
        evaluate(
            name="qa",
            cases=[Case(id="new", input="x")],
            task=_task,
            scorers=[_scorer],
            filepath="a/qa_eval.py",
        )
        declared = registry.get("qa")
        assert declared is not None
        assert [c.id for c in declared.load_cases()] == ["new"]

    def test_an_unknown_filepath_never_matches_another(self) -> None:
        """Two evals with no filepath are not evidence of being the same file."""
        evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer])
        with pytest.raises(DuplicateEvalNameError):
            evaluate(name="qa", cases=CASES, task=_task, scorers=[_scorer])

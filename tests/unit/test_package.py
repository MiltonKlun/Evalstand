"""The public API surface: everything `__all__` promises must actually work.

Written after `evalstand.trace` was found to be a Phase 2 placeholder that
raised `NotImplementedError`. Phase 3 implemented `trace()` in `tracing.py` and
never rewired the export, so the documented import crashed for four phases —
invisible because every internal caller reached past it to
`evalstand.tracing.trace`, and the only smoke test asserted a version string.

The lesson generalises: an export nothing internal uses is an export nothing
internal tests. So each name is not merely imported here but *used* the way a
reader of the README would use it.
"""

from __future__ import annotations

import evalstand


def test_the_package_reports_a_version() -> None:
    assert evalstand.__version__


def test_every_promised_name_exists() -> None:
    """`__all__` is a promise. A name listed but absent is an import error in
    someone else's code."""
    missing = [name for name in evalstand.__all__ if not hasattr(evalstand, name)]

    assert not missing, f"__all__ promises names the package does not have: {missing}"


def test_no_export_is_an_unimplemented_placeholder() -> None:
    """The bug this file exists for.

    A placeholder that raises is worse than a missing name: the import succeeds,
    so the failure arrives at runtime — in a user's task, mid-run, after they
    have already paid for the model calls that got that far.
    """
    import inspect

    for name in evalstand.__all__:
        obj = getattr(evalstand, name)
        if not callable(obj) or inspect.isclass(obj):
            continue
        try:
            source = inspect.getsource(obj)
        except (OSError, TypeError):  # pragma: no cover - C or synthesised objects
            continue
        assert "NotImplementedError" not in source, f"{name} is still a placeholder"


class TestEachExportDoesItsJob:
    """Imported *and* exercised. A name that imports but does not work is the
    exact shape of the defect this file was written for."""

    def test_trace_records_a_span(self) -> None:
        from evalstand.tracing import TraceCollector

        collector = TraceCollector()
        with collector, evalstand.trace("retrieve"):
            pass

        assert [t.name for t in collector.traces] == ["retrieve"]

    def test_trace_outside_a_case_does_nothing_rather_than_raising(self) -> None:
        """Someone experimenting at a REPL should not hit an error — which is
        also what made the broken export so easy to miss."""
        with evalstand.trace("no collector bound"):
            pass

    def test_evaluate_registers_an_eval(self) -> None:
        from evalstand.api import registry

        registry.clear()
        try:
            declared = evalstand.evaluate(
                name="package-smoke",
                cases=[evalstand.Case(id="q1", input="x", expected="y")],
                task=lambda value: "y",
                scorers=[lambda output, expected: 1.0],
            )
            assert declared.name == "package-smoke"
            assert registry.get("package-smoke") is declared
        finally:
            registry.clear()

    def test_scorer_validates_a_signature(self) -> None:
        """The decorator's purpose is to fail at import time rather than after
        a run has spent money."""

        @evalstand.scorer
        def usable(output: object, expected: object) -> float:
            return 1.0

        assert callable(usable)

    def test_the_models_construct(self) -> None:
        case = evalstand.Case(id="q1", input="x", expected="y")
        score = evalstand.Score(scorer_name="s", value=1.0, passed=True)
        result = evalstand.Result(id="r1", case_id=case.id, output="y", scores=[score])
        trace = evalstand.Trace(id="t1", name="call", duration_ms=1)

        assert result.mean_score == 1.0
        assert trace.parent_id is None

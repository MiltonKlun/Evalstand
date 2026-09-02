"""The CLI (Phase 2's exit criterion).

`run` delegates to pytest rather than reimplementing collection: a second
execution path would be a second thing to keep correct, and the two could
disagree about what an eval run means.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from evalstand.cli import app

runner = CliRunner()

PASSING_EVAL = """
from evalstand import Case, evaluate
from evalstand.scorers import exact

evaluate(
    name="cli-toy",
    cases=[Case(id="q1", input="x", expected="x"), Case(id="q2", input="y", expected="y")],
    task=lambda value: value,
    scorers=[exact],
)
"""

FAILING_EVAL = """
from evalstand import Case, evaluate
from evalstand.scorers import exact

evaluate(
    name="cli-bad",
    cases=[Case(id="q1", input="x", expected="right")],
    task=lambda value: "wrong",
    scorers=[exact],
)
"""


class TestRun:
    def test_runs_an_eval_file(self, tmp_path: Path) -> None:
        (tmp_path / "toy_eval.py").write_text(PASSING_EVAL, encoding="utf-8")
        result = runner.invoke(app, ["run", str(tmp_path)])
        assert result.exit_code == 0
        assert "cli-toy" in result.output

    def test_prints_the_summary(self, tmp_path: Path) -> None:
        (tmp_path / "toy_eval.py").write_text(PASSING_EVAL, encoding="utf-8")
        output = runner.invoke(app, ["run", str(tmp_path)]).output
        assert "scorer" in output
        assert "2/2" in output

    def test_a_failing_eval_exits_non_zero(self, tmp_path: Path) -> None:
        """CI gates on this, so it must not be swallowed."""
        (tmp_path / "bad_eval.py").write_text(FAILING_EVAL, encoding="utf-8")
        assert runner.invoke(app, ["run", str(tmp_path)]).exit_code != 0

    def test_k_selects_cases(self, tmp_path: Path) -> None:
        (tmp_path / "toy_eval.py").write_text(PASSING_EVAL, encoding="utf-8")
        output = runner.invoke(app, ["run", str(tmp_path), "-k", "q1"]).output
        assert "1 passed" in output
        assert "1 deselected" in output

    def test_a_directory_with_no_evals_is_not_an_error(self, tmp_path: Path) -> None:
        """An empty directory is a normal state, not a failure."""
        result = runner.invoke(app, ["run", str(tmp_path)])
        assert "error" not in result.output.lower()


class TestVersion:
    def test_prints_the_version(self) -> None:
        from evalstand import __version__

        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestHelp:
    def test_bare_invocation_shows_help(self) -> None:
        """Running the tool with no arguments should teach, not fail silently."""
        result = runner.invoke(app, [])
        assert "run" in result.output

    def test_run_documents_its_arguments(self) -> None:
        assert "eval" in runner.invoke(app, ["run", "--help"]).output.lower()

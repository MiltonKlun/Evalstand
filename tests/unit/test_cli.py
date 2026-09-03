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


SLOW_EVAL = """
import asyncio

from evalstand import Case, evaluate
from evalstand.scorers import exact


async def task(value):
    await asyncio.sleep(0 if value == "fast" else 30)
    return "ok"


evaluate(
    name="cli-slow",
    cases=[Case(id="fast", input="fast", expected="ok"),
           Case(id="slow", input="slow", expected="ok")],
    task=task,
    scorers=[exact],
)
"""

# The cache footer only appears when calls were actually made, so a task that
# never touches a provider cannot show whether the flag took effect. litellm is
# patched inside the eval file because `run` executes it in this process but
# through pytest, so a fixture here would not be in scope when it imports.
MODEL_EVAL = """
from unittest.mock import MagicMock, patch

from evalstand import Case, evaluate
from evalstand.llm import call
from evalstand.scorers import exact


def _response(**_):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "x"
    response.usage.prompt_tokens = 4
    response.usage.completion_tokens = 1
    response.usage.total_tokens = 5
    response.model = "gpt-4o-mini"
    return response


patch("evalstand.llm.litellm.completion", side_effect=_response).start()
patch("evalstand.llm.litellm.completion_cost", return_value=0.0).start()


def task(value):
    return call("gpt-4o-mini", [{"role": "user", "content": value}]).text


evaluate(
    name="cli-model",
    cases=[Case(id="q1", input="x", expected="x")],
    task=task,
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


class TestExecutionFlags:
    """Phase 3's controls, reachable from the CLI.

    Asserted on observable behaviour rather than on the argv the CLI builds: a
    test that only checks the forwarded string would pass just as happily if the
    plugin ignored the flag entirely, which is the failure actually worth
    catching.
    """

    def test_no_cache_switches_the_summary_to_a_bypass(self, tmp_path: Path) -> None:
        """Bypassing spends real money, so the footer must say so rather than
        show a 0% hit rate that reads like a merely cold cache."""
        (tmp_path / "model_eval.py").write_text(MODEL_EVAL, encoding="utf-8")
        output = runner.invoke(app, ["run", str(tmp_path), "--no-cache"]).output
        assert "bypassed" in output

    def test_without_no_cache_nothing_is_reported_as_bypassed(self, tmp_path: Path) -> None:
        """The other half: without the flag the word must not appear, or the
        assertion above would hold no matter what the flag did."""
        (tmp_path / "model_eval.py").write_text(MODEL_EVAL, encoding="utf-8")
        output = runner.invoke(app, ["run", str(tmp_path)]).output
        assert "cache 0/1" in output, "a normal run should report its hit rate"
        assert "bypassed" not in output

    def test_timeout_abandons_a_slow_case(self, tmp_path: Path) -> None:
        (tmp_path / "slow_eval.py").write_text(SLOW_EVAL, encoding="utf-8")
        result = runner.invoke(app, ["run", str(tmp_path), "--timeout", "1"])
        assert result.exit_code != 0
        assert "timed out after 1.0s" in result.output

    def test_without_a_timeout_a_slower_case_is_allowed_to_finish(self, tmp_path: Path) -> None:
        """No default limit: a slow model is normal, and timing out honest work
        by default would be worse than waiting. The delay here is short enough
        to keep the suite quick but longer than any default could plausibly be.
        """
        (tmp_path / "pause_eval.py").write_text(
            SLOW_EVAL.replace("30", "0.3").replace("cli-slow", "cli-pause"), encoding="utf-8"
        )
        result = runner.invoke(app, ["run", str(tmp_path)])
        assert result.exit_code == 0
        assert "timed out" not in result.output

    def test_an_invalid_concurrency_is_rejected(self, tmp_path: Path) -> None:
        """Rejected before anything runs. A run that cannot be configured must
        not quietly fall back to a default the user did not ask for."""
        (tmp_path / "toy_eval.py").write_text(PASSING_EVAL, encoding="utf-8")
        result = runner.invoke(app, ["run", str(tmp_path), "--concurrency", "0"])
        assert result.exit_code != 0
        assert "2/2" not in result.output, "a rejected run executed anyway"

    def test_concurrency_is_accepted(self, tmp_path: Path) -> None:
        (tmp_path / "toy_eval.py").write_text(PASSING_EVAL, encoding="utf-8")
        result = runner.invoke(app, ["run", str(tmp_path), "--concurrency", "2"])
        assert result.exit_code == 0
        assert "2/2" in result.output


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

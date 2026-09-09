"""The CLI (Phase 2's exit criterion).

`run` delegates to pytest rather than reimplementing collection: a second
execution path would be a second thing to keep correct, and the two could
disagree about what an eval run means.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from evalstand.cli import app
from tests.conftest import help_text

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
        assert "eval" in help_text("run").lower()


class TestFlagsReachPytest:
    """`run` builds an argv and hands it to pytest. Coverage showed the lines
    forwarding `--fail-on-error` and `--output` were never executed by a test,
    which means a typo in either would have shipped: the flag would silently
    never arrive, the gate would never fire, and the build would go green on a
    regression it was configured to catch.

    Asserted by intercepting `pytest.main`, because what matters is the argv —
    running the eval again would only re-test the plugin.
    """

    def _argv(self, monkeypatch, *flags: str) -> list[str]:
        import pytest as pytest_module

        seen: list[list[str]] = []

        def fake_main(args: list[str]) -> int:
            seen.append(args)
            return 0

        monkeypatch.setattr(pytest_module, "main", fake_main)
        runner.invoke(app, ["run", "somewhere", *flags])

        assert seen, "run did not call pytest"
        return seen[0]

    def test_fail_on_error_is_forwarded(self, monkeypatch) -> None:
        assert "--fail-on-error" in self._argv(monkeypatch, "--fail-on-error")

    def test_it_is_absent_unless_asked_for(self, monkeypatch) -> None:
        """An opt-in flag that leaked into every run would turn a flaky provider
        into a failed build for people who never asked for that."""
        assert "--fail-on-error" not in self._argv(monkeypatch)

    def test_output_markdown_is_forwarded(self, monkeypatch) -> None:
        argv = self._argv(monkeypatch, "--output", "markdown")

        assert "--output" in argv
        assert argv[argv.index("--output") + 1] == "markdown"

    def test_the_default_output_is_not_forwarded(self, monkeypatch) -> None:
        """Passing the default explicitly would work, but leaving it off keeps
        the argv the plugin sees identical to a bare `pytest` run."""
        assert "--output" not in self._argv(monkeypatch)

    def test_the_threshold_value_survives_the_hop(self, monkeypatch) -> None:
        argv = self._argv(monkeypatch, "--threshold", "0.85")

        assert argv[argv.index("--threshold") + 1] == "0.85"

    def test_the_exit_code_is_passed_through_unchanged(self, monkeypatch) -> None:
        """The whole exit-code contract depends on this one line. A CLI that
        normalised pytest's 2 to a 1 would erase the distinction between "the
        model got worse" and "nothing ran"."""
        import pytest as pytest_module

        for code in (0, 1, 2):
            monkeypatch.setattr(pytest_module, "main", lambda args, c=code: c)
            assert runner.invoke(app, ["run", "somewhere"]).exit_code == code


class TestWatchOpensTheLiveView:
    """The happy path of `watch`, which coverage showed was never reached: the
    tests only exercised its failure branches, so everything after eval
    selection — the RunConfig, the recorder, the app itself — was untested."""

    EVAL = """
from evalstand import Case, evaluate
from evalstand.scorers import exact

evaluate(
    name="watched-cli",
    cases=[Case(id="q1", input="x", expected="x")],
    task=lambda value: value,
    scorers=[exact],
)
"""

    def _launch(self, tmp_path: Path, monkeypatch, *flags: str):
        """Run `watch` with the app stubbed, returning the app it built."""
        (tmp_path / "watched_eval.py").write_text(self.EVAL, encoding="utf-8")

        built: list[object] = []
        from evalstand.tui.app import EvalApp

        def capture(self, **kwargs):
            built.append(self)

        monkeypatch.setattr(EvalApp, "run", capture)
        result = runner.invoke(app, ["watch", str(tmp_path), *flags])

        return result, built

    def test_it_opens_the_app_for_the_declared_eval(self, tmp_path: Path, monkeypatch) -> None:
        result, built = self._launch(tmp_path, monkeypatch)

        assert result.exit_code == 0
        assert len(built) == 1
        assert built[0].declared.name == "watched-cli"

    def test_the_case_count_reaches_the_progress_bar(self, tmp_path: Path, monkeypatch) -> None:
        """Resolved before the screen is taken over, so a loader that raises
        fails in plain text rather than inside a terminal UI."""
        _, built = self._launch(tmp_path, monkeypatch)

        assert built[0].state.expected == 1

    def test_once_opens_the_view_without_watching(self, tmp_path: Path, monkeypatch) -> None:
        _, built = self._launch(tmp_path, monkeypatch, "--once")

        assert built[0].watching is False

    def test_watching_is_on_by_default(self, tmp_path: Path, monkeypatch) -> None:
        _, built = self._launch(tmp_path, monkeypatch)

        assert built[0].watching is True

    def test_execution_flags_reach_the_run_config(self, tmp_path: Path, monkeypatch) -> None:
        _, built = self._launch(
            tmp_path, monkeypatch, "--concurrency", "3", "--timeout", "9", "--no-cache"
        )

        assert built[0].config.concurrency == 3
        assert built[0].config.timeout_seconds == 9
        assert built[0].config.bypass_cache is True

    def test_nothing_is_recorded_without_store(self, tmp_path: Path, monkeypatch) -> None:
        """The opposite of `run`, and deliberate: watch mode is used *while*
        editing, so every run would be tied to a commit whose code it did not
        reflect."""
        _, built = self._launch(tmp_path, monkeypatch)

        assert built[0].recorder is None

    def test_store_opens_a_recorder(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _, built = self._launch(tmp_path, monkeypatch, "--store")

        assert built[0].recorder is not None
        built[0].recorder.store.close()

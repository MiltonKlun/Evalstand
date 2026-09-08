"""The exit-code contract `docs/ci.md` publishes (task 7.1).

A documented exit code is a promise a CI job is written against. If the table
and the code drift apart, every pipeline that trusted the table starts
misreading its own builds — and the failure is silent, because a wrong exit code
still looks like a working build until someone checks what it meant.

`test_exit_codes.py` proves the codes behave correctly. This file proves the
document says what they are.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[2] / "docs" / "ci.md"


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


class TestTheTableMatchesTheCode:
    def test_the_documented_codes_are_the_constants_the_plugin_uses(self, text: str) -> None:
        """The numbers in the table are the ones the plugin actually sets."""
        from evalstand.plugin import EXIT_BELOW_THRESHOLD, EXIT_EXECUTION_ERROR

        assert EXIT_BELOW_THRESHOLD == 1
        assert EXIT_EXECUTION_ERROR == 2
        assert "| `1` |" in text
        assert "| `2` |" in text

    def test_compare_uses_the_same_code_for_nothing_to_compare(self, text: str) -> None:
        from evalstand.cli import EXIT_NOTHING_TO_COMPARE
        from evalstand.plugin import EXIT_EXECUTION_ERROR

        assert EXIT_NOTHING_TO_COMPARE == EXIT_EXECUTION_ERROR
        assert "`compare` uses `2` for the same reason" in text

    def test_zero_is_documented_as_success(self, text: str) -> None:
        assert "| `0` |" in text


class TestTheFlagsItNames:
    @pytest.mark.parametrize("flag", ["--threshold", "--fail-on-error"])
    def test_every_documented_flag_exists_on_the_cli(self, text: str, flag: str) -> None:
        from typer.testing import CliRunner

        from evalstand.cli import app

        assert flag in text
        assert flag in CliRunner().invoke(app, ["run", "--help"]).output

    def test_fail_on_error_is_opt_in(self, text: str) -> None:
        """The doc explains *why* it is opt-in, which is only honest if it is."""
        from evalstand.cli import run

        assert inspect.signature(run).parameters["fail_on_error"].default is False
        assert "opt-in rather than the default" in text

    def test_the_example_command_is_valid(self, text: str) -> None:
        """The doc shows a copyable command. A flag typo there costs a user a
        failed build and a confusing error."""
        commands = re.findall(r"^evalstand run .*$", text, re.MULTILINE)
        assert commands, "the doc should show a runnable example"

        from typer.testing import CliRunner

        from evalstand.cli import app

        help_text = CliRunner().invoke(app, ["run", "--help"]).output
        for command in commands:
            for flag in re.findall(r"--[a-z-]+", command):
                assert flag in help_text, f"{flag} is documented but not a real flag"


class TestTheClaimsAboutPrecedence:
    def test_it_states_that_two_outranks_one(self, text: str) -> None:
        """The single most important sentence in the table, because it is the
        one a reader relies on when a build goes red for two reasons at once."""
        assert "`2` outranks `1`" in text

    def test_it_says_which_test_file_proves_the_table(self, text: str) -> None:
        """A documented promise with no named proof is an assertion. This is the
        pointer that lets a sceptical reader check."""
        assert "tests/unit/test_exit_codes.py" in text

        named = DOC.parent.parent / "tests" / "unit" / "test_exit_codes.py"
        assert named.exists(), "the doc points at a test file that does not exist"


class TestTheEmptyDatabaseClaim:
    def test_it_says_threshold_works_without_history(self, text: str) -> None:
        """ADR 0005 makes the database project-local and gitignored, so CI
        starts every build with none. The doc has to say what still works."""
        assert "empty database" in text

    def test_the_workflow_it_ships_does_not_depend_on_history(self, text: str) -> None:
        """A copyable workflow that silently needed a prior run would fail on
        the very first build in a fresh clone."""
        workflows = re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)
        assert workflows, "the doc should ship a copyable workflow"

        for workflow in workflows:
            assert "evalstand compare" not in workflow


class TestTheShippedWorkflows:
    """Task 7.3 ships copyable YAML rather than a published Action.

    That makes the block itself the deliverable: a user pastes it verbatim, and
    a typo costs them a red build and a confusing error in someone else's
    repository — the one place they cannot debug it.
    """

    def _blocks(self, text: str) -> list[str]:
        return re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)

    def test_every_block_is_valid_yaml(self, text: str) -> None:
        yaml = pytest.importorskip("yaml", reason="pyyaml is only available transitively")

        blocks = self._blocks(text)
        assert blocks, "the doc should ship a copyable workflow"

        for index, block in enumerate(blocks, start=1):
            try:
                yaml.safe_load(block)
            except yaml.YAMLError as exc:  # pragma: no cover - only on a broken doc
                pytest.fail(f"yaml block {index} does not parse: {exc}")

    def test_every_evalstand_flag_used_is_real(self, text: str) -> None:
        """The recipe's whole value is that it works when pasted."""
        from typer.testing import CliRunner

        from evalstand.cli import app

        help_text = CliRunner().invoke(app, ["run", "--help"]).output

        for block in self._blocks(text):
            for command in re.findall(r"evalstand run ([^\n|]*)", block):
                for flag in re.findall(r"--[a-z-]+", command):
                    assert flag in help_text, f"{flag} is in the workflow but not a real flag"

    def test_a_comment_posting_workflow_asks_for_write_permission(self, text: str) -> None:
        """`GITHUB_TOKEN` is read-only by default. A recipe that posts a comment
        without requesting the scope fails at the last step, after the evals
        have already been paid for."""
        posting = [b for b in self._blocks(text) if "createComment" in b]
        assert posting, "the doc should show how to post the summary"

        for block in posting:
            assert "pull-requests: write" in block

    def test_the_gate_survives_the_comment_step(self, text: str) -> None:
        """`continue-on-error` without an explicit re-fail would turn the
        threshold gate off — the build would go green on a regression, which is
        the exact failure the gate exists to prevent."""
        for block in self._blocks(text):
            if "continue-on-error" not in block:
                continue
            assert "exit 1" in block, "the workflow disables the gate and never restores it"

    def test_it_warns_that_forks_cannot_comment(self, text: str) -> None:
        """Otherwise the first external contributor's PR fails at a step the
        maintainer cannot reproduce."""
        assert "Forked pull requests cannot post comments" in text

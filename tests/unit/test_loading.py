"""Finding evals without a pytest session (task 6.6).

Watch mode re-imports after every edit, so the property that matters most is
**freshness**: an edited file must produce the edited behaviour. A re-run that
measured the code the user had just replaced would look exactly like a model
ignoring their changes, and no amount of staring at the eval file would explain
it.

Freshness comes from `exec_module` re-reading the source, not from the module
name — the loader's first docstring claimed otherwise and a mutant that fixed
the name proved it wrong. Kept as a note because a plausible-sounding
explanation that is false is worse than none: it stops the next reader looking.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evalstand.api import Eval, registry
from evalstand.loading import (
    NoEvalsFoundError,
    eval_files,
    load_evals,
    select_eval,
)
from evalstand.runner import run_eval

_EVAL_SOURCE = """
from evalstand import evaluate
from evalstand.models import Score

ANSWER = "{answer}"


def task(value):
    return ANSWER


def scorer(output, expected):
    return Score(scorer_name="s", value=1.0, passed=True)


evaluate(
    name="{name}",
    cases=[{{"input": "q", "expected": "x"}}],
    task=task,
    scorers=[scorer],
)
"""


def _write(directory: Path, filename: str, *, name: str, answer: str = "first") -> Path:
    path = directory / filename
    path.write_text(_EVAL_SOURCE.format(name=name, answer=answer), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    registry.clear()


class TestFindingEvalFiles:
    def test_a_directory_is_searched_with_the_collection_pattern(self, tmp_path: Path) -> None:
        _write(tmp_path, "qa_eval.py", name="qa")
        (tmp_path / "helpers.py").write_text("x = 1", encoding="utf-8")

        assert eval_files([tmp_path]) == [tmp_path / "qa_eval.py"]

    def test_a_named_file_is_taken_whatever_it_is_called(self, tmp_path: Path) -> None:
        """The user naming a file is a clearer signal than the glob."""
        path = _write(tmp_path, "evals.py", name="qa")

        assert eval_files([path]) == [path.resolve()]

    def test_nested_directories_are_searched(self, tmp_path: Path) -> None:
        nested = tmp_path / "suites" / "deep"
        nested.mkdir(parents=True)
        _write(nested, "qa_eval.py", name="qa")

        assert eval_files([tmp_path]) == [nested / "qa_eval.py"]

    def test_generated_directories_are_skipped(self, tmp_path: Path) -> None:
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        _write(cache, "qa_eval.py", name="stale")
        _write(tmp_path, "real_eval.py", name="real")

        assert eval_files([tmp_path]) == [tmp_path / "real_eval.py"]

    def test_a_file_named_twice_is_imported_once(self, tmp_path: Path) -> None:
        """`evalstand watch . qa_eval.py` names one file two ways, and importing
        it twice would register each eval twice."""
        path = _write(tmp_path, "qa_eval.py", name="qa")

        assert eval_files([tmp_path, path]) == [path.resolve()]

    def test_a_missing_path_is_ignored_rather_than_raising(self, tmp_path: Path) -> None:
        _write(tmp_path, "qa_eval.py", name="qa")

        assert eval_files([tmp_path, tmp_path / "nope"]) == [tmp_path / "qa_eval.py"]


class TestLoadingRegistersWithoutRunning:
    def test_importing_a_file_declares_its_evals(self, tmp_path: Path) -> None:
        _write(tmp_path, "qa_eval.py", name="qa")

        assert [declared.name for declared in load_evals([tmp_path])] == ["qa"]

    def test_nothing_is_executed_by_loading(self, tmp_path: Path) -> None:
        """`evaluate()` registers and returns (ADR 0004), which is the property
        that lets watch mode ask what a file declares after every keystroke
        without paying a provider for the answer."""
        path = tmp_path / "spending_eval.py"
        marker = tmp_path / "ran.txt"
        path.write_text(
            "from evalstand import evaluate\n"
            "from evalstand.models import Score\n"
            "from pathlib import Path\n"
            f"def task(value):\n    Path(r'{marker}').write_text('ran')\n    return 'x'\n"
            "def scorer(output, expected):\n"
            "    return Score(scorer_name='s', value=1.0, passed=True)\n"
            "evaluate(name='spending', cases=[{'input': 'q'}], task=task, scorers=[scorer])\n",
            encoding="utf-8",
        )

        load_evals([tmp_path])

        assert not marker.exists(), "loading executed the task"

    def test_evals_from_a_previous_load_do_not_linger(self, tmp_path: Path) -> None:
        """An eval the user deleted or renamed must disappear. A table still
        showing cases from a file that no longer declares them is a report about
        code that is not there."""
        path = _write(tmp_path, "qa_eval.py", name="original")
        load_evals([tmp_path])

        path.unlink()
        _write(tmp_path, "qa_eval.py", name="renamed")

        assert [declared.name for declared in load_evals([tmp_path])] == ["renamed"]


class TestAnEditedFileIsActuallyReloaded:
    """The property watch mode lives on."""

    def test_a_second_load_sees_the_edit(self, tmp_path: Path) -> None:
        """The guarantee watch mode rests on, asserted end to end: load, edit,
        load again, and the behaviour must have changed.

        Asserted through `run_eval` rather than by inspecting the module,
        because what matters is the answer the user sees — not which import
        mechanism produced it.
        """
        path = _write(tmp_path, "qa_eval.py", name="qa", answer="first")

        first = select_eval(load_evals([tmp_path]))
        assert asyncio.run(_output(first)) == "first"

        path.write_text(_EVAL_SOURCE.format(name="qa", answer="second"), encoding="utf-8")

        second = select_eval(load_evals([tmp_path]))
        assert asyncio.run(_output(second)) == "second"

    def test_the_module_is_not_left_in_sys_modules(self, tmp_path: Path) -> None:
        """One entry per edit would grow without bound over a long watch
        session, and nothing ever looks the module up by name."""
        import sys

        _write(tmp_path, "qa_eval.py", name="qa")
        before = {name for name in sys.modules if name.startswith("_evalstand_eval_")}

        load_evals([tmp_path])

        after = {name for name in sys.modules if name.startswith("_evalstand_eval_")}
        assert after == before


async def _output(declared: Eval) -> object:
    run = await run_eval(declared)
    return run.results[0].output


class TestChoosingWhichEvalToWatch:
    def test_a_single_eval_needs_no_name(self, tmp_path: Path) -> None:
        _write(tmp_path, "qa_eval.py", name="only")

        assert select_eval(load_evals([tmp_path])).name == "only"

    def test_a_named_eval_is_selected(self, tmp_path: Path) -> None:
        _write(tmp_path, "a_eval.py", name="alpha")
        _write(tmp_path, "b_eval.py", name="beta")

        assert select_eval(load_evals([tmp_path]), "beta").name == "beta"

    def test_several_evals_without_a_name_is_refused(self, tmp_path: Path) -> None:
        """Guessing would silently measure something the user did not ask
        about, and every number on screen would be about the wrong thing."""
        _write(tmp_path, "a_eval.py", name="alpha")
        _write(tmp_path, "b_eval.py", name="beta")

        with pytest.raises(NoEvalsFoundError) as caught:
            select_eval(load_evals([tmp_path]))

        assert "alpha" in str(caught.value) and "beta" in str(caught.value)
        assert "--eval" in str(caught.value)

    def test_an_unknown_name_lists_what_was_found(self, tmp_path: Path) -> None:
        """A typo and an empty directory send a user looking in different
        places."""
        _write(tmp_path, "qa_eval.py", name="alpha")

        with pytest.raises(NoEvalsFoundError) as caught:
            select_eval(load_evals([tmp_path]), "alfa")

        assert "alpha" in str(caught.value)

    def test_no_evals_at_all_explains_the_pattern(self, tmp_path: Path) -> None:
        """A user whose file is called `evals.py` needs to be told the pattern,
        not shown a stack trace."""
        with pytest.raises(NoEvalsFoundError) as caught:
            select_eval(load_evals([tmp_path]))

        assert "*_eval.py" in str(caught.value)


class TestTheWatchCommand:
    def test_it_reports_a_missing_eval_plainly(self, tmp_path: Path) -> None:
        """Printed rather than raised inside a terminal UI that would then have
        to be dismissed."""
        from typer.testing import CliRunner

        from evalstand.cli import app

        result = CliRunner().invoke(app, ["watch", str(tmp_path)])

        assert result.exit_code == 1
        assert "*_eval.py" in result.output

    def test_it_reports_an_eval_file_that_raises_on_import(self, tmp_path: Path) -> None:
        (tmp_path / "broken_eval.py").write_text("raise ValueError('bad file')", encoding="utf-8")

        from typer.testing import CliRunner

        from evalstand.cli import app

        result = CliRunner().invoke(app, ["watch", str(tmp_path)])

        assert result.exit_code == 1
        assert "bad file" in result.output

    def test_it_names_the_evals_when_several_are_declared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`EvalApp.run` is stubbed out deliberately.

        Without it, a bug that let selection fall through would launch the real
        terminal app inside the test process — which does not fail, it *hangs*.
        A suite that hangs on a defect is worse than one that fails on it: the
        signal arrives as a timeout somewhere in CI rather than as a named
        assertion, and the first instinct is to blame the runner.
        """
        _write(tmp_path, "a_eval.py", name="alpha")
        _write(tmp_path, "b_eval.py", name="beta")

        launched: list[bool] = []
        monkeypatch.setattr(
            "evalstand.tui.app.EvalApp.run", lambda self, **kwargs: launched.append(True)
        )

        from typer.testing import CliRunner

        from evalstand.cli import app

        result = CliRunner().invoke(app, ["watch", str(tmp_path)])

        assert result.exit_code == 1
        assert "alpha" in result.output and "beta" in result.output
        assert launched == [], "the live view was opened despite an ambiguous selection"

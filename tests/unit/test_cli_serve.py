"""`evalstand serve` (task 8.4).

Two failures are worth catching here, and neither is a crash.

The first is a server that starts and serves the wrong database — the user
reads real-looking numbers about a project they are not in. The second is the
missing-extra path: `serve` is the only command that can fail on a correct
install, and a traceback about `fastapi` tells a user which *package* is absent
but not which *extra* supplies it, which is the only part they can act on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from evalstand.cli import app
from evalstand.models import Batch, BatchKind, BatchStatus, Result, Run, RunStatus, Score
from evalstand.storage import RunStore
from evalstand.web import MISSING_EXTRA

runner = CliRunner()

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def _populated(path: Path, *, name: str = "demo") -> None:
    with RunStore(path) as store:
        store.save_batch(
            Batch(
                id="batch-1",
                kind=BatchKind.FULL,
                status=BatchStatus.COMPLETED,
                started_at=NOW,
            )
        )
        store.save_run(
            Run(
                id="run-1",
                batch_id="batch-1",
                name=name,
                filepath="demo_eval.py",
                status=RunStatus.COMPLETED,
                started_at=NOW,
                results=[
                    Result(
                        id="r1",
                        case_id="q1",
                        output="answer",
                        scores=[Score(scorer_name="exact", value=1.0, passed=True)],
                    )
                ],
            )
        )


class TestItDoesNotRunEvals:
    """ADR 0009's rule. A page that spent money on load would be a trap."""

    def test_serving_never_reaches_the_runner(self, tmp_path: Path) -> None:
        database = tmp_path / "history.db"
        _populated(database)

        with (
            patch("uvicorn.run") as served,
            patch("evalstand.runner.run_eval") as executed,
        ):
            result = runner.invoke(app, ["serve", "--db", str(database)])

        assert result.exit_code == 0
        assert served.called, "the server never started"
        assert not executed.called, "serving executed an eval"


class TestItServesTheDatabaseItWasGiven:
    def test_the_app_reads_the_named_database(self, tmp_path: Path) -> None:
        """The failure this catches is a server that starts against the wrong
        store: every number looks plausible and belongs to another project."""
        database = tmp_path / "elsewhere.db"
        _populated(database, name="from-the-named-db")

        pytest.importorskip("fastapi", reason="the web extra is not installed")
        from fastapi.testclient import TestClient

        captured: dict[str, Any] = {}

        # Read *inside* the callback, while `serve` still holds the store open.
        # `serve` closes it on the way out — correctly, since a command that
        # returned would otherwise strand a connection — so a client built
        # after `invoke` returns gets `Cannot operate on a closed database`.
        # That is the CLI behaving, not failing.
        def _capture(application: Any, **_: Any) -> None:
            captured["names"] = TestClient(application).get("/api/evals").json()["names"]

        with patch("uvicorn.run", side_effect=_capture):
            result = runner.invoke(app, ["serve", "--db", str(database)])

        assert result.exit_code == 0
        assert captured["names"] == ["from-the-named-db"]

    def test_it_reports_how_many_runs_it_found(self, tmp_path: Path) -> None:
        """Printed before the server blocks, so a user pointed at an empty or
        wrong database learns it immediately rather than from a bare page."""
        database = tmp_path / "history.db"
        _populated(database)

        with patch("uvicorn.run"):
            output = runner.invoke(app, ["serve", "--db", str(database)]).output

        assert "1 runs" in output

    def test_an_empty_database_still_serves(self, tmp_path: Path) -> None:
        """Nothing recorded yet is a normal state. CI starts every build
        there, and so does a user on their first day."""
        with patch("uvicorn.run") as served:
            result = runner.invoke(app, ["serve", "--db", str(tmp_path / "empty.db")])

        assert result.exit_code == 0
        assert served.called


class TestTheAddressItBinds:
    def test_it_binds_localhost_by_default(self, tmp_path: Path) -> None:
        """The server exposes every recorded prompt, completion and cost on the
        machine. Reaching it from the network is a decision, not a default."""
        with patch("uvicorn.run") as served:
            runner.invoke(app, ["serve", "--db", str(tmp_path / "h.db")])

        assert served.call_args.kwargs["host"] == "127.0.0.1"

    def test_host_and_port_are_forwarded(self, tmp_path: Path) -> None:
        with patch("uvicorn.run") as served:
            runner.invoke(
                app,
                ["serve", "--db", str(tmp_path / "h.db"), "--host", "0.0.0.0", "--port", "9999"],
            )

        assert served.call_args.kwargs["host"] == "0.0.0.0"
        assert served.call_args.kwargs["port"] == 9999

    def test_the_printed_url_matches_what_it_binds(self, tmp_path: Path) -> None:
        """A URL that disagrees with the bind address sends the user to a page
        that will not load, and the server looks broken when it is fine."""
        with patch("uvicorn.run"):
            output = runner.invoke(
                app, ["serve", "--db", str(tmp_path / "h.db"), "--port", "9999"]
            ).output

        assert "http://127.0.0.1:9999" in output


class TestTheBrowser:
    def test_it_does_not_open_one_by_default(self, tmp_path: Path) -> None:
        """A command that seizes the screen when run over ssh, or in a script,
        is a nuisance the user did not ask for."""
        with patch("uvicorn.run"), patch("webbrowser.open") as opened:
            runner.invoke(app, ["serve", "--db", str(tmp_path / "h.db")])

        assert not opened.called

    def test_open_asks_for_one(self, tmp_path: Path) -> None:
        with patch("uvicorn.run"), patch("webbrowser.open") as opened:
            runner.invoke(app, ["serve", "--db", str(tmp_path / "h.db"), "--open"])

        assert opened.called


class TestTheMissingExtra:
    """`serve` is the only command that can fail on a correct install."""

    def test_it_names_the_extra_rather_than_the_package(self, tmp_path: Path) -> None:
        with patch.dict("sys.modules", {"uvicorn": None, "evalstand.web.api": None}):
            result = runner.invoke(app, ["serve", "--db", str(tmp_path / "h.db")])

        assert result.exit_code == 1
        assert "evalstand[web]" in result.output

    def test_the_message_is_installable_verbatim(self) -> None:
        """A user copies this line. If it does not name a real extra, they
        install something that does not fix their problem."""
        import tomllib

        root = Path(__file__).resolve().parents[2]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        assert "web" in config["project"]["optional-dependencies"]
        assert "evalstand[web]" in MISSING_EXTRA

    def test_it_does_not_leave_a_database_handle_open(self, tmp_path: Path) -> None:
        """The import is checked before the store is opened, so a missing extra
        cannot strand a connection on a file the user then cannot delete."""
        database = tmp_path / "h.db"

        with patch.dict("sys.modules", {"uvicorn": None, "evalstand.web.api": None}):
            runner.invoke(app, ["serve", "--db", str(database)])

        assert not database.exists(), "the store was opened before the extra was checked"

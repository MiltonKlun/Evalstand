"""What `docs/web.md` promises (task 8.2 / 8.4).

A documented endpoint that does not exist, or a flag spelled wrong, costs a
reader the time to find out — and the doc is the only place the JSON API is
described, so a drifted table is the whole specification being wrong.

The claims about honesty are asserted too. They are the reason two extra fields
exist, and a doc that stopped explaining them would leave a consumer summing
`total_cost_usd` into an understated bill.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import help_text

DOC = Path(__file__).resolve().parents[2] / "docs" / "web.md"


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


class TestTheFlagsItNames:
    @pytest.mark.parametrize("flag", ["--host", "--port", "--db", "--open"])
    def test_every_documented_flag_is_real(self, text: str, flag: str) -> None:
        assert flag in text
        assert flag in help_text("serve")

    def test_the_default_port_is_the_documented_one(self, text: str) -> None:
        """A reader copies this into a bookmark."""
        assert "8420" in text
        assert "8420" in help_text("serve")

    def test_the_example_commands_are_valid(self, text: str) -> None:
        flags = help_text("serve")
        for command in re.findall(r"^evalstand serve .*$", text, re.MULTILINE):
            for flag in re.findall(r"--[a-z-]+", command):
                assert flag in flags, f"{flag} is documented but not a real flag"


class TestTheEndpointTable:
    def test_every_documented_endpoint_exists(self, text: str) -> None:
        """The doc is the API's only specification. An endpoint it names that
        the app does not serve is a broken promise nobody else records."""
        fastapi = pytest.importorskip("fastapi", reason="the web extra is not installed")
        assert fastapi

        import tempfile

        from evalstand.storage import RunStore
        from evalstand.web.api import create_app

        with tempfile.TemporaryDirectory() as tmp, RunStore(Path(tmp) / "h.db") as store:
            routes = {
                getattr(route, "path", "")
                for route in create_app(store).routes  # type: ignore[attr-defined]
            }

        documented = set(re.findall(r"`(GET /api/[^`]+)`", text))
        assert documented, "the doc should list the endpoints"

        for entry in documented:
            path = entry.split(" ", 1)[1].split("?")[0]
            # `{id}` in prose, `{run_id}` in the route.
            pattern = re.sub(r"\{[^}]+\}", "{}", path)
            actual = {re.sub(r"\{[^}]+\}", "{}", route) for route in routes}
            assert pattern in actual, f"{entry} is documented but not served"


class TestTheHonestyClaims:
    def test_it_explains_that_cost_is_a_lower_bound(self, text: str) -> None:
        """The single most important sentence for a scripting consumer."""
        flat = " ".join(text.split())

        assert "lower bound" in flat
        assert "cost_is_complete" in text
        assert "unpriced_call_count" in text

    def test_it_says_null_is_not_zero(self, text: str) -> None:
        assert "never means zero" in text or "null` never means zero" in text

    def test_the_fields_it_names_are_really_in_the_payload(self, text: str) -> None:
        """Asserted against a rendered payload rather than the source, so a
        renamed field fails here rather than in someone's script."""
        pytest.importorskip("fastapi", reason="the web extra is not installed")

        from datetime import UTC, datetime

        from evalstand.models import Run, RunStatus
        from evalstand.web.api import run_summary

        payload = run_summary(
            Run(
                id="r",
                batch_id="b",
                name="n",
                filepath="f",
                status=RunStatus.COMPLETED,
                started_at=datetime(2026, 9, 16, tzinfo=UTC),
            )
        )

        for field in ["total_cost_usd", "cost_is_complete", "unpriced_call_count", "mean_score"]:
            assert field in payload, f"the doc names {field}, which the payload lacks"


class TestTheSafetyClaims:
    def test_it_says_serve_runs_nothing(self, text: str) -> None:
        """ADR 0009's rule, and the one a reader most needs to trust."""
        assert "does not run anything" in text.lower() or "it does not run" in text.lower()

    def test_it_says_localhost_is_the_default(self, text: str) -> None:
        assert "127.0.0.1" in text

    def test_the_default_it_documents_is_the_real_default(self) -> None:
        """A doc promising localhost over a command that bound every interface
        would be worse than no doc at all."""
        import inspect

        from evalstand.cli import serve

        assert inspect.signature(serve).parameters["host"].default == "127.0.0.1"

    def test_it_warns_what_host_exposes(self, text: str) -> None:
        """The database holds every prompt and completion. A reader reaching
        for `--host` should know that before they use it."""
        flat = " ".join(text.split())

        assert "prompt" in flat and "completion" in flat

    def test_it_says_the_page_works_offline(self, text: str) -> None:
        assert "offline" in text.lower()

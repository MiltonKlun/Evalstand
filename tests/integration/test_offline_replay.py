"""Task 1.6's acceptance criterion, demonstrated end to end.

The unit suite already passes with no keys because it mocks LiteLLM. That proves
the tests do not spend money; it does not prove the code works against a real
payload shape, because a hand-written mock only encodes what its author believed
a provider returns.

These tests run the real call path — extraction, pricing, caching — against a
committed cassette, with every provider entry point patched so nothing can reach
the network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evalstand.cache import ResponseCache
from evalstand.cassettes import Cassette, CassetteMissError, CassetteMode
from evalstand.llm import acall, call

CASSETTE = Path(__file__).parent.parent / "cassettes" / "basic_qa.json"

PARIS = [{"role": "user", "content": "What is the capital of France?"}]
MATHS = [{"role": "user", "content": "What is 2 + 2?"}]
OCEAN = [{"role": "user", "content": "Name the largest ocean."}]


@pytest.fixture
def cassette() -> Cassette:
    return Cassette(CASSETTE, mode=CassetteMode.REPLAY)


class TestOfflineReplay:
    def test_the_cassette_is_committed(self) -> None:
        """The suite depends on this file; its absence must fail loudly."""
        assert CASSETTE.exists(), f"missing cassette: {CASSETTE}"

    def test_no_provider_keys_are_needed(self, cassette: Cassette) -> None:
        """Stated as an assertion rather than an assumption about the runner."""
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_API_KEY"):
            os.environ.pop(name, None)

        with cassette.patched():
            assert call("gpt-4o-mini", PARIS, temperature=0.0).text == "Paris"

    def test_the_real_extraction_path_runs(self, cassette: Cassette) -> None:
        """Tokens and cost come out of the code under test, not the fixture."""
        with cassette.patched():
            response = call("gpt-4o-mini", PARIS, temperature=0.0)

        assert response.text == "Paris"
        assert response.input_tokens == 14
        assert response.output_tokens == 1
        assert response.cost_usd == pytest.approx(0.0000027)
        assert response.model == "gpt-4o-mini"

    @pytest.mark.parametrize(
        ("messages", "expected"),
        [(PARIS, "Paris"), (MATHS, "4"), (OCEAN, "The Pacific Ocean")],
    )
    def test_every_recorded_call_replays(
        self, cassette: Cassette, messages: list[dict[str, str]], expected: str
    ) -> None:
        with cassette.patched():
            assert call("gpt-4o-mini", messages, temperature=0.0).text == expected

    @pytest.mark.anyio
    async def test_async_calls_replay(self, cassette: Cassette) -> None:
        with cassette.patched():
            response = await acall("gpt-4o-mini", PARIS, temperature=0.0)
        assert response.text == "Paris"
        assert response.cost_usd == pytest.approx(0.0000027)

    def test_an_unrecorded_call_fails_instead_of_reaching_a_provider(
        self, cassette: Cassette
    ) -> None:
        """The failure mode that matters: a test drifting onto a new prompt must
        break, not quietly start spending money."""
        with cassette.patched(), pytest.raises(CassetteMissError, match=r"basic_qa\.json"):
            call("gpt-4o-mini", [{"role": "user", "content": "unrecorded"}], temperature=0.0)

    def test_a_differing_parameter_is_a_miss(self, cassette: Cassette) -> None:
        """Recorded at temperature 0.0; a hotter call is a different call."""
        with cassette.patched(), pytest.raises(CassetteMissError):
            call("gpt-4o-mini", PARIS, temperature=0.9)


class TestReplayThroughTheCache:
    """Cassette and cache compose: the cassette stands in for the provider, the
    cache still avoids repeat calls."""

    def test_a_cached_replay_hits_the_cassette_once(
        self, cassette: Cassette, tmp_path: Path
    ) -> None:
        cache = ResponseCache(tmp_path / "cache.db")

        with cassette.patched():
            first = call("gpt-4o-mini", PARIS, cache=cache, temperature=0.0)
            second = call("gpt-4o-mini", PARIS, cache=cache, temperature=0.0)

        assert first.cached is False
        assert second.cached is True
        assert first.text == second.text == "Paris"
        assert second.cost_usd == pytest.approx(first.cost_usd)


class TestNoNetworkAccess:
    """The guarantee stated as a test rather than as a claim in a docstring."""

    def test_replay_makes_no_outbound_connection(self, cassette: Cassette) -> None:
        import socket

        class NetworkAttemptedError(Exception):
            pass

        original = socket.socket.connect

        def blocked(self: socket.socket, *args: object, **kwargs: object) -> None:
            raise NetworkAttemptedError("replay attempted an outbound connection")

        socket.socket.connect = blocked  # type: ignore[method-assign]
        try:
            with cassette.patched():
                response = call("gpt-4o-mini", PARIS, temperature=0.0)
        finally:
            socket.socket.connect = original  # type: ignore[method-assign]

        assert response.text == "Paris"
        assert response.cost_usd == pytest.approx(0.0000027)

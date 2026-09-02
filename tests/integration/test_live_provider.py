"""Live provider tests (task 1.2's remaining acceptance criterion).

These are the only tests that reach a real provider and spend real money. They
are marked `live` and excluded by default, so CI and a normal local run never
touch them — guardrail 1 requires zero API keys and zero spend.

Run them deliberately, with a key set:

    pytest -m live

Their purpose is narrow but not decorative: everything else in the suite is
mocked or replayed against a recording, so nothing else would notice if LiteLLM
changed the shape of what it returns. These would.

Cost is a fraction of a cent per run — the model is the cheapest available and
the prompts are a few tokens.
"""

from __future__ import annotations

import os

import pytest

from evalstand.llm import acall, acall_stream, call

pytestmark = pytest.mark.live

MODEL = "gpt-4o-mini"
"""The cheapest widely available model. A live test that costs real money should
cost as little of it as possible."""

MESSAGES = [{"role": "user", "content": "Reply with exactly one word: ok"}]


@pytest.fixture(autouse=True)
def _require_a_key() -> None:
    """Skip rather than fail when no key is set.

    A missing key means these were not meant to run, which is not a defect.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; live tests need a real provider")


class TestLiveCall:
    def test_a_real_call_returns_text(self) -> None:
        response = call(MODEL, MESSAGES, temperature=0.0, max_tokens=5)
        assert response.text.strip(), "a live model returned no text"

    def test_a_real_call_reports_tokens(self) -> None:
        """The mocked tests assert we read usage the way LiteLLM presents it.
        This is the only test that checks LiteLLM still presents it that way."""
        response = call(MODEL, MESSAGES, temperature=0.0, max_tokens=5)

        assert response.input_tokens is not None, "no prompt tokens from a live call"
        assert response.output_tokens is not None, "no completion tokens from a live call"
        assert response.input_tokens > 0
        assert response.output_tokens > 0

    def test_a_real_call_is_priced(self) -> None:
        """Parity item 8 rests on LiteLLM knowing this model's price."""
        response = call(MODEL, MESSAGES, temperature=0.0, max_tokens=5)

        assert response.cost_usd is not None, f"LiteLLM could not price {MODEL}"
        assert response.cost_usd > 0
        assert response.cost_usd < 0.01, "a five-token reply should cost almost nothing"

    def test_latency_is_recorded(self) -> None:
        response = call(MODEL, MESSAGES, temperature=0.0, max_tokens=5)
        assert response.latency_ms > 0, "a network round trip took no time"

    def test_the_model_name_comes_back(self) -> None:
        response = call(MODEL, MESSAGES, temperature=0.0, max_tokens=5)
        assert MODEL in response.model


class TestLiveAsync:
    @pytest.mark.anyio
    async def test_acall_matches_call(self) -> None:
        """Sync and async must not disagree about what a provider returned."""
        synchronous = call(MODEL, MESSAGES, temperature=0.0, max_tokens=5)
        asynchronous = await acall(MODEL, MESSAGES, temperature=0.0, max_tokens=5)

        assert asynchronous.input_tokens == synchronous.input_tokens
        assert asynchronous.cost_usd == pytest.approx(synchronous.cost_usd, rel=0.5)


class TestLiveStreaming:
    @pytest.mark.anyio
    async def test_a_stream_yields_chunks_and_totals(self) -> None:
        """Task 1.3 priced streams from `cost_per_token` because streamed
        responses often omit usage. This is the only test that checks a real
        provider still sends it when asked."""
        streamer = acall_stream(MODEL, MESSAGES, temperature=0.0, max_tokens=5)
        chunks = [chunk async for chunk in streamer]

        assert chunks, "a live stream yielded nothing"
        assert streamer.response is not None
        assert streamer.response.text == "".join(chunks)
        assert streamer.response.input_tokens is not None, (
            "stream_options include_usage did not produce usage"
        )
        assert streamer.response.cost_usd is not None


class TestLiveCache:
    def test_a_second_identical_call_is_served_from_the_cache(self, tmp_path: object) -> None:
        """The cache's whole purpose, against a real provider: the second call
        costs nothing because it never happens."""
        from pathlib import Path

        from evalstand.cache import ResponseCache

        cache = ResponseCache(Path(str(tmp_path)) / "live-cache.db")

        first = call(MODEL, MESSAGES, cache=cache, temperature=0.0, max_tokens=5)
        second = call(MODEL, MESSAGES, cache=cache, temperature=0.0, max_tokens=5)

        assert first.cached is False
        assert second.cached is True
        assert second.text == first.text

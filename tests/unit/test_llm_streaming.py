"""Streaming calls (task 1.3).

The acceptance criterion is that a streamed call and a non-streamed call to the
same prompt agree on accumulated text and on cost. Streamed responses often omit
usage entirely, so the cost path here is the part worth testing hard.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evalstand.llm import acall, acall_stream

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "capital of France?"}]


def _chunk(content: str | None) -> MagicMock:
    """Shape a mock the way LiteLLM shapes a streamed delta."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    chunk.usage = None
    return chunk


def _final_chunk(input_tokens: int, output_tokens: int) -> MagicMock:
    """A terminal chunk carrying usage, as providers send when asked to."""
    chunk = _chunk(None)
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = input_tokens
    chunk.usage.completion_tokens = output_tokens
    return chunk


def _stream(*chunks: MagicMock) -> Any:
    """Build the async iterator LiteLLM returns for a streamed call."""

    async def _iterator(**_: Any) -> AsyncIterator[MagicMock]:
        for chunk in chunks:
            yield chunk

    return _iterator


def _completion(text: str, input_tokens: int, output_tokens: int) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    response.model = "gpt-4o-mini"
    return response


class TestAcallStream:
    @pytest.mark.anyio
    async def test_yields_each_chunk_as_it_arrives(self) -> None:
        """Partial output must reach the caller incrementally, not in one batch."""
        stream = _stream(_chunk("Par"), _chunk("is"), _chunk("!"))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            seen = [chunk async for chunk in acall_stream("gpt-4o-mini", MESSAGES)]
        assert seen == ["Par", "is", "!"]

    @pytest.mark.anyio
    async def test_skips_empty_deltas(self) -> None:
        """Providers interleave content-free chunks; they are not output."""
        stream = _stream(_chunk("A"), _chunk(None), _chunk(""), _chunk("B"))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            seen = [chunk async for chunk in acall_stream("gpt-4o-mini", MESSAGES)]
        assert seen == ["A", "B"]

    @pytest.mark.anyio
    async def test_exposes_the_accumulated_response_when_done(self) -> None:
        stream = _stream(_chunk("Par"), _chunk("is"), _final_chunk(12, 3))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch("evalstand.llm.litellm.cost_per_token", return_value=(0.00010, 0.00002)),
        ):
            streamer = acall_stream("gpt-4o-mini", MESSAGES)
            async for _ in streamer:
                pass
            response = streamer.response

        assert response is not None
        assert response.text == "Paris"
        assert response.input_tokens == 12
        assert response.output_tokens == 3
        assert response.cost_usd == pytest.approx(0.00012)

    @pytest.mark.anyio
    async def test_response_is_unavailable_until_the_stream_finishes(self) -> None:
        """Reading totals mid-stream would give a half-formed answer."""
        stream = _stream(_chunk("A"), _chunk("B"))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            streamer = acall_stream("gpt-4o-mini", MESSAGES)
            assert streamer.response is None
            async for _ in streamer:
                break
            assert streamer.response is None

    @pytest.mark.anyio
    async def test_requests_usage_from_the_provider(self) -> None:
        """Without asking, most providers send no usage at all for a stream."""
        stream = _stream(_chunk("A"))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream) as acompletion,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            async for _ in acall_stream("gpt-4o-mini", MESSAGES):
                pass
        kwargs = acompletion.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    @pytest.mark.anyio
    async def test_tokens_are_none_when_the_provider_reports_no_usage(self) -> None:
        """Missing usage is unknown, not zero — a zero would understate spend."""
        stream = _stream(_chunk("Paris"))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch("evalstand.llm.litellm.completion_cost", side_effect=Exception("no usage")),
        ):
            streamer = acall_stream("gpt-4o-mini", MESSAGES)
            async for _ in streamer:
                pass

        assert streamer.response is not None
        assert streamer.response.input_tokens is None
        assert streamer.response.output_tokens is None
        assert streamer.response.cost_usd is None

    @pytest.mark.anyio
    async def test_records_latency(self) -> None:
        stream = _stream(_chunk("A"))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            streamer = acall_stream("gpt-4o-mini", MESSAGES)
            async for _ in streamer:
                pass
        assert streamer.response is not None
        assert streamer.response.latency_ms >= 0


class TestStreamedCostUsesTokenCounts:
    """A stream must be priced from its token counts, not from its text.

    Pricing a stream by passing an empty prompt would charge nothing for input
    tokens, understating the cost of every streamed call.
    """

    @pytest.mark.anyio
    async def test_prices_from_reported_usage(self) -> None:
        stream = _stream(_chunk("Paris"), _final_chunk(1000, 500))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch(
                "evalstand.llm.litellm.cost_per_token", return_value=(0.00015, 0.00030)
            ) as cost_per_token,
        ):
            streamer = acall_stream("gpt-4o-mini", MESSAGES)
            async for _ in streamer:
                pass

        assert streamer.response is not None
        assert streamer.response.cost_usd == pytest.approx(0.00045)
        kwargs = cost_per_token.call_args.kwargs
        assert kwargs["prompt_tokens"] == 1000
        assert kwargs["completion_tokens"] == 500

    @pytest.mark.anyio
    async def test_is_unpriced_when_usage_is_absent(self) -> None:
        """No token counts means no honest price, so the call stays unpriced."""
        stream = _stream(_chunk("Paris"))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch(
                "evalstand.llm.litellm.cost_per_token", return_value=(0.1, 0.2)
            ) as cost_per_token,
        ):
            streamer = acall_stream("gpt-4o-mini", MESSAGES)
            async for _ in streamer:
                pass

        assert streamer.response is not None
        assert streamer.response.cost_usd is None
        cost_per_token.assert_not_called()


class TestStreamedMatchesNonStreamed:
    """Task 1.3's acceptance criterion, stated directly."""

    @pytest.mark.anyio
    async def test_same_text_and_equivalent_cost(self) -> None:
        text, input_tokens, output_tokens, cost = "Paris is the capital.", 12, 6, 0.00042

        stream = _stream(
            _chunk("Paris "),
            _chunk("is the "),
            _chunk("capital."),
            _final_chunk(input_tokens, output_tokens),
        )
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch("evalstand.llm.litellm.cost_per_token", return_value=(0.00012, 0.00030)),
        ):
            streamer = acall_stream("gpt-4o-mini", MESSAGES)
            async for _ in streamer:
                pass
            streamed = streamer.response

        async def fake_acompletion(**_: Any) -> MagicMock:
            return _completion(text, input_tokens, output_tokens)

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake_acompletion),
            patch("evalstand.llm.litellm.completion_cost", return_value=cost),
        ):
            whole = await acall("gpt-4o-mini", MESSAGES)

        assert streamed is not None
        assert streamed.text == whole.text == text
        assert streamed.input_tokens == whole.input_tokens
        assert streamed.output_tokens == whole.output_tokens
        assert streamed.cost_usd == pytest.approx(whole.cost_usd)


class TestStreamedPricingFailure:
    @pytest.mark.anyio
    async def test_a_pricing_failure_leaves_the_call_unpriced(self) -> None:
        """An unknown model must not be recorded as free, even with usage."""
        stream = _stream(_chunk("Paris"), _final_chunk(10, 5))
        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=stream),
            patch("evalstand.llm.litellm.cost_per_token", side_effect=Exception("unknown model")),
        ):
            streamer = acall_stream("some/unlisted-model", MESSAGES)
            async for _ in streamer:
                pass

        assert streamer.response is not None
        assert streamer.response.cost_usd is None
        assert streamer.response.input_tokens == 10, "tokens are still recorded"
        assert streamer.response.text == "Paris", "the text is still returned"

"""The LiteLLM wrapper: token, cost, and latency extraction (task 1.2).

Every test here mocks LiteLLM. The suite must pass with no provider API key set.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evalstand.llm import LLMResponse, acall, call


def _fake_completion(
    text: str = "Paris",
    input_tokens: int = 12,
    output_tokens: int = 3,
    model: str = "gpt-4o-mini",
) -> MagicMock:
    """Shape a mock the way LiteLLM shapes a real completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    response.usage.total_tokens = input_tokens + output_tokens
    response.model = model
    return response


MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "capital of France?"}]


class TestCall:
    def test_returns_the_text(self) -> None:
        with (
            patch("evalstand.llm.litellm.completion", return_value=_fake_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.00012),
        ):
            assert call("gpt-4o-mini", MESSAGES).text == "Paris"

    def test_extracts_token_counts(self) -> None:
        with (
            patch(
                "evalstand.llm.litellm.completion",
                return_value=_fake_completion(input_tokens=100, output_tokens=25),
            ),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            response = call("gpt-4o-mini", MESSAGES)
        assert response.input_tokens == 100
        assert response.output_tokens == 25
        assert response.total_tokens == 125

    def test_extracts_cost(self) -> None:
        with (
            patch("evalstand.llm.litellm.completion", return_value=_fake_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.00042),
        ):
            assert call("gpt-4o-mini", MESSAGES).cost_usd == pytest.approx(0.00042)

    def test_records_latency(self) -> None:
        with (
            patch("evalstand.llm.litellm.completion", return_value=_fake_completion()),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert call("gpt-4o-mini", MESSAGES).latency_ms >= 0

    def test_cost_is_none_when_litellm_cannot_price_the_model(self) -> None:
        """An unknown price must not silently become $0.00 in a run total."""
        with (
            patch("evalstand.llm.litellm.completion", return_value=_fake_completion()),
            patch("evalstand.llm.litellm.completion_cost", side_effect=Exception("unknown model")),
        ):
            assert call("some/unlisted-model", MESSAGES).cost_usd is None

    def test_survives_a_response_without_usage(self) -> None:
        """Some providers omit usage; that is missing data, not zero tokens."""
        response = _fake_completion()
        response.usage = None
        with (
            patch("evalstand.llm.litellm.completion", return_value=response),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            result = call("gpt-4o-mini", MESSAGES)
        assert result.input_tokens is None
        assert result.output_tokens is None

    def test_empty_content_becomes_empty_string(self) -> None:
        response = _fake_completion()
        response.choices[0].message.content = None
        with (
            patch("evalstand.llm.litellm.completion", return_value=response),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert call("gpt-4o-mini", MESSAGES).text == ""

    def test_passes_parameters_through_to_litellm(self) -> None:
        with (
            patch(
                "evalstand.llm.litellm.completion", return_value=_fake_completion()
            ) as completion,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            call("gpt-4o-mini", MESSAGES, temperature=0.7, max_tokens=100)
        kwargs = completion.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["model"] == "gpt-4o-mini"

    def test_keeps_the_raw_response(self) -> None:
        """The raw provider payload stays available for debugging."""
        fake = _fake_completion()
        with (
            patch("evalstand.llm.litellm.completion", return_value=fake),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert call("gpt-4o-mini", MESSAGES).raw is fake


class TestAcall:
    @pytest.mark.anyio
    async def test_returns_the_text(self) -> None:
        async def fake_acompletion(**_: Any) -> MagicMock:
            return _fake_completion()

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake_acompletion),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert (await acall("gpt-4o-mini", MESSAGES)).text == "Paris"

    @pytest.mark.anyio
    async def test_matches_the_sync_path(self) -> None:
        """Sync and async must not disagree about tokens or cost."""

        async def fake_acompletion(**_: Any) -> MagicMock:
            return _fake_completion(input_tokens=50, output_tokens=10)

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake_acompletion),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.001),
        ):
            asynchronous = await acall("gpt-4o-mini", MESSAGES)
        with (
            patch(
                "evalstand.llm.litellm.completion",
                return_value=_fake_completion(input_tokens=50, output_tokens=10),
            ),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.001),
        ):
            synchronous = call("gpt-4o-mini", MESSAGES)

        assert asynchronous.text == synchronous.text
        assert asynchronous.input_tokens == synchronous.input_tokens
        assert asynchronous.output_tokens == synchronous.output_tokens
        assert asynchronous.cost_usd == synchronous.cost_usd


class TestLLMResponse:
    def test_total_tokens_is_none_when_counts_are_missing(self) -> None:
        assert LLMResponse(text="x", model="m", latency_ms=1).total_tokens is None

    def test_is_json_serialisable_without_the_raw_payload(self) -> None:
        """The raw provider object is not serialisable; storage must not choke."""
        response = LLMResponse(
            text="x", model="m", latency_ms=1, input_tokens=1, output_tokens=2, raw=object()
        )
        assert "raw" not in response.model_dump(mode="json")


class TestMalformedProviderResponses:
    """The defensive paths, which coverage reported as covered but nothing
    asserted on. A provider returning an unexpected shape must degrade to empty
    text, never take down a run that already cost money.
    """

    @pytest.mark.parametrize(
        ("label", "build"),
        [
            ("no choices attribute", lambda r: delattr(r, "choices")),
            ("choices is empty", lambda r: setattr(r, "choices", [])),
            ("choices is None", lambda r: setattr(r, "choices", None)),
            ("message is None", lambda r: setattr(r.choices[0], "message", None)),
            ("content is a dict", lambda r: setattr(r.choices[0].message, "content", {"a": 1})),
            ("content is a number", lambda r: setattr(r.choices[0].message, "content", 42)),
        ],
    )
    def test_text_extraction_degrades_to_empty(self, label: str, build: Any) -> None:
        from evalstand.llm import _extract_text

        response = _fake_completion()
        build(response)
        assert _extract_text(response) == "", label

    @pytest.mark.parametrize(
        ("label", "build"),
        [
            ("no choices attribute", lambda c: delattr(c, "choices")),
            ("choices is empty", lambda c: setattr(c, "choices", [])),
            ("delta is None", lambda c: setattr(c.choices[0], "delta", None)),
            ("content is not a string", lambda c: setattr(c.choices[0].delta, "content", 7)),
        ],
    )
    def test_delta_extraction_degrades_to_empty(self, label: str, build: Any) -> None:
        from evalstand.llm import _extract_delta

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "text"
        build(chunk)
        assert _extract_delta(chunk) == "", label

    def test_a_malformed_response_still_produces_a_usable_result(self) -> None:
        """The call already cost money; losing the whole response to a shape
        surprise would waste it and abort the case."""
        response = _fake_completion()
        response.choices = []

        with (
            patch("evalstand.llm.litellm.completion", return_value=response),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0001),
        ):
            result = call("gpt-4o-mini", MESSAGES)

        assert result.text == ""
        assert result.cost_usd == pytest.approx(0.0001), "the spend is still recorded"
        assert result.input_tokens == 12, "usage is still recorded"

    @pytest.mark.parametrize(
        ("label", "model_value"),
        [
            ("model is None", None),
            ("model is empty", ""),
            ("model is not a string", 12345),
        ],
    )
    def test_falls_back_to_the_requested_model_name(self, label: str, model_value: Any) -> None:
        """A provider echoing a junk model name must not lose the response."""
        response = _fake_completion()
        response.model = model_value

        with (
            patch("evalstand.llm.litellm.completion", return_value=response),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert call("gpt-4o-mini", MESSAGES).model == "gpt-4o-mini", label

    @pytest.mark.parametrize(
        ("label", "build"),
        [
            ("usage is None", lambda r: setattr(r, "usage", None)),
            ("no prompt_tokens", lambda r: delattr(r.usage, "prompt_tokens")),
            ("prompt_tokens is None", lambda r: setattr(r.usage, "prompt_tokens", None)),
        ],
    )
    def test_missing_usage_is_none_never_zero(self, label: str, build: Any) -> None:
        """Zero tokens is a claim; unknown is the truth."""
        response = _fake_completion()
        build(response)

        with (
            patch("evalstand.llm.litellm.completion", return_value=response),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            result = call("gpt-4o-mini", MESSAGES)

        assert result.input_tokens is None, label
        assert result.total_tokens is None, label

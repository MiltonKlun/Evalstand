"""The model-call layer, built on LiteLLM.

Every model call `evalstand` makes goes through here, which is what lets calls
be cached, traced, and costed uniformly regardless of provider.

A note on missing data: when a token count or a price is unavailable, this
module reports `None` rather than `0`. A zero cost is a claim that the call was
free, and a run total built from such claims would understate real spend.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import litellm
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LLMResponse", "StreamedCall", "acall", "acall_stream", "call"]

logger = logging.getLogger("evalstand.llm")


class LLMResponse(BaseModel):
    """One model call's outcome.

    `raw` holds the provider's own object for debugging. It is excluded from
    serialisation because it is not JSON-safe, and storage must not choke on it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    model: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    raw: Any = Field(default=None, exclude=True)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def _as_model_name(chunk: Any) -> str | None:
    """A chunk's model name, only when it really is a string."""
    name = getattr(chunk, "model", None)
    return name if isinstance(name, str) and name else None


def _extract_text(response: Any) -> str:
    """Pull the assistant message, treating an absent one as empty."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def _extract_tokens(response: Any) -> tuple[int | None, int | None]:
    """Token counts, or None when the provider did not report them."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    return prompt, completion


def _extract_cost(response: Any) -> float | None:
    """Price the call, or report None when LiteLLM cannot.

    LiteLLM raises for models it has no pricing for. Reporting None keeps an
    unpriced call visibly unpriced instead of silently free.
    """
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception as exc:  # any pricing failure means "unknown", never "free"
        logger.debug("could not price completion: %s", exc)
        return None


def _build(response: Any, model: str, latency_ms: int) -> LLMResponse:
    input_tokens, output_tokens = _extract_tokens(response)
    return LLMResponse(
        text=_extract_text(response),
        model=_as_model_name(response) or model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_extract_cost(response),
        raw=response,
    )


def call(model: str, messages: list[dict[str, Any]], **params: Any) -> LLMResponse:
    """Make one model call, reporting text, tokens, latency, and cost."""
    started = time.perf_counter()
    response = litellm.completion(model=model, messages=messages, **params)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return _build(response, model, latency_ms)


async def acall(model: str, messages: list[dict[str, Any]], **params: Any) -> LLMResponse:
    """Async twin of `call`. Must agree with it on tokens and cost."""
    started = time.perf_counter()
    response = await litellm.acompletion(model=model, messages=messages, **params)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return _build(response, model, latency_ms)


class StreamedCall:
    """One streamed model call: an async iterator of text chunks, plus totals.

    Iterate it to render output as it arrives; read `response` once iteration
    finishes to get the accumulated text, tokens, latency, and cost. `response`
    stays `None` until the stream is exhausted, so a caller cannot mistake a
    half-formed total for a final one.
    """

    def __init__(self, model: str, messages: list[dict[str, Any]], **params: Any) -> None:
        self._model = model
        self._messages = messages
        self._params = params
        self.response: LLMResponse | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        started = time.perf_counter()
        chunks: list[str] = []
        usage: Any = None
        model = self._model

        stream = await litellm.acompletion(
            model=self._model,
            messages=self._messages,
            stream=True,
            # Without this most providers send no usage for a stream at all,
            # which would leave every streamed call unpriced.
            stream_options={"include_usage": True},
            **self._params,
        )

        async for chunk in stream:
            usage = getattr(chunk, "usage", None) or usage
            model = _as_model_name(chunk) or model
            text = _extract_delta(chunk)
            if text:
                chunks.append(text)
                yield text

        latency_ms = int((time.perf_counter() - started) * 1000)
        self.response = _build_streamed("".join(chunks), model, latency_ms, usage)


def _extract_delta(chunk: Any) -> str:
    """Pull the text from one streamed chunk, if it carries any."""
    try:
        content = chunk.choices[0].delta.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def _build_streamed(text: str, model: str, latency_ms: int, usage: Any) -> LLMResponse:
    """Assemble the final response for a stream.

    Cost is priced from the reported token counts via `cost_per_token`, not from
    the accumulated text. Pricing text alone would charge nothing for input
    tokens and understate every streamed call. Without usage there is no honest
    price, so the call stays unpriced rather than free.
    """
    input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None

    cost: float | None = None
    if input_tokens is not None and output_tokens is not None:
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            cost = float(prompt_cost) + float(completion_cost)
        except Exception as exc:  # any pricing failure means "unknown", never "free"
            logger.debug("could not price streamed completion: %s", exc)

    return LLMResponse(
        text=text,
        model=model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )


def acall_stream(model: str, messages: list[dict[str, Any]], **params: Any) -> StreamedCall:
    """Stream one model call. Iterate for chunks, then read `.response`."""
    return StreamedCall(model, messages, **params)

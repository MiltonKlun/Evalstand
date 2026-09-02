"""The model-call layer, built on LiteLLM.

Every model call `evalstand` makes goes through here, which is what lets calls
be cached, traced, and costed uniformly regardless of provider.

A note on missing data: when a token count or a price is unavailable, this
module reports `None` rather than `0`. A zero cost is a claim that the call was
free, and a run total built from such claims would understate real spend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import litellm
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from evalstand.cache import ResponseCache

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

__all__ = ["LLMResponse", "StreamedCall", "acall", "acall_stream", "cache_key", "call"]

KEY_PARAMETERS = ("temperature", "top_p", "max_tokens", "seed", "tools", "response_format")
"""The parameters that change what a model says.

Everything else a caller might pass — a timeout, an api_base, a retry setting —
affects how the request travels, not what comes back, so it stays out of the key
and two calls differing only in transport share a cached response.
"""


def cache_key(model: str, messages: list[dict[str, Any]], **params: Any) -> str:
    """Identify one model call.

    Canonical JSON with sorted keys, so a call is the same call regardless of
    the order its parameters were written in.
    """
    payload = {
        "model": model,
        "messages": messages,
        "params": {name: params[name] for name in KEY_PARAMETERS if name in params},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


MAX_ATTEMPTS = 3
"""Three attempts total, not three retries. A fourth rarely converts."""

_BASE_DELAY_SECONDS = 0.5
_MAX_DELAY_SECONDS = 8.0

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
"""Rate limits, timeouts, and server faults. Everything else is permanent:
an auth failure or a rejected prompt fails identically on every attempt, so
retrying it only burns time and rate limit."""


def _status_code(error: BaseException) -> int | None:
    """The HTTP status an error carries, if it carries one."""
    for attribute in ("status_code", "code", "http_status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_retryable(error: BaseException) -> bool:
    """Only errors known to be transient are retried.

    An error with no recognisable status is treated as permanent: without
    evidence that a retry could succeed, retrying is a guess that costs time.
    """
    status = _status_code(error)
    return status is not None and status in RETRYABLE_STATUS_CODES


def _log_retry(state: RetryCallState) -> None:
    """Report every retry. A silent one hides both latency and spend."""
    error = state.outcome.exception() if state.outcome else None
    logger.warning(
        "retry %d/%d after %s; waiting %.2fs",
        state.attempt_number,
        MAX_ATTEMPTS - 1,
        error,
        state.idle_for or 0.0,
    )


def _retry_policy() -> dict[str, Any]:
    """The shared policy, so sync and async cannot drift apart."""
    return {
        "stop": stop_after_attempt(MAX_ATTEMPTS),
        "wait": wait_exponential_jitter(initial=_BASE_DELAY_SECONDS, max=_MAX_DELAY_SECONDS),
        "retry": retry_if_exception(_is_retryable),
        "before_sleep": _log_retry,
        "reraise": True,
    }


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
    cached: bool = False
    """True when this came from the cache rather than the provider. A run
    summary needs the hit rate, so a hit must be distinguishable from a call."""
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


def call(
    model: str,
    messages: list[dict[str, Any]],
    *,
    cache: ResponseCache | None = None,
    bypass_cache: bool = False,
    **params: Any,
) -> LLMResponse:
    """Make one model call, reporting text, tokens, latency, and cost.

    Transient failures are retried; permanent ones are raised immediately.
    Latency covers only the successful attempt, so it measures the model rather
    than the retry loop.

    When a `cache` is supplied, an identical earlier call is served from it.
    `bypass_cache` skips it in both directions: neither read nor written.
    """
    key = cache_key(model, messages, **params) if cache is not None else None

    if cache is not None and key is not None:
        hit = cache.get(key, bypass=bypass_cache)
        if hit is not None:
            return hit.model_copy(update={"cached": True})

    for attempt in Retrying(**_retry_policy()):
        with attempt:
            started = time.perf_counter()
            response = litellm.completion(model=model, messages=messages, **params)
            built = _build(response, model, int((time.perf_counter() - started) * 1000))
            if cache is not None and key is not None:
                cache.set(key, model, built, bypass=bypass_cache)
            return built

    raise AssertionError("unreachable: Retrying either returns or reraises")


async def acall(
    model: str,
    messages: list[dict[str, Any]],
    *,
    cache: ResponseCache | None = None,
    bypass_cache: bool = False,
    **params: Any,
) -> LLMResponse:
    """Async twin of `call`. Shares its retry and cache behaviour."""
    key = cache_key(model, messages, **params) if cache is not None else None

    if cache is not None and key is not None:
        hit = cache.get(key, bypass=bypass_cache)
        if hit is not None:
            return hit.model_copy(update={"cached": True})

    async for attempt in AsyncRetrying(**_retry_policy()):
        with attempt:
            started = time.perf_counter()
            response = await litellm.acompletion(model=model, messages=messages, **params)
            built = _build(response, model, int((time.perf_counter() - started) * 1000))
            if cache is not None and key is not None:
                cache.set(key, model, built, bypass=bypass_cache)
            return built

    raise AssertionError("unreachable: AsyncRetrying either returns or reraises")


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

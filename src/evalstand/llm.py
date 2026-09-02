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
from typing import Any

import litellm
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LLMResponse", "acall", "call"]

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
        model=getattr(response, "model", model) or model,
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

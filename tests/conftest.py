"""Shared test configuration and fixtures.

Async tests run through `anyio`, which arrives with litellm, rather than adding
`pytest-asyncio` as a direct dependency. Only the asyncio backend is exercised:
`evalstand` targets asyncio, and running every async test twice over trio would
buy nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def make_completion(
    text: str = "Paris",
    *,
    input_tokens: int | None = 12,
    output_tokens: int | None = 3,
    model: str = "gpt-4o-mini",
) -> MagicMock:
    """Shape a mock the way LiteLLM shapes a completion response.

    Defined once here because four test modules need the same shape; a local
    copy in each drifts as soon as one is edited.
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    if input_tokens is None and output_tokens is None:
        response.usage = None
    else:
        response.usage.prompt_tokens = input_tokens
        response.usage.completion_tokens = output_tokens
        response.usage.total_tokens = (input_tokens or 0) + (output_tokens or 0)
    response.model = model
    return response


@pytest.fixture
def completion() -> Callable[..., MagicMock]:
    """A factory, not an instance: tests that need two differing responses
    should not have to reach past the fixture to build the second."""
    return make_completion


@pytest.fixture
def messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "capital of France?"}]

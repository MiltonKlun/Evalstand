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


def help_text(*command: str) -> str:
    """The `--help` output for a command, with styling removed.

    Rich styles a flag by splitting it: `--threshold` is emitted as
    `[1;2;36m-[0m[1;2;36m-threshold`, so the literal string never
    appears in the output and `"--threshold" in output` is False.

    That depends on whether colour is enabled, which depends on the terminal —
    so tests asserting on help text passed locally and failed in CI, where
    GitHub Actions turns colour on. Every such assertion goes through here.
    """
    import re

    from typer.testing import CliRunner

    from evalstand.cli import app

    output = CliRunner().invoke(app, [*command, "--help"]).output
    return re.sub(r"\[[0-9;]*m", "", output)

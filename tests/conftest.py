"""Shared test configuration.

Async tests run through `anyio`, which arrives with litellm, rather than adding
`pytest-asyncio` as a direct dependency. Only the asyncio backend is exercised:
`evalstand` targets asyncio, and running every async test twice over trio would
buy nothing.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

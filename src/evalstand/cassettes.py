"""Cassettes: committed recordings of real provider responses.

A cassette lets a test run offline against a payload shape that genuinely came
back from a provider, rather than one a test author believed a provider returns.
That distinction is not academic: the streamed-cost bug in task 1.3 existed
because a hand-written mock agreed with the wrong assumption.

Cassettes are **not** the response cache. Deleting a cassette breaks the suite;
deleting the cache only costs money. The cache is an optimisation keyed on call
identity, the cassette is a fixture that pins behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any
from unittest.mock import patch

from evalstand.llm import KEY_PARAMETERS, LLMResponse, cache_key

__all__ = ["Cassette", "CassetteMissError", "CassetteMode"]

FORMAT_VERSION = 1


class CassetteMode(StrEnum):
    RECORD = "record"
    """Call the provider for real and write what comes back."""

    REPLAY = "replay"
    """Serve from disk. Never reach a provider."""


class CassetteMissError(LookupError):
    """No recording matches this call.

    Raised rather than falling through to the provider: a silent fallthrough
    means a test quietly spending money, which is what cassettes exist to stop.
    """


class Cassette:
    """A file of recorded interactions, addressed by call identity."""

    def __init__(self, path: Path | str, *, mode: CassetteMode = CassetteMode.REPLAY) -> None:
        self.path = Path(path)
        self.mode = mode
        self._interactions: dict[str, dict[str, Any]] = {}

        if mode is CassetteMode.REPLAY and self.path.exists():
            self._interactions = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        """Read the cassette, refusing anything we cannot read correctly.

        A cassette is a fixture, not a cache. The cache degrades a damaged entry
        to a miss because losing it only costs money; a cassette silently
        failing would let a test pass against the wrong data, so every failure
        here is loud and names the file.
        """
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"cassette {self.path.name} is not valid JSON: {exc}") from exc

        version = data.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"cassette {self.path.name} declares version {version}, but this "
                f"version of evalstand reads version {FORMAT_VERSION}"
            )

        return {entry["key"]: entry for entry in data.get("interactions", [])}

    def record(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Store one interaction. Only legal in record mode."""
        if self.mode is not CassetteMode.RECORD:
            raise RuntimeError(
                "cannot record in replay mode; a replay run must not rewrite a committed cassette"
            )

        params = params or {}
        key = cache_key(model, messages, **params)
        self._interactions[key] = {
            "key": key,
            "request": {
                "model": model,
                "messages": messages,
                # Only the parameters that shape a response are kept. Anything
                # else a caller passed — an api_key above all — stays out: a
                # cassette is committed, and a recorded credential is a leaked
                # one.
                "params": {name: params[name] for name in KEY_PARAMETERS if name in params},
            },
            "response": response.model_dump(mode="json"),
        }

    def replay(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Serve a recorded response, or raise."""
        key = cache_key(model, messages, **(params or {}))
        entry = self._interactions.get(key)
        if entry is None:
            raise CassetteMissError(
                f"no recording in {self.path.name} for a {model} call; "
                f"re-record the cassette to add it"
            )
        return LLMResponse.model_validate(entry["response"])

    def save(self) -> None:
        """Write the cassette out.

        Sorted and indented: cassettes are committed, so the file must diff
        legibly and must not churn just because calls happened in a different
        order.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "interactions": [self._interactions[key] for key in sorted(self._interactions)],
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @contextmanager
    def patched(self) -> Iterator[Cassette]:
        """Serve every model call from this cassette for the duration.

        An unrecorded call raises rather than reaching a provider.
        """

        def _completion(
            *, model: str, messages: list[dict[str, Any]], **params: Any
        ) -> _RecordedCompletion:
            return _RecordedCompletion(self.replay(model, messages, params=params))

        async def _acompletion(
            *, model: str, messages: list[dict[str, Any]], **params: Any
        ) -> _RecordedCompletion:
            return _RecordedCompletion(self.replay(model, messages, params=params))

        with (
            patch("evalstand.llm.litellm.completion", side_effect=_completion),
            patch("evalstand.llm.litellm.acompletion", side_effect=_acompletion),
            patch("evalstand.llm.litellm.completion_cost", side_effect=_recorded_cost),
        ):
            yield self


class _RecordedCompletion:
    """Presents a recorded response in the shape LiteLLM returns.

    Wrapping rather than returning the `LLMResponse` directly means the
    extraction code under test runs for real, instead of being bypassed.
    """

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.model = response.model
        self.choices = [_Choice(response.text)]
        self.usage = _Usage(response.input_tokens, response.output_tokens)


class _Choice:
    def __init__(self, text: str) -> None:
        self.message = _Message(text)
        self.delta = _Message(text)


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Usage:
    def __init__(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = (input_tokens or 0) + (output_tokens or 0)


def _recorded_cost(completion_response: Any = None, **_: Any) -> float:
    """Return the cost the cassette recorded, so replays report real figures."""
    if isinstance(completion_response, _RecordedCompletion):
        return completion_response._response.cost_usd or 0.0
    return 0.0

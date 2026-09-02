"""Retry policy (task 1.4).

Retry on 429 and 5xx with exponential backoff and jitter, at most 3 attempts.
Never retry an auth or content-policy error: those will fail identically every
time, and retrying them wastes the user's time and the provider's rate limit.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tenacity import RetryCallState, wait_none

from evalstand.llm import acall, call

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]


def _completion() -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "ok"
    response.usage.prompt_tokens = 1
    response.usage.completion_tokens = 1
    response.model = "gpt-4o-mini"
    return response


class _StatusError(Exception):
    """Stand-in for a provider error carrying an HTTP status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _no_waiting() -> Any:
    """Collapse backoff so the suite does not actually sleep.

    Patches the policy itself rather than a helper: tenacity owns the waiting,
    so patching anything else would leave the tests genuinely sleeping.
    """
    from evalstand.llm import _retry_policy

    def _instant() -> dict[str, Any]:
        return {**_retry_policy(), "wait": wait_none()}

    with patch("evalstand.llm._retry_policy", side_effect=_instant):
        yield


class TestRetriesTransientFailures:
    def test_retries_a_429_then_succeeds(self) -> None:
        attempts = [_StatusError(429), _completion()]
        with (
            patch("evalstand.llm.litellm.completion", side_effect=attempts) as completion,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert call("gpt-4o-mini", MESSAGES).text == "ok"
        assert completion.call_count == 2

    @pytest.mark.parametrize("status", [500, 502, 503, 529])
    def test_retries_server_errors(self, status: int) -> None:
        attempts = [_StatusError(status), _completion()]
        with (
            patch("evalstand.llm.litellm.completion", side_effect=attempts) as completion,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert call("gpt-4o-mini", MESSAGES).text == "ok"
        assert completion.call_count == 2

    def test_gives_up_after_three_attempts(self) -> None:
        with (
            patch("evalstand.llm.litellm.completion", side_effect=_StatusError(429)) as completion,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            pytest.raises(_StatusError),
        ):
            call("gpt-4o-mini", MESSAGES)
        assert completion.call_count == 3

    def test_logs_every_retry(self, caplog: pytest.LogCaptureFixture) -> None:
        """A silent retry hides both latency and spend."""
        attempts = [_StatusError(429), _StatusError(503), _completion()]
        with (
            patch("evalstand.llm.litellm.completion", side_effect=attempts),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            caplog.at_level("WARNING", logger="evalstand.llm"),
        ):
            call("gpt-4o-mini", MESSAGES)
        assert len([r for r in caplog.records if "retry" in r.getMessage().lower()]) == 2


class TestDoesNotRetryPermanentFailures:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_fails_immediately_on_client_errors(self, status: int) -> None:
        """A bad key or a rejected prompt fails the same way every time."""
        with (
            patch(
                "evalstand.llm.litellm.completion", side_effect=_StatusError(status)
            ) as completion,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            pytest.raises(_StatusError),
        ):
            call("gpt-4o-mini", MESSAGES)
        assert completion.call_count == 1

    def test_does_not_retry_an_error_without_a_status(self) -> None:
        """An unclassifiable error is not known to be transient."""
        with (
            patch("evalstand.llm.litellm.completion", side_effect=ValueError("bad")) as completion,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            pytest.raises(ValueError),
        ):
            call("gpt-4o-mini", MESSAGES)
        assert completion.call_count == 1


class TestAsyncRetries:
    @pytest.mark.anyio
    async def test_retries_a_429_then_succeeds(self) -> None:
        attempts = [_StatusError(429), _completion()]

        async def fake(**_: Any) -> MagicMock:
            outcome = attempts.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            assert (await acall("gpt-4o-mini", MESSAGES)).text == "ok"
        assert attempts == []

    @pytest.mark.anyio
    async def test_does_not_retry_a_401(self) -> None:
        calls = 0

        async def fake(**_: Any) -> MagicMock:
            nonlocal calls
            calls += 1
            raise _StatusError(401)

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
            pytest.raises(_StatusError),
        ):
            await acall("gpt-4o-mini", MESSAGES)
        assert calls == 1


class TestBackoff:
    """Exercises the real wait policy, so the autouse patch is lifted here."""

    @pytest.fixture(autouse=True)
    def _no_waiting(self) -> None:  # type: ignore[override]
        return None

    @staticmethod
    def _sample(attempt: int, count: int = 200) -> list[float]:
        from evalstand.llm import _retry_policy

        wait = _retry_policy()["wait"]
        state = RetryCallState(None, None, (), {})  # type: ignore[arg-type]
        state.attempt_number = attempt
        return [wait(state) for _ in range(count)]

    def test_wait_grows_and_is_jittered(self) -> None:
        """Exponential growth with jitter, so retries do not arrive in lockstep."""
        first, second = self._sample(1), self._sample(3)

        assert len(set(first)) > 1, "jitter should vary the wait"
        assert max(second) > max(first), "the backoff ceiling should grow"
        assert all(w >= 0 for w in first + second)

    def test_wait_is_capped(self) -> None:
        """Backoff must not grow without bound on a long outage."""
        from evalstand.llm import _MAX_DELAY_SECONDS

        assert all(w <= _MAX_DELAY_SECONDS for w in self._sample(20, count=50))

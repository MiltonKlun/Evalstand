"""The response cache (task 1.5).

The cache exists to avoid paying twice for an identical call. It is never
load-bearing for correctness: deleting it must cost money, never break anything.
Cassettes, not this, are what make tests deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalstand.cache import ResponseCache, cache_key
from evalstand.llm import LLMResponse

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "capital of France?"}]


def _response(text: str = "Paris") -> LLMResponse:
    return LLMResponse(
        text=text,
        model="gpt-4o-mini",
        latency_ms=120,
        input_tokens=12,
        output_tokens=3,
        cost_usd=0.00012,
    )


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache.db")


class TestCacheKey:
    def test_is_stable_for_the_same_call(self) -> None:
        assert cache_key("gpt-4o-mini", MESSAGES) == cache_key("gpt-4o-mini", MESSAGES)

    def test_ignores_key_order_within_parameters(self) -> None:
        a = cache_key("m", MESSAGES, temperature=0.0, max_tokens=10)
        b = cache_key("m", MESSAGES, max_tokens=10, temperature=0.0)
        assert a == b

    def test_differs_by_model(self) -> None:
        assert cache_key("gpt-4o-mini", MESSAGES) != cache_key("gpt-4o", MESSAGES)

    def test_differs_by_messages(self) -> None:
        other = [{"role": "user", "content": "capital of Spain?"}]
        assert cache_key("m", MESSAGES) != cache_key("m", other)

    @pytest.mark.parametrize(
        "param",
        ["temperature", "top_p", "max_tokens", "seed", "tools", "response_format"],
    )
    def test_differs_by_each_significant_parameter(self, param: str) -> None:
        """These change the response, so they must change the key."""
        values: dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "seed": 42,
            "tools": [{"type": "function"}],
            "response_format": {"type": "json_object"},
        }
        assert cache_key("m", MESSAGES) != cache_key("m", MESSAGES, **{param: values[param]})

    def test_ignores_parameters_that_do_not_change_the_response(self) -> None:
        """A timeout or an api_base does not alter what the model says."""
        assert cache_key("m", MESSAGES) == cache_key("m", MESSAGES, timeout=30, api_base="http://x")

    def test_is_a_sha256_hex_digest(self) -> None:
        key = cache_key("m", MESSAGES)
        assert len(key) == 64
        assert set(key) <= set("0123456789abcdef")


class TestGetAndSet:
    def test_a_miss_returns_none(self, cache: ResponseCache) -> None:
        assert cache.get(cache_key("m", MESSAGES)) is None

    def test_stores_and_retrieves_a_response(self, cache: ResponseCache) -> None:
        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response())

        cached = cache.get(key)
        assert cached is not None
        assert cached.text == "Paris"
        assert cached.input_tokens == 12
        assert cached.output_tokens == 3
        assert cached.cost_usd == pytest.approx(0.00012)

    def test_changing_temperature_misses(self, cache: ResponseCache) -> None:
        cache.set(cache_key("m", MESSAGES, temperature=0.0), "m", _response())
        assert cache.get(cache_key("m", MESSAGES, temperature=0.7)) is None

    def test_counts_hits(self, cache: ResponseCache) -> None:
        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response())

        for _ in range(3):
            cache.get(key)

        assert cache.hit_count(key) == 3

    def test_a_stored_entry_survives_reopening(self, tmp_path: Path) -> None:
        """The point of a cache is that it outlives the process."""
        path = tmp_path / "cache.db"
        key = cache_key("m", MESSAGES)
        ResponseCache(path).set(key, "m", _response())

        cached = ResponseCache(path).get(key)
        assert cached is not None
        assert cached.text == "Paris"

    def test_setting_the_same_key_twice_overwrites(self, cache: ResponseCache) -> None:
        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response("first"))
        cache.set(key, "m", _response("second"))

        cached = cache.get(key)
        assert cached is not None
        assert cached.text == "second"


class TestVersionInvalidation:
    def test_an_entry_from_another_version_is_ignored(
        self, cache: ResponseCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A change to key canonicalisation must invalidate, not mismatch."""
        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response())
        assert cache.get(key) is not None

        monkeypatch.setattr("evalstand.cache.__version__", "999.0.0")
        assert cache.get(key) is None


class TestBypass:
    def test_bypass_neither_reads_nor_writes(self, cache: ResponseCache) -> None:
        """Bypass runs in both directions: a response produced under conditions
        that made it non-reusable must not later be served as the answer."""
        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response("stored"))

        assert cache.get(key, bypass=True) is None

        cache.set(key, "m", _response("ignored"), bypass=True)
        cached = cache.get(key)
        assert cached is not None
        assert cached.text == "stored"


class TestDisposability:
    def test_deleting_the_file_costs_money_not_correctness(self, tmp_path: Path) -> None:
        """A deleted cache must degrade to a miss, never to an error."""
        path = tmp_path / "cache.db"
        key = cache_key("m", MESSAGES)
        cache = ResponseCache(path)
        cache.set(key, "m", _response())
        cache.close()
        path.unlink()

        assert ResponseCache(path).get(key) is None

    def test_creates_its_parent_directory(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "nested" / "deeper" / "cache.db")
        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response())
        assert cache.get(key) is not None

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, cache: ResponseCache) -> None:
        """Cache damage must never take down a run."""
        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response())
        cache.connection.execute("UPDATE cache SET response_json = ? WHERE key = ?", ("{bad", key))
        cache.connection.commit()

        assert cache.get(key) is None


class TestStats:
    def test_reports_entry_count(self, cache: ResponseCache) -> None:
        assert cache.entry_count() == 0
        cache.set(cache_key("m", MESSAGES), "m", _response())
        assert cache.entry_count() == 1

    def test_clear_empties_the_cache(self, cache: ResponseCache) -> None:
        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response())
        cache.clear()
        assert cache.get(key) is None
        assert cache.entry_count() == 0


class TestCachedCalls:
    """Task 1.5's acceptance criterion, stated end to end."""

    def test_the_same_call_twice_hits_the_provider_once(self, cache: ResponseCache) -> None:
        from unittest.mock import MagicMock, patch

        from evalstand.llm import call

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Paris"
        completion.usage.prompt_tokens = 12
        completion.usage.completion_tokens = 3
        completion.model = "gpt-4o-mini"

        with (
            patch("evalstand.llm.litellm.completion", return_value=completion) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.00012),
        ):
            first = call("gpt-4o-mini", MESSAGES, cache=cache)
            second = call("gpt-4o-mini", MESSAGES, cache=cache)

        assert provider.call_count == 1, "the second call should come from the cache"
        assert first.text == second.text == "Paris"
        assert cache.hit_count(cache_key("gpt-4o-mini", MESSAGES)) == 1

    def test_changing_temperature_calls_the_provider_again(self, cache: ResponseCache) -> None:
        from unittest.mock import MagicMock, patch

        from evalstand.llm import call

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Paris"
        completion.usage.prompt_tokens = 1
        completion.usage.completion_tokens = 1
        completion.model = "gpt-4o-mini"

        with (
            patch("evalstand.llm.litellm.completion", return_value=completion) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            call("gpt-4o-mini", MESSAGES, cache=cache, temperature=0.0)
            call("gpt-4o-mini", MESSAGES, cache=cache, temperature=0.7)

        assert provider.call_count == 2

    def test_no_cache_bypasses_in_both_directions(self, cache: ResponseCache) -> None:
        from unittest.mock import MagicMock, patch

        from evalstand.llm import call

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Paris"
        completion.usage.prompt_tokens = 1
        completion.usage.completion_tokens = 1
        completion.model = "gpt-4o-mini"

        with (
            patch("evalstand.llm.litellm.completion", return_value=completion) as provider,
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            call("gpt-4o-mini", MESSAGES, cache=cache, bypass_cache=True)
            call("gpt-4o-mini", MESSAGES, cache=cache, bypass_cache=True)

        assert provider.call_count == 2
        assert cache.entry_count() == 0, "a bypassed call must not be written either"

    def test_a_cached_response_reports_that_it_was_cached(self, cache: ResponseCache) -> None:
        """A run summary needs the hit rate, so a hit must be distinguishable."""
        from unittest.mock import MagicMock, patch

        from evalstand.llm import call

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Paris"
        completion.usage.prompt_tokens = 1
        completion.usage.completion_tokens = 1
        completion.model = "gpt-4o-mini"

        with (
            patch("evalstand.llm.litellm.completion", return_value=completion),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            live = call("gpt-4o-mini", MESSAGES, cache=cache)
            cached = call("gpt-4o-mini", MESSAGES, cache=cache)

        assert live.cached is False
        assert cached.cached is True

    @pytest.mark.anyio
    async def test_acall_uses_the_cache_too(self, cache: ResponseCache) -> None:
        from unittest.mock import MagicMock, patch

        from evalstand.llm import acall

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Paris"
        completion.usage.prompt_tokens = 1
        completion.usage.completion_tokens = 1
        completion.model = "gpt-4o-mini"

        calls = 0

        async def fake(**_: Any) -> MagicMock:
            nonlocal calls
            calls += 1
            return completion

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await acall("gpt-4o-mini", MESSAGES, cache=cache)
            await acall("gpt-4o-mini", MESSAGES, cache=cache)

        assert calls == 1


class TestConcurrency:
    """The cache must survive concurrent use.

    Phase 3 runs cases concurrently. asyncio alone stays on one thread, but a
    thread pool anywhere in a user's task would otherwise hit a confusing
    SQLite "objects created in a thread can only be used in that thread" crash.
    """

    def test_survives_concurrent_writes_from_many_threads(self, cache: ResponseCache) -> None:
        import threading

        errors: list[str] = []

        def worker(index: int) -> None:
            try:
                key = cache_key("m", [{"role": "user", "content": f"q{index}"}])
                cache.set(key, "m", _response(f"answer-{index}"))
                cache.get(key)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert cache.entry_count() == 50

    def test_concurrent_reads_and_writes_do_not_corrupt_counts(self, cache: ResponseCache) -> None:
        """hit_count is read-modify-write, so it needs the lock too."""
        import threading

        key = cache_key("m", MESSAGES)
        cache.set(key, "m", _response())

        def reader() -> None:
            for _ in range(20):
                cache.get(key)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert cache.hit_count(key) == 100, "every hit must be counted exactly once"

    @pytest.mark.anyio
    async def test_survives_concurrent_asyncio_calls(self, cache: ResponseCache) -> None:
        """The shape Phase 3's runner actually uses: gather on one thread."""
        import asyncio
        from unittest.mock import MagicMock, patch

        from evalstand.llm import acall

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Paris"
        completion.usage.prompt_tokens = 1
        completion.usage.completion_tokens = 1
        completion.model = "gpt-4o-mini"

        async def fake(**_: Any) -> MagicMock:
            return completion

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=fake),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            await asyncio.gather(
                *[
                    acall("gpt-4o-mini", [{"role": "user", "content": f"q{i}"}], cache=cache)
                    for i in range(8)
                ]
            )

        assert cache.entry_count() == 8

    def test_mixed_operations_under_heavy_contention(self, cache: ResponseCache) -> None:
        """Every public method must hold the lock, not just the obvious ones.

        An earlier fix locked `set` and `get` but left `hit_count` and
        `entry_count` unguarded, which surfaced only under contention as
        `fetchone()` returning None. Unit tests that exercise one method at a
        time cannot catch that; this one can.
        """
        import random
        import threading

        errors: list[str] = []

        def churn(index: int) -> None:
            try:
                for step in range(10):
                    key = cache_key("m", [{"role": "user", "content": f"q{(index * step) % 37}"}])
                    match random.choice(["set", "get", "hit_count", "entry_count"]):
                        case "set":
                            cache.set(key, "m", _response(f"t{index}"))
                        case "get":
                            cache.get(key)
                        case "hit_count":
                            cache.hit_count(key)
                        case _:
                            cache.entry_count()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=churn, args=(i,)) for i in range(40)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert errors == []
        assert [t for t in threads if t.is_alive()] == [], "a thread deadlocked"

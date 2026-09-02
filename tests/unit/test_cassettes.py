"""Cassette record/replay (task 1.6).

A cassette is a committed recording of a real provider response. It exists so
tests can run offline against payload shapes that actually came back from a
provider, rather than shapes a test author believed a provider returns — the
streamed-cost bug in task 1.3 came from exactly that gap.

Distinct from the Cache: deleting a cassette breaks the suite, deleting the
cache only costs money.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalstand.cassettes import Cassette, CassetteMissError, CassetteMode
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


class TestReplay:
    def test_replays_a_recorded_response(self, tmp_path: Path) -> None:
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response())
        cassette.save()

        replayed = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)
        response = replayed.replay("gpt-4o-mini", MESSAGES)
        assert response.text == "Paris"
        assert response.input_tokens == 12
        assert response.cost_usd == pytest.approx(0.00012)

    def test_a_missing_recording_raises_rather_than_calling_out(self, tmp_path: Path) -> None:
        """Silence here would mean a test quietly hitting a real provider and
        spending money, which is precisely what cassettes exist to prevent."""
        (tmp_path / "qa.json").write_text('{"version": 1, "interactions": []}', encoding="utf-8")
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)

        with pytest.raises(CassetteMissError, match="no recording"):
            cassette.replay("gpt-4o-mini", MESSAGES)

    def test_a_missing_file_in_replay_mode_raises(self, tmp_path: Path) -> None:
        cassette = Cassette(tmp_path / "absent.json", mode=CassetteMode.REPLAY)
        with pytest.raises(CassetteMissError):
            cassette.replay("gpt-4o-mini", MESSAGES)

    def test_replay_is_repeatable(self, tmp_path: Path) -> None:
        """Replaying twice must give the same answer; a cassette is not a queue."""
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response())
        cassette.save()

        replayed = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)
        assert replayed.replay("gpt-4o-mini", MESSAGES).text == "Paris"
        assert replayed.replay("gpt-4o-mini", MESSAGES).text == "Paris"

    def test_different_calls_replay_independently(self, tmp_path: Path) -> None:
        other = [{"role": "user", "content": "capital of Spain?"}]
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response("Paris"))
        cassette.record("gpt-4o-mini", other, _response("Madrid"))
        cassette.save()

        replayed = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)
        assert replayed.replay("gpt-4o-mini", MESSAGES).text == "Paris"
        assert replayed.replay("gpt-4o-mini", other).text == "Madrid"


class TestOnDiskFormat:
    def test_is_human_readable_json(self, tmp_path: Path) -> None:
        """A cassette is committed, so it must diff legibly in review."""
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response())
        cassette.save()

        raw = (tmp_path / "qa.json").read_text(encoding="utf-8")
        assert "\n" in raw, "a single-line cassette produces unreadable diffs"

        data = json.loads(raw)
        assert data["version"] == 1
        assert len(data["interactions"]) == 1

    def test_records_the_request_alongside_the_response(self, tmp_path: Path) -> None:
        """A reviewer must be able to see what was asked, not just what came back."""
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response())
        cassette.save()

        interaction = json.loads((tmp_path / "qa.json").read_text(encoding="utf-8"))[
            "interactions"
        ][0]
        assert interaction["request"]["model"] == "gpt-4o-mini"
        assert interaction["request"]["messages"] == MESSAGES
        assert interaction["response"]["text"] == "Paris"

    def test_interactions_are_sorted_for_stable_diffs(self, tmp_path: Path) -> None:
        """Recording order must not churn the file; that hides real changes."""
        other = [{"role": "user", "content": "capital of Spain?"}]

        first = Cassette(tmp_path / "a.json", mode=CassetteMode.RECORD)
        first.record("gpt-4o-mini", MESSAGES, _response("Paris"))
        first.record("gpt-4o-mini", other, _response("Madrid"))
        first.save()

        second = Cassette(tmp_path / "b.json", mode=CassetteMode.RECORD)
        second.record("gpt-4o-mini", other, _response("Madrid"))
        second.record("gpt-4o-mini", MESSAGES, _response("Paris"))
        second.save()

        assert (tmp_path / "a.json").read_text(encoding="utf-8") == (tmp_path / "b.json").read_text(
            encoding="utf-8"
        )

    def test_no_api_keys_are_written(self, tmp_path: Path) -> None:
        """Cassettes are committed. A recorded credential is a leaked one."""
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record(
            "gpt-4o-mini",
            MESSAGES,
            _response(),
            params={"api_key": "sk-secret-do-not-commit", "temperature": 0.0},
        )
        cassette.save()

        raw = (tmp_path / "qa.json").read_text(encoding="utf-8")
        assert "sk-secret-do-not-commit" not in raw
        assert "api_key" not in raw
        assert "temperature" in raw, "parameters that shape the response are kept"


class TestModes:
    def test_record_mode_does_not_read_existing_recordings(self, tmp_path: Path) -> None:
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response("first"))
        cassette.save()

        rerecorded = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        rerecorded.record("gpt-4o-mini", MESSAGES, _response("second"))
        rerecorded.save()

        replayed = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)
        assert replayed.replay("gpt-4o-mini", MESSAGES).text == "second"

    def test_replay_mode_refuses_to_record(self, tmp_path: Path) -> None:
        """Otherwise a replay run could silently rewrite a committed cassette."""
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)
        with pytest.raises(RuntimeError, match="replay mode"):
            cassette.record("gpt-4o-mini", MESSAGES, _response())

    def test_parameters_distinguish_recordings(self, tmp_path: Path) -> None:
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response("cold"), params={"temperature": 0.0})
        cassette.record("gpt-4o-mini", MESSAGES, _response("warm"), params={"temperature": 0.9})
        cassette.save()

        replayed = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)
        assert replayed.replay("gpt-4o-mini", MESSAGES, params={"temperature": 0.0}).text == "cold"
        assert replayed.replay("gpt-4o-mini", MESSAGES, params={"temperature": 0.9}).text == "warm"


class TestPatching:
    def test_patched_calls_replay_without_touching_a_provider(self, tmp_path: Path) -> None:
        """The end the acceptance criterion cares about: llm.call() served
        entirely from disk, with no provider reachable."""
        from evalstand.llm import call

        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response())
        cassette.save()

        replayed = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)
        with replayed.patched():
            response = call("gpt-4o-mini", MESSAGES)

        assert response.text == "Paris"
        assert response.input_tokens == 12

    def test_an_unrecorded_call_fails_loudly_when_patched(self, tmp_path: Path) -> None:
        from evalstand.llm import call

        (tmp_path / "qa.json").write_text('{"version": 1, "interactions": []}', encoding="utf-8")
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)

        with cassette.patched(), pytest.raises(CassetteMissError):
            call("gpt-4o-mini", MESSAGES)

    @pytest.mark.anyio
    async def test_async_calls_replay_too(self, tmp_path: Path) -> None:
        from evalstand.llm import acall

        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response())
        cassette.save()

        replayed = Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY)
        with replayed.patched():
            response = await acall("gpt-4o-mini", MESSAGES)

        assert response.text == "Paris"


class TestUnreadableCassettes:
    """A cassette is a fixture, not a cache: it must fail loudly and legibly.

    The cache degrades a damaged entry to a miss because losing it only costs
    money. Losing a cassette silently would let a test pass against the wrong
    data, so these raise instead.
    """

    def test_a_corrupt_file_names_itself(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match=r"broken\.json"):
            Cassette(tmp_path / "broken.json", mode=CassetteMode.REPLAY)

    def test_a_newer_format_version_is_refused(self, tmp_path: Path) -> None:
        """Reading a format we do not understand would misinterpret it."""
        (tmp_path / "future.json").write_text(
            json.dumps({"version": 99, "interactions": []}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="version 99"):
            Cassette(tmp_path / "future.json", mode=CassetteMode.REPLAY)

    def test_an_older_format_version_is_readable(self, tmp_path: Path) -> None:
        """Only newer is refused; the current version must still load."""
        cassette = Cassette(tmp_path / "qa.json", mode=CassetteMode.RECORD)
        cassette.record("gpt-4o-mini", MESSAGES, _response())
        cassette.save()
        assert Cassette(tmp_path / "qa.json", mode=CassetteMode.REPLAY).replay(
            "gpt-4o-mini", MESSAGES
        )

    def test_a_missing_version_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "novers.json").write_text(json.dumps({"interactions": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            Cassette(tmp_path / "novers.json", mode=CassetteMode.REPLAY)

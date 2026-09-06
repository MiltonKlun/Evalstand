"""The mutation harness is itself under test.

A mutant whose anchor no longer matches the source silently tests **nothing**,
and the harness reported it as a survivor — "behaviour no test asserts on" —
which sends a reader to write a test that already exists.

One anchor went stale for four tasks: a list comprehension in `plugin.py` became
a `for` loop when per-run recording landed, and nobody noticed because the line
in the report looked like every other survivor.

The suite is only as trustworthy as the thing measuring it, so the measuring
tool gets the same scrutiny as the code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "mutate.py"


def _harness() -> Any:
    """Import `scripts/mutate.py`, which is not a package."""
    if "mutate_harness" in sys.modules:
        return sys.modules["mutate_harness"]

    spec = importlib.util.spec_from_file_location("mutate_harness", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutate_harness"] = module
    spec.loader.exec_module(module)
    return module


class TestEveryAnchorStillMatches:
    """The failure this file exists for."""

    def test_no_mutant_has_a_stale_anchor(self) -> None:
        """A stale anchor is worse than a missing mutant: the missing one is
        absent from the report, and the stale one is present and reassuring.

        This fails the moment a refactor moves the line a mutant targets, which
        is exactly when someone is in a position to fix it — rather than four
        tasks later when the score is finally read carefully.
        """
        harness = _harness()
        sources: dict[str, str] = {}
        stale: list[tuple[str, str]] = []

        for relative, description, old, _ in harness.MUTANTS:
            if relative not in sources:
                sources[relative] = (ROOT / relative).read_text(encoding="utf-8")
            if old not in sources[relative]:
                stale.append((relative, description))

        assert not stale, "these mutants test nothing:\n" + "\n".join(
            f"  {path} :: {description}" for path, description in stale
        )

    def test_every_mutant_names_a_file_that_exists(self) -> None:
        harness = _harness()
        missing = [
            relative for relative, _, _, _ in harness.MUTANTS if not (ROOT / relative).exists()
        ]

        assert missing == []

    def test_no_mutant_is_a_no_op(self) -> None:
        """A replacement identical to its anchor changes nothing and is killed
        by nothing — it would sit in the report as a permanent survivor."""
        harness = _harness()
        no_ops = [description for _, description, old, new in harness.MUTANTS if old == new]

        assert no_ops == []

    def test_descriptions_are_unique(self) -> None:
        """Two mutants sharing a description make a survivor impossible to
        trace back to the behaviour it describes."""
        harness = _harness()
        descriptions = [description for _, description, _, _ in harness.MUTANTS]

        duplicates = sorted({d for d in descriptions if descriptions.count(d) > 1})
        assert duplicates == []


class TestTheReportTellsThemApart:
    """A stale anchor and a survivor are different problems demanding different
    fixes, and the report used to call both "behaviour no test asserts on"."""

    def test_a_stale_anchor_is_not_counted_as_killed(self) -> None:
        """It tested nothing, so it cannot have killed anything."""
        harness = _harness()
        killed, note = harness.run_mutant(
            "src/evalstand/models.py",
            "an anchor that cannot match",
            "this text does not appear anywhere in the file",
            "replacement",
        )

        assert killed is False
        assert note.startswith("ANCHOR")

    def test_the_harness_source_distinguishes_the_two_in_its_report(self) -> None:
        """Asserted on the source because the reporting is in `main`, which
        runs the whole suite per mutant and cannot be called from a test."""
        text = HARNESS.read_text(encoding="utf-8")

        assert "STALE ANCHORS" in text
        assert "these mutants tested nothing" in text

    def test_a_stale_anchor_fails_the_run(self) -> None:
        """Exiting zero with a stale anchor would let it survive another four
        tasks."""
        text = HARNESS.read_text(encoding="utf-8")

        assert "return 1 if (survivors or stale) else 0" in text


class TestTheHarnessCoversWhatItClaims:
    def test_it_covers_every_source_module_that_holds_logic(self) -> None:
        """A module with no mutants has never been challenged, whatever its
        coverage percentage says."""
        harness = _harness()
        covered = {relative.replace("\\", "/") for relative, _, _, _ in harness.MUTANTS}

        substantial = {
            f"src/evalstand/{name}.py"
            for name in (
                "models",
                "runner",
                "plugin",
                "storage",
                "recording",
                "provenance",
                "comparison",
                "api",
                "llm",
                "cache",
            )
        }
        uncovered = sorted(substantial - covered)

        assert uncovered == [], f"no mutant challenges: {uncovered}"

    @pytest.mark.parametrize(
        "path",
        [
            "src/evalstand/scorers/base.py",
            "src/evalstand/scorers/text.py",
            "src/evalstand/scorers/numeric.py",
            "src/evalstand/scorers/json_field.py",
            "src/evalstand/scorers/llm.py",
            "src/evalstand/reporting/console.py",
        ],
    )
    def test_each_scorer_and_the_reporter_are_challenged(self, path: str) -> None:
        harness = _harness()
        covered = {relative.replace("\\", "/") for relative, _, _, _ in harness.MUTANTS}

        assert path in covered

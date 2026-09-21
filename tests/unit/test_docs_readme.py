"""The README's factual claims (task 7.5).

The README is the first and often only thing a reader sees, which makes a stale
claim there more expensive than anywhere else — and it goes stale silently,
because nothing imports it. This file has already caught two: a status line
saying "Phases 0-3 of 7" three phases late, and "levenshtein arrives in Phase 4"
written when it had already shipped.

Only the checkable things are asserted. This is not a prose review.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def text() -> str:
    return README.read_text(encoding="utf-8")


class TestItsExamplesWork:
    def test_every_python_block_compiles(self, text: str) -> None:
        """A reader pastes the first block they see. One that does not parse
        costs them the time to find out it was never checked."""
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        assert blocks, "the README should carry a runnable example"

        for index, block in enumerate(blocks, start=1):
            try:
                compile(block, f"<README block {index}>", "exec")
            except SyntaxError as exc:  # pragma: no cover - only on a broken README
                pytest.fail(f"block {index} does not compile: {exc}")

    def test_every_name_the_example_imports_exists(self, text: str) -> None:
        """`from evalstand import trace` was documented while it raised
        NotImplementedError. An import line in a README is a promise."""
        import importlib

        for module, names in re.findall(r"^from ([\w.]+) import (.+)$", text, re.MULTILINE):
            imported = importlib.import_module(module)
            for name in (n.strip() for n in names.split(",")):
                assert hasattr(imported, name), f"{module} has no {name}"

    def test_every_relative_link_resolves(self, text: str) -> None:
        """A broken link in the README is the most visible kind of rot."""
        broken = [
            target
            for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
            if not target.startswith("http") and not (ROOT / target.split("#")[0]).exists()
        ]

        assert not broken, f"the README links to files that do not exist: {broken}"


class TestItsCountsAreReal:
    def test_the_scorers_it_names_are_the_scorers_that_exist(self, text: str) -> None:
        """A list that drifts understates or oversells the library, and a reader
        who tries a scorer that is not there concludes the docs are stale."""
        import evalstand.scorers as scorers

        exported = {
            name
            for name in scorers.__all__
            if name not in {"Scorer", "scorer", "parse_number", "parse_object"}
        }

        # The README names them in prose ("close-to", "JSON fields"), so each is
        # matched on its distinctive stem rather than its identifier. Asserting
        # the identifier would force the prose to read like code.
        stems = {name.split("_")[0].rstrip("s") for name in exported}
        lowered = text.lower()
        missing = [stem for stem in stems if stem not in lowered]

        assert not missing, f"the README does not name these scorers: {sorted(missing)}"
        assert "Ten scorers" in text, "the count should match the library"
        assert len(exported) == 10

    def test_the_mutant_count_matches_the_harness(self, text: str) -> None:
        """A number in a README is quoted elsewhere. This one is a claim about
        how well the project is tested, so it has to be the real figure."""
        import importlib.util

        claimed = re.search(r"(\d+) mutants", text)
        assert claimed, "the README should state the mutant count"

        spec = importlib.util.spec_from_file_location("m", ROOT / "scripts" / "mutate.py")
        assert spec and spec.loader
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)

        assert int(claimed.group(1)) == len(harness.MUTANTS)

    def test_the_commands_it_shows_are_real(self, text: str) -> None:
        from evalstand.cli import app

        available = {c.name or c.callback.__name__ for c in app.registered_commands}
        shown = set(re.findall(r"^evalstand (\w+)", text, re.MULTILINE))

        missing = shown - available
        assert not missing, f"the README shows commands that do not exist: {missing}"


class TestTheLimitationsSectionIsHonest:
    """Required by the plan, and the section most likely to be quietly dropped
    once the numbers look good."""

    def test_it_exists(self, text: str) -> None:
        assert "## Limitations" in text

    def test_it_says_a_delta_is_not_a_verdict(self, text: str) -> None:
        assert "no significance testing" in text.lower()

    def test_it_says_judge_scorers_are_unvalidated(self, text: str) -> None:
        assert "unvalidated" in text.lower()

    def test_it_admits_the_baseline_is_unrecorded(self, text: str) -> None:
        """While `BASELINE.md` carries no numbers, the README must not imply it
        does. A baseline is the figure people quote."""
        baseline = (ROOT / "examples" / "pdf_extraction" / "BASELINE.md").read_text(
            encoding="utf-8"
        )
        if "Not yet recorded" in baseline:
            assert "no published baseline" in text.lower()

    def test_it_says_the_web_ui_has_no_authentication(self, text: str) -> None:
        """A security property a reader acts on. `serve` exposes every recorded
        prompt and completion, and the only thing standing between that and a
        network is a default the user can override with one flag."""
        assert "no authentication" in text.lower()

    def test_no_auth_is_still_true(self) -> None:
        """Pinned against the code, not just the prose. If authentication were
        ever added, this limitation would become a false warning — which erodes
        trust in the rest of the section."""
        import inspect

        from evalstand.cli import serve

        source = inspect.getsource(serve)
        for hint in ["password", "token=", "HTTPBasic", "Depends("]:
            assert hint not in source, f"serve now does auth; the README says it does not ({hint})"

    def test_it_admits_the_missing_demo_gif(self, text: str) -> None:
        """The plan asks for a GIF at the top. Until one exists, saying so beats
        an empty space a reader reads as a broken image."""
        if "demo.gif" not in text or text.count("demo.gif") == 1:
            assert "No demo GIF yet" in text or "demo GIF" in text


class TestTheStatusLine:
    """The line a reader trusts before trying anything.

    It previously had to assert the *opposite* of what it asserts now: while
    nothing was on PyPI, `pip install evalstand` as the headline instruction
    would have failed for every reader who tried it. The direction flipped at
    the 1.0.0 release, so the test flipped with it — the invariant is that the
    line matches reality, not that it says any particular thing.
    """

    def test_the_install_line_matches_the_declared_version(self, text: str) -> None:
        """A README promising `pip install` for a version that was never
        published sends a reader to an error message. The two claims have to
        move together, so they are checked together."""
        import evalstand

        flat = " ".join(line.lstrip("> ") for line in text.splitlines())
        flat = " ".join(flat.split())

        published = not evalstand.__version__.startswith("0.0.0")
        promises_install = "pip install evalstand" in flat

        assert promises_install == published, (
            f"__version__ is {evalstand.__version__!r} but the README "
            f"{'does not promise' if published else 'promises'} pip install"
        )

    def test_it_states_the_version_it_ships(self, text: str) -> None:
        """A status line naming a different version than the package is the
        kind of drift nobody notices until someone quotes it."""
        import evalstand

        flat = " ".join(text.splitlines())

        assert evalstand.__version__ in flat, (
            f"the status line does not name {evalstand.__version__}"
        )

    def test_the_phase_count_matches_the_plan(self, text: str) -> None:
        """The status line said "Phases 0-3 of 7" while phase 6 was finished.

        Now that every phase is done the line says "All 8 phases", so the check
        is that the number it names is the number the plan has.
        """
        plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
        phases = len(re.findall(r"^## Phase \d", plan, re.MULTILINE))

        claimed = re.search(r"All (\d+) phases", text) or re.search(r"Phases 0-\d+ of (\d+)", text)
        assert claimed, "the status line should say how far along the project is"

        # Phase 0 is counted in the plan's headings but not in the total.
        assert int(claimed.group(1)) == phases - 1, "the README's phase total is wrong"

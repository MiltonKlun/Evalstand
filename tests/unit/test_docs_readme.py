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
from typing import ClassVar

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


LIMITATIONS = ROOT / "docs" / "limitations.md"


@pytest.fixture(scope="module")
def limitations() -> str:
    return LIMITATIONS.read_text(encoding="utf-8")


class TestTheLimitationsAreStatedAndReachable:
    """The limitations live on their own docs page, linked from the README.

    They moved out of the README to keep its front page short, which makes the
    link the thing most likely to be lost next: a limitations page nobody can
    find from the front page is a limitations page nobody reads. So the link is
    checked as well as the claims.
    """

    def test_the_readme_links_to_them(self, text: str) -> None:
        assert "docs/limitations.md" in text, "the README no longer points at the limitations"

    def test_it_says_a_delta_is_not_a_verdict(self, limitations: str) -> None:
        assert "no significance testing" in limitations.lower()

    def test_it_says_judge_scorers_are_unvalidated(self, limitations: str) -> None:
        assert "unvalidated" in limitations.lower()

    def test_it_describes_the_baseline_that_actually_exists(self, limitations: str) -> None:
        """The README and `BASELINE.md` must agree about whether there is one.

        This only checked one direction while the file was a placeholder: the
        README had to admit there was no baseline. Once a real one landed, a
        README still saying "no published baseline" would have passed, and so
        would one describing a baseline without naming the method behind it.
        A baseline is the figure people quote, so both halves are checked.
        """
        baseline = (ROOT / "examples" / "pdf_extraction" / "BASELINE.md").read_text(
            encoding="utf-8"
        )
        text = limitations
        if "Not yet recorded" in baseline:
            assert "no published baseline" in text.lower()
            return

        assert "no published baseline" not in text.lower(), "the page denies a real baseline"

        model = re.search(r"\| model \| `([^`]+)` \|", baseline)
        assert model, "BASELINE.md does not name its model"
        assert model.group(1) in text, f"the page does not say {model.group(1)} produced it"

        if "Line items were not scored" in baseline:
            assert "line items" in text.lower(), "the page omits what the baseline does not cover"

    def test_it_says_the_web_ui_has_no_authentication(self, limitations: str) -> None:
        """A security property a reader acts on. `serve` exposes every recorded
        prompt and completion, and the only thing standing between that and a
        network is a default the user can override with one flag."""
        assert "no authentication" in limitations.lower()

    def test_no_auth_is_still_true(self) -> None:
        """Pinned against the code, not just the prose. If authentication were
        ever added, this limitation would become a false warning — which erodes
        trust in the rest of the section."""
        import inspect

        from evalstand.cli import serve

        source = inspect.getsource(serve)
        for hint in ["password", "token=", "HTTPBasic", "Depends("]:
            assert hint not in source, f"serve now does auth; the README says it does not ({hint})"

    def test_the_demo_gif_it_shows_exists_within_the_plans_limits(self, text: str) -> None:
        """Task 6.8: a GIF at the top, under 5 MB and 30 seconds.

        This used to be a conditional that passed whether or not the GIF
        existed. Now there is one, so the checks are about it: the path the
        README embeds resolves, the file is inside the plan's limits, and the
        README has stopped saying it is missing. Duration is read from the
        GIF's own frame delays rather than trusted from a note.
        """
        import re as _re

        embedded = _re.search(r"\]\((docs/demo\.gif)\)", text)
        assert embedded, "the README does not embed docs/demo.gif"

        gif = ROOT / embedded.group(1)
        assert gif.exists(), "the README embeds a GIF that is not in the repository"

        data = gif.read_bytes()
        assert data[:6] in (b"GIF87a", b"GIF89a"), "docs/demo.gif is not a GIF"
        assert len(data) < 5 * 1024 * 1024, f"{len(data)} bytes, over the plan's 5 MB"

        # Graphic Control Extension: 0x21 0xF9 0x04, packed byte, then the
        # frame delay in hundredths of a second, little-endian.
        delays = [
            int.from_bytes(data[match.start() + 4 : match.start() + 6], "little")
            for match in _re.finditer(rb"\x21\xf9\x04", data)
        ]
        seconds = sum(delays) / 100
        assert delays, "no frame timing found in the GIF"
        assert 5 < seconds < 30, f"{seconds:.1f}s, outside the plan's 30 seconds"

        assert "No demo GIF yet" not in text


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


class TestNothingPublicPointsAtAPrivateFile:
    """Working notes are kept out of the repository by `.gitignore`. A link to
    one from a public page would be a 404 for every visitor who followed it."""

    PRIVATE: ClassVar[list[str]] = ["PLAN.md", "RESUME.md", "JEV.md", "DEMO_SCRIPT.md"]

    def test_no_tracked_document_names_them(self) -> None:
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files", "*.md", "*.yml", "*.toml"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()

        offenders = [
            f"{path}: {name}"
            for path in tracked
            for name in self.PRIVATE
            if name in (ROOT / path).read_text(encoding="utf-8")
        ]
        assert not offenders, f"public files reference private notes: {offenders}"

    def test_the_private_files_are_ignored(self) -> None:
        """Listed in `.gitignore`, so a `git add .` cannot publish them again."""
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

        for name in self.PRIVATE:
            assert any(line.strip().rstrip("/").endswith(name) for line in ignored), (
                f"{name} is not in .gitignore"
            )

"""The docs site holds together.

A documentation site fails differently from code: nothing raises, the page just
says the wrong thing or 404s, and the reader concludes the tool is unfinished.
So the things asserted here are the ones that break silently — a nav entry
pointing at a file nobody wrote, a page that exists but is still a stub, a code
example that would not run.

`mkdocs build --strict` catches broken *links* in CI. These are the checks that
run without building.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CONFIG = ROOT / "mkdocs.yml"


@pytest.fixture(scope="module")
def config() -> dict:
    yaml = pytest.importorskip("yaml", reason="pyyaml is only available transitively")

    # mkdocs uses `!!python/name:` tags that safe_load rejects, so unknown tags
    # are ignored rather than resolved — this file only needs the nav.
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda *_: None)
    Loader.add_multi_constructor("!", lambda *_: None)
    return yaml.load(CONFIG.read_text(encoding="utf-8"), Loader=Loader)


def _nav_pages(entry: object) -> list[str]:
    """Every markdown path the nav names, at any depth."""
    if isinstance(entry, str):
        return [entry] if entry.endswith(".md") else []
    if isinstance(entry, list):
        return [page for item in entry for page in _nav_pages(item)]
    if isinstance(entry, dict):
        return [page for value in entry.values() for page in _nav_pages(value)]
    return []


class TestTheNav:
    def test_every_page_it_names_exists(self, config: dict) -> None:
        """A nav entry pointing at a missing file builds a link to a 404 — and
        `strict` catches it in CI, but only after someone pushed."""
        missing = [page for page in _nav_pages(config["nav"]) if not (DOCS / page).exists()]

        assert not missing, f"the nav names pages that do not exist: {missing}"

    def test_every_page_that_exists_is_reachable(self, config: dict) -> None:
        """A page nobody links to is a page nobody reads. Writing one and
        forgetting to add it to the nav is the easiest mistake here."""
        listed = set(_nav_pages(config["nav"]))
        on_disk = {str(path.relative_to(DOCS)).replace("\\", "/") for path in DOCS.rglob("*.md")}

        assert not on_disk - listed, f"these pages are not in the nav: {sorted(on_disk - listed)}"

    def test_every_adr_is_listed(self, config: dict) -> None:
        """The ADRs are the record of why the design is what it is. One written
        and never linked is a decision nobody can find."""
        listed = {page for page in _nav_pages(config["nav"]) if page.startswith("adr/")}
        on_disk = {f"adr/{path.name}" for path in (DOCS / "adr").glob("*.md")}

        assert on_disk == listed


class TestTheSiteCoversWhatWasPromised:
    """The pages the docs site promises. Two of them were once missing while
    everything built green.

    `TestTheNav` compares the nav against the files on disk, which agree with
    each other when a page is missing from *both* — so `architecture.md` and the
    ADR index were absent for weeks without a single failing test and
    with `mkdocs build --strict` green, because nothing linked to what nobody
    had written. Consistency is not coverage. This class asserts against the
    promise instead of against the site.
    """

    PROMISED: ClassVar[list[str]] = [
        "quickstart.md",
        "writing-evals.md",
        "scorers.md",
        "traces.md",
        "ci.md",
        "architecture.md",
        "adr/index.md",
    ]

    @pytest.mark.parametrize("page", PROMISED)
    def test_the_page_exists_and_is_in_the_nav(self, config: dict, page: str) -> None:
        assert (DOCS / page).exists(), f"the docs site promises {page}"
        assert page in _nav_pages(config["nav"]), f"{page} exists but is unreachable"

    def test_the_adr_index_lists_every_decision(self) -> None:
        """An index that silently omits a record is worse than no index: a
        reader who trusts it concludes the decision was never made.

        Asserted on the **table rows**, not on the page text. Most ADRs are also
        mentioned in the prose below the table, so a substring search over the
        whole page passes even after a row is deleted — which it did, the first
        time this was written.
        """
        index = (DOCS / "adr" / "index.md").read_text(encoding="utf-8")
        rows = [line for line in index.splitlines() if line.startswith("| [")]
        tabled = {name for line in rows for name in re.findall(r"\(([0-9]{4}-[^)]+\.md)\)", line)}

        on_disk = {adr.name for adr in (DOCS / "adr").glob("*.md") if adr.name != "index.md"}

        assert on_disk == tabled, f"the index table omits {sorted(on_disk - tabled)}"

    def test_the_architecture_page_names_the_real_modules(self) -> None:
        """A design document that describes modules which do not exist sends a
        reader to a path that is not there. These are the load-bearing ones."""
        text = (DOCS / "architecture.md").read_text(encoding="utf-8")
        source = Path(__file__).resolve().parents[2] / "src" / "evalstand"

        for module in ["runner.py", "plugin.py", "storage.py", "tracing.py", "llm.py"]:
            assert module in text, f"the architecture page never mentions {module}"
            assert (source / module).exists(), f"it names {module}, which does not exist"

    def test_it_keeps_the_cache_and_cassette_distinction(self) -> None:
        """The one thing in the codebase most likely to be collapsed by a
        well-meaning change: deleting a cassette breaks the suite, deleting the
        cache only costs money."""
        text = (DOCS / "architecture.md").read_text(encoding="utf-8")

        assert "cassettes.py" in text
        assert "cache.py" in text

    def test_it_documents_that_missing_data_is_none(self) -> None:
        """The honesty rule the reporting layer is built on. A reader who
        assumes zero-as-unknown will build a total that understates spend."""
        text = (DOCS / "architecture.md").read_text(encoding="utf-8")

        assert "None" in text and "zero" in text.lower()


class TestNoPageIsStillAStub:
    """`index.md`, `quickstart.md` and `traces.md` each sat at three lines
    saying "To be written" while the features they describe were shipped and
    tested. A site that publishes those is worse than one that omits them."""

    @pytest.mark.parametrize("page", sorted(str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md")))
    def test_it_says_something(self, page: str) -> None:
        text = (DOCS / page).read_text(encoding="utf-8")

        assert "To be written" not in text, f"{page} is still a placeholder"
        assert len(text.splitlines()) > 10, f"{page} is too short to be finished"


class TestTheExamplesCompile:
    """A code block that does not run costs a reader the time to find out."""

    @pytest.mark.parametrize("page", sorted(str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md")))
    def test_every_python_block_compiles(self, page: str) -> None:
        text = (DOCS / page).read_text(encoding="utf-8")

        for index, block in enumerate(re.findall(r"```python\n(.*?)```", text, re.DOTALL), 1):
            try:
                compile(block, f"<{page} block {index}>", "exec")
            except SyntaxError as exc:  # pragma: no cover - only on a broken doc
                pytest.fail(f"{page} block {index} does not compile: {exc}")


class TestEveryDocumentedFlagExists:
    """The architecture page described `--repeat 3` from the start. There has never
    been such a flag — repeats are `repeat=N` on `evaluate()` — so a reader who
    tried it got "No such option" from a page explaining the design.

    Every `--flag` in the published Markdown must be an option of an
    `evalstand` command, an option the pytest plugin registers, or one of the
    other tools' flags listed below by name.
    """

    # Flags the documents quote for tools that are not evalstand, each named
    # with its owner so the list is reviewed rather than grown by reflex.
    OTHER_TOOLS: ClassVar[dict[str, str]] = {
        "--lf": "pytest",
        "--maxfail": "pytest",
        "--collect-only": "pytest",
        "--no-cov": "pytest-cov",
        "--dev": "uv",
        "--extra": "uv",
        "--all-extras": "uv",
        "--seed": "examples/pdf_extraction/generate.py",
        "--check": "examples/pdf_extraction/generate.py",
        "--list": "scripts/mutate.py",
    }

    @staticmethod
    def _evalstand_flags() -> set[str]:
        import typer.main

        from evalstand import plugin
        from evalstand.cli import app

        flags: set[str] = set()
        group = typer.main.get_command(app)
        for command in group.commands.values():  # type: ignore[attr-defined]
            for param in command.params:
                flags.update(param.opts)
                flags.update(param.secondary_opts)

        class Group:
            def addoption(self, *names: str, **_: object) -> None:
                flags.update(names)

        class Parser:
            def getgroup(self, *_: object) -> Group:
                return Group()

        plugin.pytest_addoption(Parser())  # type: ignore[arg-type]
        return flags

    def test_no_document_names_an_option_that_does_not_exist(self) -> None:
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
        known = self._evalstand_flags() | set(self.OTHER_TOOLS)
        flag = re.compile(r"(?:^|[\s`(])(--[a-z][a-z-]*)")

        offenders = [
            f"{path}:{number}: {name}"
            for path in tracked
            # The changelog records history, including options since renamed.
            if path != "CHANGELOG.md"
            for number, line in enumerate(
                (ROOT / path).read_text(encoding="utf-8").splitlines(), start=1
            )
            for name in flag.findall(line)
            if name not in known
        ]
        assert not offenders, "documented options that do not exist:\n" + "\n".join(offenders)

    def test_it_knows_the_real_options(self) -> None:
        """Guards the guard: an empty set would make every flag look unknown,
        and a failure there would be read as a docs problem."""
        flags = self._evalstand_flags()

        assert {"--threshold", "--allow-dirty", "--store", "--open"} <= flags
        assert "--repeat" not in flags


class TestTheBuildStaysStrict:
    def test_strict_is_on(self, config: dict) -> None:
        """Without it, a broken internal link is published rather than caught.
        A docs site that 404s inside itself costs more trust than it saves."""
        assert config.get("strict") is True

    def test_the_workflow_builds_strictly(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

        assert "mkdocs build --strict" in workflow

    def test_it_builds_on_pull_requests_not_only_on_main(self) -> None:
        """A docs build that runs only after merge finds its broken links on
        main, where they are already public."""
        workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

        assert "pull_request" in workflow

    def test_deployment_needs_pages_permission(self) -> None:
        """Without it the deploy step fails at the end of a green build, which
        reads as a flaky pipeline rather than a missing scope."""
        workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

        assert "pages: write" in workflow
        assert "id-token: write" in workflow

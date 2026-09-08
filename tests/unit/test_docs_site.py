"""The docs site holds together (task 7.4).

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

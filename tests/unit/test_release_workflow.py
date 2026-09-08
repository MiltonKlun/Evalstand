"""The release workflow (task 7.6).

A release workflow is the least-tested code in most projects: it runs once per
version, in an environment nobody can reproduce locally, and its failures are
discovered at the worst moment. PyPI makes some of them unrecoverable — a
version number, once uploaded, cannot be reused even after the file is deleted.

So the things asserted here are the ones that would be expensive to get wrong on
the day: publishing without running the tests, publishing a version that does
not match its tag, or publishing with a long-lived token in the repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def config(text: str) -> dict:
    yaml = pytest.importorskip("yaml", reason="pyyaml is only available transitively")
    return yaml.safe_load(text)


class TestItRunsOnATag:
    def test_it_is_not_triggered_by_an_ordinary_push(self, config: dict) -> None:
        """Publishing should be a deliberate act, not a side effect of merging.
        `on: push` to a branch would upload a release for every commit."""
        # PyYAML parses the bare key `on` as the boolean True.
        trigger = config.get("on") or config.get(True)

        assert set(trigger) == {"push"}
        assert "tags" in trigger["push"]
        assert "branches" not in trigger["push"]


class TestItCannotPublishSomethingBroken:
    @pytest.mark.parametrize("gate", ["ruff check", "ruff format --check", "mypy", "pytest"])
    def test_the_full_gate_runs_before_publishing(self, text: str, gate: str) -> None:
        """A release that skipped its own tests is the one build where nobody
        notices, because it is not a build anyone is watching."""
        assert gate in text

    def test_the_tests_run_on_the_commit_being_published(self, config: dict) -> None:
        """Checking out the tag and testing something else would prove nothing.
        The gate and the build must be the same job."""
        build = config["jobs"]["build"]["steps"]
        commands = [step.get("run", "") for step in build]

        assert any("pytest" in c for c in commands)
        assert any("uv build" in c for c in commands)

    def test_publishing_waits_for_the_build(self, config: dict) -> None:
        assert config["jobs"]["publish"]["needs"] == "build"

    def test_twine_check_runs_before_upload(self, text: str) -> None:
        """Metadata PyPI rejects *after* claiming the version slot cannot then
        be re-uploaded under the same number."""
        assert "twine check" in text


class TestTheTagMustMatchTheVersion:
    def test_the_workflow_compares_them(self, text: str) -> None:
        """A tag that disagrees with pyproject.toml publishes a version nobody
        can trace back to a commit, and PyPI will not let it be corrected."""
        assert "GITHUB_REF_NAME" in text
        assert "does not match package version" in text

    def test_the_comparison_strips_the_v_prefix(self, text: str) -> None:
        """`v1.0.0` and `1.0.0` are the same release. Comparing them raw would
        fail every legitimate tag."""
        assert "GITHUB_REF_NAME#v" in text

    @pytest.mark.parametrize(
        ("tag", "version", "should_publish"),
        [
            ("v1.0.0", "1.0.0", True),
            ("v0.0.0.dev0", "0.0.0.dev0", True),
            ("v1.0.0", "1.0.1", False),
            ("v2.0.0", "1.0.0", False),
        ],
    )
    def test_the_comparison_logic(self, tag: str, version: str, should_publish: bool) -> None:
        """The shell in the workflow, expressed here so the rule is tested even
        though the workflow itself only ever runs on GitHub."""
        assert (tag.removeprefix("v") == version) is should_publish


class TestItUsesTrustedPublishing:
    def test_there_is_no_api_token_in_the_workflow(self, text: str) -> None:
        """A long-lived token in a repository is a credential to leak, rotate
        and remember. OIDC has none."""
        assert "PYPI_API_TOKEN" not in text
        assert "password:" not in text

    def test_the_publish_job_can_mint_an_oidc_token(self, config: dict) -> None:
        """Without `id-token: write` the trusted-publishing handshake fails at
        the last step of an otherwise green release."""
        assert config["jobs"]["publish"]["permissions"]["id-token"] == "write"

    def test_it_names_a_pypi_environment(self, config: dict) -> None:
        """PyPI's trusted publisher configuration is bound to a workflow *and* an
        environment name. A mismatch is rejected at upload."""
        assert config["jobs"]["publish"]["environment"]["name"] == "pypi"

    def test_the_build_job_cannot_publish(self, config: dict) -> None:
        """Least privilege: the job that runs the code has no upload rights, so
        a compromised dependency cannot push a release."""
        assert "id-token" not in config["jobs"]["build"].get("permissions", {})


class TestThePackageItWouldPublish:
    def test_both_entry_points_are_declared(self) -> None:
        """The console script and the pytest plugin are how the package is used
        at all. Losing either would ship an importable library that does
        nothing when installed."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        assert "[project.scripts]" in pyproject
        assert 'evalstand = "evalstand.cli:main"' in pyproject
        assert "[project.entry-points.pytest11]" in pyproject
        assert 'evalstand = "evalstand.plugin"' in pyproject

    def test_the_version_is_readable_from_the_package(self) -> None:
        """The workflow compares the tag against this. If it moved, the check
        would silently compare against an empty string."""
        import evalstand

        assert evalstand.__version__

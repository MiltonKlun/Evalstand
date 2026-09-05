"""Every factual claim in `docs/scorers.md` (task 4.8).

Documentation that drifts from the code is worse than none: a user who follows
a wrong example loses time and then trust. So the numbers, signatures and
behaviours the doc states are asserted here, and the doc's own code blocks are
compiled.

This is deliberately not a prose-quality check. It asserts only the things that
can silently become false when the code changes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evalstand.models import Case, Score
from evalstand.scorers import (
    close_to,
    contains,
    exact,
    factuality,
    json_fields,
    judge,
    levenshtein,
    normalised_exact,
)

DOC = Path(__file__).resolve().parents[2] / "docs" / "scorers.md"
TEXT = DOC.read_text(encoding="utf-8")
CASE = Case(id="c1", input="q", expected="Paris")


class TestTheDocumentedNumbers:
    """Every figure the doc quotes, checked against the code."""

    def test_levenshtein_kitten_sitting(self) -> None:
        """The doc says 3 edits over 7 characters, 0.5714."""
        assert "0.5714" in TEXT
        result = levenshtein("kitten", "sitting")
        assert f"{result.value:.4f}" == "0.5714"
        assert result.metadata["distance"] == 3

    def test_the_json_fields_example(self) -> None:
        """The doc shows value 0.5 with born wrong and name matched."""
        result = json_fields()({"name": "Ada", "born": 1816}, {"name": "Ada", "born": 1815})
        assert result.value == pytest.approx(0.5)
        assert result.metadata["wrong"] == ["born"]
        assert result.metadata["matched"] == ["name"]

    def test_the_normalised_exact_examples(self) -> None:
        assert normalised_exact("  PARIS.  ", "Paris").value == 1.0
        assert normalised_exact("  PARIS.  ", "Paris").passed is True
        assert normalised_exact("-5", "5").value == 0.0
        assert normalised_exact("-5", "5").passed is False

    def test_the_scorer_count(self) -> None:
        """The doc says ten."""
        import evalstand.scorers as module

        built_in = {
            name
            for name in module.__all__
            if name not in {"Scorer", "scorer", "parse_number", "parse_object"}
        }
        assert len(built_in) == 10, f"the doc says ten; the package exports {len(built_in)}"
        assert "ships ten" in TEXT


class TestTheDocumentedBehaviours:
    @pytest.mark.parametrize(
        "output,expected",
        [("-5", "5"), ("$100", "100%"), ("C++", "C")],
    )
    def test_normalisation_does_not_fold_these(self, output: str, expected: str) -> None:
        """The doc names these three as answers that must not match."""
        assert normalised_exact(output, expected).passed is False

    def test_exact_is_strict_about_types_and_case(self) -> None:
        """The doc: "1 and "1" are different, and "Paris" and "paris" are
        different"."""
        assert exact(1, "1").passed is False
        assert exact("Paris", "paris").passed is False

    def test_contains_with_an_empty_expected_scores_zero(self) -> None:
        """The doc says 0.0, not 1.0, and explains why."""
        assert contains("anything at all", "").value == 0.0

    @pytest.mark.parametrize("output", ["42 or 43", "between 10 and 20", "5 + 5"])
    def test_close_to_refuses_ambiguous_output(self, output: str) -> None:
        """The doc names these three verbatim."""
        result = close_to()(output, 42)
        assert result.value == 0.0
        assert "no single number" in result.metadata["reason"]

    def test_close_to_reads_the_documented_shapes(self) -> None:
        """The doc's two examples of embedded numbers."""
        assert close_to(rel_tol=0.01)("The widget costs $19.99.", 20.00).passed is True
        assert close_to()("about 1,000", 1000).passed is True

    def test_close_to_defaults_to_an_exact_match(self) -> None:
        """The doc: "a forgotten tolerance means an exact match rather than
        silent slack"."""
        assert close_to()(42, 42).passed is True
        assert close_to()(42.001, 42).passed is False

    def test_a_relative_tolerance_is_useless_around_zero(self) -> None:
        """The doc's stated reason for abs_tol."""
        assert close_to(rel_tol=0.5)(0.001, 0.0).passed is False
        assert close_to(abs_tol=0.01)(0.001, 0.0).passed is True

    def test_json_fields_parses_a_fenced_block(self) -> None:
        result = json_fields()('```json\n{"a": 1}\n```', {"a": 1})
        assert result.value == 1.0

    def test_json_fields_reports_volunteered_fields_without_counting_them(self) -> None:
        result = json_fields()({"a": 1, "extra": 2}, {"a": 1})
        assert result.value == 1.0
        assert result.metadata["unexpected"] == ["extra"]

    def test_fuzzy_scorers_never_set_passed(self) -> None:
        """The doc's table marks both as **no**."""
        from evalstand.scorers import ratio

        assert levenshtein("a", "b").passed is None
        assert ratio("a", "b").passed is None

    def test_binary_scorers_do_set_passed(self) -> None:
        """The doc's table marks all four text scorers as yes."""
        from evalstand.scorers import regex_match

        assert exact("a", "a").passed is True
        assert normalised_exact("a", "a").passed is True
        assert contains("a", "a").passed is True
        assert regex_match("a")("a").passed is True


class TestTheReturnValueTable:
    """The doc's table of what each return type means."""

    @pytest.mark.anyio
    async def test_true_gives_value_one_and_passed_true(self) -> None:
        from evalstand.scorers.base import call_scorer

        result = await call_scorer(lambda output: True, "o", "e", CASE)
        assert (result.value, result.passed) == (1.0, True)

    @pytest.mark.anyio
    async def test_false_gives_value_zero_and_passed_false(self) -> None:
        from evalstand.scorers.base import call_scorer

        result = await call_scorer(lambda output: False, "o", "e", CASE)
        assert (result.value, result.passed) == (0.0, False)

    @pytest.mark.anyio
    async def test_a_float_leaves_passed_unset(self) -> None:
        from evalstand.scorers.base import call_scorer

        result = await call_scorer(lambda output: 0.62, "o", "e", CASE)
        assert result.value == pytest.approx(0.62)
        assert result.passed is None

    @pytest.mark.anyio
    async def test_a_score_is_used_exactly_as_given(self) -> None:
        from evalstand.scorers.base import call_scorer

        built = Score(scorer_name="mine", value=0.3, metadata={"k": "v"})
        assert await call_scorer(lambda output: built, "o", "e", CASE) is built


class TestTheJudgeCaveat:
    def test_the_doc_states_the_scorers_are_unvalidated(self) -> None:
        """Task 4.6's note requires this to be said plainly, so its absence is
        a documentation bug worth failing the suite over."""
        assert "unvalidated" in TEXT.lower()
        assert "not been calibrated against human labels" in TEXT.lower()

    @pytest.mark.anyio
    async def test_every_judge_score_carries_the_caveat(self) -> None:
        """The doc promises `unvalidated: True` in the metadata."""

        async def responder(**_: Any) -> MagicMock:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "A"
            response.usage.prompt_tokens = 1
            response.usage.completion_tokens = 1
            response.usage.total_tokens = 2
            response.model = "gpt-4o-mini"
            return response

        with (
            patch("evalstand.llm.litellm.acompletion", side_effect=responder),
            patch("evalstand.llm.litellm.completion_cost", return_value=0.0),
        ):
            result = await judge(rubric="r", choices={"A": 1.0})("out", "exp", CASE)

        assert result.metadata["unvalidated"] is True

    def test_factuality_scores_consistency_not_equality(self) -> None:
        """The doc: an answer saying less or more still scores 1.0, and only a
        disagreement scores 0."""
        from evalstand.scorers.llm import FACTUALITY_CHOICES

        assert FACTUALITY_CHOICES["A"] == 1.0, "a subset"
        assert FACTUALITY_CHOICES["B"] == 1.0, "a superset"
        assert FACTUALITY_CHOICES["D"] == 0.0, "a disagreement"


class TestTheDocumentedSignatures:
    """A doc showing a call that no longer type-checks is worse than silence."""

    def test_the_factory_signatures_match(self) -> None:
        import inspect

        assert "rel_tol" in inspect.signature(close_to).parameters
        assert "abs_tol" in inspect.signature(close_to).parameters
        assert "require_all" in inspect.signature(json_fields).parameters
        assert "include_expected" in inspect.signature(judge).parameters
        assert "rubric" in inspect.signature(judge).parameters
        assert "choices" in inspect.signature(judge).parameters

    def test_the_documented_factory_calls_all_work(self) -> None:
        """Every factory invocation the doc shows, executed."""
        close_to(rel_tol=0.01)
        close_to(abs_tol=0.5)
        close_to()
        json_fields()
        json_fields(require_all=True)
        factuality()
        judge(
            rubric="Does the answer use a professional tone?",
            choices={"A": 1.0, "B": 0.5, "C": 0.0},
            include_expected=False,
        )


class TestTheDocsCodeBlocksAreValidPython:
    def test_every_python_block_compiles(self) -> None:
        """Catches a stale example that no longer parses. Blocks are compiled,
        not executed: several reference an `answer` task that only exists in
        the reader's own file."""
        blocks = re.findall(r"```python\n(.*?)```", TEXT, re.DOTALL)
        assert blocks, "the doc has no python examples, which cannot be right"

        for index, block in enumerate(blocks):
            try:
                compile(block, f"<docs/scorers.md block {index}>", "exec")
            except SyntaxError as exc:  # pragma: no cover - only on a bad doc
                pytest.fail(f"block {index} does not parse: {exc}\n{block}")

    def test_the_imports_the_doc_shows_all_resolve(self) -> None:
        """A user copying an import line must not hit an ImportError."""
        for line in re.findall(r"^from evalstand[.\w]* import .*$", TEXT, re.MULTILINE):
            # The input is this repo's own doc, not user data.
            exec(line, {})


class TestEveryScorerIsDocumented:
    """The doc must name every scorer the package exports, and no others.

    Catches drift in the direction the per-claim tests cannot: a scorer added
    to the library and never written up, or one renamed in code while the doc
    keeps advertising the old name.
    """

    def test_the_doc_names_every_exported_scorer(self) -> None:
        import evalstand.scorers as module

        documented = {
            name
            for name in module.__all__
            if name not in {"Scorer", "parse_number", "parse_object"}
        }
        missing = sorted(name for name in documented if name not in TEXT)

        assert not missing, f"docs/scorers.md does not mention: {missing}"

    def test_the_doc_advertises_nothing_that_no_longer_exists(self) -> None:
        """A doc naming a scorer the package dropped sends users to an
        ImportError, which is worse than an undocumented feature."""
        import evalstand.scorers as module

        # Scoped to the scorer tables and to `scorers=[...]` lines: those are
        # where the doc tells a user what to call. Prose mentioning `float(True)`
        # is not advertising a scorer, and treating it as one would make this
        # test fail on writing rather than on drift.
        advertised = set(re.findall(r"^\| `(\w+)\(", TEXT, re.MULTILINE)) | set(
            re.findall(r"scorers=\[(\w+)", TEXT)
        )
        known = set(module.__all__)
        unknown = sorted(name for name in advertised if name not in known)

        assert not unknown, f"docs/scorers.md advertises names that do not exist: {unknown}"

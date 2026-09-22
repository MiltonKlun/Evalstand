"""The showcase example (task 5.6).

The acceptance is unusual for this project: it is about **reproducibility**
rather than about what a figure claims. `python generate.py --seed 42` must
produce byte-identical PDFs, or the committed baseline in 5.7 describes a corpus
that no longer exists.

The trap that makes it non-trivial: a PDF embeds a creation timestamp by
default, so two runs of the same code differ. Found by building the same
document twice a second apart and comparing digests — not by reading reportlab's
documentation and hoping.

These tests do not call a model. They check the corpus, the ground truth, and
the wiring; the eval itself is exercised against a mocked provider.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "pdf_extraction"
GROUND_TRUTH = EXAMPLE / "ground_truth.json"
CHECKSUMS = EXAMPLE / "checksums.txt"

pytest.importorskip("reportlab", reason="the examples extra is not installed")
# No `faker` guard. The corpus was moved to `random.Random` in 16eefc4 and the
# dependency dropped from the `examples` extra — but this line outlived it and
# skipped the whole file, silently, on every platform, because `importorskip`
# on a package nothing installs any more is an unconditional skip.


def _load(name: str) -> Any:
    """Import a module from the example directory by path, once.

    Cached deliberately. `extraction_eval` calls `evaluate()` at import, and a
    second execution registers the same eval name twice — which the registry
    correctly rejects. The example is not a package and is not on `sys.path`,
    so importing it by location is what a user's own eval file effectively
    does too.
    """
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def truth() -> list[dict[str, Any]]:
    if not GROUND_TRUTH.exists():
        pytest.skip("run `python generate.py --seed 42` first")
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


class TestTheCorpusIsReproducible:
    """Task 5.6's acceptance."""

    def test_regenerating_at_the_same_seed_reproduces_the_ground_truth(self) -> None:
        """The claim the whole example rests on — and the one it can keep.

        This asserted byte-identical PDFs until the first CI run, where all 30
        checksums differed on Linux while the data inside was unchanged.
        reportlab's output depends on its build, so byte-identity was only ever
        a per-platform property; what the eval is scored against is the ground
        truth, and that regenerates identically anywhere.

        Run as a subprocess because that is what a user does, and because a
        fresh process is the only way to catch state that survives within one.
        """
        result = subprocess.run(
            [sys.executable, "generate.py", "--seed", "42", "--check"],
            cwd=EXAMPLE,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "ground truth matches" in result.stdout

    def test_a_byte_difference_alone_is_not_a_failure(self, tmp_path: Path) -> None:
        """The distinction the CI failure taught: a PDF that differs
        byte-for-byte while carrying identical data is a note, not a fault.

        Driven by corrupting a committed checksum, which is exactly what Linux
        looks like to a corpus whose checksums were recorded on Windows — same
        data, different bytes. Asserting on the source text instead would only
        check that a comment still says the right thing.
        """
        checksums = EXAMPLE / "checksums.txt"
        original = checksums.read_text(encoding="utf-8")
        lines = original.splitlines()
        lines[0] = "0" * 64 + lines[0][64:]

        checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "generate.py", "--seed", "42", "--check"],
                cwd=EXAMPLE,
                capture_output=True,
                text=True,
                timeout=300,
            )
        finally:
            checksums.write_text(original, encoding="utf-8")

        assert result.returncode == 0, "a byte difference must not fail the check"
        assert "ground truth matches" in result.stdout
        assert "differ byte-for-byte" in result.stdout, "but it must still be reported"

    def test_a_different_seed_produces_a_different_corpus(self) -> None:
        """The other half. A `--check` that passed for every seed would be
        checking nothing, and this is exactly the test that would catch it."""
        result = subprocess.run(
            [sys.executable, "generate.py", "--seed", "43", "--check"],
            cwd=EXAMPLE,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode != 0
        assert "differ from the committed truth" in result.stdout

        # Leave the committed corpus in place for the tests that follow.
        subprocess.run(
            [sys.executable, "generate.py", "--seed", "42"],
            cwd=EXAMPLE,
            capture_output=True,
            timeout=300,
        )

    def test_the_pdfs_carry_a_fixed_timestamp(self) -> None:
        """The specific trap. A PDF embeds a CreationDate by default, so two
        runs of identical code produce different bytes and a "reproducible"
        corpus drifts silently.

        `invariant=1` does not remove the field — it pins it to a fixed epoch,
        which is what makes the bytes identical. Asserted on the value rather
        than the absence, because the first version of this test asserted the
        wrong thing and passed only by accident of wording.
        """
        generate = _load("generate")
        pdfs = sorted(generate.INVOICES.glob("*.pdf"))
        if not pdfs:
            pytest.skip("run `python generate.py --seed 42` first")

        dates = {
            match.group(1)
            for path in pdfs
            if (match := re.search(rb"/CreationDate \(([^)]*)\)", path.read_bytes()))
        }

        assert len(dates) == 1, f"the corpus carries {len(dates)} different timestamps"
        assert b"20000101" in dates.pop(), "the timestamp is not the fixed epoch"

    def test_every_committed_checksum_matches_its_file(self) -> None:
        generate = _load("generate")
        if not CHECKSUMS.exists():
            pytest.skip("run `python generate.py --seed 42` first")

        expected = {
            line.split("  ", 1)[1]: line.split("  ", 1)[0]
            for line in CHECKSUMS.read_text(encoding="utf-8").strip().splitlines()
        }
        actual = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(generate.INVOICES.glob("*.pdf"))
        }

        assert actual == expected


class TestTheGroundTruthIsTrueByConstruction:
    """Nothing here is a model's opinion. The numbers were chosen first and
    then rendered, so they can be checked against each other."""

    def test_every_total_matches_its_line_items(self, truth: list[dict[str, Any]]) -> None:
        """A ground truth that disagrees with itself by a cent would make every
        extraction look wrong. Decimal throughout is what prevents it: 0.1 + 0.2
        is not 0.3 in binary floating point."""
        for entry in truth:
            summed = sum((Decimal(item["amount"]) for item in entry["line_items"]), Decimal("0"))
            assert str(summed) == entry["total"], entry["file"]

    def test_every_line_amount_is_quantity_times_unit_price(
        self, truth: list[dict[str, Any]]
    ) -> None:
        for entry in truth:
            for item in entry["line_items"]:
                expected = Decimal(item["unit_price"]) * item["quantity"]
                assert expected == Decimal(item["amount"]), entry["file"]

    def test_the_corpus_is_the_documented_size(self, truth: list[dict[str, Any]]) -> None:
        assert len(truth) == 30

    def test_every_entry_names_a_file_that_exists(self, truth: list[dict[str, Any]]) -> None:
        generate = _load("generate")
        missing = [e["file"] for e in truth if not (generate.INVOICES / e["file"]).exists()]
        assert missing == []


class TestTheDeliberateVariations:
    """The corpus exists to break extraction in specific ways. A corpus of
    thirty easy documents would measure nothing."""

    def test_some_invoices_have_no_due_date(self, truth: list[dict[str, Any]]) -> None:
        """The honest answer is null, and a model that invents a plausible date
        is wrong in a way that looks right — which is the failure mode most
        worth catching."""
        assert [e["file"] for e in truth if e["due_date"] is None] == [
            "invoice_07.pdf",
            "invoice_15.pdf",
            "invoice_22.pdf",
        ]

    def test_two_currencies_appear(self, truth: list[dict[str, Any]]) -> None:
        assert {e["currency"] for e in truth} == {"USD", "EUR"}

    def test_some_invoices_run_to_a_second_page(self, truth: list[dict[str, Any]]) -> None:
        """The total is on the last page, so a model that stops reading early
        reports items that do not sum to it."""
        import pypdfium2

        generate = _load("generate")
        multi = [
            entry["file"]
            for entry in truth
            if len(pypdfium2.PdfDocument(generate.INVOICES / entry["file"])) > 1
        ]

        assert multi == ["invoice_03.pdf", "invoice_11.pdf", "invoice_24.pdf"]

    def test_the_ambiguous_dates_are_rendered_ambiguously(
        self, truth: list[dict[str, Any]]
    ) -> None:
        """`04/03/2026` is printed; the ground truth records which reading was
        meant. Without that the model's answer could not be called wrong."""
        import pypdfium2

        generate = _load("generate")
        entry = truth[9]
        document = pypdfium2.PdfDocument(generate.INVOICES / entry["file"])
        text = "".join(page.get_textpage().get_text_range() for page in document)

        assert entry["invoice_date"] not in text, "the ISO date was printed after all"
        year, month, day = entry["invoice_date"].split("-")
        assert f"{day}/{month}/{year}" in text


class TestTheEvalIsWiredCorrectly:
    """No model is called. These check the parts a missing corpus or a renamed
    field would break."""

    def test_it_declares_one_case_per_invoice(self, truth: list[dict[str, Any]]) -> None:
        module = _load("extraction_eval")
        cases = module.load_cases()

        assert len(cases) == len(truth)
        assert cases[0].id == "invoice_00"

    def test_the_expected_value_is_the_generators_own_record(
        self, truth: list[dict[str, Any]]
    ) -> None:
        module = _load("extraction_eval")
        case = module.load_cases()[0]

        assert case.expected["invoice_number"] == truth[0]["invoice_number"]
        assert case.expected["total"] == truth[0]["total"]

    def test_line_items_are_carried_for_the_judge(self) -> None:
        """Judged separately, because comparing them field by field would score
        a reworded description the same as a missing item."""
        module = _load("extraction_eval")
        case = module.load_cases()[0]

        assert case.metadata["line_items"]
        assert "line_items" not in case.expected

    def test_a_missing_corpus_raises_rather_than_running_on_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An eval reporting 0 cases looks like a pass, which is the worst way
        for a missing corpus to announce itself."""
        module = _load("extraction_eval")
        monkeypatch.setattr(module, "GROUND_TRUTH", tmp_path / "nothing.json")

        with pytest.raises(FileNotFoundError, match=r"generate\.py"):
            module.load_cases()

    def test_a_missing_pdf_raises_and_names_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        module = _load("extraction_eval")
        monkeypatch.setattr(module, "INVOICES", tmp_path)

        with pytest.raises(FileNotFoundError, match=r"invoice_00\.pdf"):
            module.load_cases()

    def test_reading_a_pdf_is_serialised(self) -> None:
        """pypdfium2 is not thread-safe, and evalstand runs cases concurrently.
        Without the lock the interpreter dies with an access violation rather
        than raising something the runner could record — verified by watching
        it happen."""
        module = _load("extraction_eval")

        assert module._PDF_LOCK is not None

        import concurrent.futures

        names = [f"invoice_{index:02d}.pdf" for index in range(10)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            texts = list(pool.map(module.read_pdf, names))

        assert len(texts) == len(names)
        assert all(text.strip() for text in texts), "a concurrent read returned nothing"
        # Each invoice's own number must appear in its own text: a race that
        # returned another document's page would pass a length check.
        assert all(f"INV-{2026000 + index:07d}" in texts[index] for index in range(10))


class TestTheBaselineGenerator:
    """`baseline.py` turns a stored run into BASELINE.md (task 5.7).

    Generated rather than typed: a hand-written figure drifts from the code the
    moment either changes, and a baseline that quietly disagrees with the tool
    is worse than none — it is the number people quote.

    These tests build Runs directly. The real baseline needs an API key; the
    machinery that turns a run into a document does not.
    """

    @staticmethod
    def _run(results: list[Any]) -> Any:
        from datetime import UTC, datetime

        from evalstand.models import Run, RunStatus

        return Run(
            id="run-baseline",
            batch_id="b1",
            name="invoice-extraction",
            filepath="extraction_eval.py",
            status=RunStatus.COMPLETED,
            started_at=datetime.now(UTC),
            results=results,
        )

    @staticmethod
    def _result(case_id: str, matched: list[str], wrong: list[str], **kwargs: Any) -> Any:
        from evalstand.models import Result, Score, Trace

        scores = [
            Score(
                scorer_name="json_fields",
                value=len(matched) / max(len(matched) + len(wrong), 1),
                metadata={"matched": matched, "wrong": wrong, "missing": []},
            )
        ]
        if kwargs.get("judged"):
            scores.append(
                Score(scorer_name="line_items", value=1.0, metadata={"unvalidated": True})
            )

        return Result(
            id=f"r-{case_id}",
            case_id=case_id,
            scores=scores,
            traces=[
                Trace(
                    id=f"t-{case_id}",
                    name="model call",
                    model=kwargs.get("model", "gpt-4o-mini"),
                    duration_ms=10,
                    cost_usd=kwargs.get("cost", 0.0003),
                )
            ],
        )

    def test_per_field_accuracy_counts_each_field(self) -> None:
        baseline = _load("baseline")
        run = self._run(
            [
                self._result("a", matched=["total", "vendor"], wrong=[]),
                self._result("b", matched=["vendor"], wrong=["total"]),
            ]
        )

        assert baseline.field_accuracy(run) == {"total": (1, 2), "vendor": (2, 2)}

    def test_a_missing_field_counts_as_attempted_and_wrong(self) -> None:
        """The model was asked for it and did not supply it. A different
        failure from supplying the wrong value, but not a better one."""
        from evalstand.models import Result, Score

        baseline = _load("baseline")
        run = self._run(
            [
                Result(
                    id="r",
                    case_id="a",
                    scores=[
                        Score(
                            scorer_name="json_fields",
                            value=0.5,
                            metadata={"matched": ["total"], "wrong": [], "missing": ["vendor"]},
                        )
                    ],
                )
            ]
        )

        assert baseline.field_accuracy(run) == {"total": (1, 1), "vendor": (0, 1)}

    def test_an_errored_score_does_not_count_against_a_field(self) -> None:
        """A scorer that broke is not evidence the model got a field wrong —
        the same rule that keeps errored scores out of every mean."""
        from evalstand.models import Result, Score

        baseline = _load("baseline")
        run = self._run(
            [
                Result(
                    id="r",
                    case_id="a",
                    scores=[Score.from_error("json_fields", "the output was not JSON")],
                )
            ]
        )

        assert baseline.field_accuracy(run) == {}

    def test_failure_modes_group_by_document(self) -> None:
        """A document that got three fields wrong is usually one problem, and
        listing them apart hides that."""
        baseline = _load("baseline")
        run = self._run(
            [
                self._result("good", matched=["total"], wrong=[]),
                self._result("bad", matched=[], wrong=["total", "vendor"]),
            ]
        )

        modes = baseline.failure_modes(run)

        assert [case_id for case_id, _ in modes] == ["bad"]
        assert "total, vendor" in modes[0][1][0]

    def test_the_rendered_document_carries_the_three_required_sections(self) -> None:
        """Task 5.7 asks for per-field accuracy, cost per document, and
        observed failure modes."""
        baseline = _load("baseline")
        text = baseline.render(
            self._run([self._result("a", matched=["total"], wrong=["vendor"])]),
            "gpt-4o-mini",
        )

        assert "Per-field accuracy" in text
        assert "cost per document" in text
        assert "Observed failure modes" in text

    def test_it_states_that_the_judge_is_unvalidated(self) -> None:
        """The baseline is where a number gets quoted, so the caveat belongs
        beside it rather than only in the scorer documentation."""
        baseline = _load("baseline")
        run = self._run([self._result("a", ["total"], [], judged=True)])
        text = baseline.render(run)

        assert "unvalidated" in text
        assert "not scored" not in text

    def test_a_run_without_a_judge_says_line_items_were_not_scored(self) -> None:
        """The dangerous omission. A span-selection run cannot produce line
        items, and a table with no line-item row reads as a document that got
        everything right when the truth is that nobody asked.

        Previously the judge caveat was printed unconditionally, so such a run
        would have carried a warning about a scorer it never used — and said
        nothing about the scorer it was missing.
        """
        baseline = _load("baseline")
        text = baseline.render(self._run([self._result("a", ["total"], [])]))

        assert "Line items were not scored" in text
        assert "unvalidated" not in text, "it caveats a judge the run never used"

    def test_the_model_comes_from_the_run_not_a_default(self) -> None:
        """`--model` used to default to `gpt-4o-mini`, so a baseline from any
        other eval would name a model it never called."""
        baseline = _load("baseline")
        run = self._run([self._result("a", ["total"], [], model="jev-1.13.0")])
        text = baseline.render(run)

        assert "`jev-1.13.0`" in text
        assert "gpt-4o-mini" not in text

    def test_it_names_the_eval_that_produced_it(self) -> None:
        """Hard-coded as `extraction_eval.py` before. Two evals now share this
        corpus, and a baseline that named the wrong one sends a reader to the
        wrong method."""
        from evalstand.models import Run, RunStatus

        baseline = _load("baseline")
        run = Run(
            id="run-x",
            batch_id="b",
            name="invoice-extraction-jev",
            filepath="examples/pdf_extraction/jev_extraction_eval.py",
            status=RunStatus.COMPLETED,
            results=[self._result("a", ["total"], [])],
        )

        assert "`jev_extraction_eval.py`" in baseline.render(run)

    def test_it_refuses_to_call_a_number_good_or_bad(self) -> None:
        """The same restraint `compare` observes. A baseline that graded itself
        would assert more than one run can support."""
        baseline = _load("baseline")
        text = baseline.render(self._run([self._result("a", [], ["total"])]), "gpt-4o-mini").lower()

        for word in ("regress", "improve", "excellent", "poor", "acceptable"):
            assert word not in text, f"the baseline editorialised: {word}"

        # "target" appears, but only in the disclaimer that says this is not
        # one. Asserted rather than banned, because the sentence carrying it is
        # the whole point.
        assert "not a target" in text

    def test_a_run_with_no_priced_calls_is_not_reported_as_free(self) -> None:
        """`$0.0000` claims a run cost nothing. `-` says nobody knows."""
        from evalstand.models import Result

        baseline = _load("baseline")
        run = self._run([Result(id="r", case_id="a")])
        text = baseline.render(run, "gpt-4o-mini")

        assert "$0.0000" not in text

    def test_a_mocked_run_is_labelled_as_one(self) -> None:
        """A baseline generated from fixtures and committed as though it were a
        measurement would be the most misleading document in the repository.
        Every call costing exactly the same is the signal: real completions
        vary in length and therefore in price."""
        baseline = _load("baseline")
        run = self._run(
            [self._result(f"case{index}", ["total"], [], cost=0.0003) for index in range(5)]
        )

        assert "mocked provider" in baseline.render(run, "gpt-4o-mini")

    def test_a_real_looking_run_is_not_labelled(self) -> None:
        """The other half: varying costs must not trip the warning, or it
        becomes noise on every real baseline."""
        baseline = _load("baseline")
        run = self._run(
            [
                self._result(f"case{index}", ["total"], [], cost=0.0003 + index * 0.00001)
                for index in range(5)
            ]
        )

        assert "mocked provider" not in baseline.render(run, "gpt-4o-mini")


class TestTheCommittedBaseline:
    """BASELINE.md is a placeholder until a real run exists.

    Tested because the failure mode is specific and quiet: someone fills it in
    with plausible numbers, or regenerates it from a mocked run, and the file
    becomes a measurement nobody made.
    """

    def test_it_does_not_state_numbers_it_has_not_measured(self) -> None:
        text = (EXAMPLE / "BASELINE.md").read_text(encoding="utf-8")

        if "Not yet recorded" not in text:
            return

        # Scoped to table cells. A figure inside a table reads as a result
        # whatever the surrounding prose says; the same characters in a
        # sentence explaining why `$0.0000` is never printed do not.
        cells = [line for line in text.splitlines() if line.strip().startswith("|")]
        joined = "\n".join(cells)

        assert not re.search(r"\d+%", joined), "the placeholder quotes a rate"
        assert not re.search(r"\$\d+\.\d{2}", joined), "the placeholder quotes a cost"

    def test_a_real_baseline_names_the_run_it_came_from(self) -> None:
        """So a reader can open it with `evalstand show` and check."""
        text = (EXAMPLE / "BASELINE.md").read_text(encoding="utf-8")

        if "Not yet recorded" not in text:
            assert re.search(r"run-[0-9a-f]{12}", text), "no run id to trace it back to"

    def test_its_caveat_matches_what_it_reports(self) -> None:
        """A baseline is where a number gets quoted, so the caveat belongs
        beside it. Which caveat depends on what the file reports: a
        `line_items` row is an unvalidated judge's opinion, and its *absence*
        means line items were never measured. Either omission misleads."""
        text = (EXAMPLE / "BASELINE.md").read_text(encoding="utf-8")

        if "Not yet recorded" in text or "| `line_items` |" in text:
            assert "unvalidated" in text
        else:
            assert "Line items were not scored" in text

    def test_it_says_a_difference_is_not_a_verdict(self) -> None:
        """The same restraint the tool itself observes."""
        text = (EXAMPLE / "BASELINE.md").read_text(encoding="utf-8")

        assert "significance testing" in text


class TestJevSpanSelection:
    """`jev_extract.py`: regex over-finds, Jev selects, code normalises.

    Everything here runs offline. The model's half needs a key; the half that
    decides what the model is *allowed* to answer, and what its answer becomes,
    does not — and that half is where this method's first run lost three fields.
    """

    def test_a_ddmm_date_becomes_iso_with_the_day_first(self) -> None:
        """The bug the first live run found. The document prints `02/10/2018`
        and means 2 October; Jev picked that span correctly every time, and the
        code had not converted it. Day-first is a fact about this corpus, and
        the jaggedness guidance is that it belongs in code."""
        jev = _load("jev_extract")

        assert jev.normalise("invoice_date", "02/10/2018") == "2018-10-02"
        assert jev.normalise("due_date", "4/11/2016") == "2016-11-04"

    def test_every_ambiguous_date_in_the_corpus_normalises_to_its_truth(
        self, truth: list[dict[str, Any]]
    ) -> None:
        """Asserted against the ground truth rather than a hand-picked
        example, so a wrong day/month assumption cannot pass on a date like
        `01/01/2008` where the two readings agree."""
        jev = _load("jev_extract")
        read_pdf = _load("extraction_eval").read_pdf

        checked = 0
        for entry in truth:
            text = read_pdf(entry["file"])
            for span in jev.find(jev.DATE_RE, text):
                if "/" not in span:
                    continue
                assert jev.normalise("invoice_date", span) in (
                    entry["invoice_date"],
                    entry["due_date"],
                ), f"{entry['file']}: {span} normalised to a date the truth does not hold"
                checked += 1

        assert checked >= 3, "the corpus's ambiguous dates were not found"

    def test_none_becomes_null_not_a_string(self) -> None:
        """Three invoices have no due date. `"none"` compared against a null
        ground truth would fail a correct answer."""
        jev = _load("jev_extract")

        assert jev.normalise("due_date", jev.NONE) is None

    def test_a_total_loses_its_symbol_and_separators(self) -> None:
        jev = _load("jev_extract")

        assert jev.normalise("total", "$1,961.66") == "1961.66"

    def test_every_question_can_answer_none(self) -> None:
        """Without it, a field the document lacks forces a pick among values it
        has — a real date in the wrong field, wrong in a way that looks right."""
        jev = _load("jev_extract")

        assert jev.NONE in jev.options(["2018-10-02"], "a date")
        assert jev.NONE in jev.options([], "a date")

    def test_find_dedupes_in_document_order(self) -> None:
        jev = _load("jev_extract")

        found = jev.find(jev.MONEY_RE, "a $5.00 b $3.00 c $5.00")

        assert found == ["$5.00", "$3.00"]

    def test_the_regexes_cover_every_true_value_in_the_corpus(
        self, truth: list[dict[str, Any]]
    ) -> None:
        """The method's ceiling. Jev can only select a span the regex found, so
        a true value the regex misses is one no model could return — and the
        failure would look like the model's."""
        jev = _load("jev_extract")
        read_pdf = _load("extraction_eval").read_pdf

        for entry in truth:
            text = read_pdf(entry["file"])
            totals = [jev.normalise("total", s) for s in jev.find(jev.MONEY_RE, text)]
            numbers = jev.find(jev.INVOICE_NO_RE, text)
            vendors = jev.vendor_candidates(text)

            assert entry["total"] in totals, f"{entry['file']}: total not a candidate"
            assert entry["invoice_number"] in numbers, f"{entry['file']}: number not found"
            assert entry["vendor_name"] in vendors, f"{entry['file']}: vendor not a candidate"

    def test_importing_it_needs_no_sdk(self) -> None:
        """The SDK is imported inside the functions that call the service, so
        these helpers — and this file — run on a machine without it."""
        source = (EXAMPLE / "jev_extract.py").read_text(encoding="utf-8")
        top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

        assert not any("typesafe_sdk" in line for line in top_level)


class TestTheJevEval:
    def test_collecting_it_needs_no_key(self) -> None:
        """`--collect-only` must never spend money or need credentials. The
        client is built on first use, not at import."""
        import os
        import subprocess
        import sys

        env = {key: value for key, value in os.environ.items() if key != "TYPESAFE_API_KEY"}
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(EXAMPLE / "jev_extraction_eval.py"),
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "30 tests collected" in proc.stdout

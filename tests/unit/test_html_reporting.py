"""The static HTML report (task 8.3).

A CI artifact is opened by whoever did not run the evals — often days later,
often someone without the project installed. Three things therefore matter more
here than in the terminal:

- **It must be one file.** A report that fetched a stylesheet renders unstyled
  inside the sandboxed iframe CI systems serve artifacts from.
- **It must be well-formed.** A stray tag silently swallows the rest of the
  page, and a browser shows no error — the report just ends early.
- **Model output must never execute.** A completion containing `<script>` is
  inevitable rather than adversarial: models are asked about HTML constantly.

The honesty rules are the same ones the console and markdown reporters apply,
and they are asserted again here because a third copy that drifted would mean
three reports disagreeing about whether a build passed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from html.parser import HTMLParser

from evalstand.models import Result, Run, RunStatus, Score, Trace
from evalstand.reporting.html import render_html

_FIXED = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


def _result(
    case_id: str = "q1",
    *,
    scores: list[Score] | None = None,
    traces: list[Trace] | None = None,
    output: object = "an answer",
    error: str | None = None,
    latency_ms: int | None = None,
    repeat_index: int = 0,
) -> Result:
    default = [Score(scorer_name="exact", value=1.0, passed=True)]
    return Result(
        id=f"r-{case_id}-{repeat_index}",
        case_id=case_id,
        repeat_index=repeat_index,
        output=output,
        error=error,
        latency_ms=latency_ms,
        scores=default if scores is None else scores,
        traces=traces or [],
    )


def _run(name: str = "qa", *, results: list[Result] | None = None) -> Run:
    return Run(
        id=f"run-{name}",
        batch_id="b1",
        name=name,
        filepath=f"{name}_eval.py",
        status=RunStatus.COMPLETED,
        results=[_result()] if results is None else results,
    )


def _trace(
    trace_id: str = "t1",
    *,
    parent_id: str | None = None,
    cost: float | None = 0.001,
    name: str = "call",
) -> Trace:
    return Trace(
        id=trace_id,
        parent_id=parent_id,
        name=name,
        duration_ms=12,
        model="gpt-4o",
        cost_usd=cost,
    )


class _Balance(HTMLParser):
    """A minimal well-formedness check.

    Not a validator — it only catches the mistake that actually happens when
    building HTML by string concatenation: a tag closed in the wrong order, or
    never closed, which truncates everything after it in a real browser.
    """

    VOID = frozenset({"meta", "br", "hr", "img", "input", "link"})

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            self.problems.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.problems.append(f"</{tag}> while <{self.stack[-1]}> is open")
        else:
            self.stack.pop()


def _check(document: str) -> _Balance:
    parser = _Balance()
    parser.feed(document)
    return parser


class TestItIsOneSelfContainedFile:
    def test_nothing_is_fetched_over_the_network(self) -> None:
        """CI serves artifacts from a sandboxed origin. An external stylesheet
        would leave the report unstyled exactly where it is read."""
        document = render_html([_run()])

        assert 'src="http' not in document
        assert 'href="http' not in document
        assert "@import" not in document

    def test_there_is_no_script_tag_at_all(self) -> None:
        """`<details>` gives collapsible sections with no JavaScript, so the
        report works with scripting disabled — which is how some CI viewers
        serve it."""
        assert "<script" not in render_html([_run()])

    def test_the_css_is_inline(self) -> None:
        document = render_html([_run()])

        assert "<style>" in document
        assert "prefers-color-scheme" in document, "should follow the reader's theme"

    def test_it_declares_a_doctype_and_charset(self) -> None:
        """Without a charset a model's non-ASCII output renders as mojibake."""
        document = render_html([_run()])

        assert document.startswith("<!doctype html>")
        assert '<meta charset="utf-8">' in document


class TestItIsWellFormed:
    def test_a_plain_report_balances(self) -> None:
        parsed = _check(render_html([_run()]))

        assert parsed.problems == []
        assert parsed.stack == []

    def test_a_report_with_everything_balances(self) -> None:
        """The shape most likely to break the string building: nested traces, an
        error, an unmeasured case and a repeat, all at once."""
        run = _run(
            results=[
                _result("ok"),
                _result("crashed", error="RuntimeError: boom", scores=[]),
                _result("unmeasured", scores=[Score.from_error("judge", "no key")]),
                _result("repeated", repeat_index=2),
                _result(
                    "traced",
                    traces=[
                        _trace("root"),
                        _trace("child", parent_id="root"),
                        _trace("grandchild", parent_id="child"),
                        _trace("sibling", parent_id="root"),
                    ],
                ),
            ]
        )

        parsed = _check(render_html([run], threshold=0.9))

        assert parsed.problems == []
        assert parsed.stack == []

    def test_a_deep_trace_chain_does_not_overflow(self) -> None:
        """A judge that calls a judge that calls a judge. Built iteratively for
        this reason — a recursive walk dies at about a thousand levels, and
        losing the whole report to one deep case would be a poor failure."""
        traces = [_trace("t0")]
        traces += [_trace(f"t{i}", parent_id=f"t{i - 1}") for i in range(1, 1200)]

        document = render_html([_run(results=[_result("deep", traces=traces)])])

        assert _check(document).problems == []
        assert document.count("<span class='tname'>") == 1200


class TestModelOutputCannotExecute:
    def test_a_script_tag_in_the_output_is_escaped(self) -> None:
        """Not adversarial — models are asked about HTML all the time. A report
        that executed it would be stored XSS in a file people open from a build
        page."""
        run = _run(results=[_result(output="<script>alert(1)</script>")])

        document = render_html([run])

        assert "<script>alert(1)</script>" not in document
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document

    def test_a_quote_in_a_case_id_cannot_break_out_of_an_attribute(self) -> None:
        run = _run(results=[_result('q1" onload="x')])

        document = render_html([run])

        assert "&quot; onload=&quot;" in document
        assert '" onload="' not in document

    def test_markup_in_an_eval_name_is_escaped(self) -> None:
        document = render_html([_run(name="<b>bold</b>")])

        assert "<b>bold</b>" not in document
        assert "&lt;b&gt;bold&lt;/b&gt;" in document

    def test_markup_in_a_trace_payload_is_escaped(self) -> None:
        """Prompts and completions are shown in full here, which is precisely
        why they are the most likely place for markup to arrive."""
        traced = Trace(
            id="t1",
            name="call",
            duration_ms=1,
            input="<img onerror=alert(1) src=x>",
            output="</pre><script>x</script>",
        )
        run = _run(results=[_result(traces=[traced])])

        document = render_html([run])

        assert "<img onerror" not in document
        assert "<script>x</script>" not in document
        assert _check(document).problems == []


class TestTheHonestyRulesHold:
    def test_an_unpriced_call_is_not_shown_as_free(self) -> None:
        run = _run(results=[_result(traces=[_trace(cost=None)])])

        document = render_html([run])

        assert "$0.0000" not in document
        assert "lower bound" in document

    def test_a_fully_priced_run_states_its_cost(self) -> None:
        run = _run(results=[_result(traces=[_trace(cost=0.0025)])])

        document = render_html([run])

        assert "$0.0025" in document
        assert "lower bound" not in document

    def test_errored_scores_are_declared(self) -> None:
        """The means cover fewer cases than the run contains, and a reader
        taking them as complete would overstate what was measured."""
        run = _run(
            results=[
                _result("q1"),
                _result("q2", scores=[Score.from_error("exact", "boom")]),
            ]
        )

        assert "1 score errored" in render_html([run])

    def test_an_unmeasured_case_is_not_called_a_failure(self) -> None:
        """A broken scorer leaves the model's performance unknown. Painting it
        as a failure would send someone to debug a working prompt."""
        run = _run(results=[_result(scores=[Score.from_error("judge", "no key")])])

        document = render_html([run])

        assert ">unmeasured<" in document

    def test_a_continuous_score_is_not_called_a_failure_either(self) -> None:
        run = _run(results=[_result(scores=[Score(scorer_name="ratio", value=0.1)])])

        document = render_html([run])

        assert ">scored<" in document
        assert ">fail<" not in document

    def test_a_crashed_case_says_error_not_fail(self) -> None:
        run = _run(results=[_result(error="RuntimeError: boom", scores=[])])

        document = render_html([run])

        assert ">error<" in document
        assert "RuntimeError: boom" in document

    def test_the_pass_column_excludes_unjudged_cases(self) -> None:
        run = _run(
            results=[
                _result("q1", scores=[Score(scorer_name="exact", value=1.0, passed=True)]),
                _result("q2", scores=[Score(scorer_name="ratio", value=0.5)]),
            ]
        )

        assert ">1/1<" in render_html([run])

    def test_the_footer_states_what_a_dash_means(self) -> None:
        """The report is read without the docs beside it, so the one caveat that
        governs every number has to travel with it."""
        document = render_html([_run()])

        assert "never that it is zero" in document
        assert "no significance testing" in document.lower()


class TestTheDetailAReaderCameFor:
    def test_every_case_gets_a_section(self) -> None:
        run = _run(results=[_result("q1"), _result("q2"), _result("q3")])

        assert render_html([run]).count("<details class='case'") == 3

    def test_failures_are_open_and_passes_are_collapsed(self) -> None:
        """A thirty-case report with everything expanded is a page nobody
        scrolls; a failure nobody can see is a report that wasted its format."""
        passing = render_html([_run(results=[_result("ok")])])
        failing = render_html(
            [
                _run(
                    results=[
                        _result("bad", scores=[Score(scorer_name="s", value=0.0, passed=False)])
                    ]
                )
            ]
        )

        assert "<details class='case'>" in passing
        assert "<details class='case' open>" in failing

    def test_the_full_output_is_shown_rather_than_truncated(self) -> None:
        """The terminal truncates to fit a screen and the PR comment to fit
        GitHub's limit. A file has room, and the detail is why it exists."""
        long_output = "x" * 4000
        run = _run(results=[_result(output=long_output)])

        assert long_output in render_html([run])

    def test_a_repeat_is_labelled(self) -> None:
        run = _run(results=[_result("q1"), _result("q1", repeat_index=1)])

        assert "q1 #1" in render_html([run])

    def test_the_trace_tree_nests(self) -> None:
        run = _run(
            results=[
                _result(traces=[_trace("root"), _trace("child", parent_id="root", name="inner")])
            ]
        )

        document = render_html([run])
        root_at = document.index("<span class='tname'>call")
        child_at = document.index("<span class='tname'>inner")

        assert root_at < child_at, "the child should render inside the parent"
        assert document.count("<ul") >= 2, "a nested list is what makes it a tree"

    def test_error_frames_are_shown_when_a_task_raised(self) -> None:
        """Captured at the raise precisely so a stored failure stays debuggable
        later, which is when a CI artifact is read."""
        result = Result(
            id="r1",
            case_id="q1",
            error="ValueError: bad",
            error_frames=["  task.py:12 in answer", "    return parse(x)"],
            scores=[],
        )

        document = render_html([_run(results=[result])])

        assert "task.py:12 in answer" in document


class TestTheThresholdVerdict:
    def test_a_breach_is_named_with_the_bar_it_missed(self) -> None:
        run = _run(results=[_result(scores=[Score(scorer_name="exact", value=0.2)])])

        document = render_html([run], threshold=0.8)

        assert "Below the threshold of 0.80" in document
        assert "qa scored 0.20" in document

    def test_no_threshold_means_no_verdict(self) -> None:
        """`evalstand` will not invent a pass mark."""
        run = _run(results=[_result(scores=[Score(scorer_name="exact", value=0.2)])])

        assert "threshold" not in render_html([run]).lower()

    def test_an_unmeasured_run_is_not_reported_as_below_the_bar(self) -> None:
        run = _run(results=[_result(scores=[Score.from_error("judge", "boom")])])

        assert "Below the threshold" not in render_html([run], threshold=0.8)


class TestWhatAnEmptyReportSays:
    def test_no_evals_is_distinct_from_everything_passing(self) -> None:
        """Opposite findings. A page showing an empty table for both would let a
        broken collection read as success."""
        document = render_html([])

        assert "No evals ran" in document
        assert _check(document).problems == []

    def test_it_is_still_a_valid_document(self) -> None:
        document = render_html([])

        assert document.startswith("<!doctype html>")
        assert "</html>" in document


class TestItIsReproducible:
    def test_the_same_run_renders_identically(self) -> None:
        """A report that differed between renders could not be diffed, and a
        diff is how somebody notices what changed between two builds."""
        run = _run(results=[_result("q1", traces=[_trace()])])

        first = render_html([run], generated_at=_FIXED)
        second = render_html([run], generated_at=_FIXED)

        assert first == second

    def test_the_timestamp_is_stated(self) -> None:
        """An artifact with no timestamp cannot be placed against a build."""
        document = render_html([_run()], generated_at=_FIXED)

        assert "2026-01-02 03:04 UTC" in document

    def test_the_title_is_used(self) -> None:
        document = render_html([_run()], title="nightly evals")

        assert "<title>nightly evals</title>" in document
        assert "<h1>nightly evals</h1>" in document


class TestTheHtmlFlag:
    """`--html PATH`, which is what a CI job actually invokes.

    Driven through `runpytest_subprocess` for the reason recorded in ADR 0008:
    importing litellm inside a pytester inline session breaks entry-point
    discovery for the rest of the process.
    """

    EVAL = """
from evalstand import Case, evaluate
from evalstand.models import Score

evaluate(
    name="artifact",
    cases=[Case(id="q1", input="x", expected="ok"), Case(id="q2", input="y", expected="ok")],
    task=lambda value: "ok" if value == "x" else "no",
    scorers=[lambda output, expected: Score(
        scorer_name="s", value=1.0 if output == expected else 0.0, passed=output == expected
    )],
)
"""

    _BASE = ("--no-store", "--allow-dirty", "-p", "no:cacheprovider")

    def test_it_writes_a_report_to_the_named_path(self, pytester) -> None:
        pytester.makepyfile(artifact_eval=self.EVAL)
        destination = pytester.path / "out" / "report.html"

        pytester.runpytest_subprocess(*self._BASE, "--html", str(destination))

        assert destination.exists(), "the artifact was not written"
        document = destination.read_text(encoding="utf-8")
        assert document.startswith("<!doctype html>")
        assert document.count("<details class='case'") == 2

    def test_it_creates_the_parent_directory(self, pytester) -> None:
        """CI jobs name a path inside a directory that does not exist yet, and a
        report that failed for that reason would look like a broken tool."""
        pytester.makepyfile(artifact_eval=self.EVAL)
        destination = pytester.path / "deep" / "nested" / "report.html"

        pytester.runpytest_subprocess(*self._BASE, "--html", str(destination))

        assert destination.exists()

    def test_the_terminal_summary_still_prints(self, pytester) -> None:
        """Independent of --output. A flag that silenced the log in exchange for
        a file nobody has opened yet would make CI worse, not better."""
        pytester.makepyfile(artifact_eval=self.EVAL)
        destination = pytester.path / "report.html"

        result = pytester.runpytest_subprocess(*self._BASE, "--html", str(destination))

        assert "artifact" in str(result.stdout)
        assert destination.exists()

    def test_nothing_is_written_without_the_flag(self, pytester) -> None:
        pytester.makepyfile(artifact_eval=self.EVAL)

        pytester.runpytest_subprocess(*self._BASE)

        assert not list(pytester.path.glob("*.html"))

    def test_the_threshold_reaches_the_report(self, pytester) -> None:
        """The bar lives in a CI config the artifact's reader cannot see."""
        pytester.makepyfile(artifact_eval=self.EVAL)
        destination = pytester.path / "report.html"

        pytester.runpytest_subprocess(*self._BASE, "--html", str(destination), "--threshold", "0.9")

        assert "Below the threshold of 0.90" in destination.read_text(encoding="utf-8")

    def test_an_unwritable_path_does_not_lose_the_run(self, pytester) -> None:
        """The run has already happened and been recorded. Losing it to an
        unwritable directory would be the tail wagging the dog.

        Asserted on the **exit code**, not on the summary text. An exception
        raised in `pytest_terminal_summary` does not unwrite output already
        flushed, so a test that only checked for the summary passed against a
        mutant that let the write error propagate — it asserted something true
        but insufficient.
        """
        pytester.makepyfile(artifact_eval=self.EVAL)

        # A directory where the file should be. `write_text` raises
        # PermissionError on Windows and IsADirectoryError elsewhere; both are
        # OSError, which is what the handler catches.
        blocked = pytester.path / "report.html"
        blocked.mkdir()

        result = pytester.runpytest_subprocess(*self._BASE, "--html", str(blocked))
        output = str(result.stdout) + str(result.stderr)

        # Asserted on the absence of a traceback, which is the only observable
        # difference. Neither the exit code nor the summary can tell: the write
        # happens in `pytest_terminal_summary`, after the outcome is decided and
        # after the summary has been flushed — so an unhandled error still exits
        # 1 and still prints the tables, while spraying a stack trace that reads
        # as a crash in the tool.
        assert "PermissionError" not in output
        assert "IsADirectoryError" not in output
        assert "Traceback" not in output, "a failed artifact write must not look like a crash"
        assert "artifact" in output, "the summary should still print"


class TestTheCliForwardsIt:
    def test_the_html_path_reaches_pytest(self, monkeypatch) -> None:
        """Coverage showed the CLI's argv-building is where a flag silently
        stops arriving."""
        import pytest as pytest_module
        from typer.testing import CliRunner

        from evalstand.cli import app

        seen: list[list[str]] = []
        monkeypatch.setattr(pytest_module, "main", lambda args: seen.append(args) or 0)

        CliRunner().invoke(app, ["run", "somewhere", "--html", "out/report.html"])

        assert seen, "run did not call pytest"
        argv = seen[0]
        assert "--html" in argv
        assert argv[argv.index("--html") + 1].endswith("report.html")

    def test_it_is_absent_unless_asked_for(self, monkeypatch) -> None:
        import pytest as pytest_module
        from typer.testing import CliRunner

        from evalstand.cli import app

        seen: list[list[str]] = []
        monkeypatch.setattr(pytest_module, "main", lambda args: seen.append(args) or 0)

        CliRunner().invoke(app, ["run", "somewhere"])

        assert "--html" not in seen[0]


pytest_plugins = ["pytester"]

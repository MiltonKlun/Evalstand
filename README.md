# evalstand

> **Status: 1.0.0.** All 8 phases complete. Install with
> `pip install evalstand`.

Evaluating an LLM application should feel like running a test suite.

`evalstand` is a local-first LLM evaluation tool for Python. You write an eval
file, run a watch command, and results stream into a live terminal UI — scores,
nested call traces, token counts, latency, and cost. Everything runs on your
machine and persists to a local SQLite database, so you can compare a run
against the one before it.

<!-- The demo GIF belongs here. See examples/demo/README.md to record it: the
     eval is offline, so re-recording costs nothing. -->

## 60 seconds

```bash
git clone https://github.com/MiltonKlun/Evalstand && cd Evalstand
uv sync
```

```python
# qa_eval.py
from evalstand import Case, evaluate, llm
from evalstand.scorers import exact


async def answer(question: str) -> str:
    reply = await llm.acall("gpt-4o-mini", [{"role": "user", "content": question}])
    return reply.text.strip()


evaluate(
    name="capitals",
    cases=[
        Case(id="france", input="Capital of France? City only.", expected="Paris"),
        Case(id="japan", input="Capital of Japan? City only.", expected="Tokyo"),
        Case(id="peru", input="Capital of Peru? City only.", expected="Lima"),
    ],
    task=answer,
    scorers=[exact],
)
```

```bash
export OPENAI_API_KEY=sk-...
evalstand watch          # the live view, re-running when you edit
evalstand run            # one pass, prints a summary
evalstand serve          # browse past runs in a browser (needs the web extra)
pytest qa_eval.py        # the plain test runner; same runner underneath
```

Three things, and only three: **Cases** are the inputs, the **Task** is your
function under test, and **Scorers** judge what it returned.

## What you get

| | |
| --- | --- |
| **Live results** | rows appear as each case finishes, not in one batch at the end |
| **Trace trees** | a call made inside another call is its child, so you can see *which step* went wrong |
| **Cost and tokens** | per call, per case, per run — and marked as a lower bound when a call could not be priced |
| **History** | every run recorded locally; `history`, `show`, `compare` |
| **Watch mode** | edit a prompt, the eval re-runs within a second |
| **CI gates** | `--threshold` and `--fail-on-error`, with documented exit codes |
| **CI artifacts** | `--html` writes one self-contained report: full outputs, whole trace trees |
| **Ten scorers** | exact, normalised, contains, regex, levenshtein, ratio, close-to, JSON fields, judge, factuality |
| **Runs under pytest** | each `(case, repeat)` is one test item, so `-k`, `-x`, `--lf` all work |
| **A web UI, optionally** | `evalstand serve` browses history in a browser and streams a running eval into it |

Full capability list in [PLAN.md](PLAN.md) §2. Three capabilities go beyond the
tool that inspired this one:

- **Nested traces.** The reference implementation's traces are a flat list.
- **Response caching.** It has none, so iterating re-buys every answer.
- **Stable case identity.** It matches cases by position, so inserting one
  silently re-pairs every later case with the wrong history.

## Docs

- [Quickstart](docs/quickstart.md) — install, write an eval, run it
- [Writing evals](docs/writing-evals.md) — cases, tasks, repeats, custom columns
- [Scorers](docs/scorers.md) — the library, and writing your own
- [Traces](docs/traces.md) — what your task did, and what it cost
- [Watching](docs/watching.md) — the live view and watch mode
- [CI](docs/ci.md) — thresholds, exit codes, pull-request comments
- [Web UI](docs/web.md) — `serve`, the JSON API, and how to read its numbers
- [Architecture](docs/architecture.md) — how the pieces fit, for anyone changing them
- [Decisions](docs/adr/) — why the design is the way it is

## In CI

```bash
evalstand run --threshold 0.85 --fail-on-error
```

`0` met the bar, `1` fell below it, `2` something did not run. `--output
markdown` produces a body for a pull-request comment. See [docs/ci.md](docs/ci.md)
for the workflow and the full table.

## Why

Existing Python options are either heavyweight platforms that push you toward a
hosted service, or bare metric libraries with no runner, no persistence, and no
live feedback loop. `evalstand` is the middle: a real runner with a real UI that
stays on your machine.

## Limitations

Stated up front, and kept accurate as the project grows.

**A delta is not a verdict.** `compare` reports the arithmetic difference
between two runs' means. There is **no significance testing**, so a difference
between two runs of a stochastic system may be noise. Nothing here will call a
change a regression.

**LLM judge scorers are unvalidated.** `judge` and `factuality` work, but nobody
has calibrated their verdicts against human labels on your data. Treat them as a
signal, not a measurement.

**The showcase baseline covers the scalar fields only, and one method.**
`examples/pdf_extraction/BASELINE.md` records one run of span selection with
TypeSafe's `jev-1.13.0`: a regex finds candidate values and the model picks
among them. It scored 30/30 invoices on six scalar fields for $0.0015. It does
not score line items — span selection cannot produce a list of records — and
the generative eval, which does, has not yet been run against a real model. One
run of one method on a synthetic corpus is a record, not a verdict on either.

**No demo GIF yet.** `examples/demo/` holds an offline eval and a VHS tape ready
to record; the recording tooling is not installed here.

**Cost figures are lower bounds when a model is not in LiteLLM's pricing table.**
Unpriced calls are counted and declared, never silently treated as free.

**The web UI has no authentication.** `evalstand serve` binds localhost by
default for that reason. The database holds every prompt and completion your
evals sent and received, so `--host 0.0.0.0` publishes all of it to anyone who
can reach the port. There is no login, and adding one is not planned — put it
behind something that does auth if it needs to leave the machine.

## Development

```bash
uv sync --all-extras --dev
uv run pytest                       # the suite
uv run python scripts/mutate.py     # 281 mutants, all killed
uv run mkdocs serve                 # the docs site
```

The mutation harness is the real quality measure here. Every defect this project
has found ships with a mutant that reintroduces it, so a test that stops catching
its bug fails loudly rather than passing quietly. `tests/unit/test_mutation_harness.py`
holds the harness itself to the same standard — a stale anchor reports as a
broken probe, not as a survivor.

## Licence

MIT. See [LICENSE](LICENSE).

---

<sub>Inspired by [evalite](https://github.com/mattpocock/evalite) (MIT), which
showed that local LLM evals could feel like running tests. `evalstand` is an
independent Python implementation.</sub>

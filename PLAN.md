# Build Plan — `evalstand`

> **Name:** `evalstand` — verified free on PyPI (404) and GitHub (0 name matches) on 2026-09-01.
> **Audience:** an autonomous coding agent (Claude Code) executing this plan without access to the conversation that produced it.
> **Owner profile:** mid-level software engineer, 3–5 years production experience, strong in Python, Docker, and CI; new to AI engineering.
> **Budget:** ~15 hours per week. Total: 6–7 weeks for feature parity (Phases 0–7).
> **Scope:** this project is entirely self-contained. It has no dependencies on, and makes no references to, any other project. Everything needed is in this document.

---

## 1. What we are building and why

### The goal

`evalstand` — a local-first LLM evaluation tool for Python.

The core idea: evaluating an LLM application should feel exactly like running a test suite: you write an eval file, run a watch command, and a live UI updates as results stream in, showing scores, traces, token counts, and cost. Nothing in the Python ecosystem offers that experience. The closest Python tools are either heavyweight platforms that push you toward a hosted service, or bare metric libraries with no runner, no persistence, and no live feedback loop.

The design is informed by `evalite` (TypeScript, MIT), which solved this problem well. `evalstand` is an independent Python implementation, not a transliteration: the capability set in Section 2 is the target, and every design decision is made for Python idiom on its own merits. Where the reference design does not carry into Python, we adopt the Python equivalent and record why in an ADR.

### Explicit non-goals for this version

These are deliberately out of scope. They are listed so the agent does not drift into them, and revisited in Section 8.

- Statistical significance testing of run-over-run differences.
- Validation or calibration of LLM-as-judge scorers.
- Any hosted service, account system, or cloud sync.
- Distributed or multi-machine execution.
- Dataset generation or curation tooling.

### Interface choice: TUI first

`evalstand` ships a **Textual TUI first**, with a web UI as an optional stretch phase. Rationale: the TUI reaches parity of *function* much faster, keeps the entire stack in Python, and demos well in a terminal recording. Record this in an ADR.

---

## 2. Feature parity checklist

This is the definition of "done" for v1. Every row must be satisfied or consciously waived with an ADR.

| # | Capability | `evalstand` implementation | Phase | vs. reference |
| --- | --- | --- | --- | --- |
| 1 | Eval files auto-collected by the test runner | `*_eval.py` collected by a pytest plugin | 2 | parity |
| 2 | One entry function for cases, task, scorers | `evaluate(cases=, task=, scorers=)` | 2 | parity |
| 3 | Case loader returning input/expected pairs | Sync or async callable returning `Case` objects | 2 | parity |
| 4 | Task receives input, returns output | Sync or async, auto-detected | 2 | parity |
| 5 | Built-in scorer library | Native scorer set, plus an adapter interface | 4 | parity |
| 6 | Custom scorers via a simple function signature | `@scorer` decorator | 4 | parity |
| 7 | Traces — nested LLM calls captured inside a task | `trace()` context manager + auto-capture | 3 | **beyond** — reference traces are flat |
| 8 | Token and cost reporting per call and per run | Via LiteLLM cost lookup | 1, 3 | parity |
| 9 | Model response caching | SQLite-backed cache | 1 | **beyond** — reference has no response cache |
| 10 | Run each case N times | `--repeat N` / `repeat=` argument | 3 | parity |
| 11 | SQLite result storage | stdlib `sqlite3` with migrations | 5 | parity |
| 12 | Score history across runs | `history` command and TUI history view | 5 | parity |
| 13 | Live-updating UI during a run | Textual TUI | 6 | parity |
| 14 | Watch mode with file-change re-runs | `watchfiles` | 6 | parity |
| 15 | Case detail view — input, output, expected, trace | TUI detail pane | 6 | parity |
| 16 | Custom result table columns | `columns=` argument on `evaluate()` | 6 | parity |
| 17 | `--threshold` for CI pass/fail | `--threshold` with documented exit codes | 7 | parity |
| 18 | Streaming task output | LiteLLM streaming, rendered live | 3 | parity |
| 19 | Runs under the plain test runner too | Works under bare `pytest` | 2 | parity |

Three capabilities go **beyond** the reference implementation, and the README
says so: nested trace trees (its traces are a flat list), model response caching
(it has none), and stable `Case.id` identity — it matches cases positionally by
index, so inserting a case silently re-pairs every later case with the wrong
history. Durable identity is what makes `history` and `compare` trustworthy.

---

## 3. Architecture

```
evalstand/
├── pyproject.toml
├── README.md
├── LICENSE                        # MIT
├── CHANGELOG.md
├── .github/workflows/
│   ├── ci.yml                     # lint, type-check, test — no API keys
│   └── release.yml                # PyPI publish on tag
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── writing-evals.md
│   ├── scorers.md
│   ├── traces.md
│   ├── ci.md
│   └── adr/
├── src/evalstand/
│   ├── __init__.py                # public API, kept under 10 exported names
│   ├── api.py                     # evaluate(), Case, Score, @scorer
│   ├── models.py                  # Case, Score, Trace, Result, Run
│   ├── plugin.py                  # pytest collection + reporting hooks
│   ├── runner.py                  # async execution, concurrency, repeats
│   ├── tracing.py                 # trace context manager + LiteLLM callback
│   ├── llm.py                     # LiteLLM wrapper: call, tokens, cost, retry
│   ├── cache.py                   # SQLite response cache, record/replay
│   ├── storage.py                 # SQLite run store + migrations
│   ├── config.py                  # settings resolution
│   ├── scorers/
│   │   ├── base.py                # Scorer protocol, @scorer decorator
│   │   ├── text.py                # exact, normalised, contains, regex
│   │   ├── fuzzy.py               # levenshtein, ratio (rapidfuzz)
│   │   ├── numeric.py             # tolerance-based
│   │   ├── json_field.py          # per-field structured comparison
│   │   └── llm.py                 # judge-style scorers
│   ├── reporting/
│   │   ├── console.py             # rich summary tables
│   │   └── markdown.py            # CI summary body
│   ├── tui/
│   │   ├── app.py
│   │   └── widgets/
│   └── cli.py                     # typer entrypoint
├── examples/
│   ├── toy/                       # 3 cases, built in Phase 2, used throughout
│   └── pdf_extraction/            # showcase example, Phase 5
└── tests/
    ├── cassettes/                 # recorded responses, committed
    ├── unit/
    └── integration/
```

### Core abstractions

Four concepts. Resist adding a fifth.

- **Case** — one test input: `id`, `input`, `expected` (optional), `metadata`.
- **Task** — the user's function under test. Receives `Case.input`, returns any output. The LLM call happens here.
- **Scorer** — a callable `(output, expected, case) -> Score`. `Score` carries `value: float` in `[0, 1]`, optional `passed: bool`, and `metadata: dict`.
- **Run** — one execution of one eval. Persisted, immutable, comparable to previous runs.

**Trace** is a supporting concept, not a fifth abstraction: a record of one LLM call made inside a task, forming a tree when calls nest.

### Target API shape

```python
from evalstand import evaluate, Case, scorer
from evalstand.scorers import levenshtein, exact


def load_cases() -> list[Case]:
    return [
        Case(id="q1", input="What is the capital of France?", expected="Paris"),
        Case(id="q2", input="What is 2 + 2?", expected="4"),
    ]


async def answer(question: str) -> str:
    resp = await llm.acall("gpt-4o-mini", [{"role": "user", "content": question}])
    return resp.text


evaluate(
    name="basic-qa",
    cases=load_cases,
    task=answer,
    scorers=[exact, levenshtein],
)
```

Saved as `qa_eval.py`, this runs under both `evalstand run` and bare `pytest`.

---

## 4. Approved dependencies

| Purpose | Package |
| --- | --- |
| Packaging | `uv` |
| Test runner base | `pytest` |
| LLM provider layer | `litellm` |
| CLI | `typer` |
| Console rendering | `rich` |
| TUI | `textual` |
| File watching | `watchfiles` |
| Data models | `pydantic` v2 |
| Fuzzy matching | `rapidfuzz` |
| Retry | `tenacity` |
| Docs | `mkdocs-material` |
| Dev | `ruff`, `mypy`, `pytest-cov`, `pre-commit` |
| Example only | `reportlab`, `faker`, `pypdfium2` (behind an optional extra) |

Storage is stdlib `sqlite3`. **Do not add an ORM.** Do not add a dependency outside this table without an ADR.

---

## 5. Hard guardrails

The agent must not violate these.

1. **CI runs with zero API keys and zero API spend.** All tests use recorded cassettes or mocks. Live tests are marked `@pytest.mark.live` and excluded by default.
2. **Never commit secrets.** Keys come from environment variables only. A secret-scanning pre-commit hook is installed in Phase 0.
3. **Expected outputs are written by humans or produced programmatically — never generated by an LLM.** A golden answer produced by the system under test is not ground truth.
4. **No dependency outside Section 4 without an ADR.**
5. **A phase is not done until `ruff check`, `ruff format --check`, `mypy`, and `pytest` all pass locally and in CI.**
6. **Every number in the README must be reproducible by a command in the repo.**
7. **Write an ADR for every non-obvious decision** in `docs/adr/NNNN-title.md`: context, decision, consequences. Keep them short.

---

## Phase 0 — Foundations

**Goal:** a named, licensed, linted, CI-green empty package.
**Estimate:** 6 hours.

- [x] **0.1 Public name: `evalstand`.** Verified free on PyPI (`https://pypi.org/pypi/evalstand/json` -> 404) and GitHub (0 repositories matching the name) on 2026-09-01. Rejected because already taken on PyPI: `evalrig` (an AI-eval package), `proofmark` (an LLM-output testing package, uploaded 2026-08-17), `assay`, `evalbench`, `plumbline`, `rigor`, `tessera`, `proofbench`, `evalforge`, `evalloop`, `rubricon`.
  - *Acceptance:* recorded in `docs/adr/0001-name.md`, applied consistently across `pyproject.toml`, `src/`, and docs.
  - *Naming rule (binding):* the string `evalite` must not appear in any package name, module name, directory name, filename, class name, function name, or CLI command. A single attribution footnote at the bottom of the README is the only permitted occurrence in the shipped repo.
- [x] **0.2 Initialise the repo:** `uv init --lib`, Python 3.11+, `src/` layout, MIT `LICENSE`, `.gitignore`.
  - *Acceptance:* `uv sync` succeeds; `uv run python -c "import evalstand"` succeeds.
- [x] **0.3 Configure tooling:** `ruff` (lint + format), `mypy` strict on `src/`, `pytest` with `--cov`, `pre-commit` with ruff plus a secret-scanning hook.
  - *Acceptance:* `pre-commit run --all-files` passes on a clean tree.
- [x] **0.4 Write `.github/workflows/ci.yml`.** Matrix over Python 3.11, 3.12, 3.13. Steps: checkout, install uv, sync, ruff check, ruff format --check, mypy, pytest.
  - *Acceptance:* CI green on first push; the workflow file contains no `secrets.` reference.
- [x] **0.5 Create `docs/adr/0000-adr-process.md`** and a template.
- [x] **0.6 Study the reference implementation** before writing any code — `evalite` (TypeScript, MIT), specifically its entry function, CLI, SQLite layer, and web UI. Write `docs/adr/0002-design-scope.md` recording which behaviours `evalstand` adopts, which it does differently, and why (TUI instead of React, pytest instead of Vitest).
  - *Acceptance:* the ADR names specific behaviours and specific decisions, not generalities. Read it for design understanding only — do not copy code. This is an independent implementation.
- [x] **0.7 Placeholder README** with the problem statement, a "status: in development" banner, and the one-line attribution footnote. No marketing claims yet.

**Exit criteria:** CI green, pre-commit passes, name applied everywhere, ADRs 0001 and 0002 written. **— met 2026-09-01.**

Phase 0 also produced more than it planned to: `CONTEXT.md` (29 terms) and ADRs
0003-0006, from a design review that ran before any implementation. That review
corrected task 3.3, whose stated rule could not be implemented as written, and
established that three capability rows go beyond the reference rather than
matching it.

---

## Phase 1 — LLM layer, caching, and cost tracking

**Goal:** a single model call works, is cached, and reports tokens, latency, and cost.
**Estimate:** 12 hours.

- [x] **1.1 Define `models.py`.** `Case`, `Score`, `Trace`, `Result`, `Run`, `Batch` as Pydantic models. `Score.value` is a float in `[0, 1]`. **`Score.passed` is set only by a Scorer that genuinely knows pass from fail; nothing derives it from a threshold, and a continuous scorer leaves it `None`.** Every model is JSON-serialisable.
  - *Acceptance:* round-trip serialisation tests pass for each model.
- [x] **1.2 Build `llm.py` over LiteLLM.** `call(model, messages, **params) -> LLMResponse` and async `acall`, returning text, input and output token counts, latency in ms, `cost_usd` from `litellm.completion_cost()`, and the raw provider response.
  - *Acceptance:* unit tests with LiteLLM mocked verify token and cost extraction. One `@pytest.mark.live` test hits a real cheap model and is excluded from CI. **— the live tests were added later, during the post-Phase-2 audit; 1.2 was ticked without them.**
- [x] **1.3 Add streaming support.** `acall_stream` yields chunks and accumulates the final response with correct token and cost totals.
  - *Acceptance:* a streamed call and a non-streamed call to the same prompt produce identical accumulated text and equivalent cost.
- [x] **1.4 Add retry policy** with `tenacity`: retry on 429 and 5xx with exponential backoff and jitter, max 3 attempts. Never retry 4xx auth or content-policy errors. Log every retry.
  - *Acceptance:* tests cover 429-then-success and 401-immediate-fail.
- [x] **1.5 Build `cache.py`.** SQLite response cache keyed on the SHA-256 of canonical JSON of `(model, messages, temperature, top_p, max_tokens, seed, tools, response_format)`. Store the response, `created_at`, and `hit_count`. Support `--no-cache` and `--refresh-cache`.
  - *Acceptance:* the same call twice yields one provider call and one cache hit; changing temperature yields a miss.
- [x] **1.6 Add record/replay for tests.** A cassette mode writing responses to JSON on record and reading on replay.
  - *Acceptance:* the full test suite passes with every provider API key unset.
- [x] **1.7 ADR 0007:** why LiteLLM rather than provider SDKs. (0003 is already taken by the variants decision.)

**Exit criteria:** a throwaway script makes a cached model call and prints text, tokens, latency, and cost. CI green without keys. **— met 2026-09-01**, demonstrated against the committed cassette: 14/1 tokens, $0.00000270, `cached=True` on the second call.

---

## Phase 2 — The `evaluate()` API and pytest collection

**Goal:** an eval file is discovered and executed with a first-class authoring experience.
**Estimate:** 14 hours.

- [x] **2.1 Design the public API** in `api.py` to the shape in Section 3. `cases` accepts a list, a callable returning a list, or an async callable.
  - **The top-level exports are exactly seven:** `evaluate`, `Case`, `Score`, `Result`, `Trace`, `scorer`, `trace`. Scorers live in `evalstand.scorers` and do not count against the budget.
  - `Run` and `Batch` are deliberately not exported: users read them in reports, they never construct them.
  - Three slots of headroom remain against the cap of ten. Spending them needs an ADR — the cap exists to force the question.
  - *Acceptance:* `docs/writing-evals.md` contains a complete working example under 25 lines.
- [x] **2.2 Implement the pytest plugin** in `plugin.py`. Use `pytest_collect_file` to collect `*_eval.py`, generate **one item per `(case, repeat_index)`**, and use `pytest_runtest_makereport` to capture outcomes. Register through the `pytest11` entry point.
  - Item IDs carry the repeat index only when repeats are on: `q1` when `repeat=1`, `q1[repeat=2]` otherwise, so `-k q1` still selects them all by prefix.
  - An item is one execution with one outcome. Do not collapse several stochastic executions into a single pass/fail — any aggregation rule there is a judgement the user did not make.
  - Duplicate Eval names are an error raised at collection time, naming both file paths. Names are identity; paths are metadata.
  - *Acceptance:* `pytest examples/toy` discovers and runs the eval; `pytest -k q1` selects a single case; `--repeat 3 -k q1` selects three items.
- [x] **2.3 Support both sync and async tasks.** Detect with `inspect.iscoroutinefunction` and dispatch accordingly. The user should never have to think about it.
  - *Acceptance:* two identical evals, one sync and one async, produce identical results.
- [x] **2.4 Ensure bare `pytest` works.** Running `pytest` with no custom CLI must collect and execute evals and report pass/fail sensibly.
  - *Acceptance:* documented in `docs/ci.md` with a working example.
- [x] **2.5 Build `examples/toy/`** — three cases, one scorer, no external files. Every subsequent phase develops against this.
- [x] **2.6 Console reporting** in `reporting/console.py` using rich: a summary table (per-scorer mean, pass count, total cost, wall time) and a failures table.
  - *Acceptance:* the toy example prints a readable summary within one screen.

**Exit criteria:** `evalstand run examples/toy` and `pytest examples/toy` both work end to end. **— met 2026-09-02.**

Note: the exit criterion required `evalstand run`, but no task created the CLI —
the architecture lists `cli.py` and no phase assigned it. Phase 2 therefore ships
a minimal `run` command that delegates to pytest rather than reimplementing
collection. `history`, `show`, and `compare` remain Phase 5; watch mode Phase 6.

---

## Phase 3 — Runner, traces, repeats, and streaming

**Goal:** concurrent execution with the nested-call visibility that makes the UI useful.
**Estimate:** 14 hours.

- [x] **3.1 Build `runner.py`.** Async execution with an `asyncio.Semaphore` (default concurrency 8, `--concurrency` flag). Deterministic ordered result collection. Per-case timeout. Errors are captured on the result rather than aborting the run.
  - *Acceptance:* a suite where one case raises still completes and records the error; changing concurrency measurably changes wall time.
- [x] **3.2 Implement tracing** in `tracing.py`. This is the feature most worth porting carefully. Two mechanisms:
  - A `trace(name)` context manager the user can wrap around any operation.
  - **Automatic capture** of every call made through `llm.py` during a task, using a `contextvars.ContextVar` to associate calls with the currently executing case.
  - Traces nest into a tree. Each node records name, start, duration, input, output, model, tokens, and cost.
  - **Parenting rule:** a `ContextVar` holds the currently open node. Entering a scope sets it and keeps the token; leaving resets the token in a `finally`. This is the only rule that survives both `asyncio.gather` (contextvars copy per task, so siblings share a parent for free) and an exception mid-call (the `finally` reset stops a raising call corrupting its siblings' parentage).
  - **When a parent cannot be determined confidently, attach the node to the Result root.** A visible orphan is honest; a wrongly-parented node is a plausible-looking lie. If nesting proves unreliable under load, degrade to a flat list rather than ship a wrong tree.
  - *Acceptance:* a task making three nested LLM calls produces a three-node trace tree with correct parent-child relationships and per-node cost, and the sum of node costs equals the case total.
- [x] **3.3 Implement `--repeat N`** (the reference implementation calls this `trialCount`). Each case runs N times with `repeat_index` recorded on every result. **`--repeat N` with N > 1 bypasses the Cache unconditionally.**
  - The earlier draft of this plan said to bypass only when temperature > 0. That rule cannot be implemented as written: the Task calls the model itself, so its parameters live in user code the runner cannot inspect. Temperature is also not the only source of nondeterminism. Unconditional bypass is predictable and matches what asking for repeats means — receiving N identical cached rows never does.
  - This spends real money, N times over. The run summary must show it (`repeats bypassed cache: 5 x 30 calls`), because it is the easiest way to run up a bill by accident.
  - **Bypass runs in both directions:** a bypassed call is neither read from nor written to the Cache. Writing one would let a later non-repeat run serve an arbitrary sample from a repeat set as though it were the answer for that key.
  - *Acceptance:* `--repeat 5` on a temperature-0.7 task produces at least one case with 5 distinct outputs; a cache-hit counter shows zero hits for repeated cases.
- [x] **3.4 Wire streaming through the runner** so partial output is available to the reporting layer as it arrives.
  - *Acceptance:* a streaming task shows incremental output in console reporting.
- [x] **3.5 Aggregate per-run totals:** total cost, total tokens, cache hit rate, wall time, pass count.

**Exit criteria:** the toy example runs concurrently with repeats, and a trace tree is captured and printable. **— met 2026-09-02**, demonstrated against the committed cassette: 3 repeats at concurrency 3, cache correctly bypassed (6 calls rather than 2), and a two-level trace tree whose node costs sum to the case total of $0.00000510.

- [x] **3.6 Wire the runner into the execution path.** Everything above was reachable only by calling `run_eval()` directly; neither `pytest` nor `evalstand run` went through it, so users had none of it. The plugin now takes over `pytest_runtestloop` and delegates execution to the runner, and `--concurrency`, `--timeout` and `--no-cache` are exposed on both entry points. See [ADR 0008](docs/adr/0008-plugin-delegates-execution-to-the-runner.md).
  - The risk register's *"plugin fights the runner's async model"* is resolved rather than waived: only one component executes, so parity item 19 stands.
  - *Acceptance:* the same eval file run through bare `pytest` reports token totals and cost where it previously printed `-`; `--concurrency 1` measurably serialises what the default runs in parallel; `-k` executes only the selected cases. **— met 2026-09-03.** 8 cases x 0.5s: 2.52s at the default concurrency against 6.01s at `--concurrency 1`. A model-calling eval that reported `cost: -` now reports `$0.0004` and `20 in / 10 out tokens`. 18 mutants aimed at the new seam: 7/18 killed before its tests were written, 18/18 after.

---

## Phase 4 — Scorers

**Goal:** enough built-in scorers that a user never writes one on day one, plus a clean path when they do.
**Estimate:** 10 hours.

- [x] **4.1 Define the `Scorer` protocol** in `scorers/base.py` and a `@scorer` decorator that adapts a plain function. Support sync and async scorers.
  - *Acceptance:* a user-defined 3-line scorer works without importing any base class.
- [x] **4.2 String scorers** in `text.py`: `exact`, `normalised_exact` (case, whitespace, and punctuation folding), `contains`, `regex_match`.
- [x] **4.3 Fuzzy scorers** in `fuzzy.py` using `rapidfuzz`: `levenshtein` (normalised to `[0, 1]`; this is `evalstand`'s default scorer) and `ratio`.
  - *Acceptance:* `levenshtein("kitten", "sitting")` returns the documented normalised value.
- [x] **4.4 Numeric scorer** in `numeric.py`: absolute and relative tolerance, with sensible handling of `None` and unparseable output.
  - Shipped as `close_to(rel_tol=..., abs_tol=...)`, a factory like `regex_match`: the tolerance is a property of the check, not of the data. Both default to 0, so a user who forgets one is not silently given slack.
  - **An ambiguous output is unparseable, not a guess.** `"42 or 43"`, `"between 10 and 20"` and `"5 + 5"` all refuse rather than pick a candidate, because a score nobody can audit is worse than an honest failure to read.
  - Refuses `nan`/`inf` (which `float()` accepts, and which then compare false against every tolerance, reading as a wrong answer rather than an unreadable one) and `bool` (which subclasses `int`, so `float(True)` is 1.0).
  - *Acceptance:* verified through the real runner — a model answering `"The widget costs $19.99."` against an expected 20.00 passes at `rel_tol=0.01`, `"I'd estimate roughly 1,000."` parses the separator, and `"somewhere between 10 and 20"` reports as unreadable rather than guessing. **— met 2026-09-03.** 18 mutants against the scorer, 18/18 killed.
- [x] **4.5 Structured scorer** in `json_field.py`: compare two dicts field by field, returning both a macro-average and a per-field breakdown in `Score.metadata`.
  - Shipped as `json_fields(require_all=False)`. The value is a macro-average, so every field counts equally regardless of how much text it holds; `passed` stays absent unless `require_all=True`, where "every field correct" is a verdict the scorer genuinely knows.
  - **The denominator is the expected keys.** A field the model volunteered is reported in metadata but never counted — otherwise a verbose model scores worse than a terse one that answered exactly as badly.
  - **Nested dicts flatten to dotted paths**, so one wrong leaf costs one field rather than a whole subtree, and the metadata can name `person.born` instead of `person`.
  - **Missing is reported separately from wrong.** Both score zero, but a wrong field means the model answered and erred while a missing one means it never answered — different fixes.
  - Parses JSON returned as text, including ``` fences, since refusing to unwrap them would measure presentation rather than content.
  - *Acceptance:* a partial match on 3 of 5 fields returns 0.6 with the failing field names in metadata. **— met 2026-09-04**, and verified through the real runner on a four-case extraction eval. 17 mutants against the scorer, 17/17 killed.
- [x] **4.6 LLM scorers** in `llm.py`: a `judge` factory taking a rubric and returning a scorer, plus a `factuality`-style scorer comparing output against expected. Judge calls must go through `llm.py` so they are cached, traced, and costed like any other call.
  - The judge picks a **labelled choice**, never a number. Models cluster on 0.0/0.5/1.0 and cannot justify 0.7 over 0.8, but choose reliably between described options. The caller supplies what each label is worth, so the mapping from judgement to score is made once in the eval file rather than improvised per case.
  - **A reply naming no known choice, or two, is an errored Score.** Excluded from the mean rather than scored 0.0, which would report that the task did badly when what failed was the judge.
  - *Acceptance:* a judge scorer's LLM call appears in the case's trace tree. **— met 2026-09-05.** Verified through `run_eval`: a case whose task called the model once and whose judge called it once produced two traces, 70 in / 9 out tokens (20/8 from the task plus 50/1 from the judge) and a case cost summing both. A mutant that makes the judge call the provider directly, escaping capture, is killed by the suite.
  - *Note:* these scorers are unvalidated by design in this version. `docs/scorers.md` must say so plainly — and every judge Score carries `unvalidated: True` in its metadata, so the caveat travels with the number into any report that quotes it.
- [x] **4.7 Every scorer gets unit tests** covering the happy path, empty output, and `None` expected.
  - Each scorer has its own test file, plus `test_scorer_coverage.py`, which enumerates the public surface and applies the three inputs to every scorer without exception. It also asserts the library-wide Score contract: values stay in `[0, 1]`, scores are filed under the scorer's own name, a stated verdict agrees with its value, and a continuous scorer never invents one.
  - The inventory exists because per-scorer files can only test the scorers someone remembered to write tests for. This one **fails when a scorer is added without them**, naming it — which a checklist in a plan document cannot do. Verified by adding a scorer to the package and watching the suite name it as untested.
  - **— met 2026-09-05.** 92 tests across ten scorers.
- [x] **4.8 Write `docs/scorers.md`** documenting each built-in scorer and how to write a custom one.
  - Leads with writing your own, because that is the normal case rather than the advanced one. Documents all ten built-ins, what each return type means, and why `passed` is set by some scorers and not others.
  - States the judge caveat as an admonition rather than a footnote: **not calibrated against human labels**, use them to notice changes worth investigating, not to establish correctness.
  - **`test_docs_scorers.py` asserts every factual claim the doc makes** — the quoted 0.5714, the worked examples, the two behaviour tables, every documented signature and factory call, and that the doc names every exported scorer and advertises none that no longer exist. Its Python blocks are compiled and its import lines executed. Documentation that drifts from the code is worse than none, so the drift fails the suite.
  - **— met 2026-09-05.** 32 tests over the doc.

**Exit criteria:** eight or more built-in scorers, all tested and documented. **— met 2026-09-05.** Ten scorers: `exact`, `normalised_exact`, `contains`, `regex_match`, `levenshtein`, `ratio`, `close_to`, `json_fields`, `judge`, `factuality`. 829 tests passing; the mutation harness carries 105 mutants across the library.

---

### Pre-Phase-5 audit — 2026-09-05

A full review before persistence lands, hunting the failure mode that had
repeatedly slipped past a green suite: something wrong that looked right. Four
defects, none caught by the 829 tests passing at the time.

- **A continuous scorer could never fail a run.** A model answering every case
  with garbage reported `3 passed` and **exit code 0**. `render_failures` listed
  a case only when `passed is False`, and continuous scorers correctly never set
  it — so the worst possible result produced an empty table and a green CI.
  Fixed by listing zero-scoring cases as "scored 0.00" (not "failed" — the
  scorer gave no verdict), and by bringing the Threshold forward as an opt-in
  `--threshold` flag.
- **A Score could contradict itself.** `passed=True, value=0.0` was accepted and
  reached the report: counted in the pass column *and* omitted from failures, so
  a wrong answer was reported as correct and made invisible. Now rejected at the
  endpoints only; between them the verdict stays the scorer's own.
- **A Run could hold the same execution twice.** Duplicate
  `(case_id, repeat_index)` averaged both into the mean and read `1/2` for a
  single case.
- **A task returning a shared mutable object corrupted earlier Results.** All
  three stored outputs held the last value while their scores were right, so the
  report contradicted itself and the failure was undebuggable. Outputs are now
  snapshotted at capture, falling back to a repr for anything uncopyable.

Each fix carries a mutant that reintroduces the shipped behaviour: 11/11 killed.
Harness now at 116 mutants; 861 tests.

---

## Phase 5 — Storage, history, and the showcase example

**Goal:** runs persist and can be compared; there is a realistic example to demo.
**Estimate:** 14 hours.

- [x] **5.1 Design the SQLite schema** in `storage.py` with a `schema_version` table and sequential migrations under `src/evalstand/migrations/`:

  ```sql
  batches(id, kind, status, started_at, finished_at, git_sha, git_dirty)
  runs(id, batch_id, name, filepath, started_at, finished_at,
       model_config_json, repeat_n, total_cases, total_cost_usd, status)
  results(id, run_id, case_id, repeat_index, output_text, output_json,
          latency_ms, input_tokens, output_tokens, cost_usd, error)
  scores(id, result_id, scorer_name, value_float, passed, error, metadata_json)
  traces(id, result_id, parent_id, name, started_at, duration_ms,
         input_json, output_json, model, tokens_json, cost_usd)
  case_snapshots(run_id, case_id, content_hash, input_json, expected_json,
                 metadata_json)
  cache(key, model, evalstand_version, response_json, created_at, hit_count)
  ```
  Snapshot cases per run so history stays valid when the dataset changes later.

  Notes on the shape, each settled by a design decision:
  - `batches` is the invocation; `runs` is one Eval within it. `batches.kind` is
    `full` or `partial`, so a watch-mode re-run of three Evals is grouped rather
    than appearing as three unrelated executions. `batches.status` carries
    `cancelled` for a Batch a file change interrupted; `history` and `compare`
    exclude those.
  - `case_snapshots.content_hash` is `sha256(input, expected)`. It makes
    Amended Case detection an integer comparison instead of a JSON parse per
    case, and it covers `input` as well as `expected` — an edited input under
    the same `case_id` is also no longer the same test.
  - `scores.passed` is nullable and is set **only** by Scorers that genuinely
    know pass from fail. Nothing derives it from a threshold; a continuous
    scorer leaves it `NULL` rather than having the system invent a cutoff.
  - `traces.parent_id` is what makes a Result carry a tree rather than a list.
  - `cache.evalstand_version` invalidates entries when key canonicalisation
    changes, instead of silently mismatching.
  - `scores.error` holds a Scorer that raised. Such a Score is excluded from
    every mean rather than counted as zero — an infrastructure failure is not
    evidence the Task did badly — and any summary reporting a mean must also
    report how many Scores errored.

  **Migration direction.** An older database is migrated forward on open, inside
  a transaction. A database whose `schema_version` exceeds what the installed
  code understands is **refused** with an error naming both versions — never
  opened best-effort, because a silently degraded read would corrupt exactly the
  comparison data the tool exists to provide. This is unrelated to
  `cache.evalstand_version`: the Cache is disposable, run history is not.
  - *Acceptance:* the migration applies to an empty database and is idempotent; a second run does not corrupt the first; opening a database with a higher `schema_version` exits with a clear error. **— met 2026-09-05.**
  - Three SQLite behaviours were verified directly rather than assumed, each a trap that would corrupt history silently:
    - **Foreign keys are OFF by default.** Without `PRAGMA foreign_keys = ON` every `ON DELETE CASCADE` is decoration and an orphan row is accepted. The pragma is per-connection, so it is set on every open.
    - **`executescript` issues an implicit COMMIT**, ending the transaction wrapping a migration — a later failure then cannot roll back. Statements are executed one at a time instead. Verified: a script inside `BEGIN` left its tables behind and the ROLLBACK raised "cannot rollback - no transaction is active".
    - **`isolation_level=None`** is required, because Python's default opens a transaction before DML but not before DDL.
  - Storage never loses a run to a bad value: a cyclic output, or one whose `__repr__` raises, is stored as a description rather than taking down every other result in the save.
  - 15 mutants against the store, 15/15 killed — including the one bug that actually occurred (`Case.content_hash` is a method; storing it uncalled put a bound-method object in the hash column).
- [x] **5.2 Record provenance** on every run: git SHA, dirty-tree flag, model config, and a hash of the task source. Refuse to persist without a SHA unless `--allow-dirty` is passed. **— met 2026-09-05.**
  - **Absent is not the same as false.** A directory with no repository, or one with no commits yet, records a null SHA *and* a null dirty flag — "unknown", not "clean". Recording `False` there would assert a fact nobody established.
  - Not being in a repository is **not** refused: it is a normal way to try the tool, and the null SHA already tells a reader the run cannot be tied to a commit. Only a genuinely dirty tree is gated behind `--allow-dirty`.
  - **Untracked files do not count as dirty.** A scratch file or a `.env` is not a change to the code being measured, and counting it would have users passing `--allow-dirty` reflexively — which defeats the check.
  - The gate runs **before any case executes**. Discovering that results cannot be recorded after paying for them would be the worst possible ordering.
  - A storage failure never costs a measurement: by the time a run is written the money is spent, so a locked database degrades to one warning per batch. The single exception is `DatabaseTooNewError`, which stays a hard refusal.
  - New flags: `--no-store` and `--allow-dirty`.
  - 16 mutants, 16/16 killed.
- [x] **5.3 Build `evalstand history [name]`** listing runs with name, SHA, date, mean score, pass count, and cost. **— met 2026-09-05.**
  - **Whole Runs are loaded and the model's own properties compute every figure.** Aggregating in SQL would define "mean" a second time, in another language, and a history table quietly disagreeing with a fresh run about the same data is exactly the silent wrongness this project keeps finding. Measured at ~1ms for 50 runs, so there was no reason to trade correctness for speed.
  - The cost column distinguishes three claims that a single number would flatten: `$0.0002` (exact), `$0.0002+` (a lower bound — some calls were never priced), and `-` (nothing priced, so nothing has been shown to be free).
  - A dirty tree is marked `*` **and explained in a legend**: an asterisk with no legend is a puzzle, not information.
  - Cancelled batches are excluded; their aggregates describe a subset of the cases.
  - 15 mutants, 15/15 killed. Two were real gaps in the *read* path: `git_dirty=None` and an unpriced `cost_usd` were both preserved on write and silently coerced on read — a distinction that survives storage and dies on retrieval is no distinction at all.
- [x] **5.4 Build `evalstand show <run_id>`** rendering a full run: summary, per-case scores, and trace trees. **— met 2026-09-05.**
  - **The tree is built iteratively.** A recursive walk raises `RecursionError` at about a thousand levels — measured — and losing a whole run's display to a judge that called a judge would be a poor way to fail. Verified against a 3000-deep chain.
  - The Result validator already guarantees a forest (no cycles, no dangling parents, no duplicate ids), so the renderer trusts the *shape* and only has to survive its *size*. It must still handle several roots and children listed before their parents, both of which the validator permits.
  - **Prompts and completions are withheld by default**, shown as a size; `--full` prints them. One 4000-token prompt buries the tree it belongs to, and a trace input can hold a customer record nobody meant to display on a shared terminal.
  - Every figure comes from the same Run properties `history` and the live summary use, so the three views cannot disagree about one run.
  - 12 mutants, 12/12 killed — including one that reintroduces the recursive walk.
- [x] **5.5 Build `evalstand compare <run_a> <run_b>`.** Report per-scorer means for both runs, the delta, and the list of cases whose pass state flipped, with old and new output side by side.
  - **Important:** report the delta as a plain difference. Do **not** label it a regression or an improvement — this version has no significance testing, and asserting a verdict without one would be a false claim. Say "changed" and show the flipped cases. `docs/ci.md` must state this limitation explicitly.
  - *Acceptance:* comparing two runs shows the delta and flipped cases with no verdict language. **— met 2026-09-05.** Asserted mechanically rather than by eye: the rendered output is searched for every word CONTEXT.md tells us to avoid, in both directions — a tool that stays quiet about bad news but celebrates good news is not neutral.
  - **A Flip requires the expected value to be unchanged.** `case_snapshots.content_hash` is what makes that checkable; without it, a dataset edit is reported as evidence about the model. Verified directly: passing the hashes turns two apparent flips into one flip and one Amended Case.
  - Continuous scorers set no pass state, so a suite scored only by them has nothing to flip. Those cases appear under **"largest score changes"** — named for what happened, not for what it might mean.
  - The caveat is printed on **every** comparison rather than left in documentation: a reader who takes a delta as proof has been misled by the tool, and the tool is the only thing present at the moment they might do so.
  - Also warns when two runs are not comparable at all: different evals, a changed task source, or a different set of cases.
  - `docs/ci.md` states the limitation, and its claims are under test.
  - 16 mutants, 16/16 killed — three of them try to insert verdict language.
- [x] **5.6 Build the showcase example** at `examples/pdf_extraction/`: extract `invoice_number`, `vendor_name`, `invoice_date`, `total`, and `line_items[]` from documents.
  - Generate 30 synthetic invoices with `reportlab` + `faker` at a fixed seed. Ground truth is written at generation time and is therefore true by construction — programmatic, not LLM-generated.
  - Vary deliberately: multi-page documents, two currencies, a missing due date, an ambiguous date format.
  - Use `json_field` and `numeric_tolerance` scorers plus one judge scorer for line-item completeness.
  - *Acceptance:* `python generate.py --seed 42` reproduces byte-identical PDFs and golden JSON; the eval runs end to end from a clean clone with one API key set. **— met 2026-09-05.**
  - **A PDF embeds a creation timestamp by default**, so two runs of the same code produce different bytes and a "reproducible" corpus drifts every time anybody regenerates it. `invariant=1` pins it to a fixed epoch. Found by building the same document twice a second apart and comparing digests, not by reading documentation and hoping. `generate.py --check` regenerates and compares against the committed checksums; removing `invariant=1` fails two tests.
  - Ground truth is committed and reviewable; PDFs are gitignored and rebuilt on demand. So a dependency upgrade that changes the generator's output fails loudly rather than shifting the corpus underneath a committed baseline.
  - Ground truth is verified **internally consistent**: all 30 totals equal the sum of their line items, and every line amount equals quantity x unit price. Decimal throughout, because 0.1 + 0.2 is not 0.3 in binary floating point and a truth that disagrees with itself by a cent makes every extraction look wrong.
  - **A trap worth recording: `pypdfium2` is not thread-safe.** The runner offloads sync tasks to a thread pool, and eight threads in the same C library killed the interpreter with an access violation — not a Python exception the runner could record. `read_pdf` holds a lock, and the README says so, because any user with a C-backed library in their task will hit this.
  - Verified end to end against a mocked provider: 30/30 cases run, and a model that invents a plausible due date for the three invoices that have none scores 0.833 on those, with `json_fields` naming `due_date` as the wrong field.
- [~] **5.7 Commit baseline results** as `examples/pdf_extraction/BASELINE.md` with per-field accuracy, cost per document, and observed failure modes.
  - **Tooling complete; the measurement itself awaits an API key.** `baseline.py <run_id>` reads a stored run and writes the whole document — scorer means, per-field accuracy from `json_fields`' own metadata, cost per document, and every failing case with the field names that failed on it. Verified end to end against a mocked run.
  - **Generated, never typed.** A hand-written figure drifts from the code the moment either changes, and a baseline that disagrees with the tool is worse than none — it is the number people quote.
  - The committed `BASELINE.md` is a **placeholder that states no numbers**. Filling it with plausible figures would make it a measurement nobody made; a test asserts the placeholder carries no rate or cost in any table cell, and fails if someone adds one.
  - Two ways it refuses to be quietly wrong: a run whose calls all cost exactly the same is **labelled as coming from a mocked provider** (real completions vary in length and therefore price), and a run with no priced calls reports `-` rather than `$0.0000`.
  - Uses no verdict language, for the same reason `compare` does not — asserted by test.
  - **Remaining:** `pytest extraction_eval.py` with `OPENAI_API_KEY` set (~60 calls), then `python baseline.py <run_id>`.

**Exit criteria:** runs persist, history and comparison work, the showcase example runs from a clean clone. **— met 2026-09-05**, except the recorded baseline in 5.7, which needs an API key. Runs persist with provenance; `history`, `show` and `compare` all read them back; the example generates a byte-identical corpus and runs end to end.

---

### Pre-Phase-6 audit — 2026-09-05

A review before the TUI lands, on the layer where a defect now costs most:
before Phase 5 a wrong number was on screen and gone, and now it is written
down, quoted by three commands, and compared against months later. Three
defects, none caught by the 1058 tests passing at the time.

- **`compare` printed "nothing differs" while every case was crashing.** A flip
  needs a verdict on both sides and a move needs a value on both sides, so a
  case that errored fell through every branch — and `is_empty` then made an
  explicitly false statement. A user comparing a nightly run against a baseline
  would read it and move on while the eval was completely broken. Fixed with a
  fourth section, *cases whose measurement changed*, kept apart from flips
  because "was passing, now crashes" is not the task getting worse.
- **Stored results came back in the wrong order.** `runner.py` promises
  declaration order; storage read them `ORDER BY case_id`, so `q10` sorted
  before `q2` and a 30-case report could not be matched against its eval file.
  Fixed with a stored ordinal, migration 2, nullable so old databases still
  read.
- **`BatchRecorder.record()` silently discarded a mismatched run.** The
  never-lose-a-measurement policy was swallowing a programming error it was
  never meant to cover, leaving a batch row with no runs beneath it — the same
  shape as the 5.2 bug. Now raised.

12 mutants reintroducing these, 12/12 killed.

A fourth finding came from reading the harness's own score rather than the
code: one mutant reported **ANCHOR NOT FOUND** — it had been testing *nothing*
for four tasks, since a list comprehension in `plugin.py` became a `for` loop
when per-run recording landed. Worse, the report called it a survivor,
"behaviour no test asserts on", which sends a reader to write a test that
already exists. Stale anchors are now reported separately, fail the run, and
`test_mutation_harness.py` fails the moment a refactor moves a line any mutant
targets. The suite is only as trustworthy as the thing measuring it.

Harness at 202 mutants; 1089 tests.

---

## Phase 6 — TUI and watch mode

**Goal:** the live feedback loop that is the whole point of the tool.
**Estimate:** 16 hours.

- [x] **6.1 Run view** in `tui/app.py`: header with eval name and model, progress bar, a streaming table of cases (id, status, score, latency, cost) updating as results land, and a footer with running totals.
  - *Acceptance:* rows appear incrementally, not in one batch at the end.
- [x] **6.2 Summary panel:** per-scorer mean, pass count, total cost, wall time, cache hit rate.
- [x] **6.3 Case detail view:** press `enter` on a row for input, full output, expected, per-scorer breakdown with metadata, and the **trace tree** with per-node model, duration, tokens, and cost.
  - *Acceptance:* a task with nested LLM calls renders an expandable, navigable trace tree.
- [x] **6.4 History view:** browse past runs, select one to open, select two to render the Phase 5 comparison.
- [x] **6.5 Custom columns.** Support a `columns=` argument on `evaluate()` letting the user add derived columns to the results table.
  - The shape is a plain `dict[str, Callable[[Result], Any]]` — no new exported type, so this spends none of the three remaining public-API slots. Promoting to a structured `Column` type later is backward-compatible; retracting an exported type is not.
  - *Acceptance:* the showcase example adds a "fields correct" column.
- [x] **6.6 Watch mode** with `watchfiles`: re-run affected evals when an eval file, task file, or prompt file changes. Debounce 300ms. Preserve scroll position and show a "changed: <file>" indicator.
  - **A change during an in-flight Batch cancels it.** Waiting for a slow Batch would waste the feedback loop this feature exists to provide.
  - Cancellation must be honest: the Batch gets a terminal `cancelled` status, `history` hides it by default, and `compare` refuses it. A half-finished Batch that looked complete would drag every mean it touched.
  - In-flight model calls are allowed to finish and land in the Cache rather than being hard-killed. The money is already spent; discarding the response wastes it, and the next Batch will want it.
  - *Acceptance:* editing a prompt triggers a re-run within one second without restarting the process; a cancelled Batch never appears in `history` or `compare`.
- [x] **6.7 Keybindings:** `q` quit, `r` re-run, `f` filter to failures, `c` compare with previous run, `/` search, `y` copy case id.
- [ ] **6.8 Record a demo GIF** with `vhs` or `asciinema` + `agg`, embedded at the top of the README.
  - *Acceptance:* under 5 MB, showing a full run, a trace tree, and watch-mode re-run in under 30 seconds.

**Exit criteria:** the TUI satisfies every UI row in the Section 2 table. The README opens with a GIF that makes the value obvious in five seconds.

---

## Phase 7 — CI integration, docs, and release

**Goal:** installable, usable in a pipeline, and understandable.
**Estimate:** 12 hours.

- [x] **7.1 CI flags:** `--threshold <float>` (fail when the mean score falls below it) and `--fail-on-error`. Documented exit codes: `0` pass, `1` below threshold, `2` execution error.
  - The Threshold is **absolute**, and the number is a human decision taken from the committed Baseline (5.7) — never computed from the most recent Run, which would let the bar drift down every time quality dropped.
  - **The Threshold applies per Eval, never to a Batch mean.** Evals measure different things; averaging summarisation quality with extraction accuracy produces a number with no meaning, and lets a collapse in one Eval hide behind another's strength. A Batch fails if any Eval falls below the bar, and the output names which.
  - An Eval whose file fails to import fails that Eval and is reported; the other Runs in the Batch still execute and persist, consistent with 3.1.
  - Because the database is project-local and gitignored (ADR 0005), CI starts with no history. `--threshold` therefore works on an empty database, while `compare` correctly reports it has nothing to compare and exits 2 rather than falsely passing.
  - A Run whose Scores errored reports the mean over the Scores that succeeded, together with the errored count. `--fail-on-error` is what turns those into a failure; the Threshold alone must not silently pass a Run that scored 3 of 30 cases.
  - *Acceptance:* an exit-code table in the docs, each code reproducible in a test, including the empty-database case.
- [x] **7.2 Markdown summary output** in `reporting/markdown.py` — `--output markdown` produces a body suitable for a PR comment: summary table, failed cases, cost.
- [x] **7.3 Document the GitHub Actions recipe** in `docs/ci.md`: a workflow that runs evals on pull requests, posts the markdown summary as a comment, and gates on the threshold. Ship it as a copyable YAML block rather than a published Action in this version.
- [x] **7.4 Docs site** with mkdocs-material: quickstart, writing evals, scorers, traces, CI, architecture, ADR index. Deploy to GitHub Pages.
- [x] **7.5 Rewrite the README:** one-sentence problem statement, demo GIF, 60-second quickstart, feature list mapped to the Section 2 parity table, reproducible numbers from the showcase example, an honest **Limitations** section, the attribution footnote, licence.
  - The Limitations section is required. State plainly that score deltas are reported without significance testing and that LLM judge scorers are unvalidated in this version.
- [ ] **7.6 Publish to PyPI** via a tagged `release.yml` using trusted publishing. Tag `v1.0.0`.
- [ ] **7.7 Record a 3-minute demo video:** write an eval, run it in watch mode, edit the prompt, watch the re-run, open a trace tree, break something and see the threshold gate fail in CI.

**Exit criteria:** installable from PyPI, docs live, video recorded, parity table fully satisfied.

---

## Phase 8 — Optional stretch: web UI

Start only when Phases 0–7 are complete and polished. A shipped TUI beats a half-finished web app.

- [ ] **8.1** FastAPI backend exposing runs, results, scores, and traces as JSON.
- [ ] **8.2** HTMX plus server-sent events front end for the live run table, keeping the stack entirely Python.
- [ ] **8.3** Static HTML report export for CI artifacts. Worth doing even without the full web UI.
- [ ] **8.4** `evalstand serve` command.

---

## 8. Parked for a future version

Recorded here so the ideas are not lost, and so the agent does not build them now. **Do not start any of these until Phase 7 is complete and released.**

- **Statistical significance testing.** Bootstrap confidence intervals on aggregate scores and McNemar's exact test for paired run-over-run comparisons, so `compare` can distinguish a real change from sampling noise instead of reporting a bare delta. This is the single most valuable extension and the natural v2.
- **Judge calibration.** Measuring an LLM judge against human labels — Cohen's kappa, true and false positive rates reported separately, position bias, verbosity bias, self-preference bias — and refusing to run an uncalibrated judge.
- **Dataset hygiene.** Near-duplicate detection within an eval set and overlap detection against few-shot examples.
- **A published GitHub Action** rather than a copyable workflow.

---

## 9. Definition of done

A phase is complete only when all of the following hold:

1. `ruff check`, `ruff format --check`, `mypy`, and `pytest` pass locally and in CI.
2. Test coverage for new modules is at or above 80%.
3. CI runs with no API keys set.
4. Every new design decision has an ADR.
5. Public API changes are reflected in the docs.
6. `CHANGELOG.md` is updated.
7. The Section 2 parity table is updated to reflect what now works.

---

## 10. Risk register

| Risk | Mitigation |
| --- | --- |
| API costs during development | Cache aggressively from Phase 1; develop against the cheapest available model; set a monthly budget alert. Expect $20–60 total for this project. |
| Tracing turns out to be the hard part | It is. Budget the full Phase 3 allocation for it and build it against the toy example before the showcase example exists. `contextvars` behaviour under `asyncio.gather` is the specific thing to test early. |
| pytest plugin fights the runner's async model | Prototype collection and async dispatch together in Phase 2 rather than sequentially. If the plugin proves intractable, fall back to a standalone runner and keep pytest compatibility as a stretch — but record it as an ADR, since parity item 19 would be waived. |
| Scope creep into the parked v2 features | Section 8 exists for exactly this. Anything in it goes back in the parking lot. |
| TUI consumes the whole budget | Timebox Phase 6 to 16 hours. If it overruns, ship console reporting only and move the TUI to a stretch phase. |
| The tool ends up a shallow imitation | Parity item 7 (traces) and item 10 (repeats) are the two features that are genuinely hard. If those work well, the tool is real. If they are skipped, it is a wrapper. |

# Scorers

A **Scorer** judges one output and returns a [Score](#what-a-score-means): a
value in `[0, 1]`, optional metadata, and an optional pass flag.

`evalstand` ships ten. You will also write your own — that is the normal case,
not the advanced one, and it takes three lines.

```python
from evalstand import Case, evaluate
from evalstand.scorers import levenshtein


def mentions_the_year(output, expected):
    return "1815" in output


evaluate(
    name="biography",
    cases=[Case(id="q1", input="When was Ada Lovelace born?", expected="1815")],
    task=answer,
    scorers=[levenshtein, mentions_the_year],
)
```

## Writing your own

A scorer is a plain function. No base class, no decorator, no registration.

**Take only the arguments you need.** Scorers are called with what they declare:

```python
def nonempty(output):  # the output alone
    return bool(output.strip())


def matches(output, expected):  # against the Case's reference
    return output == expected


def weighted(output, expected, case):  # and the Case itself
    return 1.0 if case.metadata["easy"] else 0.5
```

`case` is how a scorer reaches per-case data — a weight, a rubric, a schema —
which is the reason the third argument exists.

**Return whatever fits.** Each return type means something specific:

| You return | `value` | `passed` |
|---|---|---|
| `True` / `False` | 1.0 / 0.0 | `True` / `False` |
| a float in `[0, 1]` | that float | not set |
| a `Score` | used exactly as given | |

The distinction matters. A bool is an unambiguous claim of pass from fail, so
it sets the flag. A float is a position on a scale — 0.62 does not say whether
0.62 is good enough, because only you know where the line sits for your task.
Nothing else in `evalstand` derives that flag.

Return a `Score` when you want metadata:

```python
from evalstand.models import Score


def with_reasons(output, expected):
    missing = [w for w in expected.split() if w not in output]
    return Score(
        scorer_name="coverage",
        value=1 - len(missing) / len(expected.split()),
        metadata={"missing_words": missing},
    )
```

**Async works.** A scorer that calls a model is a normal scorer:

```python
from evalstand import llm


async def graded(output, expected):
    reply = await llm.acall("gpt-4o-mini", [{"role": "user", "content": f"..."}])
    return 1.0 if reply.text.strip() == "yes" else 0.0
```

Calls made through `evalstand.llm` inside a scorer are traced, cached and costed
exactly like the task's own calls, and appear in the same trace tree.

**A scorer that raises does not fail the case.** The exception is recorded on the
Score and that Score is excluded from the mean. A broken scorer is not evidence
the task did badly, and counting it as zero would say that it was.

The optional `@scorer` decorator changes no behaviour. It validates the
signature at import, so a scorer that could never be called fails before a run
spends money:

```python
from evalstand.scorers import scorer


@scorer
def graded(output, expected): ...
```

## The built-in scorers

### Text

| Scorer | Matches when | Sets `passed` |
|---|---|---|
| `exact(output, expected)` | the values are equal, with no coercion | yes |
| `normalised_exact(output, expected)` | equal after folding case, whitespace and presentation punctuation | yes |
| `contains(output, expected)` | the expected text appears anywhere in the output, normalised | yes |
| `regex_match(pattern, *, full=False)` | the pattern matches | yes |

`exact` is strict on purpose: `1` and `"1"` are different, and `"Paris"` and
`"paris"` are different. It is the scorer for when you mean *exactly*. For
everything else you probably want `normalised_exact`.

Normalisation folds case, collapses whitespace, unifies Unicode forms (so a
composed `é` matches `e` plus a combining accent), and trims presentation
punctuation from the edges of words — full stops, commas, quotes, brackets,
dashes.

It deliberately does **not** strip everything non-alphanumeric. That rule reads
sensibly and behaves badly: it would make `-5` match `5`, `$100` match `100%`,
and `C++` match `C`. Each is a wrong answer scored as a perfect match. Anything
not on the trimmed list is treated as part of the answer.

```python
normalised_exact("  PARIS.  ", "Paris")  # 1.0, passed=True
normalised_exact("-5", "5")  # 0.0, passed=False
```

`contains` with an empty expected value scores **0.0**, not 1.0. Every string
contains the empty string, and the vacuous truth would silently mark a whole
suite as passing.

`regex_match` is a factory because the pattern belongs to the check, not to the
Case. Use `full=True` for "the whole output matches" — a bare search for `\d+`
is satisfied by any output containing a digit anywhere.

### Fuzzy

| Scorer | Reports | Sets `passed` |
|---|---|---|
| `levenshtein(output, expected)` | `1 - distance / max(len)` | **no** |
| `ratio(output, expected)` | indel similarity, rescaled to `[0, 1]` | **no** |

`levenshtein` is the default scorer worth reaching for first: a nearly-right
answer scores nearly 1.0 rather than falling off the cliff `exact` presents.
`levenshtein("kitten", "sitting")` is 3 edits over 7 characters — **0.5714**.

Neither sets `passed`. They report where an answer sits on a scale; deciding
which end of that scale counts as success is a judgement about your task.

### Numeric

```python
from evalstand.scorers import close_to

scorers = [close_to(rel_tol=0.01)]  # within 1%
scorers = [close_to(abs_tol=0.5)]  # within half a unit
scorers = [close_to()]  # exactly equal
```

`close_to` finds the number in an answer, so `"The widget costs $19.99."` and
`"about 1,000"` both work. Both tolerances default to zero, so a forgotten
tolerance means an exact match rather than silent slack.

Use `abs_tol` when the expected value can be zero. Any percentage of zero is
zero, so a relative tolerance around 0 admits nothing but an exact 0.

**An ambiguous output is unreadable, not a guess.** `"42 or 43"`,
`"between 10 and 20"` and `"5 + 5"` all score 0.0 with a reason in metadata,
rather than picking a candidate. A score nobody can audit is worse than an
honest failure to read one. `nan` and `inf` are refused for the same reason,
and so is `True` — `float(True)` is 1.0, which is a fact about Python rather
than an answer.

### Structured

```python
from evalstand.scorers import json_fields

scorers = [json_fields()]  # macro-average over fields
scorers = [json_fields(require_all=True)]  # and a pass/fail verdict
```

Compares an answer object against the expected one field by field, and reports
the fraction that matched. An extraction task rarely gets everything right or
everything wrong, and a single pass/fail throws away the only information worth
having: *which* field was wrong.

```python
json_fields()({"name": "Ada", "born": 1816}, {"name": "Ada", "born": 1815})
# value 0.5, metadata {"wrong": ["born"], "matched": ["name"], ...}
```

- **JSON as text is parsed**, including ```` ```json ```` fences. Refusing to
  unwrap them would measure presentation rather than content.
- **Nested objects flatten to dotted paths.** One wrong leaf out of ten costs a
  tenth, and the metadata names `person.born` rather than `person`.
- **The denominator is the expected keys.** A field the model volunteered is
  reported in `unexpected` but never counted, or a verbose model would score
  worse than a terse one that answered exactly as badly.
- **Missing is reported separately from wrong.** Both score zero, but a wrong
  field means the model answered and erred, while a missing one means it never
  answered at all — different problems, different fixes.

### LLM judges

!!! warning "These scorers are unvalidated"

    `judge` and `factuality` have **not been calibrated against human labels**.
    A judge score is a second model's opinion, not a measurement, and nothing in
    this project establishes that it agrees with a human reading the same
    answers.

    Every judge Score carries `unvalidated: True` in its metadata, so the caveat
    travels with the number into any report or database row that quotes it. Use
    them to notice changes worth investigating, not to establish that a system
    is correct.

```python
from evalstand.scorers import factuality, judge

scorers = [factuality()]

scorers = [
    judge(
        rubric="Does the answer use a professional tone?",
        choices={"A": 1.0, "B": 0.5, "C": 0.0},
        include_expected=False,
    )
]
```

The judge picks a **labelled choice**, never a number. Models cluster on 0.0,
0.5 and 1.0 and cannot justify 0.7 over 0.8, but they choose reliably between
described options. You supply what each label is worth, so the mapping from
judgement to score is made once in your eval file rather than improvised by the
model on every case.

`include_expected=False` withholds the reference answer, for rubrics that judge
the output alone — tone, safety, format — where showing a reference invites the
judge to compare against it instead of applying the rubric.

**A reply naming no known choice, or two, is an errored Score** — excluded from
the mean, never scored 0.0. Scoring zero would report that the task did badly
when what actually failed was the judge.

`factuality()` asks whether the answer *contradicts* the reference. An answer
that says less than the reference, or more, still scores 1.0; only a genuine
disagreement scores 0. Penalising a correct answer for being differently
detailed measures verbosity rather than truth.

Judge calls go through `evalstand.llm`, so they are cached, retried, traced and
costed like any other call. **A judge's request appears in the case's trace tree
and its tokens count towards the case total** — which is what stops
LLM-as-judge from being an invisible line on your bill.

## What a Score means

```python
Score(
    scorer_name="levenshtein",  # how the report groups it
    value=0.57,  # in [0, 1]
    passed=None,  # only when the scorer genuinely knows
    metadata={"distance": 3},  # anything worth reading later
)
```

Two rules are worth stating plainly, because the rest of the system depends on
them:

**An errored Score is excluded from the mean, not counted as zero.** A scorer
that broke — a rate-limited judge, a malformed rubric — is not evidence the task
did badly. A run where every scorer errored reports *no measurement*, and fails,
rather than reporting a confident zero.

**`passed` is set only by the scorer, and only when it genuinely knows.** Nothing
derives it from a value, and an absent flag stays absent. A case whose scorers
set no flag is reported as unjudged rather than as a pass — a green tick for a
case nobody graded is the most misleading thing an eval tool can produce.

**A Score cannot contradict itself.** `passed=True` with a value of 0.0 is
rejected, as is `passed=False` with 1.0. Between the endpoints the verdict is
the scorer's own: it may pass at 0.8 or fail at 0.3, and nothing second-guesses
that.

## Failing a run

Continuous scorers never set `passed`, so on their own they can never fail a
run — a model answering every case badly would still exit zero. Two things
prevent that being silent:

- Any case scoring **0.00** is listed under *needs attention*, described as
  "scored 0.00" rather than "failed", because the scorer gave no verdict.
- `--threshold` fails the run when an eval's mean falls below a value **you**
  choose:

```bash
pytest qa_eval.py --threshold 0.7      # exit 1 if the mean drops below 0.7
evalstand run qa_eval.py --threshold 0.7
```

It is opt-in because `evalstand` will not invent a pass mark, and it judges the
aggregate only — it never marks an individual case as failed, since no scorer
said it was. A run whose scorers all errored has no mean, and is reported as
unmeasured rather than as below the threshold.

## Choosing one

- Comparing against a known answer, and formatting should not matter?
  **`normalised_exact`**, or **`levenshtein`** if you want partial credit.
- The answer is embedded in a sentence? **`contains`**.
- A number? **`close_to`**, with a tolerance you chose deliberately.
- Structured extraction? **`json_fields`**.
- A quality no string comparison captures — tone, reasoning, faithfulness?
  **`judge`**, with the caveat above firmly in mind.
- Something specific to your task? **Write one.** It is three lines, and it will
  be a better measurement than anything generic.

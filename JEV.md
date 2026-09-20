# Evaluating TypeSafe's Jev as a scorer backend

Investigated 2026-09-20. **Recommendation: do not build.** Not deferred to a
named phase — deferred until the evidence that would justify it exists, which is
a calibration study rather than a model swap.

This file records the reasoning so the decision can be revisited without
re-running the investigation. It is not an ADR: no code changed, so there is no
decision in the codebase to record. If Jev is ever adopted, that adoption needs
its own ADR.

## What Jev is

A "System One" model from TypeSafe. It does not generate text. You send a
`state` plus a map of typed `questions`; it returns one typed answer per
question, evaluated in parallel against that state in a single pass. Code keeps
control flow; the model supplies bounded semantic judgment.

Three question primitives, verified against
<https://docs.typesafe.ai/primitives.md>:

| Type | Answers | Returns |
| --- | --- | --- |
| Choice | Which of these options? | `choice`, `probabilities`, `confidence` |
| Score | Which level? | `score`, `legend`, `probabilities`, `confidence` |
| Noul | Is this true? | `noul` (0 to 1) |

Pricing is $0.042 per million input tokens with output tokens free. Model ID
`jev-1.13.0`; aliases move, so a pinned version matters if thresholds are tuned
against it.

## Why it looked worth evaluating

`src/evalstand/scorers/llm.py` already implements Jev's core idea without Jev.
Its docstring says the judge asks the model to pick a **labelled choice** rather
than emit a number, because models cluster on 0.0/0.5/1.0 and cannot justify 0.7
over 0.8, and that the label-to-score mapping belongs to the caller. That is a
Choice question implemented by prompt and convention.

Jev would make the same contract structural rather than prompted. The question
was whether structural is better here.

## What could not be verified

Two claims in the brief did not hold in this environment, and both blocked the
live experiments:

- **`TYPESAFE_API_KEY` was not set.** Absent from Bash, from PowerShell, and
  from the persistent User scope. No live call was made.
- **`typesafe-sdk` was not installed** in this project's venv.

The docs *were* reachable and were read directly, so the API shape, the
primitives table, the fan-out claim and the jaggedness page below are verified
at source rather than taken from the brief.

Everything in the recommendation rests on measurements of **this** codebase,
which needed no key.

## The finding that settles it

The strongest argument for adopting Jev was that an out-of-schema answer becomes
unrepresentable, making the "a reply that names no known choice is an errored
Score" path unreachable. The brief itself said to measure how often that path
fires, and that if it is already near zero the argument is worth nothing.

**It is near zero.** `_read_choice` in `scorers/llm.py` is deliberately
forgiving about packaging and strict about content — it finds any label as a
standalone token, case-insensitively:

```
realistic replies parsed:   20/20
pathological refused:        7/7
```

Parsed: `A`, `A.`, `(B)`, `Answer: C`, `answer: d`, `**E**`,
`The answer is A.`, `I would say B.`, ```` ```\nC\n``` ````, `Choice: D`, and
the rest. The error path fires only when a reply names **zero** known labels or
**two or more**.

### Adopting Jev would remove a safety property, not add one

This is the part that turns a neutral trade into a regression.

A reply of "Either A or B could apply" is refused today, deliberately, for the
same reason `parse_number` refuses `"42 or 43"`: picking one invents a verdict
the judge never committed to. The result is an errored Score, which is
**excluded from the mean** rather than scored 0.0.

A Choice constrained to its schema cannot produce that outcome. It must return a
label. An ambiguous judgement therefore arrives as a confident one, and the
distinction between "the judge was unreadable" and "the judge decided" is lost —
silently, in a number that looks exactly like a real measurement.

That is the false-pass shape this project exists to hunt.

## On using the probability as the score

Jev returns a probability for every option, so a score could be the probability
itself rather than the caller's label-to-float bucket. Rejected as **false
precision**.

A probability of 0.73 for choice A is the model's confidence in a *label*. It is
not a claim that the answer is 0.73 correct. What a label is worth is a
judgement about the task, made once by the eval author and visible in the eval
file. Substituting confidence-in-a-label for value-of-a-label conflates two
different quantities — and with calibration unvalidated on our data, it puts a
number nobody chose into a database row that will be quoted later.

## The batching decision

Framed in the brief as the substance of the task. It only bites if a Jev scorer
is worth building, but recorded here so it need not be re-derived.

**Accept N calls, one per scorer.** Both alternatives were rejected:

- **One scorer returning several Scores** breaks the protocol.
  `ScorerResult = Score | float | bool` in `scorers/base.py`; widening it to
  `list[Score]` touches the runner, `pass_counts`, all four renderers and the
  storage schema. A core change to serve an optional extra.
- **A batching layer** would require the runner to know which scorers are
  TypeSafe-backed and hold them until all are collected. That inverts the "a
  plain function is a scorer" promise — the runner would have to introspect
  scorer *provenance*, which nothing currently does.
- **N calls** cost roughly 100ms and fractions of a cent each. The fan-out
  saving is real but buys nothing here: the binding constraint on this project
  is honesty of the number, not latency of the judgement.

## Where Jev must never be used here

From <https://docs.typesafe.ai/model-jaggedness/jev-1.13.md>, mapped against the
ten shipped scorers. **Eight must stay deterministic, and it is not close:**

| Scorer | Why it stays in code |
| --- | --- |
| `close_to` | Jaggedness #2 — "Jev is not a calculator" |
| `levenshtein`, `ratio` | #2 counting — "does not count reliably" |
| `exact`, `normalised_exact`, `contains` | Exact comparison; a model adds only the chance of being wrong |
| `regex_match` | Deterministic by definition |
| `json_fields` | #2 counting — the denominator is a count of keys |

Only `judge` and `factuality` were ever candidates.

The other documented failure modes that would apply if this is revisited:
literal reading, date comparison as text, multi-hop indirection, accuracy
falling as irrelevant state grows, adversarial content treated as data, and no
structural invariants — the same question as a Noul and as a Choice may
disagree, so a threshold must never be carried across types.

## The pdf_extraction experiment

The brief called this the highest-value first experiment for evaluating Jev as a
scorer. It is not, and the conflation matters.

**If Jev extracts invoice fields, Jev is the system under test.** The scorer
would still be `json_fields`, deterministic as it is today. That experiment
measures Jev's *extraction accuracy* — genuinely interesting — but says nothing
about whether Jev should back a scorer.

It is, separately, a good candidate for **gate 5.7**, the unrecorded showcase
baseline. The TypeSafe cookbook shape — find candidates in code with a parser or
regex, then use a question to *select* the intended one, never to produce the
value — matches an invoice corpus closely, and `examples/pdf_extraction/` is a
ready-made labelled corpus with ground truth written at generation time. If it
is used there, `BASELINE.md` must name which model produced the numbers.

## What would change this answer

A measured **disagreement rate between `judge` and human labels on real data**,
and the same measurement for Jev.

That is a calibration study, and it is the work the `unvalidated: True` marker
has been waiting for since Phase 4. Without it, replacing one uncalibrated judge
with another uncalibrated judge is motion rather than progress — and it is worth
noting that TypeSafe's calibration claim is *their* calibration, not ours, on
data that is not ours.

## Limitations paragraph, ready to paste into README.md

Holds whether or not Jev is ever adopted:

```markdown
**Judge scorers use a general model, not a constrained-output one.**
`judge` and `factuality` ask a model to name a labelled choice and parse the
reply. Purpose-built classification models return the choice as a typed value,
which removes the parse — but the parse is not where this fails. A reply naming
no known label, or two, becomes an errored Score excluded from the mean, and a
constrained model cannot produce that outcome: it must return a label, so an
ambiguous judgement arrives as a confident one. Nothing here is calibrated
against human labels either way, so the constraint would buy type safety at the
cost of a verdict nobody reached.
```

## If this is revisited

Re-verify everything above. The docs said "last reviewed 2026-09-17" on the
jaggedness page when this was written, the model was `jev-1.13.0`, and the brief
noted the API's rate limits adjust without notice. The measurement of
`_read_choice` should also be re-run — if that parser ever becomes stricter, the
first argument in this file stops being dead.

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

## What was verified, and when

The recommendation below was reached on 2026-09-20 **without** a live call: the
key was not set in any scope and the SDK was not installed. It rests entirely on
measurements of *this* codebase plus the docs, which were reachable and were
read at source.

The key arrived on 2026-09-22 and the extraction experiment at the end of this
file is live, against `jev-1.13.0`. It confirmed the API shape from the docs
exactly — `usage.input_tokens` reported, the pinned model ID returned, and
`choice` + `probabilities` + `confidence` on every Choice answer.

**Nothing in the live run changes the recommendation**, because it measures a
different role: extraction is a task, not a scorer.

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

## The pdf_extraction experiment — run 2026-09-22

The brief called this the highest-value first experiment for evaluating Jev as a
scorer. It is not, and the conflation matters: **if Jev extracts invoice fields,
Jev is the system under test.** The scorer would still be `json_fields`,
deterministic as it is today. This measures Jev's *extraction accuracy*, which
says nothing about whether Jev should back a scorer — and the recommendation
above is unchanged by it.

As an extraction task, it worked.

### Results

Thirty invoices, six scalar fields each, `jev-1.13.0`, one request per invoice:

| | |
| --- | --- |
| Fields correct | **180/180** |
| Invoices fully correct | **30/30** |
| Input tokens | 35,597 |
| Cost | **$0.001495** exact ($0.0000498/invoice) |
| Wall time | 10.7s (0.36s/invoice) |

Per field: `invoice_number`, `vendor_name`, `invoice_date`, `due_date`,
`currency`, `total` — all 30/30.

The corpus's three deliberate traps were all handled:

- **Three invoices state no due date.** The honest answer is null and an
  invented plausible date would look right. Answered `none` all three times, at
  confidence 1.0, 1.0 and 0.99.
- **Three print an ambiguous `DD/MM/YYYY` date.** See below.
- **Multi-page invoices put the total on page two.** No total was confused with
  a line item or a unit price.

### The one interesting failure, and whose fault it was

The first run scored **177/180**. All three misses were the same field on the
three ambiguous-date invoices: Jev answered `02/10/2018` where ground truth
says `2018-10-02`.

**Jev was right and the code was wrong.** The document prints `02/10/2018`;
Jev picked that span verbatim, which is exactly what the cookbook promises — it
selects among spans a regex found and cannot invent or transpose a value. What
was missing was step 3 of the cookbook: *code copies the picked value and
normalises it*. There was no `DD/MM/YYYY` → ISO conversion.

Adding six lines of date normalisation took it to 180/180. This is the
jaggedness page's own advice working as documented: Jev reads dates as text,
so extract the components and compare in code.

### Method

Following `cookbooks/pre_parsed_value_extraction_cookbook.md`:

1. A regex tuned to over-find collects candidate spans — money, dates, invoice
   numbers — plus short header lines as vendor candidates.
2. **One** `system_one` request per invoice carries all six Choice questions
   against the same state. They are independent, so they run in parallel; this
   is the fan-out pattern, and it is why thirty invoices cost a tenth of a cent.
3. Every question includes a `none` option, so "no candidate fits" is
   representable rather than forced.
4. Code copies the picked span and normalises it.

### Cost is exact here, not a lower bound

Verified against a live response: `usage.input_tokens` is reported, and pricing
is $0.042 per million input tokens with output free. So a Jev call's cost is
computable exactly, unlike a LiteLLM call whose model may be missing from the
pricing table — which is the case `total_cost_usd` reports as a lower bound.

### Confidence behaved as documented

`vendor_name` was the only field where confidence varied meaningfully: mean
0.94, minimum 0.31. **Every low-confidence answer was still correct**, which is
the point the docs make about confidence being distribution concentration
rather than a correctness probability — several header lines are plausibly a
company name, so the probability spreads without the pick being wrong.

### What this does and does not license

It does **not** revisit the recommendation above. A scorer and a task are
different roles, and nothing here measures agreement with a human on a
judgement call — these are verbatim spans checked against ground truth that is
true by construction.

It does make Jev a credible candidate for **gate 5.7**, the unrecorded showcase
baseline. If used there, `BASELINE.md` must name `jev-1.13.0` as the model that
produced the numbers, and should say that this task is span selection rather
than free generation — a reader comparing it against a generative model's
baseline is comparing two different methods.

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

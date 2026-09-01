# evalstand

A local-first LLM evaluation tool for Python: you write evals, run them like
tests, and watch results stream into a terminal UI with scores, traces, and cost.

## Language

### Authoring

These are the three things a user writes.

**Case**:
One test input, with an optional reference output. Identified by `id` within
its Eval.
_Avoid_: test, sample, example, row, datapoint

**Task**:
The user's function under test. Receives a Case's input and returns an output.
The LLM call happens inside it.
_Avoid_: system under test, target, subject, function

**Scorer**:
A callable that judges one output and returns a Score. Some Scorers consult the
Case's expected value; others judge the output alone.
_Avoid_: metric, grader, evaluator, check

### Reading

These are what a user inspects after execution.

**Score**:
One Scorer's judgement of one Result, carrying a value in `[0, 1]`, metadata,
and an optional pass flag. A Scorer sets that flag only when it genuinely knows
pass from fail; nothing else derives it, and an absent flag stays absent.
_Avoid_: grade, rating, mark

**Threshold**:
A minimum mean Score, supplied at the command line, that decides whether a Batch
passes or fails. It judges aggregates only — it never sets a Score's pass flag.
_Avoid_: cutoff, bar, gate, minimum

**Result**:
What happened when one Case met the Task once: the output produced, plus
latency, tokens, cost, and any error. Scores and Traces hang off it. With
repeats, one Case produces several Results in the same Run, distinguished by
`repeat_index`.
_Avoid_: outcome, attempt, trial, execution

**Trace**:
A record of one LLM call made inside a Task, capturing its input, output,
model, duration, tokens, and cost. Traces nest: a call made inside another call
is its child, so a Result carries a tree rather than a list.
_Avoid_: span, event, log, step

### Structure

**Eval**:
One `evaluate()` call, identified by its name. A single file may declare
several. The unit that `history` and `compare` operate on.
_Avoid_: suite, experiment, benchmark, test

**Eval file**:
A `*_eval.py` module. Informal term for where Evals are declared — it is not
itself an addressable concept.

**Run**:
One execution of one Eval, covering all its Cases. Persisted, immutable, and
comparable to earlier Runs of the same Eval.
_Avoid_: session, job, execution, invocation

**Batch**:
One invocation of the tool, containing every Run it produced. Either *full*
(everything collected) or *partial* (only the Evals a file change affected).
Exists so watch-mode re-runs are grouped rather than orphaned.
_Avoid_: run, session, job, sweep

**Variant**:
A named alternative configuration — a different prompt or model — that the same
Cases are run against for comparison. **Not implemented in v1** (see ADR 0003);
recorded here so the term is not reused for something else.

**Ground truth**:
A provenance property, not a thing: a value is ground truth when a human wrote
it or a program produced it. Never when a language model generated it. A Case's
expected value must be ground truth when present.
_Avoid_: golden answer, label, reference truth

**Expected**:
A Case's optional reference output. Meaningful only to Scorers that consult it;
its absence does not make a Case invalid.
_Avoid_: correct answer, gold, target, truth

### Comparison

**Flip**:
A Case whose pass state differs between two Runs *while its expected value
stayed the same* — so the change is evidence about the Task, not about the
dataset.
_Avoid_: regression, improvement, change (all imply a verdict we cannot support)

**Amended Case**:
A Case whose expected value was edited between two Runs. Reported separately
from Flips, because a pass state that moved because the expectation moved says
nothing about the Task.
_Avoid_: changed case, edited case, dirty case

**Delta**:
The plain arithmetic difference between two Runs' mean Scores. It carries no
verdict: `evalstand` has no significance testing, so a Delta is never called a
regression or an improvement.
_Avoid_: regression, improvement, gain, drop

### Determinism

**Cache**:
A store of previous model responses, keyed on the exact call, that exists purely
to avoid paying twice for the same request. Always safe to delete; never
load-bearing for correctness.
_Avoid_: store, memo, replay

**Cassette**:
A committed recording of model responses that lets the test suite run offline
and deterministically. Distinct from the Cache: deleting a Cassette breaks the
tests, deleting the Cache costs only money.
_Avoid_: fixture, mock, recording, snapshot

### Identity and lifecycle

**Case ID**:
The name that makes a Case the same Case across Runs. Supplied by the user, or
assigned by enumeration order when Cases arrive as a bare iterable. Never
derived from content — a content-derived ID would make an edited Case look like
a different Case, which is exactly what Amended Case detection must catch.
_Avoid_: key, index, position, slug

**Case Snapshot**:
The `(input, expected)` a Case had at the moment a Run executed, stored with
that Run so history stays meaningful after the dataset is edited.
_Avoid_: copy, version, frozen case

**Registry**:
The set of Evals a module declared when it was imported. `evaluate()` writes to
it and returns; nothing executes until a runner reads it. This is what lets a
tool enumerate Evals without running them.
_Avoid_: collection, catalogue, manifest

**Eval name**:
An Eval's identity, unique across the project. `history` and `compare` address
Evals by it, so two Evals sharing a name is an error raised at collection time,
not a silent merge. A file path is metadata, never identity — renaming a file
preserves its Eval's history.
_Avoid_: id, key, label, title

**Baseline**:
A committed, human-chosen record of what acceptable performance looks like,
against which a Threshold is set. Chosen by a person reading results — never
computed from the most recent Run, which would let the bar drift downward every
time quality dropped.
_Avoid_: benchmark, target, reference run

**Scorer error**:
A Scorer that raised instead of returning a Score. Recorded on the Score and
excluded from means, never conflated with a low Score: an infrastructure failure
is not evidence that the Task did badly. Any reported mean must also report how
many Scores errored.
_Avoid_: failure, bad score, null score

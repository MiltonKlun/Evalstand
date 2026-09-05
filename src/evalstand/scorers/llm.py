"""LLM-as-judge scorers.

**These scorers are unvalidated.** They have not been calibrated against human
labels, so a judge score is a second model's opinion, not a measurement. Every
Score they produce carries `unvalidated: True` in its metadata, so the caveat
travels with the number into any report or database row that quotes it — a
caveat separated from its figure stops being a caveat.

The judge asks the model to pick a **labelled choice**, not to emit a number.
Models are poor at producing calibrated floats — they cluster on 0.0, 0.5 and
1.0 and cannot justify 0.7 over 0.8 — but reliable at choosing between described
options. The caller supplies what each label is worth, so the mapping from
judgement to score is the caller's, made once and visible in the eval file
rather than improvised by the model on every case.

A reply that names no known choice is an **errored Score**, never a guess. An
errored Score is excluded from the mean; scoring 0.0 would say the task did
badly when what actually happened is that the judge was unreadable.

Judge calls go through `llm.py` like any other call, so they are cached, traced,
retried and costed automatically. A judge's request therefore appears in the
case's trace tree alongside the task's own calls, and its cost is counted in the
case total — which is what stops LLM-as-judge from being an invisible line on
the bill.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Coroutine
from typing import Any

from evalstand.llm import acall
from evalstand.models import Case, Score

__all__ = ["factuality", "judge"]

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
"""Small and cheap by default. A judge runs once per case, so an expensive
default would quietly multiply the cost of every suite that uses one."""

FACTUALITY_CHOICES: dict[str, float] = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "D": 0.0,
    "E": 1.0,
}
"""The factuality rubric's scoring, following the shape the reference
implementation uses.

A submission that is a subset, a superset, or a restatement of the reference
answer all score 1.0: each is factually consistent, and the extra or missing
*detail* is a different question from whether the content is true. Only D — a
genuine disagreement — scores 0.
"""

FACTUALITY_RUBRIC = """\
Compare the factual content of the submitted answer with the expert answer.
Ignore differences in style, grammar, or punctuation.

(A) The submitted answer is a subset of the expert answer and is fully
    consistent with it.
(B) The submitted answer is a superset of the expert answer and is fully
    consistent with it.
(C) The submitted answer contains all the same details as the expert answer.
(D) There is a disagreement between the submitted answer and the expert answer.
(E) The answers differ, but those differences do not matter from the perspective
    of factuality."""


def judge(
    *,
    rubric: str,
    choices: dict[str, float],
    model: str = DEFAULT_JUDGE_MODEL,
    name: str = "judge",
    include_expected: bool = True,
    **params: Any,
) -> Callable[[Any, Any, Case], Coroutine[Any, Any, Score]]:
    """Build a scorer that asks a model to grade the output against a rubric.

    `choices` maps each label the rubric offers to the value it is worth. The
    caller owns that mapping: whether a partially-correct answer is worth 0.5 or
    0.2 is a judgement about the task, and letting the model decide would put a
    number in its mouth that nobody chose.

    `include_expected=False` is for rubrics that judge the output alone — tone,
    safety, format — where showing a reference answer invites the judge to
    compare against it instead of applying the rubric.
    """
    if not choices:
        raise ValueError("a judge needs at least one choice to pick between")

    for label, value in choices.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"choice {label!r} is worth {value}, outside the score range [0, 1]")

    async def judge(output: Any, expected: Any, case: Case) -> Score:
        prompt = _prompt(rubric, choices, output, expected, case, include_expected)
        response = await acall(model, [{"role": "user", "content": prompt}], **params)

        choice = _read_choice(response.text, choices)
        if choice is None:
            # Not scored 0.0: the task may have been fine, and what failed was
            # the judge. An errored Score is excluded from the mean rather than
            # dragging it down with a verdict nobody reached.
            return Score.from_error(
                name, f"the judge did not name a known choice; it said {response.text.strip()!r}"
            )

        return Score(
            scorer_name=name,
            value=choices[choice],
            metadata={
                # Travels with the number, so a report cannot quote the score
                # without the caveat that qualifies it.
                "unvalidated": True,
                "choice": choice,
                "judge_model": response.model,
                # The judge's own words, so a surprising score can be read
                # rather than merely disbelieved.
                "judge_reply": response.text.strip(),
            },
        )

    judge.__name__ = name
    return judge


def factuality(
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    **params: Any,
) -> Callable[[Any, Any, Case], Coroutine[Any, Any, Score]]:
    """A judge that asks whether the output contradicts the expected answer.

    Factual *consistency*, not equality: an answer that says less than the
    reference, or more, is still consistent with it. Only a genuine
    disagreement scores 0. That is the question worth asking of a model, since
    penalising a correct answer for being differently detailed measures
    verbosity rather than truth.
    """
    return judge(
        rubric=FACTUALITY_RUBRIC,
        choices=FACTUALITY_CHOICES,
        model=model,
        name="factuality",
        **params,
    )


def _prompt(
    rubric: str,
    choices: dict[str, float],
    output: Any,
    expected: Any,
    case: Case,
    include_expected: bool,
) -> str:
    """Assemble the judge's request.

    The data is delimited and labelled so the judge can tell the material it is
    grading from the instructions it is following. Without that, an output
    containing something that reads like an instruction becomes one.
    """
    sections = [rubric, "", f"[Question]\n{case.input}", "", f"[Submission]\n{output}"]
    if include_expected:
        sections += ["", f"[Expert answer]\n{expected}"]

    labels = ", ".join(sorted(choices))
    sections += [
        "",
        f"Reply with a single letter, one of: {labels}. Give no other text.",
    ]
    return "\n".join(sections)


def _read_choice(reply: str, choices: dict[str, float]) -> str | None:
    """The choice the judge named, or None when it named none.

    Deliberately forgiving about *packaging* and strict about *content*. A model
    told to answer "A" may reply "A", "A.", "(A)" or "Answer: A", and refusing
    those would throw away a judgement that was perfectly clear. But a reply
    naming two different choices is refused rather than resolved by taking the
    first, for the same reason `parse_number` refuses "42 or 43": picking one
    invents a verdict the judge never committed to.
    """
    text = reply.strip()
    if not text:
        return None

    # Case-insensitive: a model told to answer "yes" may reply "Yes.", and the
    # capital is packaging like the full stop is. Labels differing only by case
    # would be an unusable rubric anyway, since the model cannot be relied on to
    # honour the distinction.
    named = {
        label
        for label in choices
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE
        )
    }
    if len(named) == 1:
        return named.pop()
    return None

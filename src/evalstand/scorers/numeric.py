"""Numeric scorers.

A model asked for a number rarely returns one. It returns "The answer is 42.",
or "$1,000", or "about 3.14" — so the hard part of scoring a number is finding
it, and the rules for giving up are what make the scorer trustworthy.

The governing decision is that **an ambiguous output is unparseable, not a
guess**. "42 or 43" contains two candidates, and picking the first would invent
a measurement the model never committed to. A scorer that says "I could not read
this" is honest; one that quietly picks a number produces a score nobody can
audit.

`nan` and `inf` are refused for the same reason. `float("nan")` succeeds, and a
NaN then compares false against everything — so it would silently score 0.0 and
look like a wrong answer rather than an unreadable one. `inf` is not a
measurement either.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from evalstand.models import Score

__all__ = ["close_to", "parse_number"]

_NUMBER = re.compile(
    r"""
    (?<![\w.])          # not mid-token: the "5" in "a5" is not a number
    [+-]?               # sign, which is part of the value
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,000 or 1,234,567.89
      | \d+\.\d*                        # 42. or 42.5
      | \.\d+                           # .5
      | \d+                             # 42
    )
    (?:[eE][+-]?\d+)?   # scientific notation
    (?!\w)              # not followed by more of a token
    (?!\.\d)            # nor by more number
    """,
    re.VERBOSE,
)
"""One number, with the sign and separators that belong to it.

The two lookaheads are separate on purpose. A single `(?![\\w.])` also rejects a
number followed by a **sentence-ending period**, which is the commonest shape a
model produces: "It costs 19.99." parsed as nothing at all. Splitting them lets
a full stop end the sentence while still refusing "1.2.3", where a trailing dot
is followed by more number.
"""


def parse_number(value: Any) -> float | None:
    """The single number in `value`, or None when there is not exactly one.

    Returns None rather than raising, because an unreadable output is a normal
    thing for a model to produce and the caller has to decide what it means.

    A `bool` is refused. `bool` subclasses `int` in Python, so `float(True)` is
    1.0 — and a task returning True would otherwise be scored as the number one,
    which is a coincidence of the type system rather than an answer.
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return _finite(float(value))

    matches = _NUMBER.findall(str(value))
    if len(matches) != 1:
        # Nothing to read, or several candidates and choosing between them
        # would be inventing an answer.
        #
        # Counted by occurrence, not by distinct value. Deduplicating first
        # would accept "I said 42, yes 42" — but it would equally accept
        # "5 + 5", reading an unevaluated expression as the number 5. Both are
        # outputs that did not commit to a single answer, and refusing them is
        # the same rule that refuses "42 or 43".
        return None

    return _finite(float(matches[0].replace(",", "")))


def _finite(number: float) -> float | None:
    """None for NaN and infinity, which are not measurements.

    NaN in particular has to be caught here: it compares false against every
    tolerance, so letting it through would score 0.0 and read as a wrong answer
    rather than an unreadable one.
    """
    return number if math.isfinite(number) else None


def close_to(
    *,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
) -> Callable[[Any, Any], Score]:
    """Build a scorer that accepts a number within a tolerance of the expected.

    A factory rather than a scorer, for the same reason as `regex_match`: the
    tolerance is a property of the check, not of the data, and `expected`
    already belongs to the Case.

    Both tolerances default to zero, which asks for an exact match — the
    strictest reading, so a user who forgets to set one is not silently given
    slack they did not ask for.

    `rel_tol` scales with the expected value, which is what you want for prices
    or token counts. `abs_tol` does not, which is the only thing that works when
    the expected value is zero: a relative tolerance around 0 admits nothing but
    an exact 0, since any percentage of zero is zero.

    Comparison is `math.isclose`, rather than a hand-rolled `abs(a - b) <= t`.
    It already handles the sign, the zero case, and the infinities correctly,
    and rederiving that is how the zero trap gets reintroduced.
    """
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError(f"tolerances cannot be negative, got rel_tol={rel_tol}, abs_tol={abs_tol}")

    def close_to(output: Any, expected: Any) -> Score:
        target = parse_number(expected)
        if target is None:
            return Score(
                scorer_name="close_to",
                value=0.0,
                passed=False,
                metadata={"reason": f"the expected value is not a number: {expected!r}"},
            )

        actual = parse_number(output)
        if actual is None:
            # Distinguished from a wrong number: the task may have answered
            # correctly in a form this scorer cannot read, and the metadata is
            # what tells the user which of the two happened.
            return Score(
                scorer_name="close_to",
                value=0.0,
                passed=False,
                metadata={"reason": f"no single number found in the output: {output!r}"},
            )

        within = math.isclose(actual, target, rel_tol=rel_tol, abs_tol=abs_tol)
        return Score(
            scorer_name="close_to",
            value=1.0 if within else 0.0,
            passed=within,
            metadata={
                "parsed_output": actual,
                "expected": target,
                "difference": actual - target,
                "rel_tol": rel_tol,
                "abs_tol": abs_tol,
            },
        )

    return close_to

"""String scorers.

The distinction that matters here is between scorers that *know* pass from fail
and scorers that report a position on a scale. Everything in this module is
binary — a string either matches or it does not — so all of these set `passed`.
The continuous ones live in `fuzzy.py` and deliberately leave it absent.

Every scorer here coerces its arguments with `str()` rather than requiring
strings. A task that returns an int and an expected value written as an int
should compare equal, and failing on a type the user never thought about is a
worse outcome than comparing their text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from evalstand.models import Score

__all__ = ["contains", "exact", "normalised_exact", "regex_match"]

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def exact(output: Any, expected: Any) -> Score:
    """1.0 when the output equals the expected value, 0.0 otherwise.

    Compares the values as given, without coercion: `exact` is the scorer for
    when you mean exactly, and quietly treating `1` and `"1"` as equal would
    make it something else. Use `normalised_exact` for text comparison.
    """
    matched = output == expected
    return Score(scorer_name="exact", value=1.0 if matched else 0.0, passed=matched)


def normalise(text: Any) -> str:
    """Fold case, whitespace, punctuation, and Unicode form.

    NFKC first, so that visually identical strings written differently — a
    composed "é" against "e" plus a combining accent, or a full-width digit
    against an ASCII one — do not count as a wrong answer. A model's choice of
    Unicode encoding is not a fact about whether it answered correctly.

    `None` becomes the empty string, not the word "None". A task that fell
    through without returning would otherwise score a perfect match against a
    Case whose expected value is the literal text "None" — a false pass, which
    is the most damaging thing a scorer can produce because it looks like
    success and needs no investigating.
    """
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    folded = _PUNCTUATION.sub(" ", folded)
    return _WHITESPACE.sub(" ", folded).strip()


def normalised_exact(output: Any, expected: Any) -> Score:
    """`exact`, but ignoring case, surrounding whitespace, and punctuation.

    This is usually what a user means by "did it say Paris": `"Paris."`,
    `" paris "` and `"PARIS"` are the same answer, and only an exercise in
    string formatting distinguishes them.
    """
    matched = normalise(output) == normalise(expected)
    return Score(
        scorer_name="normalised_exact",
        value=1.0 if matched else 0.0,
        passed=matched,
        metadata={"normalised_output": normalise(output)},
    )


def contains(output: Any, expected: Any) -> Score:
    """1.0 when the expected text appears anywhere in the output, normalised.

    For tasks whose answer is embedded in a sentence — "The capital is Paris."
    contains "Paris" — where insisting on an exact match would measure the
    model's verbosity rather than its correctness.

    An empty expected value scores 0.0 rather than 1.0. Every string contains
    the empty string, so the vacuous truth would silently mark every case as a
    pass, which is the most damaging way for a scorer to be wrong.
    """
    needle = normalise(expected)
    if not needle:
        return Score(
            scorer_name="contains",
            value=0.0,
            passed=False,
            metadata={"reason": "the expected value is empty, so containment is vacuous"},
        )

    found = needle in normalise(output)
    return Score(scorer_name="contains", value=1.0 if found else 0.0, passed=found)


def regex_match(pattern: str | re.Pattern[str], *, full: bool = False) -> Callable[[Any], Score]:
    """Build a scorer that tests the output against a regular expression.

    A factory rather than a scorer, because the pattern is the user's and there
    is nowhere else to put it: `expected` belongs to the Case, and a pattern is
    a property of the check, not of the data.

    `full=True` requires the pattern to match the entire output, which is what
    people usually mean by "matches" — a bare `search` for `\\d+` is satisfied
    by any output containing a digit anywhere.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    def regex_match(output: Any) -> Score:
        text = str(output)
        matched = compiled.fullmatch(text) if full else compiled.search(text)
        return Score(
            scorer_name="regex_match",
            value=1.0 if matched else 0.0,
            passed=bool(matched),
            metadata={"pattern": compiled.pattern, "full": full},
        )

    return regex_match

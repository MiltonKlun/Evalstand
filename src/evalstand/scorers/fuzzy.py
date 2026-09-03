"""Fuzzy string scorers, built on `rapidfuzz`.

These are `evalstand`'s continuous scorers, and unlike everything in `text.py`
they **never set `passed`**. A similarity of 0.62 is a position on a scale; where
the pass line sits is a judgement about the task, and nobody has told the scorer
where to draw it. Deciding that for the user is exactly the inference CONTEXT.md
says nothing may make.

`levenshtein` is `evalstand`'s default scorer: it degrades gracefully, so a
nearly-right answer scores nearly 1.0 rather than falling off the cliff that
`exact` presents.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from evalstand.models import Score
from evalstand.scorers.text import normalise

__all__ = ["levenshtein", "ratio"]


def levenshtein(output: Any, expected: Any) -> Score:
    """Edit distance normalised into `[0, 1]`, where 1.0 is identical.

    The value is `1 - distance / max(len(output), len(expected))`, so
    `levenshtein("kitten", "sitting")` is 3 edits over 7 characters — 0.5714.

    Text is normalised first (case, whitespace, punctuation, Unicode form), so
    the score measures how close the answer is rather than how closely its
    formatting matches. Two empty strings score 1.0, which is what
    `rapidfuzz` reports and the only defensible reading: nothing was expected
    and nothing is what arrived.
    """
    left, right = normalise(output), normalise(expected)
    similarity = Levenshtein.normalized_similarity(left, right)
    return Score(
        scorer_name="levenshtein",
        value=float(similarity),
        metadata={"distance": Levenshtein.distance(left, right)},
    )


def ratio(output: Any, expected: Any) -> Score:
    """`rapidfuzz`'s indel-based similarity, normalised into `[0, 1]`.

    Differs from `levenshtein` in that a substitution costs two operations
    rather than one, so it penalises replaced text more heavily than inserted
    text. Useful when a shorter-but-correct answer should score better than one
    of the right length with the wrong words in it.

    `fuzz.ratio` reports a percentage; dividing by 100 puts it on the `[0, 1]`
    scale every Score uses.
    """
    left, right = normalise(output), normalise(expected)
    return Score(scorer_name="ratio", value=fuzz.ratio(left, right) / 100.0)

"""Built-in scorers.

These live outside the seven-name public surface: importing a scorer is an
explicit act, and the library will grow past what a top-level namespace should
carry.

Re-exported here so `from evalstand.scorers import levenshtein` works without
the user having to know which module a scorer lives in — that is an
implementation detail, and moving one should not break their eval file.
"""

from evalstand.scorers.base import Scorer, scorer
from evalstand.scorers.fuzzy import levenshtein, ratio
from evalstand.scorers.json_field import json_fields, parse_object
from evalstand.scorers.numeric import close_to, parse_number
from evalstand.scorers.text import contains, exact, normalised_exact, regex_match

__all__ = [
    "Scorer",
    "close_to",
    "contains",
    "exact",
    "json_fields",
    "levenshtein",
    "normalised_exact",
    "parse_number",
    "parse_object",
    "ratio",
    "regex_match",
    "scorer",
]

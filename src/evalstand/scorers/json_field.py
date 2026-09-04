"""Structured scorers: compare an answer field by field.

An extraction task rarely gets everything right or everything wrong. A model
asked for five fields typically returns four good ones and a bad one, and a
single pass/fail throws away the only information worth having — *which* field
it got wrong. So this scorer reports a macro-average over the fields the Case
asked for, and names every failure in `Score.metadata`.

Three decisions shape it:

**The denominator is the expected keys.** The Case defines what was asked for.
A field the model volunteered that nobody wanted is not a wrong answer, and
counting it would let a verbose model score worse than a terse one that answered
exactly as badly.

**Nested dicts flatten to dotted paths.** `person.born` is its own field, so one
wrong leaf out of ten costs a tenth rather than the whole subtree. It also lets
the metadata name the exact path that failed, which is the point of the
breakdown.

**A missing field is not the same as a wrong one.** Both score zero, but the
report distinguishes them: a wrong field means the model answered and erred; a
missing field means it never answered at all. Those call for different fixes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from evalstand.models import Score
from evalstand.scorers.text import normalise

__all__ = ["json_fields", "parse_object"]

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)


def parse_object(value: Any) -> dict[str, Any] | None:
    """The mapping in `value`, or None when there is not one.

    A model asked for JSON returns *text* that looks like JSON, usually wrapped
    in a ``` fence because that is how it was taught to present code. Refusing
    to unwrap it would make this scorer report a total failure for an answer
    that was entirely correct — measuring the presentation rather than the
    content.
    """
    if isinstance(value, Mapping):
        return dict(value)

    if not isinstance(value, str):
        return None

    text = value.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group("body").strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    return parsed if isinstance(parsed, dict) else None


def flatten(obj: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested keys become dotted paths: `{"a": {"b": 1}}` -> `{"a.b": 1}`.

    An empty dict keeps its own path as a leaf. Otherwise a Case expecting
    `{"meta": {}}` would flatten to nothing and silently drop the requirement,
    scoring an output that omitted `meta` entirely as a perfect match.
    """
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _matches(output_value: Any, expected_value: Any) -> bool:
    """Whether one field's value counts as correct.

    Compared after normalisation, for the same reason `normalised_exact` exists:
    a model that serialised its answer through JSON returns `"1843"` where the
    Case says `1843`, and scoring that as wrong would measure the round trip
    rather than the model. `None` is compared identically, so a null and a
    missing value are not conflated with the string "None".

    Lists compare element-wise in order, and dicts should never arrive here —
    `flatten` has already turned them into their own fields.
    """
    if isinstance(output_value, list) and isinstance(expected_value, list):
        return len(output_value) == len(expected_value) and all(
            _matches(a, b) for a, b in zip(output_value, expected_value, strict=True)
        )

    if output_value is None or expected_value is None:
        # A null is only equal to a null. Falling through to `normalise` would
        # make None equal the empty string, so a model that omitted a value
        # would match a Case expecting one.
        return output_value is None and expected_value is None

    left, right = _as_number(output_value), _as_number(expected_value)
    if left is not None and right is not None:
        # Compared as numbers when both sides *are* numbers, so 1843 matches
        # 1843.0 and "1843". Normalising first would stringify them to "1843"
        # and "1843.0" and call a value equal to itself wrong — while the
        # sloppier int-against-string case passed, which is backwards.
        return left == right

    return normalise(output_value) == normalise(expected_value)


def _as_number(value: Any) -> float | None:
    """The number `value` *is*, never a number extracted from prose.

    Deliberately not `numeric.parse_number`, which finds a number anywhere in
    the text. That is right for a numeric scorer and wrong here: it would make
    "Paris 1" equal "Lyon 1", so two different answers match because they happen
    to contain the same digit.

    `bool` is excluded for the reason it always is — it subclasses `int`, so
    `float(True)` is 1.0, and True would match the number one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def json_fields(*, require_all: bool = False) -> Callable[[Any, Any], Score]:
    """Build a scorer that compares an answer against the expected object.

    The value is the fraction of expected fields that matched — a macro-average,
    so every field counts equally regardless of how much text it holds.

    `passed` is set only with `require_all=True`, where "all fields correct" is
    an unambiguous verdict the scorer genuinely knows. Left absent otherwise:
    0.6 is a position on a scale, and where the line sits between "good enough"
    and "not" is a judgement about the task that nobody has given the scorer.
    """

    def json_fields(output: Any, expected: Any) -> Score:
        target = parse_object(expected)
        if target is None:
            return Score(
                scorer_name="json_fields",
                error=f"the expected value is not an object: {expected!r}",
            )

        fields = flatten(target)
        if not fields:
            # 0 of 0 has no defensible value: 1.0 would claim everything
            # matched and 0.0 that nothing did, when in truth nothing was
            # asked. An errored Score is excluded from means rather than
            # dragging them toward an invented number.
            return Score(
                scorer_name="json_fields",
                error="the expected object has no fields to compare",
            )

        actual = parse_object(output)
        if actual is None:
            return Score(
                scorer_name="json_fields",
                value=0.0,
                passed=False if require_all else None,
                metadata={
                    "reason": f"the output is not an object: {output!r}",
                    "matched": [],
                    "wrong": [],
                    "missing": sorted(fields),
                },
            )

        given = flatten(actual)
        matched, wrong, missing = [], [], []
        for path, expected_value in fields.items():
            if path not in given:
                missing.append(path)
            elif _matches(given[path], expected_value):
                matched.append(path)
            else:
                wrong.append(path)

        value = len(matched) / len(fields)
        return Score(
            scorer_name="json_fields",
            value=value,
            passed=(not wrong and not missing) if require_all else None,
            metadata={
                "matched": sorted(matched),
                # Kept apart because they call for different fixes: a wrong
                # field means the model answered and erred, a missing one that
                # it never answered at all.
                "wrong": sorted(wrong),
                "missing": sorted(missing),
                "field_count": len(fields),
                # Volunteered fields are reported but never counted: the Case
                # defines what was asked for, so a verbose model must not score
                # worse than a terse one that answered exactly as badly.
                "unexpected": sorted(set(given) - set(fields)),
            },
        )

    return json_fields

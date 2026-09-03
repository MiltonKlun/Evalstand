"""The Scorer protocol and the adapter that makes a plain function into one.

A Scorer judges one output and returns a Score. The design goal is that writing
one requires no import: a three-line function taking the arguments it actually
needs is a scorer, and `@scorer` is available for explicitness rather than
required for correctness.

Two adaptations happen here, both of them about meeting users where they are:

**Arguments.** Scorers are called with whatever they declare. A scorer that only
looks at the output takes `(output)`; one comparing against the reference takes
`(output, expected)`; one needing the Case's metadata takes `(output, expected,
case)`. Forcing every scorer to accept three arguments would mean a two-line
scorer carrying two parameters it never reads.

**Return values.** A scorer may return a `Score`, a float in `[0, 1]`, or a
bool. A bool is an unambiguous claim of pass from fail, so it sets `passed`; a
float is a position on a scale and leaves `passed` absent, because a continuous
scorer does not know where the line is. Nothing else derives the flag.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from evalstand.models import Case, Score

__all__ = ["Scorer", "ScorerResult", "call_scorer", "scorer"]

ScorerResult = Score | float | bool
"""What a scorer may return. See the module docstring for how each is read."""


@runtime_checkable
class Scorer(Protocol):
    """The shape of a scorer.

    Declared with `...` arguments because scorers are adapted to their own
    signature — see `call_scorer`. The protocol exists to name the concept and
    to type public APIs, not to be subclassed: a plain function is a Scorer.
    """

    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> ScorerResult | Any: ...


def scorer(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a function as a scorer.

    Optional. A plain function passed to `evaluate(scorers=[...])` is already
    treated as one; this decorator only makes the intent explicit and gives the
    signature a single place to be validated, so a scorer with an impossible
    signature fails at import rather than mid-run after money has been spent.
    """
    _parameters(fn)  # raises now rather than during a run
    fn.__evalstand_scorer__ = True  # type: ignore[attr-defined]
    return fn


def scorer_name(fn: Any) -> str:
    """The name a Score is filed under, which the user reads in every report.

    A callable object is named by its class, since a judge built by a factory is
    usually an instance or a closure and "scorer" in every row would be useless.
    A lambda has no meaningful name, so it gets the generic one rather than
    "function" — which is the name of its *type*, and would read as though the
    scorer were actually called that.
    """
    name = getattr(fn, "__name__", None)
    if name and name != "<lambda>":
        return str(name)

    if name is None and not isinstance(fn, type):
        # A callable instance: `FactualityJudge()` reports as FactualityJudge.
        cls = type(fn)
        if cls is not type:
            return cls.__name__

    return "scorer"


async def call_scorer(fn: Any, output: Any, expected: Any, case: Case) -> Score:
    """Invoke one scorer and normalise whatever it returns into a Score.

    A scorer that raises produces an errored Score rather than ending the case:
    a broken scorer is not evidence the task did badly, and an errored Score is
    excluded from means rather than counted as zero.
    """
    name = scorer_name(fn)
    try:
        result = fn(*_arguments(fn, output, expected, case))
        if inspect.isawaitable(result):
            result = await result
        return as_score(result, name)
    except Exception as exc:
        return Score.from_error(name, f"{type(exc).__name__}: {exc}")


def as_score(result: Any, name: str) -> Score:
    """Read a scorer's return value as a Score.

    `bool` is checked before `float` because `bool` is a subclass of `int` in
    Python: without the order, `True` would be read as the number 1.0 and its
    pass/fail claim silently discarded.
    """
    if isinstance(result, Score):
        return result

    if isinstance(result, bool):
        return Score(scorer_name=name, value=1.0 if result else 0.0, passed=result)

    if isinstance(result, int | float):
        return Score(scorer_name=name, value=float(result))

    raise TypeError(
        f"a scorer must return a Score, a float in [0, 1], or a bool; "
        f"{name} returned {type(result).__name__}"
    )


def _arguments(fn: Any, output: Any, expected: Any, case: Case) -> tuple[Any, ...]:
    """The prefix of (output, expected, case) that this scorer asked for."""
    return (output, expected, case)[: _parameters(fn)]


# Bounded rather than unbounded: the cache holds references to the functions
# it has seen, and a suite that builds scorers dynamically (a judge factory
# per case, say) would otherwise keep every one of them alive for the life of
# the process. A few hundred distinct scorers is already far past realistic.
@lru_cache(maxsize=512)
def _cached_parameter_count(fn: Any) -> int:
    return _count_parameters(fn)


def _parameters(fn: Any) -> int:
    """How many of (output, expected, case) to pass.

    Memoised where possible: a scorer runs once per case, and re-inspecting the
    same function for every case in a 500-case suite is pure waste. Unhashable
    callables fall through to the uncached path rather than failing.
    """
    try:
        return _cached_parameter_count(fn)
    except TypeError:  # an unhashable callable cannot be cached
        return _count_parameters(fn)


def _count_parameters(fn: Any) -> int:
    """Read the signature, rejecting shapes that cannot be a scorer.

    `*args` means "as many as you have", so such a scorer gets all three: it has
    declared it can take them, and passing fewer would be a guess.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # A builtin or C function with no introspectable signature. Assume the
        # full form rather than refusing outright.
        return 3

    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return 3
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if parameter.default is inspect.Parameter.empty:
                raise TypeError(
                    f"{scorer_name(fn)} requires keyword-only argument "
                    f"{parameter.name!r}, which a scorer is never called with"
                )
            continue
        if parameter.default is inspect.Parameter.empty:
            positional += 1
        else:
            # A defaulted parameter is the scorer's own configuration, not
            # something the runner supplies. `contains(output, *, fold=True)`
            # and `contains(output, fold=True)` should behave the same way.
            break

    if positional > 3:
        raise TypeError(
            f"{scorer_name(fn)} takes {positional} required arguments; a scorer "
            f"receives at most three: (output, expected, case)"
        )
    return positional

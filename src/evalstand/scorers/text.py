"""String scorers.

Phase 2 ships only `exact`, the minimum needed to make the toy example real.
The rest — normalised_exact, contains, regex_match — arrive in Phase 4 with the
scorer protocol.
"""

from __future__ import annotations

from typing import Any

from evalstand.models import Case, Score

__all__ = ["exact"]


def exact(output: Any, expected: Any, case: Case) -> Score:
    """1.0 when the output equals the expected value, 0.0 otherwise.

    Sets `passed`, because a binary scorer genuinely knows pass from fail. A
    continuous scorer must not — see CONTEXT.md on Score.
    """
    matched = output == expected
    return Score(
        scorer_name="exact",
        value=1.0 if matched else 0.0,
        passed=matched,
    )

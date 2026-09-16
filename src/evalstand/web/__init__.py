"""The optional web UI (Phase 8).

Imported only by `evalstand serve`. Everything here depends on the `web` extra,
which a default install does not have — see ADR 0009 for why that is deliberate.
"""

from __future__ import annotations

MISSING_EXTRA = (
    "the web UI needs the 'web' extra, which is not installed.\n"
    "Install it with:  pip install 'evalstand[web]'\n"
    "or, with uv:      uv add 'evalstand[web]'"
)
"""What a user sees when `serve` is run on a default install.

Named here rather than written at each import site so the CLI and the app agree
on the wording. A traceback about `fastapi` would tell a user which package is
absent but not which extra supplies it, which is the only part they can act on.
"""

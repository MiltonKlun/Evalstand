"""Finding the Evals a file or directory declares, without running them.

The pytest plugin does this through pytest's own collector, which is right for
`evalstand run`. The TUI and watch mode need the same answer without a pytest
session, and this is that path.

It works only because `evaluate()` registers and returns (ADR 0004). Importing a
file populates the registry and spends nothing — which is exactly the property
that deferral was chosen for, and the reason watch mode can ask "what does this
file declare?" after every edit without paying a provider for the answer.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from evalstand.api import Eval, _current_eval_file, registry

__all__ = ["EVAL_GLOB", "NoEvalsFoundError", "eval_files", "load_evals", "select_eval"]

EVAL_GLOB = "*_eval.py"
"""The same pattern the plugin collects, so the two cannot disagree about what
an eval file is."""


class NoEvalsFoundError(RuntimeError):
    """Nothing to run.

    Its own type because the CLI answers it with guidance rather than a
    traceback: a user whose file is named `evals.py` needs to be told the
    pattern, not shown a stack.
    """


def eval_files(paths: Iterable[Path | str] | None = None) -> list[Path]:
    """Every eval file under the given paths, in a stable order.

    A file given explicitly is taken as an eval file whatever it is called: the
    user naming it is a clearer signal than the glob. A directory is searched
    with the pattern.
    """
    found: list[Path] = []
    for raw in paths or [Path()]:
        path = Path(raw).resolve()
        if path.is_dir():
            found.extend(sorted(path.rglob(EVAL_GLOB)))
        elif path.exists():
            found.append(path)

    # Deduplicated, because `evalstand watch . qa_eval.py` names one file twice
    # and importing it twice would register each eval twice.
    seen: dict[Path, None] = {}
    for path in found:
        if _is_importable(path):
            seen.setdefault(path, None)
    return list(seen)


def _is_importable(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return path.suffix == ".py" and not parts & {"__pycache__", ".venv", ".git", "node_modules"}


def load_evals(paths: Iterable[Path | str] | None = None) -> list[Eval]:
    """Import the eval files under `paths` and return what they declared.

    The registry is cleared first. Watch mode re-imports after every edit, and
    an eval the user has deleted or renamed must disappear rather than linger
    from a previous import — a table still showing a case that no longer exists
    is a report about code that is no longer there.
    """
    registry.clear()

    for path in eval_files(paths):
        token = _current_eval_file.set(str(path))
        try:
            _import_file(path)
        finally:
            _current_eval_file.reset(token)

    return registry.evals()


def _import_file(path: Path) -> None:
    """Import one file under a name of its own.

    Freshness is *not* what the unique name buys, despite how it looks.
    `exec_module` on a newly built module object re-reads the source every
    time, so an edited file produces edited behaviour whatever the module is
    called — verified by running an edit through both.

    The name matters because the entry is visible in `sys.modules` while the
    file executes, and eval files are imported one after another. Under a
    shared name, a file that imports a sibling — or anything resolving a
    class's `__module__` mid-import — would find the *previous* eval file's
    module object sitting under the name it expects to be its own.

    Removed in the `finally` because nothing looks these up afterwards, and one
    entry per edit would grow without bound across a long watch session.
    """
    module_name = f"_evalstand_eval_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unimportable path
        raise ImportError(f"could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Not left behind: one entry per edit would grow without bound over a
        # long watch session, and nothing ever looks the module up by name.
        sys.modules.pop(module_name, None)


def select_eval(evals: list[Eval], name: str | None = None) -> Eval:
    """Pick the one Eval to run in a live view.

    A named eval is looked up. With no name, a single declared Eval is the
    obvious answer and several is refused — guessing would silently measure
    something the user did not ask about, and every number on screen would be
    about the wrong thing.
    """
    if not evals:
        raise NoEvalsFoundError(
            f"no evals found. Files are collected by the pattern {EVAL_GLOB!r}, "
            f"or you can name one directly."
        )

    if name is not None:
        for declared in evals:
            if declared.name == name:
                return declared
        known = ", ".join(sorted(declared.name for declared in evals))
        raise NoEvalsFoundError(f"no eval named {name!r}. Found: {known}")

    if len(evals) > 1:
        known = ", ".join(sorted(declared.name for declared in evals))
        raise NoEvalsFoundError(
            f"several evals were found, so there is nothing to default to: {known}. "
            f"Choose one with --eval NAME."
        )

    return evals[0]

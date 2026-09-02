"""The command line interface.

Phase 2 ships only `run`, which is what the phase's exit criterion requires. It
delegates to pytest rather than reimplementing collection: the plugin already
knows how to find and execute evals, and a second execution path would be a
second thing to keep correct.

`history`, `show`, and `compare` arrive in Phase 5; watch mode and the TUI in
Phase 6.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="evalstand",
    help="Run LLM evals like a test suite.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def run(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Eval files or directories. Defaults to the current directory."),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Less pytest output.")] = False,
    select: Annotated[
        str | None, typer.Option("-k", help="Only run cases matching this expression.")
    ] = None,
) -> None:
    """Run evals and print a summary."""
    import pytest

    args: list[str] = [str(path) for path in (paths or [Path()])]
    if quiet:
        args.append("-q")
    if select:
        args += ["-k", select]

    # Evals are not tests-with-assertions; a failing case is a reported result,
    # not a stack trace worth printing twice.
    args += ["--no-header", "-p", "no:cacheprovider"]

    raise typer.Exit(code=pytest.main(args))


@app.command()
def version() -> None:
    """Print the installed version."""
    from evalstand import __version__

    typer.echo(__version__)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()

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
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", help="Cases to execute at once (default: 8).")
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Abandon a case after this many seconds."),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Call the provider even when a cached response exists."),
    ] = False,
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help="Fail when an eval's mean score falls below this."),
    ] = None,
) -> None:
    """Run evals and print a summary."""
    import pytest

    args: list[str] = [str(path) for path in (paths or [Path()])]
    if quiet:
        args.append("-q")
    if select:
        args += ["-k", select]

    # Forwarded rather than re-validated: the plugin owns these, and a second
    # copy of the rules here would be a second thing to keep in step.
    if concurrency is not None:
        args += ["--concurrency", str(concurrency)]
    if timeout is not None:
        args += ["--timeout", str(timeout)]
    if no_cache:
        args.append("--no-cache")
    if threshold is not None:
        args += ["--threshold", str(threshold)]

    # Evals are not tests-with-assertions; a failing case is a reported result,
    # not a stack trace worth printing twice.
    args += ["--no-header", "-p", "no:cacheprovider"]

    raise typer.Exit(code=pytest.main(args))


@app.command()
def history(
    name: Annotated[
        str | None,
        typer.Argument(help="Only show runs of this eval. Omit for all of them."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many runs to show.")] = 20,
    database: Annotated[
        Path | None,
        typer.Option("--db", help="Read a database other than the project's own."),
    ] = None,
) -> None:
    """List past runs: when they ran, from which commit, and what they measured."""
    from rich.console import Console

    from evalstand.reporting.console import render_history
    from evalstand.storage import DatabaseTooNewError, RunStore

    console = Console()

    try:
        store = RunStore(database) if database is not None else RunStore()
    except DatabaseTooNewError as exc:
        # Refused rather than read best-effort: a partial read would produce a
        # table that looks right and answers a different question.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]could not open the history database: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    with store:
        runs = store.runs_for(name, limit=limit)
        if not runs:
            # Told apart deliberately. "No runs yet" and "no runs of *that*
            # eval" send a user looking in completely different places.
            known = store.eval_names()
            if name is not None and known:
                console.print(f"no runs recorded for [bold]{name}[/bold].")
                console.print(f"known evals: {', '.join(known)}")
            else:
                console.print("no runs recorded yet.")
            return

        entries = [(run, store.batch_for(run.id)) for run in runs]

    console.print(render_history(entries))
    if any(batch is not None and batch.git_dirty for _, batch in entries):
        console.print(
            "[dim]* the working tree had uncommitted changes, so that run "
            "cannot be reproduced from its commit[/dim]"
        )


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="The run to display, from `evalstand history`.")],
    full: Annotated[
        bool,
        typer.Option("--full", help="Print each call's prompts and completions in full."),
    ] = False,
    database: Annotated[
        Path | None,
        typer.Option("--db", help="Read a database other than the project's own."),
    ] = None,
) -> None:
    """Show one run: its summary, per-case scores, and trace trees."""
    from rich.console import Console

    from evalstand.reporting.console import render_run_detail
    from evalstand.storage import DatabaseTooNewError, RunStore

    console = Console()

    try:
        store = RunStore(database) if database is not None else RunStore()
    except DatabaseTooNewError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]could not open the history database: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    with store:
        run = store.load_run(run_id)
        if run is None:
            # A typo and an empty database send a user looking in different
            # places, so they are answered differently.
            console.print(f"no run with id [bold]{run_id}[/bold].")
            if store.run_count():
                console.print("run [bold]evalstand history[/bold] to see what is recorded.")
            else:
                console.print("no runs recorded yet.")
            raise typer.Exit(code=1)

        batch = store.batch_for(run.id)
        rendered = render_run_detail(run, batch, full=full)

    console.print(rendered)


@app.command()
def compare(
    run_a: Annotated[str, typer.Argument(help="The earlier run.")],
    run_b: Annotated[str, typer.Argument(help="The later run.")],
    database: Annotated[
        Path | None,
        typer.Option("--db", help="Read a database other than the project's own."),
    ] = None,
) -> None:
    """Show what differs between two runs.

    Reports differences, never verdicts: `evalstand` has no significance
    testing, so it cannot tell a real change from noise.
    """
    from rich.console import Console

    from evalstand.comparison import compare_runs
    from evalstand.reporting.console import render_comparison
    from evalstand.storage import DatabaseTooNewError, RunStore

    console = Console()

    try:
        store = RunStore(database) if database is not None else RunStore()
    except DatabaseTooNewError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]could not open the history database: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    with store:
        before, after = store.load_run(run_a), store.load_run(run_b)
        missing = [run_id for run_id, run in ((run_a, before), (run_b, after)) if run is None]
        if missing:
            console.print(f"no run with id [bold]{', '.join(missing)}[/bold].")
            console.print("run [bold]evalstand history[/bold] to see what is recorded.")
            raise typer.Exit(code=1)

        assert before is not None and after is not None
        comparison = compare_runs(
            before,
            after,
            # Passed always, because without them every edited case would be
            # reported as evidence about the task.
            hashes_before=store.case_hashes(before.id),
            hashes_after=store.case_hashes(after.id),
        )

    console.print(render_comparison(comparison))


@app.command()
def version() -> None:
    """Print the installed version."""
    from evalstand import __version__

    typer.echo(__version__)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()

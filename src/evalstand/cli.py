"""The command line interface.

Phase 2 ships only `run`, which is what the phase's exit criterion requires. It
delegates to pytest rather than reimplementing collection: the plugin already
knows how to find and execute evals, and a second execution path would be a
second thing to keep correct.

`history`, `show`, and `compare` arrive in Phase 5; `watch` — the live view and
file-change re-runs — in Phase 6.
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

EXIT_NOTHING_TO_COMPARE = 2
"""`compare` was asked about runs it cannot compare.

Deliberately the same number `run` uses for an execution error, because it is
the same message: no measurement exists. Exit 1 is reserved for "we measured,
and the answer is worse than your bar" — a build that cannot tell those apart
sends whoever reads it to investigate the model when the real problem is that
nothing ran.
"""


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
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error",
            help="Exit 2 when any case or scorer errored, whatever the means say.",
        ),
    ] = False,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Summary format: 'terminal' (default) or 'markdown' for a PR comment.",
        ),
    ] = "terminal",
) -> None:
    """Run evals and print a summary.

    Exit codes, which `docs/ci.md` documents and a CI job can rely on:
    `0` everything met its bar, `1` an eval fell below `--threshold`,
    `2` something did not run (with `--fail-on-error`).
    """
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
    if fail_on_error:
        args.append("--fail-on-error")
    if output != "terminal":
        args += ["--output", output]

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

    Exits **2** when there is nothing to compare — a run id that is not
    recorded, or one whose Batch never finished. That is the same code `run`
    uses for "something did not run", and for the same reason: a CI job that
    got 1 could not tell "these runs differ badly" from "there was no
    measurement here at all", and those send a reader to different places.
    """
    from rich.console import Console

    from evalstand.comparison import NotComparableError, compare_runs, refuse_partial_runs
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
            raise typer.Exit(code=EXIT_NOTHING_TO_COMPARE)

        assert before is not None and after is not None

        # Refused rather than warned about. The membership note only fires when
        # the two runs cover different cases, so a batch cancelled *after* every
        # case had scored compared clean and printed "nothing differs between
        # these runs" — a partial run presented as a complete one.
        try:
            refuse_partial_runs((before, after), store.batch_for)
        except NotComparableError as exc:
            console.print(f"[red]{exc}[/red]")
            console.print("run [bold]evalstand history[/bold] to see what is comparable.")
            raise typer.Exit(code=EXIT_NOTHING_TO_COMPARE) from exc

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
def watch(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Eval files or directories. Defaults to the current directory."),
    ] = None,
    eval_name: Annotated[
        str | None,
        typer.Option("--eval", help="Which eval to watch, when a path declares several."),
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
    once: Annotated[
        bool,
        typer.Option("--once", help="Open the live view without watching for changes."),
    ] = False,
    store: Annotated[
        bool,
        typer.Option("--store", help="Record these runs in the history database."),
    ] = False,
) -> None:
    """Run one eval in a live view, re-running when files change.

    Not recorded by default, which is the opposite of `run`. Watch mode exists
    to be used *while editing*, so the tree is dirty by construction and every
    run would be tied to a commit whose code it did not reflect. Filling history
    with dozens of unreproducible runs would bury the deliberate ones the
    feature is measured against — so persistence is opt-in with `--store`,
    where the user is choosing it rather than getting it by default.
    """
    from rich.console import Console

    from evalstand.loading import NoEvalsFoundError, load_evals, select_eval
    from evalstand.recording import open_recorder
    from evalstand.runner import RunConfig
    from evalstand.tui.app import EvalApp
    from evalstand.tui.watch import watch_roots

    console = Console()
    targets = list(paths or [Path()])

    try:
        declared = select_eval(load_evals(targets), eval_name)
    except NoEvalsFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        # An eval file that raises on import. Reported plainly rather than
        # inside a terminal UI that would then have to be dismissed.
        console.print(f"[red]could not load evals: {type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        expected = len(declared.load_cases()) * declared.repeat
    except Exception as exc:
        console.print(f"[red]could not load cases for {declared.name!r}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    recorder = None
    if store:
        try:
            # `allow_dirty` because watch mode is *for* editing. The batch still
            # records that the tree was dirty, so no reader is misled into
            # thinking the run can be reproduced from its commit.
            recorder = open_recorder(allow_dirty=True)
        except Exception as exc:
            console.print(f"[red]could not open the history database: {exc}[/red]")
            raise typer.Exit(code=1) from exc

    config = RunConfig(
        concurrency=concurrency if concurrency is not None else RunConfig().concurrency,
        timeout_seconds=timeout,
        bypass_cache=no_cache,
    )

    EvalApp(
        declared,
        config=config,
        expected=expected,
        watch=not once,
        watch_roots=watch_roots(targets),
        recorder=recorder,
    ).run()

    if recorder is not None:
        recorder.finish()


@app.command()
def version() -> None:
    """Print the installed version."""
    from evalstand import __version__

    typer.echo(__version__)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()

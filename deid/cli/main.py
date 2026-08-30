from __future__ import annotations

import sys
import warnings

import typer

from deid.cli import commands

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = typer.Typer(
    name="deid",
    help="De-Identification Toolkit — run pipeline or explore results.",
    pretty_exceptions_enable=False,
)

app.add_typer(commands.run_app, name="run")
app.add_typer(commands.list_app, name="list")
app.add_typer(commands.select_app, name="select")

app.command(name="show")(commands.cmd_show)
app.command(name="migrate")(commands.cmd_migrate)
app.command(name="migrate-structure")(commands.cmd_migrate_structure)
app.command(name="explore")(commands.cmd_explore)

from deid.cli import verify as verify_cmd
app.command(name="verify")(verify_cmd.main)

from deid.cli import serve
app.add_typer(serve.app, name="serve")


def version_callback(value: bool) -> None:
    if value:
        from deid import __version__
        typer.echo(f"deid-toolkit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", callback=version_callback, is_eager=True
    ),
) -> None:
    ctx.obj = {}


if __name__ == "__main__":
    app()

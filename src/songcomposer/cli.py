import typer

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Songwriter and proof-of-concept generator. Each stage writes an inspectable file.")


@app.callback()
def _main() -> None:
    """Stages: ingest → analyze → brief → compose → generate → pick → chart (or `run` for all)."""

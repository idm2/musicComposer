import typer

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Songwriter and proof-of-concept generator. Each stage writes an inspectable file.")


@app.callback()
def _main() -> None:
    """Stages: ingest → analyze → brief → compose → generate → pick → chart (or `run` for all)."""


@app.command()
def ingest(song: str,
           from_: str = typer.Option(..., "--from", help="URL, audio file or video file"),
           force: bool = typer.Option(False, "--force", help="re-ingest even if 00-source.wav exists")) -> None:
    """Reference -> work/<song>/00-source.wav + 00-source.json."""
    from .ingest import run_ingest
    run_ingest(song, from_, force=force)

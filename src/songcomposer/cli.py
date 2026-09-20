import sys

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Songwriter and proof-of-concept generator. Each stage writes an inspectable file.")


def _force_utf8_console() -> None:
    """Windows consoles default stdout/stderr to cp1252, which raises UnicodeEncodeError
    on the non-ASCII characters (→ “ ” — × …) later stages print. Force UTF-8."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.callback()
def _main() -> None:
    """Stages: ingest → analyze → brief → compose → generate → pick → chart (or `run` for all)."""
    _force_utf8_console()


@app.command()
def ingest(song: str,
           from_: str = typer.Option(..., "--from", help="URL, audio file or video file"),
           force: bool = typer.Option(False, "--force", help="re-ingest even if 00-source.wav exists")) -> None:
    """Reference -> work/<song>/00-source.wav + 00-source.json."""
    from .ingest import run_ingest
    run_ingest(song, from_, force=force)


@app.command()
def analyze(song: str,
            ears: str = typer.Option("", "--ears", help="comma list; default = every ear installed")) -> None:
    """Two-ear analysis of the reference → work/<song>/01-analysis.json."""
    from .analysis import analyze as run
    from .config import load_config
    from .jsonio import write_model
    from .paths import SongPaths
    p = SongPaths(song)
    wav = p.require(p.source_wav, "ingest")
    chosen = {e.strip() for e in ears.split(",") if e.strip()} or None
    write_model(p.analysis, run(wav, p.cache, load_config(), chosen))
    print(f"  → {p.analysis}")


@app.command()
def brief(song: str,
          from_: str = typer.Option(None, "--from", help="markdown/text file"),
          text: str = typer.Option(None, "--text", help="inline brief or one-liner")) -> None:
    """Your brief → work/<song>/02-brief.json."""
    from .brief import run_brief
    run_brief(song, from_file=from_, text=text)

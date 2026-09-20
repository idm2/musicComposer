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


def main() -> None:
    """Console-script entry point. Forces UTF-8 before Typer renders anything (--help is an
    eager option that runs before the callback), and turns expected failures (missing
    upstream files, bad input, pre-flight rejections) into a one-line message instead of
    a raw traceback."""
    _force_utf8_console()
    try:
        app()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)


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


@app.command()
def compose(song: str, force: bool = typer.Option(False, "--force", help="rewrite an existing spec")) -> None:
    """Brief + analysis → lyrics and song spec (work/<song>/03-spec.json). Edit the file freely afterwards."""
    from .compose import run_compose
    run_compose(song, force=force)


@app.command()
def generate(song: str,
             provider: str = typer.Option(None, "--provider", help="suno | elevenlabs (default: songcomposer.toml)"),
             regen: bool = typer.Option(False, "--regen", help="pay for fresh takes from this provider"),
             fidelity: str = typer.Option(None, "--fidelity", help="loose | medium | close (asked if omitted)"),
             note: str = typer.Option("", "--note", help="extra direction for this run")) -> None:
    """Spec → takes. Validates, prints estimated cost, and waits for an explicit 'yes' before spending."""
    from .generate import run_generate
    run_generate(song, provider_name=provider, regen=regen, fidelity=fidelity, note=note)


@app.command()
def pick(song: str, take: int = typer.Option(..., "--take", help="take number from 04-takes/")) -> None:
    """Choose a take → 05-chosen.json and out/<song>/<song>.mp3."""
    from .pick import run_pick
    run_pick(song, take)

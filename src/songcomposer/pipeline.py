"""Stage: run — the whole pipeline. Every stage is a no-op if its output exists, so re-running resumes."""
from typing import Callable

from .analysis import analyze
from .brief import run_brief
from .chart import run_chart
from .compose import run_compose
from .config import load_config
from .generate import run_generate
from .ingest import run_ingest
from .jsonio import read_json, write_model
from .models import TakesManifest
from .paths import SongPaths
from .pick import run_pick


def run_all(song: str, src: str, brief_file: str | None = None, brief_text: str | None = None,
            provider_name: str | None = None, fidelity: str | None = None,
            input_fn: Callable[[str], str] = input, provider=None) -> None:
    paths = SongPaths(song)
    run_ingest(song, src)
    if not paths.analysis.exists():
        write_model(paths.analysis, analyze(paths.source_wav, paths.cache, load_config()))
    if not paths.brief.exists():
        run_brief(song, from_file=brief_file, text=brief_text)
    run_compose(song)
    run_generate(song, provider_name=provider_name, fidelity=fidelity, input_fn=input_fn, provider=provider)
    if not paths.chosen.exists():
        takes = TakesManifest(**read_json(paths.takes_json)).all_takes()
        print("Listen to the takes in", paths.takes_dir)
        for t in takes:
            print(f"  take {t.index}: {t.provider}, {t.duration_s:.0f}s")
        run_pick(song, int(input_fn("Which take? ").strip()))
    run_chart(song)

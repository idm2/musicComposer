"""Stage: chart. Transcribe the chosen take, then render every guitar deliverable into out/<song>/."""
from pathlib import Path

from .jsonio import read_json, write_json
from .models import LOW_CONFIDENCE, SongSpec
from .paths import SongPaths
from .render.chordsheet import render_chordsheet
from .render.lilypond import write_pdf
from .render.lyricsjson import build_lyrics_json
from .render.musicxml import write_musicxml
from .render.tab import render_tab
from .transcribe import run_transcribe


def run_chart(song: str, force: bool = False) -> dict[str, Path]:
    paths = SongPaths(song)
    title = SongSpec(**read_json(paths.require(paths.spec, "compose"))).title
    t = run_transcribe(song, force=force)
    paths.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def attempt(name: str, path: Path, render) -> None:
        """A renderer that raises must not take the rest of the stage down with it — a chord chart
        derived from probabilistic transcription data is genuinely fallible (music21 especially).
        Print an actionable line, leave `written` (and the file) alone, and let the caller carry on."""
        try:
            render()
            written[name] = path
        except Exception as e:  # noqa: BLE001 — deliberately broad: any renderer may raise on messy data
            print(f"! {name} failed: {e} — the other deliverables were still written")

    attempt("chords", paths.chords_txt,
            lambda: paths.chords_txt.write_text(render_chordsheet(title, t), encoding="utf-8"))
    attempt("tab", paths.tab_txt,
            lambda: paths.tab_txt.write_text(render_tab(title, t), encoding="utf-8"))
    attempt("lyrics", paths.lyrics_json,
            lambda: write_json(paths.lyrics_json, build_lyrics_json(song, title, t)))

    if len(t.analysis.beats) >= 2:
        attempt("musicxml", paths.out_musicxml, lambda: write_musicxml(title, t, paths.out_musicxml))
        if write_pdf(title, t, paths.chart_ly, paths.chart_pdf):
            written["pdf"] = paths.chart_pdf
    else:
        print("! no beat grid detected — skipped MusicXML and PDF (they need bar positions)")

    chords = t.analysis.chords
    low = sum(1 for c in chords if c.confidence < LOW_CONFIDENCE)
    unaligned = sum(1 for l in t.lines if l.start is None)
    print(f"chart: {len(chords)} chords ({low} low-confidence, marked ?), "
          f"{len(t.lines) - unaligned}/{len(t.lines)} lyric lines timed")
    for name, path in written.items():
        print(f"  {name:<9}→ {path}")
    return written

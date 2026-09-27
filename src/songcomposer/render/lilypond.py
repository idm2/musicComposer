"""chart.pdf — chord names, fretboard diagrams, melody, lyrics and tab, engraved by LilyPond from text we generate."""
import shutil
import subprocess
from pathlib import Path

from ..models import LOW_CONFIDENCE, Chord, Transcription
from ..chordsym import pitch_class
from . import guitar
from .guitar import into_range
from .leadsheet import chord_events, melody_events, melody_selection_note, selected_melody
from .timing import BeatMap, split_at_bars, split_sixteenths

SHARP = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
FLAT = ["c", "des", "d", "ees", "e", "f", "ges", "g", "aes", "a", "bes", "b"]
DUR16 = {16: "1", 12: "2.", 8: "2", 6: "4.", 4: "4", 3: "8.", 2: "8", 1: "16"}
QUALITY = {"maj": "", "min": ":m", "dim": ":dim", "aug": ":aug", "5": ":1.5", "1": ":1.5", "7": ":7", "maj7": ":maj7",
           "min7": ":m7", "minmaj7": ":m7+", "dim7": ":dim7", "hdim7": ":m7.5-", "6": ":6", "maj6": ":6", "min6": ":m6",
           "9": ":9", "maj9": ":maj9", "min9": ":m9", "11": ":11", "min11": ":m11", "13": ":13", "maj13": ":maj13",
           "min13": ":m13", "sus2": ":sus2", "sus4": ":sus4"}


def _name(note_name: str) -> str:
    return (FLAT if "b" in note_name[1:] else SHARP)[pitch_class(note_name)]


def ly_pitch(midi: int, flats: bool) -> str:
    octave = midi // 12 - 4
    return (FLAT if flats else SHARP)[midi % 12] + ("'" * octave if octave > 0 else "," * -octave)


def ly_chord(chord: Chord, beats: int) -> str:
    out = f"{_name(chord.root)}{DUR16[beats * 4]}{QUALITY.get(chord.quality, '')}"
    return out + (f"/{_name(chord.bass)}" if chord.bass else "")


def diagram_frets(chord: Chord) -> str | None:
    """The tab's voicing for this chord as a LilyPond fret string ("x;0;2;2;2;0;"). None when there is no shape."""
    frets = guitar.voicing(chord)
    return "".join(f"{f};" for f in frets) if frets else None


def _diagrams(t: Transcription, beatmap: BeatMap) -> str:
    """Every chord's diagram once, in a row above the music (not a tiny box over every change)."""
    cells, seen = [], set()
    for e in chord_events(t, beatmap):
        if e.chord is None or e.chord.symbol in seen:
            continue
        frets = diagram_frets(e.chord)
        if not frets:
            continue
        seen.add(e.chord.symbol)
        terse = ";".join("o" if f == "0" else f for f in frets.rstrip(";").split(";")) + ";"
        cells.append(f'\\center-column {{ \\bold {_quote(e.chord.symbol)} \\fret-diagram-terse #"{terse}" }}')
    if not cells:
        return ""
    return "\\markup { \\override #'(baseline-skip . 2) \\fill-line { " + " ".join(cells) + " } }"


def _uses_flats(key_name: str) -> bool:
    tonic, _, mode = key_name.partition(" ")
    if "b" in tonic[1:]:
        return True
    return tonic == "F" if mode == "major" else tonic in ("D", "G", "C", "F")


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _melody(t: Transcription, beatmap: BeatMap, flats: bool) -> tuple[str, str]:
    music, words = [], []
    for e in melody_events(t, beatmap):
        pieces = [(s, n) for s, length in split_at_bars(e.start16, e.len16, beatmap.bar16) for n in split_sixteenths(length)]
        for i, (_, n) in enumerate(pieces):
            if e.pitch is None:
                music.append(f"r{DUR16[n]}")
                continue
            head = "\\parenthesize " if e.low and i == 0 else ""
            tie = "~" if i + 1 < len(pieces) else ""
            music.append(f"{head}{ly_pitch(into_range(e.pitch), flats)}{DUR16[n]}{tie}")
        if e.pitch is not None:
            words.append(_quote(e.lyric) if e.lyric else "_")
    return " ".join(music), " ".join(words)


def _harmonies(t: Transcription, beatmap: BeatMap, mark_doubt: bool) -> str:
    out = []
    for e in chord_events(t, beatmap):
        for i, (_, beats16) in enumerate(split_at_bars(e.start_beat * 4, e.beats * 4, beatmap.bar16)):
            for n in split_sixteenths(beats16):
                if e.chord is None:
                    out.append(f"s{DUR16[n]}")
                    continue
                if mark_doubt and e.chord.confidence < LOW_CONFIDENCE:
                    out.append("\\once \\override ChordName.color = #grey")
                out.append(ly_chord(e.chord, 1).replace("4", DUR16[n], 1))
    return " ".join(out)


def build_ly(title: str, t: Transcription) -> str:
    a = t.analysis
    beatmap = BeatMap.from_analysis(a)
    g = a.global_info
    flats = _uses_flats(g.key) if g and g.key is not None else False
    music, words = _melody(t, beatmap, flats)
    setup = []
    if g and g.key is not None:
        tonic, _, mode = g.key.partition(" ")
        if mode in ("major", "minor"):
            setup.append(f"\\key {_name(tonic)} \\{mode}")
    if g and g.time_signature is not None:
        setup.append(f"\\time {g.time_signature}")
    if g and g.tempo_bpm is not None:
        setup.append(f"\\tempo 4 = {round(g.tempo_bpm)}")
    _, kept, total = selected_melody(t)
    tagline = ("Grey chord names and parenthesised notes are low-confidence detections. Melody is shown in "
              f"guitar range. {melody_selection_note(kept, total)}.")
    return f"""\\version "2.24.0"
\\header {{
  title = {_quote(title)}
  composer = {_quote(f"Song Composer — transcribed from take {t.take}")}
  tagline = {_quote(tagline)}
}}
{_diagrams(t, beatmap)}
harmonies = \\chordmode {{ {_harmonies(t, beatmap, True)} }}
melody = {{ {' '.join(setup)} {music} }}
words = \\lyricmode {{ {words} }}
\\score {{
  <<
    \\new ChordNames {{ \\set chordChanges = ##t \\harmonies }}
    \\new Staff {{ \\clef "treble_8" \\new Voice = "mel" {{ \\melody }} }}
    \\new Lyrics \\lyricsto "mel" {{ \\words }}
    \\new TabStaff {{ \\melody }}
  >>
  \\layout {{ }}
}}
"""


def write_pdf(title: str, t: Transcription, ly_path: Path, pdf_path: Path) -> bool:
    ly_path.parent.mkdir(parents=True, exist_ok=True)
    ly_path.write_text(build_ly(title, t), encoding="utf-8")
    if shutil.which("lilypond") is None:
        print(f"! lilypond not found — wrote {ly_path} but no PDF. Install with `scoop install lilypond` and re-run `chart`.")
        return False
    proc = subprocess.run(["lilypond", "-o", str(pdf_path.with_suffix("")), str(ly_path)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not pdf_path.exists():
        print(f"! lilypond failed — {ly_path} is kept for inspection:\n{proc.stderr[-800:]}")
        return False
    return True
